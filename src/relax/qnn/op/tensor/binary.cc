/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file binary.cc
 * \brief QNN Binary operators implementation
 */

#include "binary.h"
#include "tvm/relax/block_builder.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/struct_info.h"
#include "tvm/runtime/data_type.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/stmt.h"
#include "tvm/relax/qnn/attrs.h"

// Forward declaration of layout inference function
namespace tvm {
namespace relax {

// Function defined in /src/relax/op/tensor/binary.cc
extern InferLayoutOutput InferLayoutBinaryEwise(
    const Call& call,
    const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map);

}  // namespace relax
}  // namespace tvm

namespace tvm {
namespace relax {
namespace qnn {

// BroadcastAttrs reflection registration
TVM_FFI_STATIC_INIT_BLOCK() { BroadcastAttrs::RegisterReflection(); }

template <typename FType>
StructInfo InferStructInfoQnnBinary(const Call& call, const BlockBuilder& ctx,
                                    FType f_compute_out_dtype) {
  const auto* attrs = call->attrs.as<BroadcastAttrs>();
  ICHECK(attrs) << "Attributes for QNN binary op are missing.";
  int l_axis = attrs->lhs_axis;
  int r_axis = attrs->rhs_axis;

  // Validation helper for per-tensor quantization
  auto check_per_tensor = [&](const TensorStructInfo& sinfo, const std::string& name) {
    if (sinfo->ndim != 0) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "QNN binary op " << name
                       << " must be a scalar (ndim=0) for per-tensor quantization.");
    }
  };

  // Validate all scale and zero-point tensors are scalars
  check_per_tensor(GetInputTensorStructInfo(call, 2, ctx), "lhs_scale");
  check_per_tensor(GetInputTensorStructInfo(call, 3, ctx), "lhs_zero_point");
  check_per_tensor(GetInputTensorStructInfo(call, 4, ctx), "rhs_scale");
  check_per_tensor(GetInputTensorStructInfo(call, 5, ctx), "rhs_zero_point");
  check_per_tensor(GetInputTensorStructInfo(call, 6, ctx), "output_scale");
  check_per_tensor(GetInputTensorStructInfo(call, 7, ctx), "output_zero_point");
  // Per Tensor Quantization Validation
  ICHECK(GetInputTensorStructInfo(call, 2, ctx)->dtype.is_float()) << "LHS scale must be float.";
  ICHECK(GetInputTensorStructInfo(call, 4, ctx)->dtype.is_float()) << "RHS scale must be float.";
  ICHECK(GetInputTensorStructInfo(call, 6, ctx)->dtype.is_float()) << "Output scale must be float.";

  // Currently only per-tensor quantization is supported
  if (l_axis != -1 || r_axis != -1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "The current implementation only supports per-tensor quantization "
                        "(axis=-1).");
  }

  // Compute output dtype using the provided function
  DataType out_dtype = f_compute_out_dtype(call, ctx,
    GetInputTensorStructInfo(call, 0, ctx),
    GetInputTensorStructInfo(call, 1, ctx),
    GetInputTensorStructInfo(call, 7, ctx));

  // Compute output shape using broadcast rules
  auto output_shape = InferBinaryBroadcastShape(call, ctx,
    (GetInputTensorStructInfo(call, 0, ctx)->GetShape()).value(),
    (GetInputTensorStructInfo(call, 1, ctx)->GetShape()).value());

  if (!output_shape.defined()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Cannot infer shape for QNN binary broadcast operator.");
  }

  ShapeExpr output_shape_expr = ShapeExpr(output_shape.value());

  // Maintain device consistency
  VDevice vdev = GetInputTensorStructInfo(call, 0, ctx)->vdevice.value_or(VDevice());

  return TensorStructInfo(output_shape_expr, out_dtype, vdev);
}

StructInfo InferStructInfoQnnBinaryArith(const Call& call, const BlockBuilder& ctx) {
  return InferStructInfoQnnBinary(
      call, ctx,
      [](const Call& call, const BlockBuilder& ctx, const TensorStructInfo& lhs_sinfo,
         const TensorStructInfo& rhs_sinfo, const TensorStructInfo& out_zp_sinfo) {
        // For arithmetic operations, output dtype is determined by output zero point
        return out_zp_sinfo->dtype;
      });
}


/***************** Operator Call Functions *****************/
#define RELAX_QNN_BINARY_ARITH_OP(OpName)                                                     \
  Expr OpName(Expr lhs, Expr rhs, Expr lhs_scale, Expr lhs_zero_point, Expr rhs_scale,       \
              Expr rhs_zero_point, Expr output_scale, Expr output_zero_point, int lhs_axis,   \
              int rhs_axis) {                                                                  \
    static const Op& op = Op::Get("relax.qnn." #OpName);                                      \
    auto attrs = ffi::make_object<BroadcastAttrs>();                                        \
    attrs->lhs_axis = lhs_axis;                                                               \
    attrs->rhs_axis = rhs_axis;                                                               \
    return Call(op,                                                                           \
                {std::move(lhs), std::move(rhs), std::move(lhs_scale),                        \
                 std::move(lhs_zero_point), std::move(rhs_scale), std::move(rhs_zero_point),  \
                 std::move(output_scale), std::move(output_zero_point)},                      \
                Attrs(attrs), {});                                                            \
  }                                                                                           \
  TVM_FFI_STATIC_INIT_BLOCK() {                                                               \
    namespace refl = tvm::ffi::reflection;                                                    \
    refl::GlobalDef().def("relax.qnn.op." #OpName, OpName);                                   \
  }                                                                                           \
  TVM_REGISTER_OP("relax.qnn." #OpName)                                                       \
      .set_num_inputs(kNumQnnBinaryOpInputs)                                                  \
      .add_argument("lhs", "Tensor", "The left hand side quantized tensor.")                  \
      .add_argument("rhs", "Tensor", "The right hand side quantized tensor.")                 \
      .add_argument("lhs_scale", "Tensor", "Scale for dequantizing the LHS tensor.")          \
      .add_argument("lhs_zero_point", "Tensor",                                               \
                    "Zero point for dequantizing the LHS tensor.")                            \
      .add_argument("rhs_scale", "Tensor", "Scale for dequantizing the RHS tensor.")          \
      .add_argument("rhs_zero_point", "Tensor",                                               \
                    "Zero point for dequantizing the RHS tensor.")                            \
      .add_argument("output_scale", "Tensor", "Scale for quantizing the output tensor.")      \
      .add_argument("output_zero_point", "Tensor",                                            \
                    "Zero point for quantizing the output tensor.")                           \
      .set_attrs_type<BroadcastAttrs>()                                                     \
      .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoQnnBinaryArith)          \
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)               \
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow) \
      .set_attr<Bool>("FPurity", Bool(true))

// Define and register all operations
RELAX_QNN_BINARY_ARITH_OP(add);
RELAX_QNN_BINARY_ARITH_OP(subtract);
RELAX_QNN_BINARY_ARITH_OP(multiply);

}  // namespace qnn
}  // namespace relax
}  // namespace tvm
