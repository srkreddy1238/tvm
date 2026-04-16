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
 * \file dense.cc
 * \brief QNN Dense operator implementation
 */

#include "linear_algebra.h"

#include <string>
#include <utility>

#include "tvm/relax/attrs/linear_algebra.h"
#include "tvm/relax/block_builder.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/qnn/attrs.h"
#include "tvm/relax/struct_info.h"
#include "tvm/runtime/data_type.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace relax {
namespace qnn {

StructInfo InferStructInfoQnnDense(const Call& call, const BlockBuilder& ctx) {
  const auto* attrs = call->attrs.as<MatmulAttrs>();
  TVM_FFI_ICHECK(attrs) << "Attributes for QNN dense op are missing.";

  TensorStructInfo data_sinfo = GetInputTensorStructInfo(call, 0, ctx);
  TensorStructInfo weight_sinfo = GetInputTensorStructInfo(call, 1, ctx);

  auto check_per_tensor = [&](const TensorStructInfo& sinfo, const std::string& name) {
    if (sinfo->ndim != 0) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "QNN dense op " << name
                       << " must be a scalar (ndim=0) for per-tensor quantization.");
    }
  };

  check_per_tensor(GetInputTensorStructInfo(call, 3, ctx), "input_zero_point");
  check_per_tensor(GetInputTensorStructInfo(call, 5, ctx), "kernel_zero_point");
  check_per_tensor(GetInputTensorStructInfo(call, 2, ctx), "input_scale");
  check_per_tensor(GetInputTensorStructInfo(call, 4, ctx), "kernel_scale");

  TVM_FFI_ICHECK(GetInputTensorStructInfo(call, 2, ctx)->dtype.is_float())
      << "Input scale must be float.";
  TVM_FFI_ICHECK(GetInputTensorStructInfo(call, 4, ctx)->dtype.is_float())
      << "Kernel scale must be float.";

  TVM_FFI_ICHECK(data_sinfo->ndim >= 1 && data_sinfo->ndim <= 2)
      << "QNN dense expects data to be 1-D or 2-D, but got " << data_sinfo->ndim;
  TVM_FFI_ICHECK(weight_sinfo->ndim == 2)
      << "QNN dense expects weight to be 2-D, but got " << weight_sinfo->ndim;

  return InferStructInfoMatmul(call, ctx);
}

// relax.qnn.dense
Expr dense(Expr data, Expr weight, Expr input_scale, Expr input_zero_point, Expr kernel_scale,
           Expr kernel_zero_point, DataType out_dtype) {
  static const Op& op = Op::Get("relax.qnn.dense");
  auto attrs = ffi::make_object<MatmulAttrs>();
  attrs->out_dtype = out_dtype;
  return Call(op,
              {std::move(data), std::move(weight), std::move(input_scale),
               std::move(input_zero_point), std::move(kernel_scale), std::move(kernel_zero_point)},
              Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.qnn.op.dense", dense);
}

TVM_REGISTER_OP("relax.qnn.dense")
    .set_num_inputs(6)
    .add_argument("data", "Tensor", "The input quantized tensor.")
    .add_argument("weight", "Tensor", "The weight quantized tensor (already transposed).")
    .add_argument("input_scale", "Tensor", "Scale for the input tensor (scalar float).")
    .add_argument("input_zero_point", "Tensor", "Zero point for the input tensor (scalar).")
    .add_argument("kernel_scale", "Tensor", "Scale for the weight/kernel tensor (scalar float).")
    .add_argument("kernel_zero_point", "Tensor",
                  "Zero point for the weight/kernel tensor (scalar).")
    .set_attrs_type<MatmulAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoQnnDense)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kNever)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace qnn
}  // namespace relax
}  // namespace tvm
