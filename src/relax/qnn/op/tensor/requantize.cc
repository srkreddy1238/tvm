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
 * \file src/relax/op/qnn/requantize.cc
 * \brief QNN Requantize operator implementation
 */

#include "./requantize.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace tvm {
namespace relax {
namespace qnn {

Expr requantize(Expr data, Expr input_scale, Expr input_zero_point, Expr output_scale,
                Expr output_zero_point, int axis, DataType out_dtype) {
  ObjectPtr<QuantizeAttrs> attrs = tvm::ffi::make_object<QuantizeAttrs>();
  attrs->axis = axis;
  attrs->out_dtype = out_dtype;

  static const Op& op = Op::Get("relax.qnn.requantize");
  return Call(op, {data, input_scale, input_zero_point, output_scale, output_zero_point},
              Attrs(attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.qnn.op.requantize", requantize);
}

StructInfo InferStructInfoRequantize(const Call& call, const BlockBuilder& ctx) {
  // Expect: data, input_scale, input_zero_point, output_scale, output_zero_point
  if (call->args.size() != 5) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "qnn.requantize expects 5 arguments "
                        "(data, input_scale, input_zero_point, output_scale, output_zero_point). "
                        "However, got "
                     << call->args.size());
  }
  // Attributes check
  const auto* attrs = call->attrs.as<QuantizeAttrs>();
  if (attrs == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call) << "qnn.requantize is missing QuantizeAttrs.");
  }
  auto args = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo data_sinfo = args[0];
  TensorStructInfo iscale_sinfo = args[1];
  TensorStructInfo izp_sinfo = args[2];
  TensorStructInfo oscale_sinfo = args[3];
  TensorStructInfo ozp_sinfo = args[4];

  // Dtype check
  const std::string op_name = "relax.qnn.requantize";
  CheckIntegerInputDtype(call, ctx, data_sinfo, op_name);
  // Scales must be floating-point
  CheckScaleDtype(call, ctx, iscale_sinfo, op_name, "input_scale");
  CheckScaleDtype(call, ctx, oscale_sinfo, op_name, "output_scale");
  // Zero points must be integer types
  CheckZeroPointDtype(call, ctx, izp_sinfo, op_name, "input zero point");
  CheckZeroPointDtype(call, ctx, ozp_sinfo, op_name, "output zero point");

  // Normalize Axis
  int ndim = data_sinfo->ndim, axis = 0;
  if (ndim == kUnknownNDim || ndim == 0) {
    axis = 0;
  } else {
    axis = attrs->axis;
    if (axis < 0) {
      axis += ndim;
    }
    if (axis < 0 || axis >= ndim) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "relax.qnn.requantize: axis param is out of range. Got axis="
                       << attrs->axis << ", but input tensor has ndim=" << ndim);
    }
  }

  // Param check
  auto check_param_size = [&](const TensorStructInfo& param_sinfo, const char* param_name) {
    if (!data_sinfo->shape.defined() || !param_sinfo->shape.defined()) return;

    const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
    const auto* param_shape = param_sinfo->shape.as<ShapeExprNode>();
    if (!data_shape || !param_shape) return;

    // For scalar tensor no need of checking
    if (IsScalarTensor(param_sinfo)) return;
    // Only 1D per-axis parameters are supported
    if (param_sinfo->ndim != 1 || param_shape->values.size() != 1) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "relax.qnn.requantize: " << param_name
                       << " must be either a scalar or a 1D tensor. However, got ndim="
                       << param_sinfo->ndim);
    }
    if (data_sinfo->IsUnknownNdim() || axis < 0 || axis >= data_sinfo->ndim) {
      return;
    }

    const PrimExpr& param_dim = param_shape->values[0];
    const PrimExpr& data_dim = data_shape->values[axis];
    if (!ctx->GetAnalyzer()->CanProveEqual(param_dim, data_dim)) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "relax.qnn.requantize: Size mismatch at axis " << axis
                       << ". The input tensor shape at this axis is '" << data_dim
                       << "', but size of " << param_name << " is '" << param_dim << "'");
    }
  };
  check_param_size(iscale_sinfo, "input_scale");
  check_param_size(izp_sinfo, "input_zero_point");
  check_param_size(oscale_sinfo, "output_scale");
  check_param_size(ozp_sinfo, "output_zero_point");

  ffi::Optional<VDevice> vdevice =
      InferBinaryArithOpOutVDevice(call, ctx, data_sinfo, iscale_sinfo);
  vdevice = InferBinaryArithOpOutVDevice(call, ctx, data_sinfo, oscale_sinfo);

  // Requantize preserves shape, vdevice only dtype changes
  auto out_sinfo = ffi::make_object<TensorStructInfoNode>(*data_sinfo.get());
  out_sinfo->dtype = attrs->out_dtype;
  out_sinfo->vdevice = vdevice;  // Explicitly set the consistent vdevice
  return TensorStructInfo(out_sinfo);
}

InferLayoutOutput InferLayoutRequantize(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  // No special desired layout for this op
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  LayoutDecision data_layout = GetLayoutDecision(var_layout_map, call->args[0]);

  // Requantize is elementwise: output layout == input data layout
  return InferLayoutOutput({data_layout}, {data_layout}, call->attrs);
}

TVM_REGISTER_OP("relax.qnn.requantize")
    .set_attrs_type<QuantizeAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("input_scale", "Tensor", "Input scale.")
    .add_argument("input_zero_point", "Tensor", "Input zero point.")
    .add_argument("output_scale", "Tensor", "Output scale.")
    .add_argument("output_zero_point", "Tensor", "Output zero point.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoRequantize)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutRequantize)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace qnn
}  // namespace relax
}  // namespace tvm
