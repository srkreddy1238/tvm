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
 * \file src/relax/op/nn/convolution.cc
 * \brief Convolution operators
 */

#include "convolution.h"

#include <tvm/ffi/reflection/registry.h>

#include <cstddef>
#include <vector>

#include "../../../op/nn/convolution.h"
#include "tvm/relax/block_builder.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/struct_info.h"
#include "tvm/runtime/data_type.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace relax {
namespace qnn {

/* relax.qnn.conv2d */
Expr conv2d(Expr data, Expr weight, Expr input_zero_pt, Expr weight_zero_pt,
            ffi::Optional<Expr> input_scale, ffi::Optional<Expr> weight_scale,
            ffi::Array<int64_t> strides, ffi::Array<int64_t> padding, ffi::Array<int64_t> dilation,
            int groups, ffi::String data_layout, ffi::String kernel_layout,
            ffi::Optional<ffi::String> out_layout, DataType out_dtype) {
  padding = GetCompletePadding2D(std::move(padding));
  if (strides.size() == 1) {
    strides.push_back(strides[0]);
  }
  if (dilation.size() == 1) {
    dilation.push_back(dilation[0]);
  }

  TVM_FFI_CHECK_GT(groups, 0, ValueError)
      << "The number of groups in convolution is expected to be positive. However, "
         "the given number of groups is "
      << groups;
  TVM_FFI_CHECK_EQ(strides.size(), 2, ValueError)
      << "The input strides length is expected to be 2. However, the given strides is " << strides;
  TVM_FFI_CHECK_EQ(dilation.size(), 2, ValueError)
      << "The input dilation length is expected to be 2. However, the given dilation is "
      << dilation;

  if (static_cast<bool>(input_scale) ^ static_cast<bool>(weight_scale)) {
    LOG(FATAL) << "Expected both input_scale and weight_scale together be present or absent. Found "
                  "otherwise";
  }

  return MakeConv<Conv2DAttrs>(
      std::move(data), std::move(weight), std::move(input_zero_pt), std::move(weight_zero_pt),
      input_scale, weight_scale, std::move(strides), std::move(padding), std::move(dilation),
      groups, data_layout, std::move(kernel_layout), out_layout.value_or(data_layout), out_dtype,
      /*op_name=*/"relax.qnn.conv2d");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.qnn.op.conv2d", conv2d);
}

StructInfo InferStructInfoQnnConv2d(const Call& call, const BlockBuilder& ctx) {
  size_t num_args = call->args.size();
  TVM_FFI_ICHECK(num_args == 4 || num_args == 6)
      << "Expected Number of Argument to be 4 or 6 arguments but found " << num_args;

  bool has_scale = num_args == 6;
  TensorStructInfo data_zero_pt_sinfo = GetInputTensorStructInfo(call, 2, ctx);
  TensorStructInfo weight_zero_pt_sinfo = GetInputTensorStructInfo(call, 3, ctx);

  TVM_FFI_ICHECK(GetElementDType(data_zero_pt_sinfo) == GetElementDType(weight_zero_pt_sinfo))
      << "Mismatch between Data Zero Point and Weight Zero Point DataType";
  TVM_FFI_ICHECK(data_zero_pt_sinfo.as<TensorStructInfoNode>()->ndim == 0)
      << "Data Zero Point Must be a Scalar";
  TVM_FFI_ICHECK(weight_zero_pt_sinfo.as<TensorStructInfoNode>()->ndim == 0)
      << "Weight Zero Point Must be a Scalar";

  ffi::Optional<TensorStructInfo> data_scale_sinfo, weight_scale_sinfo;
  if (has_scale) {
    data_scale_sinfo = GetInputTensorStructInfo(call, 4, ctx);
    weight_scale_sinfo = GetInputTensorStructInfo(call, 5, ctx);

    TVM_FFI_ICHECK(GetElementDType(data_scale_sinfo.value()) ==
                   GetElementDType(weight_scale_sinfo.value()))
        << "Mismatch between Input and Weight DataType";
    TVM_FFI_ICHECK(data_scale_sinfo.as<TensorStructInfoNode>()->ndim == 0)
        << "Data Scale Must be a Scalar";
    TVM_FFI_ICHECK(weight_scale_sinfo.as<TensorStructInfoNode>()->ndim == 0 ||
                   weight_scale_sinfo.as<TensorStructInfoNode>()->ndim == 1)
        << "Weight Scale Point Must be a Scalar/1-D Tensor";
  }

  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  DataType resolved_out_dtype;
  if (!attrs->out_dtype.is_void()) {
    resolved_out_dtype = attrs->out_dtype;
  } else if (has_scale) {
    resolved_out_dtype = GetElementDType(data_scale_sinfo.value()).value();
  } else {
    resolved_out_dtype = DataType::Int(32);
  }

  if (has_scale && weight_scale_sinfo.as<TensorStructInfoNode>()->ndim == 1) {
    auto [weight_layout, weight2OIHW] =
        CheckTensorLayout(call, ctx, attrs->kernel_layout, "OIHW", "kernel");
    TensorStructInfo weight_sinfo = GetInputTensorStructInfo(call, 1, ctx);
    ffi::Optional<ShapeExpr> weight_shape =
        CheckNdimPerLayoutAndGetShape(call, ctx, weight_sinfo, weight_layout);
    if (weight_shape.defined()) {
      ffi::Array<PrimExpr> weight_OIHW_shape =
          weight2OIHW.ForwardShape(weight_shape.value()->values);
      arith::Analyzer* analyzer = ctx->GetAnalyzer();
      PrimExpr scale_axis = weight_scale_sinfo.as<TensorStructInfoNode>()->GetShape().value()[0];
      PrimExpr weight_out_channel = weight_OIHW_shape[0];
      if (!analyzer->CanProveEqual(weight_out_channel, scale_axis)) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "Qnn Conv2d expects weight scale dim(" << scale_axis
                         << ") to match the out channel dim(" << weight_out_channel << ")");
      }
    }
  }

  ObjectPtr<Conv2DAttrs> new_attrs = ffi::make_object<Conv2DAttrs>(*attrs);
  new_attrs->out_dtype = resolved_out_dtype;  // bake in the resolved dtype
  static const Op& base_conv2d_op = Op::Get("relax.nn.conv2d");
  Call base_call = Call(base_conv2d_op, {call->args[0], call->args[1]},  // data + weight only
                        Attrs(new_attrs), call->sinfo_args);

  return InferStructInfoConv2d(base_call, ctx);
}

InferLayoutOutput InferLayoutQnnConv2d(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  auto remapped = desired_layouts;
  const auto& it = desired_layouts.find("relax.qnn.conv2d");
  if (it != desired_layouts.end()) {
    remapped.Set("relax.nn.conv2d", (*it).second);
  }

  static const Op& base_conv2d_op = Op::Get("relax.nn.conv2d");  // <-- key fix
  Call base_call =
      Call(base_conv2d_op, {call->args[0], call->args[1]}, call->attrs, call->sinfo_args);

  return InferLayoutConv2d(base_call, remapped, var_layout_map);
}

Call InferMixedPrecisionQnnConv2d(const Call& call, const DataType& out_dtype) {
  const auto* conv2d_attrs = call->attrs.as<Conv2DAttrs>();
  bool has_scale = call->args.size() == 6;
  return Downcast<Call>(conv2d(call->args[0], call->args[1], call->args[2], call->args[3],
                               (has_scale ? ffi::Optional<Expr>(call->args[4]) : std::nullopt),
                               (has_scale ? ffi::Optional<Expr>(call->args[5]) : std::nullopt),
                               conv2d_attrs->strides, conv2d_attrs->padding, conv2d_attrs->dilation,
                               conv2d_attrs->groups, conv2d_attrs->data_layout,
                               conv2d_attrs->kernel_layout, conv2d_attrs->out_layout, out_dtype));
}

TVM_REGISTER_OP("relax.qnn.conv2d")
    .set_num_inputs(6)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .add_argument("data_zero_point", "Tensor", "Zero Point used for Dequantizing the Input")
    .add_argument("weight_zero_point", "Tensor", "Zero Point used for Dequantizing the Kernel")
    .add_argument("data_scale", "Optional<Tensor>", "Optional scale for Input Tensor")
    .add_argument("weight_scale", "Optional<Tensor>", "Optional scale for for Weight Tensor")
    .set_attrs_type<Conv2DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoQnnConv2d)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutQnnConv2d)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kAlways)
    .set_attr<FInferMixedPrecision>("FInferMixedPrecision", InferMixedPrecisionQnnConv2d)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.qnn.conv2d_transpose */

Expr conv2d_transpose(Expr data, Expr weight, Expr input_zero_pt, Expr weight_zero_pt,
                      ffi::Optional<Expr> input_scale, ffi::Optional<Expr> weight_scale,
                      ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
                      ffi::Array<int64_t> output_padding, ffi::Array<int64_t> dilation, int groups,
                      ffi::String data_layout, ffi::String kernel_layout,
                      ffi::Optional<ffi::String> out_layout, DataType out_dtype) {
  padding = GetCompletePadding2D(std::move(padding));

  if (strides.size() == 1) strides.push_back(strides[0]);
  if (dilation.size() == 1) dilation.push_back(dilation[0]);
  if (output_padding.size() == 1) output_padding.push_back(output_padding[0]);

  TVM_FFI_CHECK_GT(groups, 0, ValueError)
      << "The number of groups must be positive, got " << groups;
  TVM_FFI_CHECK_EQ(strides.size(), 2, ValueError) << "strides must have length 2, got " << strides;
  TVM_FFI_CHECK_EQ(dilation.size(), 2, ValueError)
      << "dilation must have length 2, got " << dilation;
  TVM_FFI_CHECK_EQ(output_padding.size(), 2, ValueError)
      << "output_padding must have length 2, got " << output_padding;

  if (static_cast<bool>(input_scale) ^ static_cast<bool>(weight_scale)) {
    LOG(FATAL) << "input_scale and weight_scale must both be present or both absent.";
  }

  return MakeConvTranspose<Conv2DTransposeAttrs>(
      std::move(data), std::move(weight), std::move(input_zero_pt), std::move(weight_zero_pt),
      input_scale, weight_scale, std::move(strides), std::move(padding), std::move(output_padding),
      std::move(dilation), groups, data_layout, std::move(kernel_layout),
      out_layout.value_or(data_layout), out_dtype,
      /*op_name=*/"relax.qnn.conv2d_transpose");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.qnn.op.conv2d_transpose", conv2d_transpose);
}

StructInfo InferStructInfoQnnConv2dTranspose(const Call& call, const BlockBuilder& ctx) {
  size_t num_args = call->args.size();
  TVM_FFI_ICHECK(num_args == 4 || num_args == 6) << "Expected 4 or 6 arguments, got " << num_args;

  bool has_scale = (num_args == 6);

  TensorStructInfo data_zp_sinfo = GetInputTensorStructInfo(call, 2, ctx);
  TensorStructInfo weight_zp_sinfo = GetInputTensorStructInfo(call, 3, ctx);

  TVM_FFI_ICHECK(GetElementDType(data_zp_sinfo) == GetElementDType(weight_zp_sinfo))
      << "data_zero_point and weight_zero_point dtype mismatch";
  TVM_FFI_ICHECK(data_zp_sinfo.as<TensorStructInfoNode>()->ndim == 0)
      << "data_zero_point must be a scalar";
  TVM_FFI_ICHECK(weight_zp_sinfo.as<TensorStructInfoNode>()->ndim == 0)
      << "weight_zero_point must be a scalar";

  ffi::Optional<TensorStructInfo> data_scale_sinfo, weight_scale_sinfo;
  if (has_scale) {
    data_scale_sinfo = GetInputTensorStructInfo(call, 4, ctx);
    weight_scale_sinfo = GetInputTensorStructInfo(call, 5, ctx);

    TVM_FFI_ICHECK(GetElementDType(data_scale_sinfo.value()) ==
                   GetElementDType(weight_scale_sinfo.value()))
        << "data_scale and weight_scale dtype mismatch";
    TVM_FFI_ICHECK(data_scale_sinfo.as<TensorStructInfoNode>()->ndim == 0)
        << "data_scale must be a scalar";
    TVM_FFI_ICHECK(weight_scale_sinfo.as<TensorStructInfoNode>()->ndim == 0)
        << "weight_scale must be a scalar";
  }

  const auto* attrs = call->attrs.as<Conv2DTransposeAttrs>();
  DataType resolved_out_dtype;
  if (!attrs->out_dtype.is_void()) {
    resolved_out_dtype = attrs->out_dtype;
  } else if (has_scale) {
    resolved_out_dtype = GetElementDType(data_scale_sinfo.value()).value();
  } else {
    resolved_out_dtype = DataType::Int(32);
  }

  Attrs base_attrs;
  if (attrs->out_dtype.is_void()) {
    ObjectPtr<Conv2DTransposeAttrs> new_attrs = ffi::make_object<Conv2DTransposeAttrs>(*attrs);
    new_attrs->out_dtype = resolved_out_dtype;
    base_attrs = Attrs(new_attrs);
  } else {
    base_attrs = call->attrs;
  }

  static const Op& base_op = Op::Get("relax.nn.conv2d_transpose");
  Call base_call = Call(base_op, {call->args[0], call->args[1]}, base_attrs, call->sinfo_args);

  return InferStructInfoConv2dTranspose(base_call, ctx);
}

InferLayoutOutput InferLayoutQnnConv2dTranspose(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  const auto* attrs = call->attrs.as<Conv2DTransposeAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  auto remapped = desired_layouts;
  const auto& it = desired_layouts.find("relax.qnn.conv2d_transpose");
  if (it != desired_layouts.end()) {
    remapped.Set("relax.nn.conv2d_transpose", (*it).second);
  }

  static const Op& base_op = Op::Get("relax.nn.conv2d_transpose");
  Call base_call = Call(base_op, {call->args[0], call->args[1]}, call->attrs, call->sinfo_args);

  return InferLayoutConv2dTranspose(base_call, remapped, var_layout_map);
}

Call InferMixedPrecisionQnnConv2dTranspose(const Call& call, const DataType& out_dtype) {
  const auto* attrs = call->attrs.as<Conv2DTransposeAttrs>();
  bool has_scale = (call->args.size() == 6);
  return Downcast<Call>(conv2d_transpose(
      call->args[0], call->args[1], call->args[2], call->args[3],
      has_scale ? ffi::Optional<Expr>(call->args[4]) : std::nullopt,
      has_scale ? ffi::Optional<Expr>(call->args[5]) : std::nullopt, attrs->strides, attrs->padding,
      attrs->output_padding, attrs->dilation, attrs->groups, attrs->data_layout,
      attrs->kernel_layout, attrs->out_layout, out_dtype));
}

TVM_REGISTER_OP("relax.qnn.conv2d_transpose")
    .set_num_inputs(6)
    .add_argument("data", "Tensor", "Input feature map.")
    .add_argument("weight", "Tensor", "Convolution kernel.")
    .add_argument("data_zero_point", "Tensor", "Scalar zero-point for dequantising the input.")
    .add_argument("weight_zero_point", "Tensor", "Scalar zero-point for dequantising the kernel.")
    .add_argument("data_scale", "Optional<Tensor>", "Optional scale for the input tensor.")
    .add_argument("weight_scale", "Optional<Tensor>", "Optional scale for the weight tensor.")
    .set_attrs_type<Conv2DTransposeAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoQnnConv2dTranspose)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutQnnConv2dTranspose)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kAlways)
    .set_attr<FInferMixedPrecision>("FInferMixedPrecision", InferMixedPrecisionQnnConv2dTranspose)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace qnn
}  // namespace relax
}  // namespace tvm
