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

#include "tvm/relax/block_builder.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/struct_info.h"
#include "tvm/runtime/data_type.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace relax {

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

  TensorStructInfo data_sinfo = GetInputTensorStructInfo(call, 0, ctx);
  TensorStructInfo weight_sinfo = GetInputTensorStructInfo(call, 1, ctx);
  TensorStructInfo data_zero_pt_sinfo = GetInputTensorStructInfo(call, 2, ctx);
  TensorStructInfo weight_zero_pt_sinfo = GetInputTensorStructInfo(call, 3, ctx);
  ffi::Optional<TensorStructInfo> data_scale_sinfo, weight_scale_sinfo;
  if (has_scale) {
    data_scale_sinfo = GetInputTensorStructInfo(call, 4, ctx);
    weight_scale_sinfo = GetInputTensorStructInfo(call, 5, ctx);
  }

  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, ctx, attrs->data_layout,  //
                                                    /*tgt_layout=*/"NCHW",          //
                                                    /*tensor_name=*/"data");
  auto [weight_layout, weight2OIHW] = CheckTensorLayout(call, ctx, attrs->kernel_layout,  //
                                                        /*tgt_layout=*/"OIHW",            //
                                                        /*tensor_name=*/"kernel");
  auto [out_layout, out2NCHW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                  /*tgt_layout=*/"NCHW",         //
                                                  /*tensor_name=*/"output");

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_sinfo, data_layout);
  ffi::Optional<ShapeExpr> weight_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, weight_sinfo, weight_layout);

  TVM_FFI_ICHECK(GetElementDType(data_sinfo) == GetElementDType(weight_sinfo) ||
                 !attrs->out_dtype.is_void())
      << "Cannot Infer Output Datatype";
  TVM_FFI_ICHECK(GetElementDType(data_zero_pt_sinfo) == GetElementDType(weight_zero_pt_sinfo))
      << "Mismatch between Data Zero Point and Weight Zero Point DataType";
  TVM_FFI_ICHECK(data_zero_pt_sinfo.as<TensorStructInfoNode>()->ndim == 0)
      << "Data Zero Point Must be a Scalar";
  TVM_FFI_ICHECK(weight_zero_pt_sinfo.as<TensorStructInfoNode>()->ndim == 0)
      << "Weight Zero Point Must be a Scalar";
  if (has_scale) {
    TVM_FFI_ICHECK(GetElementDType(data_scale_sinfo.value()) ==
                   GetElementDType(weight_scale_sinfo.value()))
        << "Mismatch between Input and Weight DataType";
    TVM_FFI_ICHECK(data_scale_sinfo.as<TensorStructInfoNode>()->ndim == 0)
        << "Data Scale Must be a Scalar";
    TVM_FFI_ICHECK(weight_scale_sinfo.as<TensorStructInfoNode>()->ndim == 0 ||
                   weight_scale_sinfo.as<TensorStructInfoNode>()->ndim == 1)
        << "Weight Scale Point Must be a Scalar/1-D Tensor";
  }

  DataType out_dtype =
      attrs->out_dtype.is_void()
          ? GetElementDType((has_scale ? data_scale_sinfo.value() : data_sinfo)).value()
          : attrs->out_dtype;

  ffi::Optional<VDevice> vdevice =
      InferBinaryArithOpOutVDevice(call, ctx, data_sinfo, weight_sinfo);
  if (!data_shape.defined() || !weight_shape.defined()) {
    return TensorStructInfo(out_dtype, out_layout.ndim(), vdevice);
  }

  ffi::Array<PrimExpr> data_NCHW_shape = data2NCHW.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> weight_OIHW_shape = weight2OIHW.ForwardShape(weight_shape.value()->values);

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  PrimExpr input_channel_data = data_NCHW_shape[1];
  PrimExpr input_channel_kernel = weight_OIHW_shape[1];
  if (has_scale && weight_scale_sinfo.as<TensorStructInfoNode>()->ndim == 1) {
    PrimExpr scale_axis = weight_scale_sinfo.as<TensorStructInfoNode>()->GetShape().value()[0];
    PrimExpr weight_out_channel = weight_OIHW_shape[0];
    if (!analyzer->CanProveEqual(weight_out_channel, scale_axis)) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "Qnn Conv2d expects weight scale dim(" << scale_axis
                       << ") to match the out channel dim(" << weight_out_channel << ")");
    }
  }
  if (analyzer->CanProve(input_channel_data != input_channel_kernel * attrs->groups)) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "The channel size of the data should equal to the product of input channel size of the "
           "weight and the number of groups. However, the data channel size is "
        << input_channel_data << " while the weight input channel size and number of groups are "
        << input_channel_kernel << " and " << attrs->groups);
  } else if (!analyzer->CanProveEqual(input_channel_data, input_channel_kernel * attrs->groups)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(floormod(weight_OIHW_shape[0], attrs->groups) != 0)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Qnn Conv2d expects the number of output channels to be divisible by the "
                        "number of groups. However, the number of output channels is "
                     << weight_OIHW_shape[0] << " while the number of groups is " << attrs->groups);
  } else if (!analyzer->CanProveEqual(floormod(weight_OIHW_shape[0], attrs->groups), 0)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }

  PrimExpr input_h = data_NCHW_shape[2];
  PrimExpr input_w = data_NCHW_shape[3];
  PrimExpr kernel_h = weight_OIHW_shape[2];
  PrimExpr kernel_w = weight_OIHW_shape[3];
  PrimExpr padding_h = Integer(attrs->padding[0] + attrs->padding[2]);
  PrimExpr padding_w = Integer(attrs->padding[1] + attrs->padding[3]);

  std::vector<PrimExpr> out_NCHW_shape;
  out_NCHW_shape.resize(4);
  out_NCHW_shape[0] = data_NCHW_shape[0];
  out_NCHW_shape[1] = weight_OIHW_shape[0];

  PrimExpr numerator_h = input_h + padding_h - Integer(attrs->dilation[0]) * (kernel_h - 1) - 1;
  PrimExpr numerator_w = input_w + padding_w - Integer(attrs->dilation[1]) * (kernel_w - 1) - 1;
  out_NCHW_shape[2] = analyzer->Simplify(floordiv(numerator_h, attrs->strides[0]) + 1);
  out_NCHW_shape[3] = analyzer->Simplify(floordiv(numerator_w, attrs->strides[1]) + 1);

  ffi::Array<PrimExpr> out_shape = out2NCHW.BackwardShape(out_NCHW_shape);
  return TensorStructInfo(ShapeExpr(out_shape), out_dtype, vdevice);
}

InferLayoutOutput InferLayoutQnnConv2d(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  const auto& it = desired_layouts.find("relax.qnn.conv2d_part");
  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision data_layout, weight_layout, output_layout;
  data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  weight_layout = GetLayoutDecision(var_layout_map, call->args[1]);
  ObjectPtr<Conv2DAttrs> new_attrs = ffi::make_object<Conv2DAttrs>(*attrs);

  if (it != desired_layouts.end()) {
    // We have a desired layout for conv2d.
    Layout desired_data_layout = (*it).second[0];
    Layout desired_weight_layout = (*it).second[1];
    Layout desired_output_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
    tir::Layout input_layout(attrs->data_layout, DataType::Int(64));
    tir::Layout kernel_layout(attrs->kernel_layout, DataType::Int(64));
    tir::Layout out_layout(attrs->out_layout, DataType::Int(64));

    if ((desired_data_layout.ndim() == input_layout.ndim()) &&
        (desired_weight_layout.ndim() == kernel_layout.ndim()) &&
        (desired_output_layout.ndim() == out_layout.ndim())) {
      // Just a transpose
      data_layout = TransposeLike(InitialLayout(4), attrs->data_layout, desired_data_layout);
      weight_layout = TransposeLike(InitialLayout(4), attrs->kernel_layout, desired_weight_layout);
      output_layout = TransposeLike(InitialLayout(4), attrs->out_layout, desired_output_layout);
      new_attrs->data_layout = (*it).second[0];
      new_attrs->kernel_layout = (*it).second[1];
      new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
      return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
    } else {
      // Layout Transform
      auto data_si = GetStructInfo(call->args[0]);
      auto kernel_si = GetStructInfo(call->args[1]);
      TensorStructInfo data_sinfo = data_si.as<TensorStructInfo>().value();
      TensorStructInfo kernel_sinfo = kernel_si.as<TensorStructInfo>().value();
      ffi::Optional<ShapeExpr> data_shape =
          ffi::GetRef<ShapeExpr>(data_sinfo->shape.as<ShapeExprNode>());
      ffi::Optional<ShapeExpr> kernel_shape =
          ffi::GetRef<ShapeExpr>(kernel_sinfo->shape.as<ShapeExprNode>());

      bool can_data_proved =
          CanProveLayoutTransform(input_layout, desired_data_layout, data_shape.value()->values);
      bool can_kernel_proved = CanProveLayoutTransform(kernel_layout, desired_weight_layout,
                                                       kernel_shape.value()->values);

      if (can_data_proved && can_kernel_proved) {
        data_layout = TransposeSubLayoutLike(InitialLayout(4), input_layout, desired_data_layout);
        weight_layout =
            TransposeSubLayoutLike(InitialLayout(4), kernel_layout, desired_weight_layout);
        output_layout = TransposeSubLayoutLike(InitialLayout(4), out_layout, desired_output_layout);
        new_attrs->data_layout = (*it).second[0];
        new_attrs->kernel_layout = (*it).second[1];
        new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
        return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
      } else {
        data_layout = LayoutDecision(InitialLayout(4));
        weight_layout = LayoutDecision(InitialLayout(4));
      }
    }
  }

  // We don't have a desired layout for conv2d or desired layouts not compatible.
  // We can just propagate the layout from the input.

  output_layout = data_layout;
  new_attrs->data_layout =
      TransposeLike(attrs->data_layout, InitialLayout(4), data_layout->layout).name();
  new_attrs->kernel_layout =
      TransposeLike(attrs->kernel_layout, InitialLayout(4), weight_layout->layout).name();
  new_attrs->out_layout =
      TransposeLike(attrs->out_layout, InitialLayout(4), output_layout->layout).name();
  return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
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

}  // namespace relax
}  // namespace tvm
