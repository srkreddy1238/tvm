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
 * \file tvm/relax/transform/legalize_ops/convolution_transpose.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */
#include <tvm/relax/attrs/nn.h>
#include <tvm/topi/utils.h>

#include <numeric>
#include <vector>

#include "nn.h"

namespace tvm {
namespace relax {

// Conv1DTranspose
std::pair<te::Tensor, te::Tensor> Conv1DTransposeNCWPreprocess(
    const te::Tensor& data, const te::Tensor& kernel, const ffi::Array<tvm::PrimExpr>& strides_arr,
    const ffi::Array<tvm::PrimExpr>& padding, const tvm::runtime::DataType& out_dtype,
    const ffi::Array<tvm::PrimExpr>& output_padding_arr) {
  auto channels_in = data->shape[1];
  auto channels_out = kernel->shape[1];
  auto kernel_width = kernel->shape[2];
  ffi::Map<ffi::String, ffi::Any> attrs({});

  auto stride = strides_arr[0];
  auto output_padding = output_padding_arr[0];

  arith::Analyzer analyzer;
  channels_out = analyzer.Simplify(channels_out);
  auto data_dilate = Dilate(data,
                            ffi::Array<tvm::PrimExpr>({tvm::IntImm(stride->dtype, 1),
                                                       tvm::IntImm(stride->dtype, 1), stride}),
                            tvm::tir::make_const(data->dtype, 0.0), "data_dilate");
  auto [pad_begin, pad_end] = GetPadTupleGeneric(padding, {kernel_width});
  tvm::PrimExpr pad_left =
      analyzer.Simplify(kernel_width - tvm::tir::make_const(kernel_width->dtype, 1) - pad_begin[0]);
  tvm::PrimExpr pad_right = analyzer.Simplify(
      kernel_width - tvm::tir::make_const(kernel_width->dtype, 1) - pad_begin[0] + output_padding);

  auto data_pad = PadTE(ffi::Array<ffi::Any>(
      {data_dilate,
       ffi::Array<tvm::PrimExpr>({tvm::tir::make_const(pad_left->dtype, 0),
                                  tvm::tir::make_const(pad_left->dtype, 0), pad_left}),
       ffi::Array<tvm::PrimExpr>({tvm::tir::make_const(pad_left->dtype, 0),
                                  tvm::tir::make_const(pad_left->dtype, 0), pad_right}),
       tvm::tir::make_const(data_dilate->dtype, 0.0), "data_pad", attrs}))[0];

  auto kernel_dilate = tvm::te::compute(
      ffi::Array<tvm::PrimExpr>({channels_out, channels_in, kernel_width}),
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        ffi::Array<tvm::PrimExpr> index_tuple;
        index_tuple.push_back(indices[1]);
        index_tuple.push_back(indices[0]);
        index_tuple.push_back(
            analyzer.Simplify(kernel_width - tvm::IntImm(kernel_width->dtype, 1) - indices[2]));

        return kernel(index_tuple);
      },
      "kernel");

  return std::make_pair(data_pad, kernel_dilate);
}

ffi::Array<te::Tensor> Conv1DTransposeNCWTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto kernel = args[1].cast<te::Tensor>();
  auto strides_i = args[2].cast<ffi::Array<int64_t>>();
  auto padding_i = args[3].cast<ffi::Array<int64_t>>();
  auto out_dtype = args[4].cast<tvm::runtime::DataType>();
  auto out_padding_i = args[5].cast<ffi::Array<int64_t>>();

  // Convert all int types to PrimExpr
  ffi::Array<tvm::PrimExpr> strides;
  ffi::Array<tvm::PrimExpr> padding;
  ffi::Array<tvm::PrimExpr> out_padding;
  arith::Analyzer analyzer;
  for (auto val : strides_i) {
    strides.push_back(tvm::IntImm(data->shape[0]->dtype, val));
  }
  for (auto val : padding_i) {
    padding.push_back(tvm::IntImm(data->shape[0]->dtype, val));
  }
  for (auto val : out_padding_i) {
    out_padding.push_back(tvm::IntImm(data->shape[0]->dtype, val));
  }
  if (out_dtype == DataType::Void()) {
    out_dtype = data->dtype;
  }

  auto batch = data->shape[0];
  auto channels_in = data->shape[1];
  auto channels_out = kernel->shape[1];
  auto kernel_width = kernel->shape[2];

  auto [data_pad, transformed_kernel] =
      Conv1DTransposeNCWPreprocess(data, kernel, strides, padding, out_dtype, out_padding);

  auto data_width = data_pad->shape[2];
  tvm::PrimExpr out_w =
      analyzer.Simplify(data_width - kernel_width + tvm::tir::make_const(kernel_width->dtype, 1));
  auto dc = tir::IterVar(tvm::Range(tvm::tir::make_const(channels_in->dtype, 0), channels_in),
                         tir::Var("dc", channels_in->dtype), tir::IterVarType::kCommReduce, "");
  auto dw = tir::IterVar(tvm::Range(tvm::tir::make_const(channels_in->dtype, 0), kernel_width),
                         tir::Var("dw", channels_in->dtype), tir::IterVarType::kCommReduce, "");

  return {tvm::te::compute(
      ffi::Array<tvm::PrimExpr>({batch, channels_out, out_w}),
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        ffi::Array<tvm::PrimExpr> data_index =
            ffi::Array<tvm::PrimExpr>({indices[0], dc, indices[2] + dw});
        ffi::Array<tvm::PrimExpr> kernel_index = ffi::Array<tvm::PrimExpr>({indices[1], dc, dw});
        return tvm::sum(tvm::cast(out_dtype, data_pad(data_index)) *
                            tvm::cast(out_dtype, transformed_kernel(kernel_index)),
                        {dc, dw});
      },
      "conv1d_transpose_ncw")};
}

ffi::Array<te::Tensor> GroupConv1DTransposeNCWTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto kernel = args[1].cast<te::Tensor>();
  auto strides_i = args[2].cast<ffi::Array<int64_t>>();
  auto padding_i = args[3].cast<ffi::Array<int64_t>>();
  auto out_dtype = args[4].cast<tvm::runtime::DataType>();
  auto out_padding_i = args[5].cast<ffi::Array<int64_t>>();
  auto groups_i = args[6].cast<int>();

  // Convert all int types to PrimExpr
  ffi::Array<tvm::PrimExpr> strides;
  ffi::Array<tvm::PrimExpr> padding;
  ffi::Array<tvm::PrimExpr> out_padding;
  tvm::PrimExpr groups;
  arith::Analyzer analyzer;

  for (auto val : strides_i) {
    strides.push_back(tvm::IntImm(data->shape[0]->dtype, val));
  }
  for (auto val : padding_i) {
    padding.push_back(tvm::IntImm(data->shape[0]->dtype, val));
  }
  for (auto val : out_padding_i) {
    out_padding.push_back(tvm::IntImm(data->shape[0]->dtype, val));
  }
  groups = tvm::IntImm(data->shape[0]->dtype, groups_i);

  if (out_dtype == DataType::Void()) {
    out_dtype = data->dtype;
  }

  auto in_channels = data->shape[1];
  TVM_FFI_ICHECK(analyzer.CanProve(tvm::tir::Mod(in_channels, groups) ==
                                   tvm::tir::make_const(in_channels->dtype, 0)))
      << "input channels " << in_channels << " must device group size " << groups;

  auto [data_pad, transformed_kernel] =
      Conv1DTransposeNCWPreprocess(data, kernel, strides, padding, out_dtype, out_padding);

  auto batch = data_pad->shape[0];
  in_channels = data_pad->shape[1];
  auto in_w = data_pad->shape[2];
  auto out_c = transformed_kernel->shape[0];
  auto filter_w = transformed_kernel->shape[2];

  auto out_channels = analyzer.Simplify(out_c * groups);
  auto out_w = analyzer.Simplify(in_w - filter_w + tvm::tir::make_const(filter_w->dtype, 1));
  auto dc = tir::IterVar(tvm::Range(tvm::tir::make_const(in_channels->dtype, 0),
                                    tvm::tir::FloorDiv(in_channels, groups)),
                         tir::Var("dc", in_channels->dtype), tir::IterVarType::kCommReduce, "");
  auto dw = tir::IterVar(tvm::Range(tvm::tir::make_const(in_channels->dtype, 0), filter_w),
                         tir::Var("dw", in_channels->dtype), tir::IterVarType::kCommReduce, "");

  return {tvm::te::compute(
      ffi::Array<tvm::PrimExpr>({batch, out_channels, out_w}),
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        ffi::Array<tvm::PrimExpr> data_index = ffi::Array<tvm::PrimExpr>(
            {indices[0],
             analyzer.Simplify(
                 (tvm::tir::FloorDiv(indices[1], tvm::tir::FloorDiv(out_channels, groups)) *
                  tvm::tir::FloorDiv(in_channels, groups)) +
                 dc),
             analyzer.Simplify(indices[2] + dw)});
        ffi::Array<tvm::PrimExpr> kernel_index = ffi::Array<tvm::PrimExpr>(
            {analyzer.Simplify(
                 tvm::tir::FloorMod(indices[1], tvm::tir::FloorDiv(out_channels, groups))),
             analyzer.Simplify(
                 (tvm::tir::FloorDiv(indices[1], tvm::tir::FloorDiv(out_channels, groups)) *
                  tvm::tir::FloorDiv(in_channels, groups)) +
                 dc),
             dw});
        return tvm::sum(tvm::cast(out_dtype, data_pad(data_index)) *
                            tvm::cast(out_dtype, transformed_kernel(kernel_index)),
                        {dc, dw});
      },
      "group_conv1d_transpose_ncw")};
}

Expr LegalizeNNConv1DTranspose(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<Conv1DTransposeAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;

  if (attrs->out_layout != attrs->data_layout) {
    LOG(WARNING) << "TOPI conv1d_transpose does not support different input-output layouts, "
                 << "and thus cannot be legalized by TOPI";
    return call;
  }

  if ((attrs->data_layout != "NCW") || (attrs->kernel_layout != "IOW")) {
    LOG(WARNING) << "conv1d_transpose supportd only NCW/IOW layout";
    return call;
  }

  if (attrs->dilation.size() != 1 || attrs->dilation[0] != 1) {
    LOG(WARNING) << "conv1d_transpose dilation should be of size 1 and value 1";
    return call;
  }

  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(attrs->strides);
  args.push_back(attrs->padding);
  args.push_back(attrs->out_dtype);
  args.push_back(attrs->output_padding);

  if (attrs->groups == 1) {
    return m_te.Make(args, FTOPIHandler(Conv1DTransposeNCWTE), std::string("conv1d_transpose"));
  } else {
    args.push_back(attrs->groups);
    return m_te.Make(args, FTOPIHandler(GroupConv1DTransposeNCWTE),
                     std::string("conv1d_transpose"));
  }
}
TVM_REGISTER_OP("relax.nn.conv1d_transpose")
    .set_attr<FLegalize>("FLegalize", LegalizeNNConv1DTranspose, TVM_LEGALIZE_CPP_LEVEL);

// Conv2DTranspose
std::pair<te::Tensor, te::Tensor> Conv2DTransposeNCHWPreprocess(
    const te::Tensor& data, const te::Tensor& kernel, const ffi::Array<tvm::PrimExpr>& strides,
    const ffi::Array<tvm::PrimExpr>& padding, const tvm::runtime::DataType& out_dtype,
    const ffi::Array<tvm::PrimExpr>& output_padding) {
  auto batch = data->shape[0];
  auto in_c = data->shape[1];
  auto in_h = data->shape[2];
  auto in_w = data->shape[3];
  auto out_c = kernel->shape[1];
  auto filter_h = kernel->shape[2];
  auto filter_w = kernel->shape[3];
  auto stride_h = strides[0];
  auto stride_w = strides[1];
  auto opad_h = output_padding[0];
  auto opad_w = output_padding[1];
  arith::Analyzer analyzer;

  TVM_FFI_ICHECK(analyzer.CanProve(opad_h < stride_h) && analyzer.CanProve(opad_w < stride_w))
      << "output paddings" << output_padding << " must be less than strides " << strides;

  // Dilate
  auto data_dilate =
      Dilate(data,
             ffi::Array<tvm::PrimExpr>({tvm::IntImm(stride_h->dtype, 1),
                                        tvm::IntImm(stride_h->dtype, 1), stride_h, stride_w}),
             tvm::tir::make_const(data->dtype, 0.0), "data_dilate");

  // Pad
  auto [pad_begin, pad_end] = GetPadTupleGeneric(padding, {filter_h, filter_w});
  auto fpad_top = pad_begin[0];
  auto fpad_left = pad_begin[1];
  auto fpad_bottom = pad_end[0];
  auto fpad_right = pad_end[1];

  auto bpad_top = analyzer.Simplify(filter_h - tvm::tir::make_const(filter_h->dtype, 1) - fpad_top);
  auto bpad_bottom =
      analyzer.Simplify(filter_h - tvm::tir::make_const(filter_h->dtype, 1) - fpad_bottom + opad_h);
  auto bpad_left =
      analyzer.Simplify(filter_w - tvm::tir::make_const(filter_w->dtype, 1) - fpad_left);
  auto bpad_right =
      analyzer.Simplify(filter_w - tvm::tir::make_const(filter_w->dtype, 1) - fpad_right + opad_w);

  ffi::Map<ffi::String, ffi::Any> attrs({});
  auto data_pad = PadTE(ffi::Array<ffi::Any>(
      {data_dilate,
       ffi::Array<tvm::PrimExpr>({tvm::tir::make_const(bpad_left->dtype, 0),
                                  tvm::tir::make_const(bpad_left->dtype, 0), bpad_top, bpad_left}),
       ffi::Array<tvm::PrimExpr>({tvm::tir::make_const(bpad_left->dtype, 0),
                                  tvm::tir::make_const(bpad_left->dtype, 0), bpad_bottom,
                                  bpad_right}),
       tvm::tir::make_const(data_dilate->dtype, 0.0), "data_pad", attrs}))[0];

  auto kernel_dilate = tvm::te::compute(
      ffi::Array<tvm::PrimExpr>({out_c, in_c, filter_h, filter_w}),
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        ffi::Array<tvm::PrimExpr> index_tuple;
        index_tuple.push_back(indices[1]);
        index_tuple.push_back(indices[0]);
        index_tuple.push_back(
            analyzer.Simplify(filter_h - tvm::IntImm(filter_h->dtype, 1) - indices[2]));
        index_tuple.push_back(
            analyzer.Simplify(filter_w - tvm::IntImm(filter_w->dtype, 1) - indices[3]));

        return kernel(index_tuple);
      },
      "kernel_transform");

  return std::make_pair(data_pad, kernel_dilate);
}

ffi::Array<te::Tensor> GroupConv2DTransposeNCHWTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto kernel = args[1].cast<te::Tensor>();
  auto strides_i = args[2].cast<ffi::Array<int64_t>>();
  auto padding_i = args[3].cast<ffi::Array<int64_t>>();
  auto out_dtype = args[4].cast<tvm::runtime::DataType>();
  auto out_padding_i = args[5].cast<ffi::Array<int64_t>>();
  auto groups_i = args[6].cast<int>();

  // Convert all int types to PrimExpr
  ffi::Array<tvm::PrimExpr> strides;
  ffi::Array<tvm::PrimExpr> padding;
  ffi::Array<tvm::PrimExpr> output_padding;
  tvm::PrimExpr groups;
  arith::Analyzer analyzer;

  for (auto val : strides_i) {
    strides.push_back(tvm::IntImm(data->shape[0]->dtype, val));
  }
  for (auto val : padding_i) {
    padding.push_back(tvm::IntImm(data->shape[0]->dtype, val));
  }
  for (auto val : out_padding_i) {
    output_padding.push_back(tvm::IntImm(data->shape[0]->dtype, val));
  }
  groups = tvm::IntImm(data->shape[0]->dtype, groups_i);

  if (out_dtype == DataType::Void()) {
    out_dtype = data->dtype;
  }

  auto in_channels = data->shape[1];
  TVM_FFI_ICHECK(analyzer.CanProve(tvm::tir::Mod(in_channels, groups) ==
                                   tvm::tir::make_const(in_channels->dtype, 0)))
      << "input channels " << in_channels << " must device group size " << groups;

  auto [data_pad, kernel_transform] =
      Conv2DTransposeNCHWPreprocess(data, kernel, strides, padding, out_dtype, output_padding);

  auto batch = data_pad->shape[0];
  in_channels = data_pad->shape[1];
  auto in_h = data_pad->shape[2];
  auto in_w = data_pad->shape[3];
  auto out_c = kernel_transform->shape[0];
  auto filter_h = kernel_transform->shape[2];
  auto filter_w = kernel_transform->shape[3];

  auto out_channels = analyzer.Simplify(out_c * groups);
  auto out_h = analyzer.Simplify(in_h - filter_h + tvm::tir::make_const(filter_h->dtype, 1));
  auto out_w = analyzer.Simplify(in_w - filter_w + tvm::tir::make_const(filter_w->dtype, 1));
  auto dc = tir::IterVar(tvm::Range(tvm::tir::make_const(in_channels->dtype, 0),
                                    tvm::tir::FloorDiv(in_channels, groups)),
                         tir::Var("dc", in_channels->dtype), tir::IterVarType::kCommReduce, "");
  auto dh = tir::IterVar(tvm::Range(tvm::tir::make_const(filter_h->dtype, 0), filter_h),
                         tir::Var("dh", filter_h->dtype), tir::IterVarType::kCommReduce, "");
  auto dw = tir::IterVar(tvm::Range(tvm::tir::make_const(filter_w->dtype, 0), filter_w),
                         tir::Var("dw", filter_w->dtype), tir::IterVarType::kCommReduce, "");

  return {tvm::te::compute(
      ffi::Array<tvm::PrimExpr>({batch, out_channels, out_h, out_w}),
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        ffi::Array<tvm::PrimExpr> data_index = ffi::Array<tvm::PrimExpr>(
            {indices[0],
             analyzer.Simplify(
                 (tvm::tir::FloorDiv(indices[1], tvm::tir::FloorDiv(out_channels, groups)) *
                  tvm::tir::FloorDiv(in_channels, groups)) +
                 dc),
             analyzer.Simplify(indices[2] + dh), analyzer.Simplify(indices[3] + dw)});
        ffi::Array<tvm::PrimExpr> kernel_index = ffi::Array<tvm::PrimExpr>(
            {analyzer.Simplify(
                 tvm::tir::FloorMod(indices[1], tvm::tir::FloorDiv(out_channels, groups))),
             analyzer.Simplify(
                 (tvm::tir::FloorDiv(indices[1], tvm::tir::FloorDiv(out_channels, groups)) *
                  tvm::tir::FloorDiv(in_channels, groups)) +
                 dc),
             dh, dw});
        return tvm::sum(tvm::cast(out_dtype, data_pad(data_index)) *
                            tvm::cast(out_dtype, kernel_transform(kernel_index)),
                        {dc, dh, dw});
      },
      "group_conv2d_transpose_nchw")};
}

ffi::Array<te::Tensor> Conv2DTransposeNCHWTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto kernel = args[1].cast<te::Tensor>();
  auto strides_i = args[2].cast<ffi::Array<int64_t>>();
  auto padding_i = args[3].cast<ffi::Array<int64_t>>();
  auto out_dtype = args[4].cast<tvm::runtime::DataType>();
  auto out_padding_i = args[5].cast<ffi::Array<int64_t>>();

  // Convert all int types to PrimExpr
  ffi::Array<tvm::PrimExpr> strides;
  ffi::Array<tvm::PrimExpr> padding;
  ffi::Array<tvm::PrimExpr> output_padding;
  arith::Analyzer analyzer;
  for (auto val : strides_i) {
    strides.push_back(tvm::IntImm(data->shape[0]->dtype, val));
  }
  for (auto val : padding_i) {
    padding.push_back(tvm::IntImm(data->shape[0]->dtype, val));
  }
  for (auto val : out_padding_i) {
    output_padding.push_back(tvm::IntImm(data->shape[0]->dtype, val));
  }
  if (out_dtype == DataType::Void()) {
    out_dtype = data->dtype;
  }

  auto [data_pad, kernel_transform] =
      Conv2DTransposeNCHWPreprocess(data, kernel, strides, padding, out_dtype, output_padding);

  auto batch = data_pad->shape[0];
  auto in_c = data_pad->shape[1];
  auto in_h = data_pad->shape[2];
  auto in_w = data_pad->shape[3];
  auto out_c = analyzer.Simplify(kernel_transform->shape[0]);
  auto filter_h = kernel_transform->shape[2];
  auto filter_w = kernel_transform->shape[3];

  tvm::PrimExpr out_h =
      analyzer.Simplify(in_h - filter_h + tvm::tir::make_const(filter_h->dtype, 1));
  tvm::PrimExpr out_w =
      analyzer.Simplify(in_w - filter_w + tvm::tir::make_const(filter_h->dtype, 1));
  auto dc = tir::IterVar(tvm::Range(tvm::tir::make_const(in_c->dtype, 0), in_c),
                         tir::Var("dc", in_c->dtype), tir::IterVarType::kCommReduce, "");
  auto dh = tir::IterVar(tvm::Range(tvm::tir::make_const(filter_h->dtype, 0), filter_h),
                         tir::Var("dh", filter_h->dtype), tir::IterVarType::kCommReduce, "");
  auto dw = tir::IterVar(tvm::Range(tvm::tir::make_const(filter_w->dtype, 0), filter_w),
                         tir::Var("dw", filter_w->dtype), tir::IterVarType::kCommReduce, "");

  return {tvm::te::compute(
      ffi::Array<tvm::PrimExpr>({batch, out_c, out_h, out_w}),
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        ffi::Array<tvm::PrimExpr> data_index =
            ffi::Array<tvm::PrimExpr>({indices[0], dc, indices[2] + dh, indices[3] + dw});
        ffi::Array<tvm::PrimExpr> kernel_index =
            ffi::Array<tvm::PrimExpr>({indices[1], dc, dh, dw});
        return tvm::sum(tvm::cast(out_dtype, data_pad(data_index)) *
                            tvm::cast(out_dtype, kernel_transform(kernel_index)),
                        {dc, dh, dw});
      },
      "conv2d_transpose_nchw")};
}

Expr LegalizeNNConv2DTranspose(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<Conv2DTransposeAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;

  if (attrs->out_layout != attrs->data_layout) {
    LOG(WARNING) << "TOPI conv2d_transpose does not support different input-output layouts, "
                 << "and thus cannot be legalized by TOPI";
    return call;
  }

  if ((attrs->data_layout != "NCHW") || (attrs->kernel_layout != "IOHW")) {
    LOG(WARNING) << "conv2d_transpose supportd only NCHW/IOHW layout";
    return call;
  }

  if (attrs->dilation.size() != 2 || attrs->dilation[0] != 1 || attrs->dilation[1] != 1) {
    LOG(WARNING) << "conv2d_transpose dilation support only 1";
    return call;
  }

  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(attrs->strides);
  args.push_back(attrs->padding);
  args.push_back(attrs->out_dtype);
  args.push_back(attrs->output_padding);

  if (attrs->groups == 1) {
    return m_te.Make(args, FTOPIHandler(Conv2DTransposeNCHWTE), std::string("conv2d_transpose"));
  } else {
    args.push_back(attrs->groups);
    return m_te.Make(args, FTOPIHandler(GroupConv2DTransposeNCHWTE),
                     std::string("conv2d_transpose"));
  }
}
TVM_REGISTER_OP("relax.nn.conv2d_transpose")
    .set_attr<FLegalize>("FLegalize", LegalizeNNConv2DTranspose, TVM_LEGALIZE_CPP_LEVEL);

}  // namespace relax
}  // namespace tvm
