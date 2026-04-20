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
 * \file tvm/relax/transform/legalize_ops/convolution.cc
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

/*!
 * \brief Return the indices that would sort `array` in ascending order.
 *
 * \param array The values to sort.
 * \return A vector of indices such that `array[result[i]]` is non-decreasing.
 */
template <typename T>
std::vector<size_t> ArgSort(const std::vector<T>& array) {
  std::vector<size_t> indices(array.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&array](size_t i1, size_t i2) { return array[i1] < array[i2]; });

  return indices;
}

/*!
 * \brief Apply constant-value padding to a tensor via the PadTE handler.
 *
 * Thin wrapper that forwards to PadTE with a fixed constant fill value.
 *
 * \param data The tensor to pad.
 * \param pad_before Per-axis leading pad extents.
 * \param pad_after Per-axis trailing pad extents.
 * \param pad_value The scalar fill value.
 * \param name Optional name for the output tensor.
 * \param attrs Optional attribute map forwarded to the TE compute.
 * \return The padded output tensor.
 */
te::Tensor Pad(const te::Tensor& data, const ffi::Array<tvm::PrimExpr>& pad_before,
               const ffi::Array<tvm::PrimExpr>& pad_after, const tvm::PrimExpr& pad_value,
               const ffi::String& name = "PadInput",
               const ffi::Map<ffi::String, ffi::Any>& attrs = ffi::Map<ffi::String, ffi::Any>({})) {
  return PadTE(ffi::Array<ffi::Any>({data, pad_before, pad_after, pad_value, name, attrs}))[0];
}

ffi::Array<te::Tensor> ConvTE(const ffi::Array<ffi::Any> args) {
  auto inp = args[0].cast<te::Tensor>();
  auto filt = args[1].cast<te::Tensor>();
  auto strides_i = args[2].cast<ffi::Array<int64_t>>();
  auto padding_i = args[3].cast<ffi::Array<int64_t>>();
  auto dilation_i = args[4].cast<ffi::Array<int64_t>>();
  auto groups_i = args[5].cast<int>();
  auto data_layout = args[6].cast<ffi::String>();
  auto kernel_layout = args[7].cast<ffi::String>();
  auto out_dtype = args[8].cast<tvm::runtime::DataType>();
  ffi::String auto_scheduler_rewritten_layout = "";
  if (args[9] != nullptr) {
    auto_scheduler_rewritten_layout = args[9].cast<ffi::String>();
  }
  ffi::Array<tvm::PrimExpr> meta_schedule_original_shape;
  if (args[10] != nullptr) {
    meta_schedule_original_shape = args[10].cast<ffi::Array<tvm::PrimExpr>>();
  }
  auto auto_scheduler_should_rewrite_layout = args[11].cast<bool>();

  // Convert all int types to PrimExpr
  ffi::Array<tvm::PrimExpr> strides;
  ffi::Array<tvm::PrimExpr> padding;
  ffi::Array<tvm::PrimExpr> dilation;
  tvm::PrimExpr groups;
  for (auto val : strides_i) {
    strides.push_back(tvm::IntImm(inp->shape[0]->dtype, val));
  }
  for (auto val : padding_i) {
    padding.push_back(tvm::IntImm(inp->shape[0]->dtype, val));
  }
  for (auto val : dilation_i) {
    dilation.push_back(tvm::IntImm(inp->shape[0]->dtype, val));
  }
  groups = tvm::IntImm(inp->shape[0]->dtype, groups_i);

  if (out_dtype == DataType::Void()) {
    out_dtype = inp->dtype;
  }

  // Checks
  auto dim = (inp->shape.size() - 2);
  TVM_FFI_ICHECK(strides.size() == dim);
  TVM_FFI_ICHECK(dilation.size() == dim);

  // Transform indices from data_layout to NCHW
  std::vector<size_t> data_permutation_to;
  data_permutation_to.push_back(data_layout.find("N"));
  data_permutation_to.push_back(data_layout.find("C"));
  for (size_t i = 0; i < data_layout.length(); ++i) {
    if (data_layout.at(i) != 'N' && data_layout.at(i) != 'C') {
      data_permutation_to.push_back(i);
    }
  }

  // Transform indices from NCHW to data_layout
  auto data_permutation_from = ArgSort<size_t>(data_permutation_to);

  // Transform indices from CHW to data_layout
  std::vector<size_t> data_permutation_from_reductions(data_permutation_from.begin() + 1,
                                                       data_permutation_from.end());

  for (size_t i = 0; i < data_permutation_from_reductions.size(); ++i) {
    if (data_permutation_from_reductions[i] > data_permutation_from[0]) {
      data_permutation_from_reductions[i] -= 1;
    }
  }

  // Transform indices from kernel_layout to OIHW
  TVM_FFI_ICHECK(kernel_layout != "") << "kernel_layout is empty";
  std::vector<size_t> kernel_permutation_to;
  kernel_permutation_to.push_back(kernel_layout.find("O"));
  kernel_permutation_to.push_back(kernel_layout.find("I"));
  for (size_t i = 0; i < kernel_layout.length(); ++i) {
    if (kernel_layout.at(i) != 'O' && kernel_layout.at(i) != 'I') {
      kernel_permutation_to.push_back(i);
    }
  }

  // Transform indices from OIHW to kernel_layout
  auto kernel_permutation_from = ArgSort<size_t>(kernel_permutation_to);
  if (meta_schedule_original_shape.size()) {
    LOG(FATAL) << "LEGACY-FLOW triggered, to be removed";
  }
  auto inp_shape = inp->shape;
  auto batch = inp_shape[data_permutation_to[0]];
  auto in_channel = inp_shape[data_permutation_to[1]];
  ffi::Array<tvm::PrimExpr> dimensions;
  for (size_t i = 2; i < data_permutation_to.size(); ++i) {
    dimensions.push_back(inp_shape[data_permutation_to[i]]);
  }

  auto filt_shape = filt->shape;
  auto num_filter = filt_shape[kernel_permutation_to[0]];
  ffi::Array<tvm::PrimExpr> kernel_dimensions;
  for (size_t i = 2; i < kernel_permutation_to.size(); ++i) {
    kernel_dimensions.push_back(filt_shape[kernel_permutation_to[i]]);
  }

  if (auto_scheduler_rewritten_layout.size()) {
    LOG(FATAL) << "LEGACY-FLOW triggered, to be removed";
  }

  // Conv operation checks
  arith::Analyzer analyzer;
  TVM_FFI_ICHECK(analyzer.CanProve(tvm::tir::Mod(in_channel, groups) ==
                                   tvm::tir::make_const(in_channel->dtype, 0)))
      << "Input channels must divide group size";
  TVM_FFI_ICHECK(analyzer.CanProve(tvm::tir::Mod(num_filter, groups) ==
                                   tvm::tir::make_const(num_filter->dtype, 0)))
      << "Output channels must divide group size";

  // Dilations
  ffi::Array<tvm::PrimExpr> dilated_kernel_dimensions;
  for (size_t i = 0; i < kernel_dimensions.size(); ++i) {
    dilated_kernel_dimensions.push_back(analyzer.Simplify(
        (((kernel_dimensions[i] - tvm::tir::make_const(kernel_dimensions[i]->dtype, 1)) *
          dilation[i]) +
         tvm::tir::make_const(kernel_dimensions[i]->dtype, 1))));
  }

  // Padding
  auto [pad_begin, pad_end] = GetPadTupleGeneric(padding, dilated_kernel_dimensions);

  // Compure output dimensions
  auto out_channel = num_filter;
  ffi::Array<tvm::PrimExpr> out_dimensions;
  for (size_t i = 0; i < dimensions.size(); ++i) {
    out_dimensions.push_back(analyzer.Simplify(
        tvm::tir::FloorDiv(
            ((dimensions[i] -
              (kernel_dimensions[i] - tvm::tir::make_const(dimensions[i]->dtype, 1)) *
                  dilation[i]) -
             tvm::tir::make_const(dimensions[i]->dtype, 1) + pad_begin[i] + pad_end[i]),
            strides[i]) +
        tvm::tir::make_const(dimensions[i]->dtype, 1)));
    if (auto ival = out_dimensions[i].as<tvm::IntImmNode>()) {
      TVM_FFI_ICHECK(ival->value > 0)
          << "Invalid conv parameters: lead to negative output shape :" << ival->value;
    }
  }

  // Pad the data as needed.
  ffi::Array<tvm::PrimExpr> pad_begin_ext = {tvm::tir::make_const(pad_begin[0]->dtype, 0),
                                             tvm::tir::make_const(pad_begin[0]->dtype, 0)};
  for (auto val : pad_begin) {
    pad_begin_ext.push_back(val);
  }
  ffi::Array<tvm::PrimExpr> pad_end_ext = {tvm::tir::make_const(pad_end[0]->dtype, 0),
                                           tvm::tir::make_const(pad_end[0]->dtype, 0)};
  for (auto val : pad_end) {
    pad_end_ext.push_back(val);
  }
  ffi::Array<tvm::PrimExpr> pad_before;
  ffi::Array<tvm::PrimExpr> pad_after;
  for (size_t i = 0; i < data_permutation_from.size(); ++i) {
    pad_before.push_back(pad_begin_ext[data_permutation_from[i]]);
    pad_after.push_back(pad_end_ext[data_permutation_from[i]]);
  }
  auto temp = Pad(inp, pad_before, pad_after, tvm::tir::make_const(inp->dtype, 0), "pad_temp");

  // Create reduction variables
  auto rc = tir::IterVar(tvm::Range(tvm::tir::make_const(in_channel->dtype, 0),
                                    tvm::tir::FloorDiv(in_channel, groups)),
                         tir::Var("rc", in_channel->dtype), tir::IterVarType::kCommReduce, "");
  ffi::Array<tir::IterVar> rs;
  ffi::Array<ffi::String> th_id = {"y", "x", "z"};
  for (size_t i = 0; i < kernel_dimensions.size(); ++i) {
    rs.push_back(
        tir::IterVar(tvm::Range(tvm::tir::make_const(in_channel->dtype, 0), kernel_dimensions[i]),
                     tir::Var(ffi::String("r") + th_id[i], kernel_dimensions[i]->dtype),
                     tir::IterVarType::kCommReduce, ""));
  }

  // compute output shape
  ffi::Array<tvm::PrimExpr> all_out_dims = {batch, out_channel};
  for (auto val : out_dimensions) {
    all_out_dims.push_back(val);
  }
  ffi::Array<tvm::PrimExpr> out_shape;
  for (size_t i = 0; i < data_permutation_from.size(); ++i) {
    out_shape.push_back(all_out_dims[data_permutation_from[i]]);
  }

  // Operation tag and names
  ffi::String tag_ = "";
  ffi::String name_ = "";
  ffi::Map<ffi::String, ffi::Any> attrs;
  if (groups_i > 1) {
    tag_ = tag_ + "group_";
  }
  std::string layout_str = std::string(data_layout.c_str());
  std::transform(layout_str.begin(), layout_str.end(), layout_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  tag_ = tag_ + "conv" + std::to_string(dim) + "d_" + layout_str;
  name_ = tag_;

  if (auto_scheduler_should_rewrite_layout) {
    attrs.Set("layout_free_placeholders", ffi::Array<te::Tensor>({filt}));
  }

  // Compute definition
  return {tvm::te::compute(
      out_shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        // Pick all indices
        auto nn = indices[data_permutation_to[0]];
        auto ff = indices[data_permutation_to[1]];
        ffi::Array<tvm::tir::Var> dim_indices;
        for (size_t i = 2; i < data_permutation_to.size(); ++i) {
          dim_indices.push_back(indices[data_permutation_to[i]]);
        }

        // Channel indice based on groups
        tvm::PrimExpr simplified_channel_index;
        if (1 == groups_i) {
          simplified_channel_index = rc;
        } else {
          simplified_channel_index =
              analyzer.Simplify((tvm::tir::FloorDiv(ff, tvm::tir::FloorDiv(num_filter, groups)) *
                                 tvm::tir::FloorDiv(in_channel, groups)) +
                                rc);
        }

        // Data indices
        ffi::Array<tvm::PrimExpr> data_index_all = {nn, simplified_channel_index};
        for (size_t i = 0; i < dilation.size(); ++i) {
          data_index_all.push_back(
              analyzer.Simplify((dim_indices[i] * strides[i]) + (rs[i] * dilation[i])));
        }
        ffi::Array<tvm::PrimExpr> data_index;
        for (auto val : data_permutation_from) {
          data_index.push_back(data_index_all[val]);
        }

        // Filter indices
        ffi::Array<tvm::PrimExpr> filt_index_all = {ff, rc};
        for (auto val : rs) {
          filt_index_all.push_back(val);
        }
        ffi::Array<tvm::PrimExpr> filt_index;
        for (auto val : kernel_permutation_from) {
          filt_index.push_back(filt_index_all[val]);
        }

        // Reduction axis
        ffi::Array<tir::IterVar> reduction_axis_all = {rc};
        for (auto val : rs) {
          reduction_axis_all.push_back(val);
        }
        ffi::Array<tir::IterVar> reduction_axis;
        for (auto val : data_permutation_from_reductions) {
          reduction_axis.push_back(reduction_axis_all[val]);
        }

        // Convolution computation
        return tvm::sum(
            tvm::cast(out_dtype, temp(data_index)) * tvm::cast(out_dtype, filt(filt_index)),
            reduction_axis);
      },
      name_, tag_, attrs)};
}

/*!
 * \brief Legalize relax.nn.conv1d to call_tir via ConvTE.
 *
 * \param bb The block builder.
 * \param call The relax.nn.conv1d call to legalize.
 * \return The legalized call_tir expression, or the original call if the
 *         layout or group configuration is not supported.
 */
Expr LegalizeNNConv1D(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<Conv1DAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;

  if (attrs->out_layout != attrs->data_layout) {
    LOG(WARNING) << "TOPI Conv2D does not support different input-output layouts, "
                 << "and thus cannot be legalized by TOPI";
    return call;
  }

  if ((attrs->out_layout.length() != 3) || (attrs->data_layout.length() != 3)) {
    LOG(WARNING) << "Conv1D where data layout or kernel layout have channel chunk "
                 << "cannot be legalized by TOPI at this moment.";
    return call;
  }

  if (attrs->groups != 1) {
    auto ic = GetStructInfo(call->args[0])
                  .as<TensorStructInfoNode>()
                  ->shape.as<ShapeExprNode>()
                  ->values[attrs->data_layout.find("C")];
    auto oc = GetStructInfo(call->args[1])
                  .as<TensorStructInfoNode>()
                  ->shape.as<ShapeExprNode>()
                  ->values[attrs->kernel_layout.find("O")];
    if (!(ic.as<tir::IntImmNode>() && oc.as<tir::IntImmNode>())) {
      LOG(WARNING) << "Conv1D where number of groups is more than one and input or output "
                   << "channel size is symbolic cannot be legalized by TOPI at this moment.";
      return call;
    }
  }

  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(attrs->strides);
  args.push_back(attrs->padding);
  args.push_back(attrs->dilation);
  args.push_back(attrs->groups);
  args.push_back(attrs->data_layout);
  args.push_back(attrs->kernel_layout);
  args.push_back(attrs->out_dtype);
  args.push_back(nullptr);  // auto_scheduler_rewritten_layout
  args.push_back(nullptr);  // meta_schedule_original_shape
  args.push_back(false);    // auto_scheduler_should_rewrite_layout

  auto call_ret = m_te.Make(args, FTOPIHandler(ConvTE), std::string("conv1d"));
  return call_ret;
}
TVM_REGISTER_OP("relax.nn.conv1d")
    .set_attr<FLegalize>("FLegalize", LegalizeNNConv1D, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Legalize relax.nn.conv2d to call_tir via ConvTE.
 *
 * \param bb The block builder.
 * \param call The relax.nn.conv2d call to legalize.
 * \return The legalized call_tir expression, or the original call if the
 *         layout or group configuration is not supported.
 */
Expr LegalizeNNConv2D(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;

  if (attrs->out_layout != attrs->data_layout) {
    return call;
  }

  if ((attrs->out_layout.length() != 4) || (attrs->data_layout.length() != 4)) {
    return call;
  }

  if (attrs->groups != 1) {
    auto ic = GetStructInfo(call->args[0])
                  .as<TensorStructInfoNode>()
                  ->shape.as<ShapeExprNode>()
                  ->values[attrs->data_layout.find("C")];
    auto oc = GetStructInfo(call->args[1])
                  .as<TensorStructInfoNode>()
                  ->shape.as<ShapeExprNode>()
                  ->values[attrs->kernel_layout.find("O")];
    if (!(ic.as<tir::IntImmNode>() && oc.as<tir::IntImmNode>())) {
      LOG(WARNING) << "Conv2D where number of groups is more than one and input or output "
                   << "channel size is symbolic cannot be legalized by TOPI at this moment.";
      return call;
    }
  }

  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(attrs->strides);
  args.push_back(attrs->padding);
  args.push_back(attrs->dilation);
  args.push_back(attrs->groups);
  args.push_back(attrs->data_layout);
  args.push_back(attrs->kernel_layout);
  args.push_back(attrs->out_dtype);
  args.push_back(nullptr);  // auto_scheduler_rewritten_layout
  args.push_back(nullptr);  // meta_schedule_original_shape
  args.push_back(false);    // auto_scheduler_should_rewrite_layout

  auto call_ret = m_te.Make(args, FTOPIHandler(ConvTE), std::string("conv2d"));
  return call_ret;
}
TVM_REGISTER_OP("relax.nn.conv2d")
    .set_attr<FLegalize>("FLegalize", LegalizeNNConv2D, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Legalize relax.nn.conv3d to call_tir via ConvTE.
 *
 * \param bb The block builder.
 * \param call The relax.nn.conv3d call to legalize.
 * \return The legalized call_tir expression, or the original call if the
 *         layout or group configuration is not supported.
 */
Expr LegalizeNNConv3D(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<Conv3DAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;

  if (attrs->out_layout != attrs->data_layout) {
    LOG(WARNING) << "TOPI Conv3D does not support different input-output layouts, "
                 << "and thus cannot be legalized by TOPI";
    return call;
  }

  if ((attrs->out_layout.length() != 5) || (attrs->data_layout.length() != 5)) {
    LOG(WARNING) << "Conv3D where data layout or kernel layout have channel chunk "
                 << "cannot be legalized by TOPI at this moment.";
    return call;
  }

  if (attrs->groups != 1) {
    auto ic = GetStructInfo(call->args[0])
                  .as<TensorStructInfoNode>()
                  ->shape.as<ShapeExprNode>()
                  ->values[attrs->data_layout.find("C")];
    auto oc = GetStructInfo(call->args[1])
                  .as<TensorStructInfoNode>()
                  ->shape.as<ShapeExprNode>()
                  ->values[attrs->kernel_layout.find("O")];
    if (!(ic.as<tir::IntImmNode>() && oc.as<tir::IntImmNode>())) {
      LOG(WARNING) << "Conv3D where number of groups is more than one and input or output "
                   << "channel size is symbolic cannot be legalized by TOPI at this moment.";
      return call;
    }
  }

  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(attrs->strides);
  args.push_back(attrs->padding);
  args.push_back(attrs->dilation);
  args.push_back(attrs->groups);
  args.push_back(attrs->data_layout);
  args.push_back(attrs->kernel_layout);
  args.push_back(attrs->out_dtype);
  args.push_back(nullptr);  // auto_scheduler_rewritten_layout
  args.push_back(nullptr);  // meta_schedule_original_shape
  args.push_back(false);    // auto_scheduler_should_rewrite_layout

  auto call_ret = m_te.Make(args, FTOPIHandler(ConvTE), std::string("conv3d"));
  return call_ret;
}
TVM_REGISTER_OP("relax.nn.conv3d")
    .set_attr<FLegalize>("FLegalize", LegalizeNNConv3D, TVM_LEGALIZE_CPP_LEVEL);

}  // namespace relax
}  // namespace tvm
