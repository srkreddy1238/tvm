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
 * \file src/relax/transform/legalize_ops/adreno/convolution.cc
 * \brief Legalize high-level convolution operator calls in Relax functions to
 * call_tir with corresponding low-level TIR PrimFuncs for Adreno GPU targets.
 */

#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>
#include <tvm/s_tir/data_layout.h>
#include <tvm/topi/utils.h>

#include <string>

#include "../../../transform/legalize_ops/nn.h"
#include "../../../transform/legalize_ops/utils.h"

namespace tvm {
namespace relax {

/*!
 * \brief Check whether a relax.nn.conv2d call can be lowered to the
 *        NCHWc / OIHWo texture-layout compute.
 *
 * \param call                  The relax.nn.conv2d call to inspect.
 * \param disable_layout_checks When true, skip the data/kernel/out layout
 *                              string checks.
 * \return True if the call satisfies all layout, dtype, and channel
 *         divisibility conditions for the NCHWc/OIHWo path.
 */
static bool SupportsConv2dNCHWcOIHWo(const Call& call, bool disable_layout_checks = false) {
  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  if (!attrs) return false;

  if (!disable_layout_checks) {
    if (attrs->data_layout != "NCHW4c") return false;
    if (attrs->kernel_layout != "OIHW4o") return false;
    if (attrs->out_layout != "NCHW4c") return false;
  }

  auto data_sinfo = GetStructInfo(call->args[0]).as<TensorStructInfoNode>();
  auto weight_sinfo = GetStructInfo(call->args[1]).as<TensorStructInfoNode>();
  if (!data_sinfo || !weight_sinfo) return false;

  std::string data_dtype = DLDataTypeToString(data_sinfo->dtype);
  std::string weight_dtype = DLDataTypeToString(weight_sinfo->dtype);
  std::string out_dtype_str;
  if (attrs->out_dtype != DataType::Void()) {
    out_dtype_str = DLDataTypeToString(attrs->out_dtype);
  } else {
    out_dtype_str = data_dtype;
  }

  if (data_dtype != weight_dtype) return false;
  if (data_dtype != out_dtype_str) return false;
  if (data_dtype != "float16" && data_dtype != "float32") return false;

  auto data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (!data_shape) return false;

  auto out_sinfo =
      call->struct_info_.defined() ? call->struct_info_.as<TensorStructInfoNode>() : nullptr;
  if (!out_sinfo || !out_sinfo->shape.defined()) return false;
  auto out_shape_node = out_sinfo->shape.value().as<ShapeExprNode>();
  if (!out_shape_node) return false;

  // Always map through bijective layout to canonical NCHW, exactly as Python
  // does with s_tir.bijective_layout().forward_shape().  This is safe whether
  // the tensor is still in its original layout (disable_layout_checks=true)
  // or has already been converted to NCHW4c (disable_layout_checks=false).
  tir::BijectiveLayout data_bij(tir::Layout(std::string(attrs->data_layout)), tir::Layout("NCHW"));
  tir::BijectiveLayout out_bij(tir::Layout(std::string(attrs->out_layout)), tir::Layout("NCHW"));

  auto data_nchw = data_bij.ForwardShape(data_shape->values);
  auto out_nchw = out_bij.ForwardShape(out_shape_node->values);

  auto in_channel = data_nchw[1];
  auto out_channel = out_nchw[1];

  if (!EqualsConstInt(tir::FloorMod(in_channel, IntImm(in_channel->dtype, 4)), 0)) return false;
  if (!EqualsConstInt(tir::FloorMod(out_channel, IntImm(out_channel->dtype, 4)), 0)) return false;

  return true;
}

/*!
 * \brief TE compute for conv2d with NCHWc data layout and OIHWo kernel layout.
 *
 * \param args Packed argument list:
 *   - args[0]: data tensor, shape [N, ic_chunk, IH, IW, ic_bn]
 *   - args[1]: kernel tensor, shape [oc_chunk, ic, kH, kW, oc_bn]
 *   - args[2]: strides as Array<int64_t>
 *   - args[3]: padding as Array<int64_t>
 *   - args[4]: dilation as Array<int64_t>
 *   - args[5]: output DataType
 * \return Single-element array containing the output tensor of shape
 *         [N, oc_chunk, OH, OW, oc_bn].
 */
static ffi::Array<te::Tensor> Conv2dNCHWcOIHWoTE(const ffi::Array<ffi::Any>& args) {
  auto data = args[0].cast<te::Tensor>();
  auto kernel = args[1].cast<te::Tensor>();
  auto strides_arr = args[2].cast<ffi::Array<int64_t>>();
  auto padding_arr = args[3].cast<ffi::Array<int64_t>>();
  auto dilation_arr = args[4].cast<ffi::Array<int64_t>>();
  auto out_dtype = args[5].cast<DataType>();

  if (out_dtype == DataType::Void()) out_dtype = data->dtype;

  int64_t HSTR = strides_arr.size() >= 2 ? strides_arr[0] : strides_arr[0];
  int64_t WSTR = strides_arr.size() >= 2 ? strides_arr[1] : strides_arr[0];
  int64_t dilation_h = dilation_arr.size() >= 2 ? dilation_arr[0] : dilation_arr[0];
  int64_t dilation_w = dilation_arr.size() >= 2 ? dilation_arr[1] : dilation_arr[0];

  auto data_shape = data->shape;
  auto n = data_shape[0];
  auto ic_chunk = data_shape[1];
  auto ih = data_shape[2];
  auto iw = data_shape[3];
  auto ic_bn = data_shape[4];

  arith::Analyzer analyzer;
  auto in_channel = analyzer.Simplify(ic_chunk * ic_bn);

  // kernel shape [oc_chunk, ic, kH, kW, oc_bn]
  auto kernel_shape = kernel->shape;
  auto oc_chunk = kernel_shape[0];
  auto ic_total = kernel_shape[1];
  auto kernel_height = kernel_shape[2];
  auto kernel_width = kernel_shape[3];
  auto oc_bn = kernel_shape[4];

  auto groups = analyzer.Simplify(tir::FloorDiv(in_channel, ic_total));

  // Dilated kernel extents
  auto one = tir::make_const(ih->dtype, 1);
  auto dil_h_expr = tir::make_const(ih->dtype, dilation_h);
  auto dil_w_expr = tir::make_const(iw->dtype, dilation_w);
  auto dilated_kh = analyzer.Simplify((kernel_height - one) * dil_h_expr + one);
  auto dilated_kw = analyzer.Simplify((kernel_width - one) * dil_w_expr + one);

  // Padding
  ffi::Array<PrimExpr> padding_pexpr;
  for (auto v : padding_arr) padding_pexpr.push_back(tir::make_const(ih->dtype, v));
  auto [pad_begin, pad_end] = GetPadTupleGeneric(padding_pexpr, {dilated_kh, dilated_kw});
  auto pad_top = pad_begin[0], pad_left = pad_begin[1];
  auto pad_down = pad_end[0], pad_right = pad_end[1];

  auto HPAD = analyzer.Simplify(pad_top + pad_down);
  auto WPAD = analyzer.Simplify(pad_left + pad_right);

  // Output spatial dimensions
  auto hstr_expr = tir::make_const(ih->dtype, HSTR);
  auto wstr_expr = tir::make_const(iw->dtype, WSTR);
  auto out_height = analyzer.Simplify(tir::FloorDiv(ih + HPAD - dilated_kh, hstr_expr) + one);
  auto out_width = analyzer.Simplify(tir::FloorDiv(iw + WPAD - dilated_kw, wstr_expr) + one);

  ffi::Array<PrimExpr> oshape = {n, oc_chunk, out_height, out_width, oc_bn};

  // Pad the data as needed
  te::Tensor data_pad;
  if (!EqualsConstInt(HPAD, 0) || !EqualsConstInt(WPAD, 0)) {
    auto zero = tir::make_const(ih->dtype, 0);
    ffi::Array<PrimExpr> pad_before_5d = {zero, zero, pad_top, pad_left, zero};
    ffi::Array<PrimExpr> pad_after_5d = {zero, zero, pad_down, pad_right, zero};
    data_pad = PadTE(ffi::Array<ffi::Any>(
        {data, pad_before_5d, pad_after_5d, tir::make_const(data->dtype, 0),
         ffi::String("conv2d_data_pad"), ffi::Map<ffi::String, ffi::Any>()}))[0];
  } else {
    data_pad = data;
  }

  // Create reduction variables
  auto ic_per_group = analyzer.Simplify(tir::FloorDiv(in_channel, groups));
  auto kh_rv = tir::IterVar(Range(tir::make_const(ih->dtype, 0), kernel_height),
                            tir::Var("kh", ih->dtype), tir::kCommReduce, "");
  auto kw_rv = tir::IterVar(Range(tir::make_const(iw->dtype, 0), kernel_width),
                            tir::Var("kw", iw->dtype), tir::kCommReduce, "");
  auto ic_rv = tir::IterVar(Range(tir::make_const(in_channel->dtype, 0), ic_per_group),
                            tir::Var("ic", in_channel->dtype), tir::kCommReduce, "");

  // Compute definition
  return {tvm::te::compute(
      oshape,
      [&](const ffi::Array<tir::Var>& idx) {
        auto nn = idx[0];
        auto occ = idx[1];
        auto oh = idx[2];
        auto ow = idx[3];
        auto ocb = idx[4];

        // Channel index into the padded data
        PrimExpr ic_chunk_idx;
        auto groups_val = analyzer.Simplify(groups);
        if (EqualsConstInt(groups_val, 1)) {
          ic_chunk_idx = tir::FloorDiv(ic_rv, ic_bn);
        } else {
          auto oc_chunk_per_group = analyzer.Simplify(tir::FloorDiv(oc_chunk, groups_val));
          auto ic_chunk_per_group = analyzer.Simplify(tir::FloorDiv(ic_chunk, groups_val));
          ic_chunk_idx =
              analyzer.Simplify(tir::FloorDiv(occ, oc_chunk_per_group) * ic_chunk_per_group +
                                tir::FloorDiv(ic_rv, ic_bn));
        }
        auto ic_block_idx = tir::FloorMod(ic_rv, ic_bn);
        auto h_idx = analyzer.Simplify(oh * hstr_expr + kh_rv * dil_h_expr);
        auto w_idx = analyzer.Simplify(ow * wstr_expr + kw_rv * dil_w_expr);

        PrimExpr data_val = data_pad({nn, ic_chunk_idx, h_idx, w_idx, ic_block_idx});
        PrimExpr kernel_val = kernel({occ, ic_rv, kh_rv, kw_rv, ocb});

        return tvm::sum(tvm::cast(out_dtype, data_val) * tvm::cast(out_dtype, kernel_val),
                        {ic_rv, kh_rv, kw_rv});
      },
      "conv2d_NCHWc_OIHWo", "conv2d_NCHWc_OIHWo")};
}

/*!
 * \brief Legalize relax.nn.conv2d to call_tir via Conv2dNCHWcOIHWoTE for
 *        Adreno texture-layout inputs.
 *
 * \param bb The block builder.
 * \param call The relax.nn.conv2d call to legalize.
 * \return The legalized call_tir expression, or the original call if the
 *         layout or dtype conditions are not satisfied.
 */
Expr LegalizeConv2dNCHWcOIHWo(const BlockBuilder& bb, const Call& call) {
  if (!SupportsConv2dNCHWcOIHWo(call)) return call;

  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  auto m_te = MakeCallTE(bb, call);
  ffi::Array<ffi::Any> args;

  DataType out_dtype = attrs->out_dtype;
  if (out_dtype == DataType::Void()) {
    out_dtype = GetStructInfo(call->args[0]).as<TensorStructInfoNode>()->dtype;
  }

  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(attrs->strides);
  args.push_back(attrs->padding);
  args.push_back(attrs->dilation);
  args.push_back(out_dtype);

  // Pass call->sinfo_args[0] as the output StructInfo override, mirroring
  // Python's sinfo_args=call.sinfo_args in bb.call_te.  This preserves the
  // symbolic shape variables from the original call on the call_tir node.
  ffi::Optional<StructInfo> out_sinfo;
  if (!call->sinfo_args.empty()) {
    out_sinfo = call->sinfo_args[0];
  }

  return m_te.Make(args, FTOPIHandler(Conv2dNCHWcOIHWoTE), "conv2d_NCHWc_OIHWo", out_sinfo);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.backend.adreno.legalize.conv2d_NCHWc_OIHWo",
                        LegalizeConv2dNCHWcOIHWo);
}

/*!
 * \brief Check whether a relax.nn.conv2d call can be lowered to the
 *        cooperative-matrix matmul compute path.
 *
 * \param call                  The relax.nn.conv2d call to inspect.
 * \param disable_layout_checks When true, skip the data/kernel/out layout
 *                              string checks (used by the layout-conversion
 *                              callback which runs before layouts are set).
 * \return True if all intrinsic, texture, and (optionally) layout conditions
 *         are satisfied.
 */
static bool SupportsConv2dMatmul(const Call& call, bool disable_layout_checks = false) {
  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  if (!attrs) return false;

  auto data_sinfo = GetStructInfo(call->args[0]).as<TensorStructInfoNode>();
  auto weight_sinfo = GetStructInfo(call->args[1]).as<TensorStructInfoNode>();
  if (!data_sinfo || !weight_sinfo) return false;

  std::string data_dtype = DLDataTypeToString(data_sinfo->dtype);
  std::string weight_dtype = DLDataTypeToString(weight_sinfo->dtype);
  if (data_dtype != weight_dtype) return false;

  std::string out_dtype_str;
  if (attrs->out_dtype != DataType::Void()) {
    out_dtype_str = DLDataTypeToString(attrs->out_dtype);
  } else {
    out_dtype_str = data_dtype;
  }

  auto get_tiles_fn = tvm::ffi::Function::GetGlobal("tir.tensor_intrin.adreno.GetWmmaTileSizes");
  if (!get_tiles_fn.has_value()) return false;
  ffi::Any tiles_any;
  try {
    tiles_any = (*get_tiles_fn)(data_dtype, out_dtype_str);
  } catch (...) {
    return false;
  }
  if (tiles_any == nullptr) return false;
  ffi::Array<Integer> tiles;
  try {
    tiles = tiles_any.cast<ffi::Array<Integer>>();
  } catch (...) {
    return false;
  }
  if (tiles.size() != 3) return false;

  int64_t M_tile = tiles[0]->value;
  int64_t N_tile = tiles[1]->value;
  int64_t K_tile = tiles[2]->value;

  auto data_shape_node = data_sinfo->shape.as<ShapeExprNode>();
  auto weight_shape_node = weight_sinfo->shape.as<ShapeExprNode>();
  auto out_sinfo =
      call->struct_info_.defined() ? call->struct_info_.as<TensorStructInfoNode>() : nullptr;
  if (!data_shape_node || !weight_shape_node) return false;

  tir::BijectiveLayout data_bij(tir::Layout(std::string(attrs->data_layout)), tir::Layout("NCHW"));
  tir::BijectiveLayout kernel_bij(tir::Layout(std::string(attrs->kernel_layout)),
                                  tir::Layout("OIHW"));
  tir::BijectiveLayout out_bij(tir::Layout(std::string(attrs->out_layout)), tir::Layout("NCHW"));

  ffi::Array<PrimExpr> data_nchw = data_bij.ForwardShape(data_shape_node->values);
  ffi::Array<PrimExpr> weight_oihw = kernel_bij.ForwardShape(weight_shape_node->values);

  arith::Analyzer analyzer;
  auto input_channel = data_nchw[1];
  auto kernel_channel = weight_oihw[1];

  if (out_sinfo && out_sinfo->shape.defined()) {
    auto out_shape_node = out_sinfo->shape.value().as<ShapeExprNode>();
    if (!out_shape_node) return false;
    ffi::Array<PrimExpr> out_nchw = out_bij.ForwardShape(out_shape_node->values);

    auto groups_expr = tir::make_const(out_nchw[1]->dtype, static_cast<int64_t>(attrs->groups));
    auto M = analyzer.Simplify(out_nchw[2] * out_nchw[3]);
    auto N_expr = analyzer.Simplify(tir::FloorDiv(out_nchw[1], groups_expr));

    const double pad_threshold = 0.75;
    auto M_imm = M.as<IntImmNode>();
    auto N_imm = N_expr.as<IntImmNode>();
    auto K_imm = kernel_channel.as<IntImmNode>();
    if (M_imm && M_imm->value < static_cast<int64_t>(pad_threshold * M_tile)) return false;
    if (N_imm && N_imm->value < static_cast<int64_t>(pad_threshold * N_tile)) return false;
    if (K_imm && K_imm->value % K_tile != 0) return false;

    auto oc_imm = out_nchw[1].as<IntImmNode>();
    if (oc_imm && oc_imm->value % 4 != 0) return false;
  }

  auto in_ch_imm = input_channel.as<IntImmNode>();
  auto k_ch_imm = kernel_channel.as<IntImmNode>();
  if (in_ch_imm && in_ch_imm->value % 4 != 0) return false;
  if (k_ch_imm && k_ch_imm->value % 4 != 0) return false;

  if (!disable_layout_checks) {
    if (attrs->data_layout != "NCHW4c") return false;
    if (attrs->kernel_layout != "OIHW4i") return false;
    if (attrs->out_layout != "NCHW4c") return false;
  }

  return true;
}

/*!
 *
 * The compute proceeds in four stages:
 *   1. input_imed      \u2013 reshape the padded input into [N, G, kH, kW, padded_HW, IC].
 *   2. weight_imed     \u2013 reshape the weight into [G, kH, kW, padded_OC, IC].
 *   3. conv_matrix     \u2013 batched GEMM producing [N, G, padded_HW, padded_OC].
 *   4. conv_reinterpret \u2013 reshape back to NCHW4c [N, oc_chunk, OH, OW, oc_bn].
 *
 * \param args Packed argument list:
 *   - args[0]: data tensor,   shape [N, ic_chunk, IH, IW, ic_bn]   (NCHW4c)
 *   - args[1]: weight tensor, shape [oc, ic_chunk, kH, kW, ic_bn]  (OIHW4i)
 *   - args[2]: tiles_size as Array<Integer> [M_tile, N_tile, K_tile]
 *   - args[3]: strides  as Array<int64_t>
 *   - args[4]: padding  as Array<int64_t>
 *   - args[5]: dilation as Array<int64_t>
 *   - args[6]: in_dtype  as DataType
 *   - args[7]: out_dtype as DataType
 * \return Single-element array containing the output tensor of shape
 *         [N, oc_chunk, OH, OW, oc_bn].
 */
static ffi::Array<te::Tensor> Conv2dMatmulTE(const ffi::Array<ffi::Any>& args) {
  auto data = args[0].cast<te::Tensor>();
  auto weight = args[1].cast<te::Tensor>();
  auto tiles = args[2].cast<ffi::Array<Integer>>();
  auto strides_arr = args[3].cast<ffi::Array<int64_t>>();
  auto padding_arr = args[4].cast<ffi::Array<int64_t>>();
  auto dilation_arr = args[5].cast<ffi::Array<int64_t>>();
  auto in_dtype = args[6].cast<DataType>();
  auto out_dtype = args[7].cast<DataType>();

  int64_t M_tile = tiles[0]->value;
  int64_t N_tile = tiles[1]->value;
  // K_tile is enforced by SupportsConv2dMatmul; not needed in the TE compute.

  int64_t HSTR = strides_arr.size() >= 2 ? strides_arr[0] : strides_arr[0];
  int64_t WSTR = strides_arr.size() >= 2 ? strides_arr[1] : strides_arr[0];
  int64_t dilation_h = dilation_arr.size() >= 2 ? dilation_arr[0] : dilation_arr[0];
  int64_t dilation_w = dilation_arr.size() >= 2 ? dilation_arr[1] : dilation_arr[0];

  // data:   [num_batch, in_channel_chunk, in_height, in_width, in_channel_block]  (NCHW4c)
  auto num_batch = data->shape[0];
  auto in_channel_chunk = data->shape[1];
  auto in_height = data->shape[2];
  auto in_width = data->shape[3];
  auto in_channel_block = data->shape[4];

  // weight: [out_channel, kernel_channel_chunk, kH, kW, kernel_channel_block]  (OIHW4i)
  auto out_channel = weight->shape[0];
  auto kernel_channel_chunk = weight->shape[1];
  auto kernel_height = weight->shape[2];
  auto kernel_width = weight->shape[3];
  auto kernel_channel_block = weight->shape[4];

  arith::Analyzer analyzer;

  auto in_channel = analyzer.Simplify(in_channel_chunk * in_channel_block);
  auto kernel_channel = analyzer.Simplify(kernel_channel_chunk * kernel_channel_block);
  auto groups = analyzer.Simplify(tir::FloorDiv(in_channel, kernel_channel));
  auto block_size = in_channel_block;  // == kernel_channel_block

  auto one = tir::make_const(in_height->dtype, 1);
  auto dil_h_e = tir::make_const(in_height->dtype, dilation_h);
  auto dil_w_e = tir::make_const(in_width->dtype, dilation_w);
  auto dilated_kh = analyzer.Simplify((kernel_height - one) * dil_h_e + one);
  auto dilated_kw = analyzer.Simplify((kernel_width - one) * dil_w_e + one);

  ffi::Array<PrimExpr> padding_pexpr;
  for (auto v : padding_arr) padding_pexpr.push_back(tir::make_const(in_height->dtype, v));
  auto [pad_begin, pad_end] = GetPadTupleGeneric(padding_pexpr, {dilated_kh, dilated_kw});
  auto pad_top = pad_begin[0], pad_left = pad_begin[1];
  auto pad_down = pad_end[0], pad_right = pad_end[1];
  auto hpad = analyzer.Simplify(pad_top + pad_down);
  auto wpad = analyzer.Simplify(pad_left + pad_right);

  auto hstr_e = tir::make_const(in_height->dtype, HSTR);
  auto wstr_e = tir::make_const(in_width->dtype, WSTR);
  auto out_height = analyzer.Simplify(tir::FloorDiv(in_height + hpad - dilated_kh, hstr_e) + one);
  auto out_width = analyzer.Simplify(tir::FloorDiv(in_width + wpad - dilated_kw, wstr_e) + one);

  auto out_channel_chunk = analyzer.Simplify(tir::FloorDiv(out_channel, block_size));
  auto out_channel_block = block_size;

  // Padded GEMM dimensions (ceil-divide to next multiple of tile size)
  auto out_height_width = analyzer.Simplify(out_height * out_width);
  auto M_tile_e = tir::make_const(out_height_width->dtype, M_tile);
  auto N_tile_e = tir::make_const(out_channel->dtype, N_tile);
  auto padded_out_hw =
      analyzer.Simplify(tir::FloorDiv(out_height_width + M_tile_e - one, M_tile_e) * M_tile_e);
  auto group_out_channel = analyzer.Simplify(tir::FloorDiv(out_channel, groups));
  auto padded_group_outc =
      analyzer.Simplify(tir::FloorDiv(group_out_channel + N_tile_e - one, N_tile_e) * N_tile_e);

  // Pad data if needed
  te::Tensor data_pad;
  if (!EqualsConstInt(hpad, 0) || !EqualsConstInt(wpad, 0)) {
    auto zero = tir::make_const(in_height->dtype, 0);
    ffi::Array<PrimExpr> pad_before_5d = {zero, zero, pad_top, pad_left, zero};
    ffi::Array<PrimExpr> pad_after_5d = {zero, zero, pad_down, pad_right, zero};
    data_pad = PadTE(
        ffi::Array<ffi::Any>({data, pad_before_5d, pad_after_5d, tir::make_const(data->dtype, 0),
                              ffi::String("data_pad"), ffi::Map<ffi::String, ffi::Any>()}))[0];
  } else {
    data_pad = data;
  }

  // ---- Stage 1: input_imed ----
  // Shape: [num_batch, groups, kH, kW, padded_out_hw, kernel_channel]
  te::Tensor input_imed = tvm::te::compute(
      {num_batch, groups, kernel_height, kernel_width, padded_out_hw, kernel_channel},
      [&](const ffi::Array<tir::Var>& idx) -> PrimExpr {
        auto nn = idx[0];
        auto gg = idx[1];
        auto kh = idx[2];
        auto kw = idx[3];
        auto pad_oh_ow = idx[4];
        auto kc = idx[5];

        auto oh_idx = analyzer.Simplify(tir::FloorDiv(pad_oh_ow, out_width) * hstr_e +
                                        tir::Cast(hstr_e->dtype, kh) * dil_h_e);
        auto ow_idx = analyzer.Simplify(tir::FloorMod(pad_oh_ow, out_width) * wstr_e +
                                        tir::Cast(wstr_e->dtype, kw) * dil_w_e);
        auto ic_chunk_idx =
            analyzer.Simplify(gg * kernel_channel_chunk + tir::FloorDiv(kc, in_channel_block));
        auto ic_block_idx = tir::FloorMod(kc, in_channel_block);

        PrimExpr in_bounds = tir::LT(pad_oh_ow, out_height_width);
        PrimExpr data_val = data_pad({nn, ic_chunk_idx, oh_idx, ow_idx, ic_block_idx});
        return tvm::if_then_else(in_bounds, data_val, tir::make_const(in_dtype, 0));
      },
      "input_imed", "input_imed");

  // ---- Stage 2: weight_imed ----
  // Shape: [groups, kH, kW, padded_group_outc, kernel_channel]
  te::Tensor weight_imed = tvm::te::compute(
      {groups, kernel_height, kernel_width, padded_group_outc, kernel_channel},
      [&](const ffi::Array<tir::Var>& idx) -> PrimExpr {
        auto gg = idx[0];
        auto kh = idx[1];
        auto kw = idx[2];
        auto pad_gg_oc = idx[3];
        auto kc = idx[4];

        auto oc_idx = analyzer.Simplify(gg * group_out_channel + pad_gg_oc);
        auto kc_chunk_idx = tir::FloorDiv(kc, kernel_channel_block);
        auto kc_block_idx = tir::FloorMod(kc, kernel_channel_block);

        PrimExpr in_bounds = tir::LT(pad_gg_oc, group_out_channel);
        PrimExpr weight_val = weight({oc_idx, kc_chunk_idx, kh, kw, kc_block_idx});
        return tvm::if_then_else(in_bounds, weight_val, tir::make_const(in_dtype, 0));
      },
      "weight_imed", "weight_imed");

  // ---- Stage 3: conv_matrix (batched GEMM) ----
  // Shape: [num_batch, groups, padded_out_hw, padded_group_outc]
  auto kh_rv = tir::IterVar(Range(tir::make_const(kernel_height->dtype, 0), kernel_height),
                            tir::Var("kh", kernel_height->dtype), tir::kCommReduce, "");
  auto kw_rv = tir::IterVar(Range(tir::make_const(kernel_width->dtype, 0), kernel_width),
                            tir::Var("kw", kernel_width->dtype), tir::kCommReduce, "");
  auto kc_rv = tir::IterVar(Range(tir::make_const(kernel_channel->dtype, 0), kernel_channel),
                            tir::Var("kc", kernel_channel->dtype), tir::kCommReduce, "");

  te::Tensor conv_matrix = tvm::te::compute(
      {num_batch, groups, padded_out_hw, padded_group_outc},
      [&](const ffi::Array<tir::Var>& idx) -> PrimExpr {
        auto nn = idx[0];
        auto gg = idx[1];
        auto pad_oh_ow = idx[2];
        auto pad_gg_oc = idx[3];
        return tvm::sum(tvm::cast(out_dtype, input_imed({nn, gg, kh_rv, kw_rv, pad_oh_ow, kc_rv}) *
                                                 weight_imed({gg, kh_rv, kw_rv, pad_gg_oc, kc_rv})),
                        {kh_rv, kw_rv, kc_rv});
      },
      "conv_matrix", "conv_matrix");

  // ---- Stage 4: conv_reinterpret ----
  // Reshape [N, G, padded_HW, padded_OC] -> [N, oc_chunk, OH, OW, oc_bn]
  te::Tensor out = tvm::te::compute(
      {num_batch, out_channel_chunk, out_height, out_width, out_channel_block},
      [&](const ffi::Array<tir::Var>& idx) -> PrimExpr {
        auto nn = idx[0];
        auto occ = idx[1];
        auto oh = idx[2];
        auto ow = idx[3];
        auto ocb = idx[4];

        auto flat_oc =
            analyzer.Simplify(occ * out_channel_block + tir::Cast(out_channel_block->dtype, ocb));
        auto gg = analyzer.Simplify(tir::FloorDiv(flat_oc, group_out_channel));
        auto oc_in_grp = analyzer.Simplify(tir::FloorMod(flat_oc, group_out_channel));
        auto hw_idx = analyzer.Simplify(oh * out_width + tir::Cast(out_width->dtype, ow));

        return conv_matrix({nn, gg, hw_idx, oc_in_grp});
      },
      "conv_reinterpret", "conv_reinterpret");

  return {out};
}

/*!
 * \brief Legalize relax.nn.conv2d to call_tir via Conv2dMatmulTE for
 *        Adreno cooperative-matrix texture-layout inputs.
 *
 * \param bb   The block builder.
 * \param call The relax.nn.conv2d call to legalize.
 * \return The legalized call_tir expression, or the original call if the
 *         layout or dtype conditions are not satisfied.
 */
Expr LegalizeConv2dMatmul(const BlockBuilder& bb, const Call& call) {
  if (!SupportsConv2dMatmul(call)) return call;

  const auto* attrs = call->attrs.as<Conv2DAttrs>();

  auto get_tiles_fn = tvm::ffi::Function::GetGlobal("tir.tensor_intrin.adreno.GetWmmaTileSizes");
  TVM_FFI_ICHECK(get_tiles_fn.has_value())
      << "Global function not found: tir.tensor_intrin.adreno.GetWmmaTileSizes";

  auto data_sinfo = GetStructInfo(call->args[0]).as<TensorStructInfoNode>();
  DataType in_dtype = data_sinfo->dtype;
  DataType out_dtype = (attrs->out_dtype != DataType::Void()) ? attrs->out_dtype : in_dtype;

  auto tiles_any = (*get_tiles_fn)(DLDataTypeToString(in_dtype), DLDataTypeToString(out_dtype));
  auto tiles = tiles_any.cast<ffi::Array<Integer>>();

  auto m_te = MakeCallTE(bb, call);
  ffi::Array<ffi::Any> te_args;
  te_args.push_back(call->args[0]);
  te_args.push_back(call->args[1]);
  te_args.push_back(tiles);
  te_args.push_back(attrs->strides);
  te_args.push_back(attrs->padding);
  te_args.push_back(attrs->dilation);
  te_args.push_back(in_dtype);
  te_args.push_back(out_dtype);

  return m_te.Make(te_args, FTOPIHandler(Conv2dMatmulTE), "conv2d_matmul");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.backend.adreno.legalize.conv2d_matmul", LegalizeConv2dMatmul);
}

/*!
 * \brief Build a ConvertLayout callback that selects the appropriate
 *        conv2d layout based on target capabilities.
 *
 * The returned callback is passed to relax::transform::ConvertLayout as
 * the \p layout_cb argument.  For each relax.nn.conv2d call it returns:
 *   - {"relax.nn.conv2d": ["NCHW4c", "OIHW4i", "NCHW4c"]}  when
 *     use_matmul is true and the call satisfies SupportsConv2dMatmul.
 *   - {"relax.nn.conv2d": ["NCHW4c", "OIHW4o", "NCHW4c"]}  when
 *     the call satisfies SupportsConv2dNCHWcOIHWo.
 *   - {}  otherwise (no layout conversion for this call).
 *
 * \param use_matmul  When true, prefer the cooperative-matrix matmul path
 *                    over the NCHWc/OIHWo path when both are applicable.
 * \return A LayoutCb callable suitable for ConvertLayout.
 */
static tvm::relax::transform::LayoutCb Conv2dConvertLayout(bool use_matmul) {
  return [use_matmul](const Call& call) -> ffi::Map<ffi::String, ffi::Array<ffi::String>> {
    const auto* op_node = call->op.as<OpNode>();
    if (!op_node || op_node->name != "relax.nn.conv2d") {
      return {};
    }
    if (use_matmul && SupportsConv2dMatmul(call, /*disable_layout_checks=*/true)) {
      return {{"relax.nn.conv2d", {"NCHW4c", "OIHW4i", "NCHW4c"}}};
    }
    if (SupportsConv2dNCHWcOIHWo(call, /*disable_layout_checks=*/true)) {
      return {{"relax.nn.conv2d", {"NCHW4c", "OIHW4o", "NCHW4c"}}};
    }
    return {};
  };
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.backend.adreno.legalize.conv2d_convert_layout", Conv2dConvertLayout);
}

}  // namespace relax
}  // namespace tvm
