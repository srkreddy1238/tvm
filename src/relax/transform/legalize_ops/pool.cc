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
 * \file tvm/relax/transform/legalize_ops/pool.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */
#include <tvm/relax/attrs/nn.h>
#include <tvm/topi/nn/pooling.h>

#include "utils.h"

namespace tvm {
namespace relax {

// ---------------------------------------------------------------------------
// Shared helper: convert an Array<int64_t> attribute to Array<PrimExpr>
// ---------------------------------------------------------------------------
/*!
 * \brief Convert an Array<int64_t> attribute to Array<PrimExpr> with the given dtype.
 *
 * \param arr The integer array to convert.
 * \param dtype The target PrimExpr integer data type.
 * \return The converted PrimExpr array.
 */
static ffi::Array<tvm::PrimExpr> Int64ArrayToPrimExpr(const ffi::Array<int64_t>& arr,
                                                      const DataType& dtype) {
  ffi::Array<tvm::PrimExpr> out;
  for (auto v : arr) out.push_back(tvm::IntImm(dtype, v));
  return out;
}

// ---------------------------------------------------------------------------
// TE handlers
// ---------------------------------------------------------------------------

/*!
 * \brief TE handler for 1-D pooling (max or avg).
 *
 * Expected args layout:
 *   [0]  data              – te::Tensor
 *   [1]  kernel_size       – Array<int64_t>  (length 1)
 *   [2]  stride_size       – Array<int64_t>  (length 1)
 *   [3]  dilation_size     – Array<int64_t>  (length 1)
 *   [4]  padding_size      – Array<int64_t>  (length 2: head, tail)
 *   [5]  pool_type         – String ("max" | "avg")
 *   [6]  ceil_mode         – bool
 *   [7]  layout            – String
 *   [8]  count_include_pad – bool
 */
ffi::Array<te::Tensor> Pool1DTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto kernel_i = args[1].cast<ffi::Array<int64_t>>();
  auto stride_i = args[2].cast<ffi::Array<int64_t>>();
  auto dilation_i = args[3].cast<ffi::Array<int64_t>>();
  auto padding_i = args[4].cast<ffi::Array<int64_t>>();
  auto pool_type_str = args[5].cast<ffi::String>();
  auto ceil_mode = args[6].cast<bool>();
  auto layout = args[7].cast<ffi::String>();
  auto count_include_pad = args[8].cast<bool>();

  auto dtype = data->shape[0]->dtype;
  // pool1d expects padding as [head_pad_w, tail_pad_w] (length == 2 * k_size == 2).
  topi::nn::PoolType pool_type = (pool_type_str == "max") ? topi::nn::kMaxPool : topi::nn::kAvgPool;

  return {topi::nn::pool1d(
      data, Int64ArrayToPrimExpr(kernel_i, dtype), Int64ArrayToPrimExpr(stride_i, dtype),
      Int64ArrayToPrimExpr(dilation_i, dtype), Int64ArrayToPrimExpr(padding_i, dtype), pool_type,
      ceil_mode, std::string(layout), count_include_pad)};
}

/*!
 * \brief TE handler for 2-D pooling (max or avg).
 *
 * Expected args layout:
 *   [0]  data              – te::Tensor
 *   [1]  kernel_size       – Array<int64_t>  (length 2)
 *   [2]  stride_size       – Array<int64_t>  (length 2)
 *   [3]  dilation_size     – Array<int64_t>  (length 2)
 *   [4]  padding_size      – Array<int64_t>  (length 4: head_h, head_w, tail_h, tail_w)
 *   [5]  pool_type         – String ("max" | "avg")
 *   [6]  ceil_mode         – bool
 *   [7]  layout            – String
 *   [8]  count_include_pad – bool
 */
ffi::Array<te::Tensor> Pool2DTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto kernel_i = args[1].cast<ffi::Array<int64_t>>();
  auto stride_i = args[2].cast<ffi::Array<int64_t>>();
  auto dilation_i = args[3].cast<ffi::Array<int64_t>>();
  auto padding_i = args[4].cast<ffi::Array<int64_t>>();
  auto pool_type_str = args[5].cast<ffi::String>();
  auto ceil_mode = args[6].cast<bool>();
  auto layout = args[7].cast<ffi::String>();
  auto count_include_pad = args[8].cast<bool>();

  auto dtype = data->shape[0]->dtype;
  topi::nn::PoolType pool_type = (pool_type_str == "max") ? topi::nn::kMaxPool : topi::nn::kAvgPool;

  return {topi::nn::pool2d(
      data, Int64ArrayToPrimExpr(kernel_i, dtype), Int64ArrayToPrimExpr(stride_i, dtype),
      Int64ArrayToPrimExpr(dilation_i, dtype), Int64ArrayToPrimExpr(padding_i, dtype), pool_type,
      ceil_mode, std::string(layout), count_include_pad)};
}

/*!
 * \brief TE handler for 3-D pooling (max or avg).
 *
 * Expected args layout:
 *   [0]  data              – te::Tensor
 *   [1]  kernel_size       – Array<int64_t>  (length 3)
 *   [2]  stride_size       – Array<int64_t>  (length 3)
 *   [3]  dilation_size     – Array<int64_t>  (length 3)
 *   [4]  padding_size      – Array<int64_t>  (length 6: head_{d,h,w}, tail_{d,h,w})
 *   [5]  pool_type         – String ("max" | "avg")
 *   [6]  ceil_mode         – bool
 *   [7]  layout            – String
 *   [8]  count_include_pad – bool
 */
ffi::Array<te::Tensor> Pool3DTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto kernel_i = args[1].cast<ffi::Array<int64_t>>();
  auto stride_i = args[2].cast<ffi::Array<int64_t>>();
  auto dilation_i = args[3].cast<ffi::Array<int64_t>>();
  auto padding_i = args[4].cast<ffi::Array<int64_t>>();
  auto pool_type_str = args[5].cast<ffi::String>();
  auto ceil_mode = args[6].cast<bool>();
  auto layout = args[7].cast<ffi::String>();
  auto count_include_pad = args[8].cast<bool>();

  auto dtype = data->shape[0]->dtype;
  topi::nn::PoolType pool_type = (pool_type_str == "max") ? topi::nn::kMaxPool : topi::nn::kAvgPool;

  return {topi::nn::pool3d(
      data, Int64ArrayToPrimExpr(kernel_i, dtype), Int64ArrayToPrimExpr(stride_i, dtype),
      Int64ArrayToPrimExpr(dilation_i, dtype), Int64ArrayToPrimExpr(padding_i, dtype), pool_type,
      ceil_mode, std::string(layout), count_include_pad)};
}

/*!
 * \brief TE handler for 1-D adaptive average pooling.
 *
 * Mirrors the Python te_adaptive_avg_pool1d inner function:
 * when output_size is None the spatial width of the input is used as the
 * output size (i.e. a no-op identity pool), matching the Python fallback
 * that reads data.shape[idx_W].
 *
 * Expected args layout:
 *   [0]  data        – te::Tensor
 *   [1]  output_size – Optional<Array<int64_t>>  (nullopt → use input W)
 *   [2]  layout      – String
 */
ffi::Array<te::Tensor> AdaptiveAvgPool1DTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto output_size_opt = args[1].cast<ffi::Optional<ffi::Array<int64_t>>>();
  auto layout_str = args[2].cast<ffi::String>();

  ffi::Array<tvm::PrimExpr> output_size;
  if (output_size_opt.defined()) {
    auto dtype = data->shape[0]->dtype;
    for (auto v : output_size_opt.value()) output_size.push_back(tvm::IntImm(dtype, v));
  } else {
    // output_size is None: derive W from the input shape using the layout.
    int width_axis = -1;
    TVM_FFI_ICHECK(topi::nn::find_width(std::string(layout_str), &width_axis))
        << "adaptive_avg_pool1d: unsupported layout " << layout_str;
    output_size.push_back(data->shape[width_axis]);
  }

  return {
      topi::nn::adaptive_pool1d(data, output_size, topi::nn::kAvgPool, std::string(layout_str))};
}

/*!
 * \brief TE handler for 2-D adaptive average pooling.
 *
 * When output_size is None the (H, W) of the input are used, matching the
 * Python fallback that reads data.shape[idx_H] and data.shape[idx_W].
 *
 * Expected args layout:
 *   [0]  data        – te::Tensor
 *   [1]  output_size – Optional<Array<int64_t>>  (nullopt → use input H×W)
 *   [2]  layout      – String
 */
ffi::Array<te::Tensor> AdaptiveAvgPool2DTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto output_size_opt = args[1].cast<ffi::Optional<ffi::Array<int64_t>>>();
  auto layout_str = args[2].cast<ffi::String>();

  ffi::Array<tvm::PrimExpr> output_size;
  if (output_size_opt.defined()) {
    auto dtype = data->shape[0]->dtype;
    for (auto v : output_size_opt.value()) output_size.push_back(tvm::IntImm(dtype, v));
  } else {
    // output_size is None: derive H and W from the input shape.
    int height_axis = -1, width_axis = -1;
    TVM_FFI_ICHECK(topi::nn::find_height_width(std::string(layout_str), &height_axis, &width_axis))
        << "adaptive_avg_pool2d: unsupported layout " << layout_str;
    output_size.push_back(data->shape[height_axis]);
    output_size.push_back(data->shape[width_axis]);
  }

  return {topi::nn::adaptive_pool(data, output_size, topi::nn::kAvgPool, std::string(layout_str))};
}

/*!
 * \brief TE handler for 3-D adaptive average pooling.
 *
 * When output_size is None the (D, H, W) of the input are used, matching the
 * Python fallback that reads data.shape[idx_D], data.shape[idx_H], and
 * data.shape[idx_W].
 *
 * Expected args layout:
 *   [0]  data        – te::Tensor
 *   [1]  output_size – Optional<Array<int64_t>>  (nullopt → use input D×H×W)
 *   [2]  layout      – String
 */
ffi::Array<te::Tensor> AdaptiveAvgPool3DTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto output_size_opt = args[1].cast<ffi::Optional<ffi::Array<int64_t>>>();
  auto layout_str = args[2].cast<ffi::String>();

  ffi::Array<tvm::PrimExpr> output_size;
  if (output_size_opt.defined()) {
    auto dtype = data->shape[0]->dtype;
    for (auto v : output_size_opt.value()) output_size.push_back(tvm::IntImm(dtype, v));
  } else {
    // output_size is None: derive D, H and W from the input shape.
    int depth_axis = -1, height_axis = -1, width_axis = -1;
    TVM_FFI_ICHECK(topi::nn::find_depth_height_width(std::string(layout_str), &depth_axis,
                                                     &height_axis, &width_axis))
        << "adaptive_avg_pool3d: unsupported layout " << layout_str;
    output_size.push_back(data->shape[depth_axis]);
    output_size.push_back(data->shape[height_axis]);
    output_size.push_back(data->shape[width_axis]);
  }

  return {
      topi::nn::adaptive_pool3d(data, output_size, topi::nn::kAvgPool, std::string(layout_str))};
}

// ---------------------------------------------------------------------------
// Shared helper: build the args array and call MakeCallTE for regular pooling.
// Used by all six max/avg pool 1-D/2-D/3-D legalizers.
// ---------------------------------------------------------------------------
/*!
 * \brief Shared legalizer for regular (non-adaptive) pooling operators.
 *
 * Builds the argument list from the operator attributes and forwards to the
 * provided TE handler. Returns the original call unchanged if the input and
 * output layouts differ.
 *
 * \tparam AttrsT The concrete pooling attribute type (e.g. Pool2DAttrs).
 * \param bb The block builder.
 * \param call The pooling call to legalize.
 * \param op_name Human-readable operator name used in warning messages.
 * \param te_handler The TE compute handler for this pooling variant.
 * \param pool_type_str Either "max" or "avg".
 * \return The legalized call_tir expression, or the original call if the
 *         layout constraint is not satisfied.
 */
template <typename AttrsT>
static Expr LegalizePool(const BlockBuilder& bb, const Call& call, const char* op_name,
                         FTOPIHandler te_handler, const ffi::String& pool_type_str) {
  const auto* attrs = call->attrs.as<AttrsT>();

  if (attrs->out_layout != attrs->layout) {
    LOG(WARNING) << "TOPI " << op_name
                 << " does not support different input-output "
                    "layouts, and thus cannot be legalized by TOPI";
    return call;
  }

  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;

  args.push_back(call->args[0]);
  args.push_back(attrs->pool_size);
  args.push_back(attrs->strides);
  args.push_back(attrs->dilation);
  args.push_back(attrs->padding);
  args.push_back(pool_type_str);
  args.push_back(attrs->ceil_mode);
  args.push_back(attrs->layout);
  // For max pooling count_include_pad is irrelevant; pass the attr value
  // directly so avg pooling gets the correct setting from the same path.
  args.push_back(attrs->count_include_pad);

  return m_te.Make(args, te_handler, std::string(op_name));
}

// ---------------------------------------------------------------------------
// Legalizers
// ---------------------------------------------------------------------------

/*!
 * \brief Legalize relax.nn.max_pool1d to call_tir via Pool1DTE.
 *
 * \param bb The block builder.
 * \param call The relax.nn.max_pool1d call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeNNMaxPool1D(const BlockBuilder& bb, const Call& call) {
  return LegalizePool<Pool1DAttrs>(bb, call, "max_pool1d", FTOPIHandler(Pool1DTE),
                                   ffi::String("max"));
}
TVM_REGISTER_OP("relax.nn.max_pool1d")
    .set_attr<FLegalize>("FLegalize", LegalizeNNMaxPool1D, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Legalize relax.nn.max_pool2d to call_tir via Pool2DTE.
 *
 * \param bb The block builder.
 * \param call The relax.nn.max_pool2d call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeNNMaxPool2D(const BlockBuilder& bb, const Call& call) {
  return LegalizePool<Pool2DAttrs>(bb, call, "max_pool2d", FTOPIHandler(Pool2DTE),
                                   ffi::String("max"));
}
TVM_REGISTER_OP("relax.nn.max_pool2d")
    .set_attr<FLegalize>("FLegalize", LegalizeNNMaxPool2D, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Legalize relax.nn.max_pool3d to call_tir via Pool3DTE.
 *
 * \param bb The block builder.
 * \param call The relax.nn.max_pool3d call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeNNMaxPool3D(const BlockBuilder& bb, const Call& call) {
  return LegalizePool<Pool3DAttrs>(bb, call, "max_pool3d", FTOPIHandler(Pool3DTE),
                                   ffi::String("max"));
}
TVM_REGISTER_OP("relax.nn.max_pool3d")
    .set_attr<FLegalize>("FLegalize", LegalizeNNMaxPool3D, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Legalize relax.nn.avg_pool1d to call_tir via Pool1DTE.
 *
 * \param bb The block builder.
 * \param call The relax.nn.avg_pool1d call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeNNAvgPool1D(const BlockBuilder& bb, const Call& call) {
  return LegalizePool<Pool1DAttrs>(bb, call, "avg_pool1d", FTOPIHandler(Pool1DTE),
                                   ffi::String("avg"));
}
TVM_REGISTER_OP("relax.nn.avg_pool1d")
    .set_attr<FLegalize>("FLegalize", LegalizeNNAvgPool1D, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Legalize relax.nn.avg_pool2d to call_tir via Pool2DTE.
 *
 * \param bb The block builder.
 * \param call The relax.nn.avg_pool2d call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeNNAvgPool2D(const BlockBuilder& bb, const Call& call) {
  return LegalizePool<Pool2DAttrs>(bb, call, "avg_pool2d", FTOPIHandler(Pool2DTE),
                                   ffi::String("avg"));
}
TVM_REGISTER_OP("relax.nn.avg_pool2d")
    .set_attr<FLegalize>("FLegalize", LegalizeNNAvgPool2D, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Legalize relax.nn.avg_pool3d to call_tir via Pool3DTE.
 *
 * \param bb The block builder.
 * \param call The relax.nn.avg_pool3d call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeNNAvgPool3D(const BlockBuilder& bb, const Call& call) {
  return LegalizePool<Pool3DAttrs>(bb, call, "avg_pool3d", FTOPIHandler(Pool3DTE),
                                   ffi::String("avg"));
}
TVM_REGISTER_OP("relax.nn.avg_pool3d")
    .set_attr<FLegalize>("FLegalize", LegalizeNNAvgPool3D, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Legalize relax.nn.adaptive_avg_pool1d to call_tir via AdaptiveAvgPool1DTE.
 *
 * \param bb The block builder.
 * \param call The relax.nn.adaptive_avg_pool1d call to legalize.
 * \return The legalized call_tir expression, or the original call if the
 *         layout constraint is not satisfied.
 */
Expr LegalizeNNAdaptiveAvgPool1D(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<AdaptivePool1DAttrs>();

  if (attrs->out_layout != attrs->layout) {
    LOG(WARNING) << "TOPI adaptive_avg_pool1d does not support different input-output "
                 << "layouts, and thus cannot be legalized by TOPI";
    return call;
  }

  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;

  args.push_back(call->args[0]);
  args.push_back(attrs->output_size);  // Optional<Array<int64_t>>: nullopt handled in TE handler
  args.push_back(attrs->layout);

  return m_te.Make(args, FTOPIHandler(AdaptiveAvgPool1DTE), std::string("adaptive_avg_pool1d"));
}
TVM_REGISTER_OP("relax.nn.adaptive_avg_pool1d")
    .set_attr<FLegalize>("FLegalize", LegalizeNNAdaptiveAvgPool1D, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Legalize relax.nn.adaptive_avg_pool2d to call_tir via AdaptiveAvgPool2DTE.
 *
 * \param bb The block builder.
 * \param call The relax.nn.adaptive_avg_pool2d call to legalize.
 * \return The legalized call_tir expression, or the original call if the
 *         layout constraint is not satisfied.
 */
Expr LegalizeNNAdaptiveAvgPool2D(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<AdaptivePool2DAttrs>();

  if (attrs->out_layout != attrs->layout) {
    LOG(WARNING) << "TOPI adaptive_avg_pool2d does not support different input-output "
                 << "layouts, and thus cannot be legalized by TOPI";
    return call;
  }

  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;

  args.push_back(call->args[0]);
  args.push_back(attrs->output_size);  // Optional<Array<int64_t>>: nullopt handled in TE handler
  args.push_back(attrs->layout);

  return m_te.Make(args, FTOPIHandler(AdaptiveAvgPool2DTE), std::string("adaptive_avg_pool2d"));
}
TVM_REGISTER_OP("relax.nn.adaptive_avg_pool2d")
    .set_attr<FLegalize>("FLegalize", LegalizeNNAdaptiveAvgPool2D, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Legalize relax.nn.adaptive_avg_pool3d to call_tir via AdaptiveAvgPool3DTE.
 *
 * \param bb The block builder.
 * \param call The relax.nn.adaptive_avg_pool3d call to legalize.
 * \return The legalized call_tir expression, or the original call if the
 *         layout constraint is not satisfied.
 */
Expr LegalizeNNAdaptiveAvgPool3D(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<AdaptivePool3DAttrs>();

  if (attrs->out_layout != attrs->layout) {
    LOG(WARNING) << "TOPI adaptive_avg_pool3d does not support different input-output "
                 << "layouts, and thus cannot be legalized by TOPI";
    return call;
  }

  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;

  args.push_back(call->args[0]);
  args.push_back(attrs->output_size);  // Optional<Array<int64_t>>: nullopt handled in TE handler
  args.push_back(attrs->layout);

  return m_te.Make(args, FTOPIHandler(AdaptiveAvgPool3DTE), std::string("adaptive_avg_pool3d"));
}
TVM_REGISTER_OP("relax.nn.adaptive_avg_pool3d")
    .set_attr<FLegalize>("FLegalize", LegalizeNNAdaptiveAvgPool3D, TVM_LEGALIZE_CPP_LEVEL);

}  // namespace relax
}  // namespace tvm
