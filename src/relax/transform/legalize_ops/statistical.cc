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
 * \file tvm/relax/transform/legalize_ops/statistical.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */
#include <tvm/relax/attrs/statistical.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/utils.h>

#include "utils.h"

namespace tvm {
namespace relax {

// median
// Std
// Variance
// Mean

inline PrimExpr _compute_shape_prod(const te::Tensor& x, ffi::Optional<ffi::Array<Integer>>& axis) {
  auto shape_prod = tvm::tir::make_const(x->shape[0]->dtype, 1);
  std::vector<int64_t> axis_arr;
  if (axis.has_value()) {
    for (auto val : axis.value()) {
      axis_arr.push_back(val.IntValue());
    }
  } else {
    for (size_t i = 0; i < x->shape.size(); ++i) {
      axis_arr.push_back(i);
    }
  }

  for (auto dim : axis_arr) {
    shape_prod = shape_prod * x->shape[dim];
  }
  return shape_prod;
}

ffi::Array<te::Tensor> _te_mean(const ffi::Array<ffi::Any> args) {
  te::Tensor x = args[0].cast<te::Tensor>();
  auto axis = tvm::topi::ArrayOrInt(args[1]);
  bool keepdims = args[2].cast<bool>();

  auto shape_prod = _compute_shape_prod(x, axis);
  auto res_sum = tvm::topi::sum(x, axis, keepdims);
  return {tvm::topi::divide(res_sum, shape_prod)};
}

ffi::Array<te::Tensor> _te_variance(const ffi::Array<ffi::Any> args) {
  /*
   * This version has better memory locality and performance
   * But may trigger some precision problems, so we will use the previous version now
   * mean = _te_mean(x, axis, keepdims)
   * return _te_mean(x * x, axis, keepdims) - mean * mean
   */
  te::Tensor x = args[0].cast<te::Tensor>();

  auto dev = tvm::topi::subtract(x, _te_mean({args[0], args[1], true})[0]);
  return _te_mean({tvm::topi::multiply(dev, dev), args[1], args[2]});
}

ffi::Array<te::Tensor> _te_std(const ffi::Array<ffi::Any> args) {
  return {tvm::topi::sqrt(_te_variance(args)[0])};
}

#if 0
te::Tensor ArgSort(te::Tensor x, int64_t axis, bool is_ascend, runtime::DataType dtype) {
  auto data_buf = tvm::tir::decl_buffer(x->shape, x->dtype, "data_buf", data_alignment=8)
  auto out_buf = tvm::tir::decl_buffer(x->shape, dtype, "out_buf", data_alignment=8)


/*

out = te.extern(
  data.shape,
  [data],
  lambda ins, outs: tvm.tir.call_packed(
    "tvm.contrib.sort.argsort", ins[0], outs[0], axis, is_ascend
  ),
  dtype=dtype,
  in_buffers=[data_buf],
  out_buffers=out_buf,
  name="argsort_cpu",
  tag="argsort_cpu",
)
*/
  auto out = tvm::te:ExternOp(std::string name, std::string tag, ffi::Map<ffi::String, ffi::Any> attrs,
                   ffi::Array<Tensor> inputs, ffi::Array<Buffer> input_placeholders,
                   ffi::Array<Buffer> output_placeholders, Stmt body);

}

ffi::Array<te::Tensor> _te_median(const ffi::Array<ffi::Any> args) {
  /*
   * currently only supports one axis or no axis ~ same pytorch
   * todo: support multiple axis ~ same numpy
   */
  te::Tensor x = args[0].cast<te::Tensor>();
  auto axis = tvm::topi::ArrayOrInt(args[1]);
  bool keepdims = args[2].cast<bool>();

  auto shape_prod = _compute_shape_prod(x, axis);
  auto mid_index = (shape_prod - tvm::tir::make_const(x->shape[0]->dtype, 1)) / tvm::tir::make_const(x->shape[0]->dtype, 2);

  int64_t ax;
  if (axis.has_value()) {
    x = tvm::topi::reshape(x, ffi::Array<PrimExpr>({shape_prod}));
    //ax = tvm::tir::make_const(x->shape[0]->dtype, -1);
    ax = -1;
  } else {
    ax = axis.value()[0]->value;
  }

  auto index_sorted = ArgSort(x, ax, true, x->shape[0]->dtype);
  auto x_sorted = tvm::topi::gather(x, ax, index_sorted);
  auto new_shape = x->shape;
  new_shape[ax] = 1;

  auto indices = tvm::topi::full(new_shape, mid_index, x->shape[0]->dtype);

  auto median_val = tvm::topi::gather(x_sorted, ax, indices);
  auto median_idx = tvm::topi::gather(index_sorted, ax, indices);

  if (!axis.has_value()) {
    if (keepdims) {
      return median_val;
    } else {
      tvm::topi::squeeze(median_val, axis);
    }
  }

  if(!keepdims) {
    median_val = tvm::topi::squeeze(median_val, axis);
    median_idx = tvm::topi::squeeze(median_idx, axis);
  }
  return {median_val, median_idx}
}
#endif

#define TVM_LEGALIZE_STATISTICAL_OP(OpName, TopiHandler)                                  \
  Expr MAKE_NAME(LegalizeStatistical, OpName)(const BlockBuilder& bb, const Call& call) { \
    const auto* attrs = call->attrs.as<StatisticalAttrs>();                               \
    auto m_te = MakeCallTE(bb, call);                                                     \
    tvm::ffi::Array<tvm::ffi::Any> args;                                                  \
    args.push_back(call->args[0]);                                                        \
    args.push_back(attrs->axis);                                                          \
    args.push_back(attrs->keepdims);                                                      \
    auto call_ret = m_te.Make(args, FTOPIHandler(TopiHandler), std::string(#OpName));     \
    return call_ret;                                                                      \
  }                                                                                       \
  TVM_REGISTER_OP("relax." #OpName)                                                       \
      .set_attr<FLegalize>("FLegalize", MAKE_NAME(LegalizeStatistical, OpName),           \
                           TVM_LEGALIZE_CPP_LEVEL);

TVM_LEGALIZE_STATISTICAL_OP(mean, _te_mean);
TVM_LEGALIZE_STATISTICAL_OP(variance, _te_variance);
TVM_LEGALIZE_STATISTICAL_OP(std, _te_std);
// TVM_LEGALIZE_STATISTICAL_OP(median, _te_median);

#define TVM_LEGALIZE_STATISTICAL_REDUCTION_OP(OpName, TopiHandler)                        \
  Expr MAKE_NAME(LegalizeStatistical, OpName)(const BlockBuilder& bb, const Call& call) { \
    const auto* attrs = call->attrs.as<StatisticalAttrs>();                               \
    auto m_te = MakeCallTE(bb, call);                                                     \
    tvm::ffi::Array<tvm::ffi::Any> args;                                                  \
    args.push_back(call->args[0]);                                                        \
    if (attrs->axis.has_value()) {                                                        \
      args.push_back(attrs->axis.value());                                                \
    } else {                                                                              \
      args.push_back(nullptr);                                                            \
    }                                                                                     \
    args.push_back(attrs->keepdims);                                                      \
    auto call_ret = m_te.Make(args, ffi::String(#TopiHandler), std::string(#OpName));     \
    return call_ret;                                                                      \
  }                                                                                       \
  TVM_REGISTER_OP("relax." #OpName)                                                       \
      .set_attr<FLegalize>("FLegalize", MAKE_NAME(LegalizeStatistical, OpName),           \
                           TVM_LEGALIZE_CPP_LEVEL);

TVM_LEGALIZE_STATISTICAL_REDUCTION_OP(min, topi.min);
TVM_LEGALIZE_STATISTICAL_REDUCTION_OP(max, topi.max);
TVM_LEGALIZE_STATISTICAL_REDUCTION_OP(prod, topi.prod);
TVM_LEGALIZE_STATISTICAL_REDUCTION_OP(sum, topi.sum);

}  // namespace relax
}  // namespace tvm
