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
 * \file tvm/relax/transform/legalize_ops/nn.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */
#include <tvm/relax/attrs/nn.h>
#include <tvm/topi/utils.h>

#include <numeric>
#include <vector>

#include "utils.h"

namespace tvm {
namespace relax {

/*!
 * \brief Compute the output shape after applying symmetric padding.
 *
 * \param data The tensor to be padded.
 * \param pad_before Per-axis leading pad extents.
 * \param pad_after Per-axis trailing pad extents.
 * \return The padded output shape.
 */
ffi::Array<tvm::PrimExpr> GetPaddedShape(const te::Tensor& data,
                                         const ffi::Array<tvm::PrimExpr>& pad_before,
                                         const ffi::Array<tvm::PrimExpr>& pad_after) {
  auto n = data->shape.size();
  TVM_FFI_ICHECK(pad_before.size() == n);
  TVM_FFI_ICHECK(pad_after.size() == n);

  ffi::Array<tvm::PrimExpr> out_shape;
  arith::Analyzer analyzer;
  for (size_t i = 0; i < n; ++i) {
    out_shape.push_back(analyzer.Simplify(data->shape[i] + pad_before[i] + pad_after[i]));
  }

  return out_shape;
}

/*!
 * \brief TE handler for reflect padding.
 *
 * Indices outside the original tensor are reflected back inward.
 *
 * \param args Packed argument list: [data, pad_before, pad_after,
 *             optional name, optional attrs].
 * \return A single-element array containing the padded output tensor.
 */
ffi::Array<te::Tensor> PadReflectTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto pad_before = args[1].cast<ffi::Array<tvm::PrimExpr>>();
  auto pad_after = args[2].cast<ffi::Array<tvm::PrimExpr>>();
  ffi::String name = "ReflectPadInput";
  if (args.size() > 3 && args[3].as<ffi::String>()) {
    name = args[3].cast<ffi::String>();
  }
  ffi::Map<ffi::String, ffi::Any> attrs({});
  if (args.size() > 3 && args[3].as<ffi::Map<ffi::String, ffi::Any>>()) {
    attrs = args[3].cast<ffi::Map<ffi::String, ffi::Any>>();
  }
  if (args.size() > 4 && args[4].as<ffi::Map<ffi::String, ffi::Any>>()) {
    attrs = args[4].cast<ffi::Map<ffi::String, ffi::Any>>();
  }

  auto n = data->shape.size();
  arith::Analyzer analyzer;
  auto out_shape = GetPaddedShape(data, pad_before, pad_after);

  return {tvm::te::compute(
      out_shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        ffi::Array<tvm::PrimExpr> index_tuple;

        for (size_t i = 0; i < n; ++i) {
          tvm::PrimExpr idx = indices[i];
          tvm::PrimExpr size = data->shape[i];
          tvm::PrimExpr before = pad_before[i];
          tvm::PrimExpr orig_idx = idx - before;

          tvm::PrimExpr reflect_idx = analyzer.Simplify(
              tvm::if_then_else(orig_idx < tvm::tir::make_const(idx->dtype, 0),
                                orig_idx * tvm::tir::make_const(idx->dtype, -1),
                                tvm::if_then_else(orig_idx >= size,
                                                  ((tvm::tir::make_const(idx->dtype, 2) * size) -
                                                   tvm::tir::make_const(idx->dtype, 2)) -
                                                      orig_idx,
                                                  orig_idx)));

          index_tuple.push_back(reflect_idx);
        }
        return data(index_tuple);
      },
      name, kInjective + ",pad", attrs)};
}

/*!
 * \brief TE handler for replicate (edge) padding.
 *
 * Indices outside the original tensor are clamped to the nearest edge value.
 *
 * \param args Packed argument list: [data, pad_before, pad_after,
 *             optional name, optional attrs].
 * \return A single-element array containing the padded output tensor.
 */
ffi::Array<te::Tensor> PadReplicateTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto pad_before = args[1].cast<ffi::Array<tvm::PrimExpr>>();
  auto pad_after = args[2].cast<ffi::Array<tvm::PrimExpr>>();
  ffi::String name = "ReplicatePadInput";
  if (args.size() > 3 && args[3].as<ffi::String>()) {
    name = args[3].cast<ffi::String>();
  }
  ffi::Map<ffi::String, ffi::Any> attrs({});
  if (args.size() > 3 && args[3].as<ffi::Map<ffi::String, ffi::Any>>()) {
    attrs = args[3].cast<ffi::Map<ffi::String, ffi::Any>>();
  }
  if (args.size() > 4 && args[4].as<ffi::Map<ffi::String, ffi::Any>>()) {
    attrs = args[4].cast<ffi::Map<ffi::String, ffi::Any>>();
  }

  auto n = data->shape.size();
  arith::Analyzer analyzer;
  auto out_shape = GetPaddedShape(data, pad_before, pad_after);

  return {tvm::te::compute(
      out_shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        ffi::Array<tvm::PrimExpr> index_tuple;

        for (size_t i = 0; i < n; ++i) {
          tvm::PrimExpr idx = indices[i];
          tvm::PrimExpr size = data->shape[i];
          tvm::PrimExpr before = pad_before[i];
          tvm::PrimExpr orig_idx = idx - before;

          tvm::PrimExpr clamped_idx = analyzer.Simplify(tvm::if_then_else(
              orig_idx < tvm::tir::make_const(idx->dtype, 0), tvm::tir::make_const(idx->dtype, 0),
              tvm::if_then_else(orig_idx >= size, (size - tvm::tir::make_const(idx->dtype, 1)),
                                orig_idx)));

          index_tuple.push_back(clamped_idx);
        }
        return data(index_tuple);
      },
      name, kInjective + ",pad", attrs)};
}

/*!
 * \brief TE handler for circular padding.
 *
 * Indices outside the original tensor are wrapped using floor-modulo.
 *
 * \param args Packed argument list: [data, pad_before, pad_after,
 *             optional name, optional attrs].
 * \return A single-element array containing the padded output tensor.
 */
ffi::Array<te::Tensor> PadCircularTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto pad_before = args[1].cast<ffi::Array<tvm::PrimExpr>>();
  auto pad_after = args[2].cast<ffi::Array<tvm::PrimExpr>>();
  ffi::String name = "CircularPadInput";
  if (args.size() > 3 && args[3].as<ffi::String>()) {
    name = args[3].cast<ffi::String>();
  }
  ffi::Map<ffi::String, ffi::Any> attrs({});
  if (args.size() > 3 && args[3].as<ffi::Map<ffi::String, ffi::Any>>()) {
    attrs = args[3].cast<ffi::Map<ffi::String, ffi::Any>>();
  }
  if (args.size() > 4 && args[4].as<ffi::Map<ffi::String, ffi::Any>>()) {
    attrs = args[4].cast<ffi::Map<ffi::String, ffi::Any>>();
  }

  auto n = data->shape.size();
  arith::Analyzer analyzer;
  auto out_shape = GetPaddedShape(data, pad_before, pad_after);

  return {tvm::te::compute(
      out_shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        ffi::Array<tvm::PrimExpr> index_tuple;

        for (size_t i = 0; i < n; ++i) {
          tvm::PrimExpr idx = indices[i];
          tvm::PrimExpr size = data->shape[i];
          tvm::PrimExpr before = pad_before[i];
          tvm::PrimExpr orig_idx = idx - before;

          tvm::PrimExpr wrapped_idx = analyzer.Simplify(tvm::floormod(orig_idx + size, size));

          index_tuple.push_back(wrapped_idx);
        }
        return data(index_tuple);
      },
      name, kInjective + ",pad", attrs)};
}

ffi::Array<te::Tensor> PadTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto pad_before = args[1].cast<ffi::Array<tvm::PrimExpr>>();
  auto pad_after = args[2].cast<ffi::Array<tvm::PrimExpr>>();
  auto pad_value = args[3].cast<tvm::PrimExpr>();
  ffi::String name = "PadInput";
  if (args.size() > 4 && args[4].as<ffi::String>()) {
    name = args[4].cast<ffi::String>();
  }
  ffi::Map<ffi::String, ffi::Any> attrs({});
  if (args.size() > 4 && args[4].as<ffi::Map<ffi::String, ffi::Any>>()) {
    attrs = args[4].cast<ffi::Map<ffi::String, ffi::Any>>();
  }
  if (args.size() > 5 && args[5].as<ffi::Map<ffi::String, ffi::Any>>()) {
    attrs = args[5].cast<ffi::Map<ffi::String, ffi::Any>>();
  }

  auto n = data->shape.size();
  arith::Analyzer analyzer;
  auto out_shape = GetPaddedShape(data, pad_before, pad_after);

  return {tvm::te::compute(
      out_shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        ffi::Array<tvm::PrimExpr> not_zero;
        ffi::Array<tvm::PrimExpr> index_tuple;
        for (size_t i = 0; i < n; ++i) {
          auto before_val_s = analyzer.Simplify(pad_before[i]);
          auto after_val_s = analyzer.Simplify(pad_after[i]);
          if (before_val_s.as<IntImmNode>() && after_val_s.as<IntImmNode>() &&
              before_val_s.as<IntImmNode>()->value == 0 &&
              after_val_s.as<IntImmNode>()->value == 0) {
            index_tuple.push_back(indices[i]);
          } else {
            index_tuple.push_back(indices[i] - pad_before[i]);
            not_zero.push_back(indices[i] >= pad_before[i]);
            not_zero.push_back(indices[i] < data->shape[i] + pad_before[i]);
          }
        }
        if (not_zero.size()) {
          tvm::PrimExpr not_zero_all = not_zero[0];
          for (size_t i = 1; i < not_zero.size(); ++i) {
            not_zero_all = tvm::logical_and(analyzer.Simplify(not_zero_all), not_zero[i]);
          }
          return tvm::if_then_else(not_zero_all, data(index_tuple), pad_value);
        }
        return data(index_tuple);
      },
      name, kInjective + ",pad", attrs)};
}

/*!
 * \brief Legalize relax.nn.pad to call_tir.
 *
 * Dispatches to PadReflectTE, PadReplicateTE, PadCircularTE, or PadTE
 * depending on the `pad_mode` attribute.
 *
 * \param bb The block builder.
 * \param call The relax.nn.pad call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeNNPad(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<PadAttrs>();
  ffi::Array<tvm::PrimExpr> pad_before;
  ffi::Array<tvm::PrimExpr> pad_after;

  for (size_t i = 0; i < attrs->pad_width.size() / 2; ++i) {
    pad_before.push_back(attrs->pad_width[i * 2]);
    pad_after.push_back(attrs->pad_width[i * 2 + 1]);
  }

  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;

  args.push_back(call->args[0]);
  args.push_back(pad_before);
  args.push_back(pad_after);

  if (attrs->pad_mode == "reflect") {
    return m_te.Make(args, FTOPIHandler(PadReflectTE), std::string("reflect_pad"));
  } else if (attrs->pad_mode == "replicate" || attrs->pad_mode == "edge") {
    return m_te.Make(args, FTOPIHandler(PadReplicateTE), std::string("replicate_pad"));
  } else if (attrs->pad_mode == "circular") {
    return m_te.Make(args, FTOPIHandler(PadCircularTE), std::string("circular_pad"));
  } else {
    args.push_back(attrs->pad_value);
    return m_te.Make(args, FTOPIHandler(PadTE), std::string("pad"));
  }
}
TVM_REGISTER_OP("relax.nn.pad")
    .set_attr<FLegalize>("FLegalize", LegalizeNNPad, TVM_LEGALIZE_CPP_LEVEL);

}  // namespace relax
}  // namespace tvm
