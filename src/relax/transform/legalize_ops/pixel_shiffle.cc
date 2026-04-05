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
 * \file tvm/relax/transform/legalize_ops/pixel_shuffle.cc
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

ffi::Array<te::Tensor> PixelShuffleTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto upscale_factor_i = args[1].cast<int>();
  TVM_FFI_ICHECK(upscale_factor_i > 0) << "upscale factor should be > 0";
  tvm::PrimExpr upscale_factor = tvm::IntImm(data->shape[0]->dtype, upscale_factor_i);

  auto n = data->shape.size();
  TVM_FFI_ICHECK(n >= 3) << "pixel_shuffle input shape dims should be >= 3";

  arith::Analyzer analyzer;
  auto c_in = data->shape[-3];
  auto h_in = data->shape[-2];
  auto w_in = data->shape[-1];

  auto c_out = analyzer.Simplify(tvm::tir::FloorDiv(c_in, upscale_factor * upscale_factor));
  auto h_out = analyzer.Simplify(h_in * upscale_factor);
  auto w_out = analyzer.Simplify(w_in * upscale_factor);

  ffi::Array<tvm::PrimExpr> out_shape;
  for (size_t i = 0; i < n - 3; ++i) {
    out_shape.push_back(data->shape[i]);
  }
  out_shape.push_back(c_out);
  out_shape.push_back(h_out);
  out_shape.push_back(w_out);

  return {tvm::te::compute(
      out_shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        ffi::Array<tvm::PrimExpr> index_tuple;
        for (size_t i = 0; i < n - 3; ++i) {
          index_tuple.push_back(indices[i]);
        }
        tvm::PrimExpr c_out_idx = indices[-3];
        tvm::PrimExpr h_out_idx = indices[-2];
        tvm::PrimExpr w_out_idx = indices[-1];

        auto h_idx = tvm::tir::FloorDiv(h_out_idx, upscale_factor);
        auto h_offset = tvm::tir::FloorMod(h_out_idx, upscale_factor);

        auto w_idx = tvm::tir::FloorDiv(w_out_idx, upscale_factor);
        auto w_offset = tvm::tir::FloorMod(w_out_idx, upscale_factor);

        auto c_in_idx = analyzer.Simplify((c_out_idx * upscale_factor * upscale_factor) +
                                          (h_offset * upscale_factor) + w_offset);

        index_tuple.push_back(c_in_idx);
        index_tuple.push_back(h_idx);
        index_tuple.push_back(w_idx);

        return data(index_tuple);
      },
      "PixelShuffle")};
}

// PixelShuffle
Expr LegalizeNNPixelShuffle(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<PixelShuffleAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;

  args.push_back(call->args[0]);
  args.push_back(attrs->upscale_factor);

  return m_te.Make(args, FTOPIHandler(PixelShuffleTE), std::string("pixel_shuffle"));
}
TVM_REGISTER_OP("relax.nn.pixel_shuffle")
    .set_attr<FLegalize>("FLegalize", LegalizeNNPixelShuffle, TVM_LEGALIZE_CPP_LEVEL);

}  // namespace relax
}  // namespace tvm
