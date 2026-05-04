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
 * \file src/relax/frontend/nn/llm/kv_cache_attn_common.h
 * \brief Shared TIR builder helpers for attention kernels.
 *
 * Provides:
 *   - RoPE expression builder (CPU and GPU variants)
 *   - GPU prefill kernel template (shared by paged, ragged, MLA)
 *   - GPU decode kernel template
 *   - Paged KV buffer/indptr helpers
 */

#ifndef TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_ATTN_COMMON_H_
#define TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_ATTN_COMMON_H_

#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include "../../../../tir/ir/script/script_complete.h"
#include "kv_cache_common.h"
#include "position_embedding.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

using namespace tvm::tir;

// ============================================================================
// RoPE helpers
// ============================================================================

/*!
 * \brief Build the RoPE-applied expression for a single element.
 *
 * Matches the TVMScript pattern:
 *   T.if_then_else(rotary_mode == 1,
 *     T.Let(cast<dtype>(cos(freq)*cast<f32>(elem) + sin(freq)*cast<f32>(
 *       T.if_then_else(d < d/2, elem[d+d/2]*(-1), elem[d-d/2]))),
 *       where={freq: cast<f32>(pos)*rope_scale / pow(rope_theta, cast<f32>(d*2%d_range)/d_range)}),
 *     elem)
 *
 * For float32 dtype the outer cast is omitted.
 *
 * \param elem_expr      The current element (e.g. q[cur_L, h, j]).
 * \param partner_expr   The rotated partner element (e.g. q[cur_L, h, j+64] or j-64).
 * \param pos_expr       The rope position (int32 PrimExpr).
 * \param d_idx          The dimension index variable (int32).
 * \param d              The head dimension (static int64).
 * \param rope_scale     The rope_scale parameter var (float32).
 * \param rope_theta     The rope_theta parameter var (float32).
 * \param rotary_mode    The rotary_mode parameter var (int32).
 * \param dtype          The element dtype string.
 * \param rope_scaling   The rope scaling config.
 * \return PrimExpr for the (possibly rotated) element value in dtype.
 */
inline PrimExpr BuildRopeExpr(PrimExpr elem_expr, PrimExpr partner_expr, PrimExpr pos_expr,
                              tir::Var d_idx, int64_t d, tir::Var rope_scale, tir::Var rope_theta,
                              tir::Var rotary_mode, const std::string& dtype,
                              const ffi::Map<ffi::String, ffi::Any>& rope_scaling) {
  bool is_f16 = (dtype == "float16");

  // pos_f32 = cast<float32>(pos_expr) * rope_scale
  PrimExpr pos_f32 = CastTo(pos_expr, "float32") * rope_scale;

  RopeFreqFunc rope_freq_func = SwitchRopeFreqFunc(rope_scaling);
  auto freq_result = rope_freq_func(pos_f32, d_idx, d, rope_theta, dtype, rope_scaling);

  // cos_part = cos_freq * cast<f32>(elem)  (cos_freq is in dtype for f16)
  // sin_part = sin_freq * cast<f32>(partner * (-1) or partner)
  PrimExpr cos_part, sin_part;
  if (is_f16) {
    // cos_freq is float16, elem is float16 → cast both to float32 for arithmetic
    cos_part = CastTo(freq_result.cos_freq, "float32") * CastTo(elem_expr, "float32");
    sin_part = CastTo(freq_result.sin_freq, "float32") * CastTo(partner_expr, "float32");
  } else {
    cos_part = freq_result.cos_freq * elem_expr;
    sin_part = freq_result.sin_freq * partner_expr;
  }

  PrimExpr rope_val = cos_part + sin_part;

  // Wrap in Let bindings (innermost first, then outer)
  for (auto it = freq_result.var_map.rbegin(); it != freq_result.var_map.rend(); ++it) {
    rope_val = tir::Let(it->first, it->second, rope_val);
  }

  // Cast result to dtype
  PrimExpr rope_typed = is_f16 ? CastTo(rope_val, dtype) : rope_val;

  // if_then_else(rotary_mode == 1, rope_typed, elem_expr)
  return tvm::if_then_else(rotary_mode == I32(1), rope_typed, elem_expr);
}

/*!
 * \brief Build the partner element expression for standard RoPE (non-gptj).
 *
 * partner = if d < d/2: elem[d + d/2] * (-1)  else: elem[d - d/2]
 *
 * \param buf        The buffer to load from.
 * \param indices    The base indices (all except the d dimension).
 * \param d_idx      The dimension index variable.
 * \param d          The head dimension.
 * \param dtype      The element dtype string.
 * \return PrimExpr for the partner element.
 */
inline PrimExpr BuildRopePartner(tir::Buffer buf, ffi::Array<PrimExpr> base_indices, tir::Var d_idx,
                                 int64_t d, const std::string& dtype) {
  int64_t half_d = d / 2;
  DataType dt = DataType(runtime::StringToDLDataType(dtype));
  PrimExpr neg_one = tir::make_const(dt, -1.0);

  // indices with d+half_d
  ffi::Array<PrimExpr> idx_plus = base_indices;
  idx_plus.push_back(d_idx + I32(half_d));
  // indices with d-half_d
  ffi::Array<PrimExpr> idx_minus = base_indices;
  idx_minus.push_back(d_idx - I32(half_d));

  return tvm::if_then_else(d_idx < I32(half_d), tir::BufferLoad(buf, idx_plus) * neg_one,
                           tir::BufferLoad(buf, idx_minus));
}

// ============================================================================
// GPU prefill kernel template parameters
// ============================================================================

struct PrefillKernelConfig {
  int64_t h_kv;         // number of KV heads
  int64_t h_q;          // number of Q heads
  int64_t d_qk;         // QK head dimension
  int64_t d_v;          // V head dimension (= d_qk for standard attention)
  std::string dtype;    // element dtype
  bool sliding_window;  // whether sliding window is used
  int64_t page_size;    // page size (16)
  bool is_ragged;       // ragged (no paged KV) vs paged KV
  bool is_mla;          // MLA variant
  bool is_sequence;     // sequence prefill (non-ragged, non-paged)
  int64_t causal;       // causal flag (for sequence prefill: 0=non-causal, 1=causal)

  // Derived
  int64_t h_qo_per_group() const { return h_q / h_kv; }  // GQA ratio
  // Tile sizes (fixed from Python implementation)
  int64_t tile_q() const { return 32; }
  int64_t tile_kv() const { return 32; }
  // Thread config
  int64_t ty() const { return 4; }
  int64_t tx() const { return 32; }
  // blockIdx.y = h_kv (for paged/ragged) or h_kv (for MLA: 1)
  int64_t by_extent() const {
    if (is_mla) return 1;
    return h_kv;
  }
  // blockIdx.x = 16 (fixed)
  int64_t bx_extent() const { return 16; }
  // Q tile rows per thread group
  int64_t q_per_group() const { return is_mla ? h_q : h_qo_per_group(); }
};

// ============================================================================
// GPU decode kernel template parameters
// ============================================================================

struct DecodeKernelConfig {
  int64_t h_kv;
  int64_t h_q;
  int64_t d;
  std::string dtype;
  bool sliding_window;
  int64_t page_size;

  int64_t h_qo_per_group() const { return h_q / h_kv; }
  // Thread config from Python: ty=8, tx=32, tz=2, by=h_kv
  int64_t ty() const { return 8; }
  int64_t tx() const { return 32; }
  int64_t tz() const { return 2; }
  int64_t by_extent() const { return h_kv; }
};

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_ATTN_COMMON_H_
