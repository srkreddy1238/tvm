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
 * \file src/relax/frontend/nn/llm/kv_cache_attention_decode_gpu_helpers.h
 * \brief Helper types and functions for the GPU batched-decode attention kernel.
 *
 * Shared between kv_cache_attention_decode_gpu.cc and any future variants.
 *
 * Responsibilities:
 *   - DecodeGpuConfig  : thread/tile configuration derived from kernel parameters.
 *   - ComputeDecodeGpuConfig : compute the config from (H_kv, H_qo, D, dtype, target).
 *   - BuildDecodeRopeExpr    : build the RoPE-applied element expression for one lane.
 *   - BuildDecodeKVLoadBody  : build the KV-load if/else body for one tile row.
 *   - BuildDecodeAllreduceBlock : build the bdz>1 allreduce block.
 */

#ifndef TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_ATTENTION_DECODE_GPU_HELPERS_H_
#define TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_ATTENTION_DECODE_GPU_HELPERS_H_

#include <tvm/target/target.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include <string>

#include "kv_cache_common.h"
#include "position_embedding.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

using namespace tvm::tir;

// ============================================================================
// Thread / tile configuration
// ============================================================================

/*!
 * \brief All static thread/tile parameters for the GPU decode kernel.
 *
 * Mirrors the Python computation in _attention_decode():
 *   VEC_SIZE = min(max(8 // dtype_bytes, D // 32), 4)
 *   bdx = D // VEC_SIZE
 *   bdy = GROUP_SIZE  (reduced while bdx*bdy > thread_limit)
 *   gdz = GROUP_SIZE // bdy
 *   bdz = max(thread_limit, bdx*bdy) // (bdx*bdy)
 *   tile_size_per_bdx = 2 if GROUP_SIZE==1 else 1  (1 on adreno)
 */
struct DecodeGpuConfig {
  int64_t VEC_SIZE;
  int64_t bdx;                ///< threadIdx.x extent  (= D / VEC_SIZE)
  int64_t bdy;                ///< threadIdx.y extent  (= reduced GROUP_SIZE)
  int64_t bdz;                ///< threadIdx.z extent
  int64_t gdz;                ///< blockIdx.y multiplier (= GROUP_SIZE / bdy)
  int64_t tile_size_per_bdx;  ///< KV rows loaded per (bdx,bdy) tile
  int64_t GROUP_SIZE;         ///< H_qo / H_kv
  std::string global_symbol;  ///< kernel name
};

/*!
 * \brief Compute DecodeGpuConfig from kernel parameters.
 *
 * \param H_kv          Number of KV heads.
 * \param H_qo          Number of QO heads.
 * \param D             Head dimension.
 * \param dtype         Element dtype string.
 * \param sliding_window Whether sliding-window attention is used.
 * \param target        Compilation target (used for thread-limit query).
 * \return              Fully populated DecodeGpuConfig.
 */
DecodeGpuConfig ComputeDecodeGpuConfig(int64_t H_kv, int64_t H_qo, int64_t D,
                                       const std::string& dtype, bool sliding_window,
                                       Target target);

// ============================================================================
// RoPE expression builder
// ============================================================================

/*!
 * \brief Build the RoPE-applied element expression for one vectorized lane.
 *
 * Produces:
 *   if_then_else(rotary_mode == 1,
 *     Let(d_var, d_expr,
 *       Let(freq_vars...,
 *         cast<dtype>(cos*cast<f32>(elem) + sin*cast<f32>(partner)))),
 *     elem)
 *
 * where partner = if d < D/2: buf[..., d+D/2]*(-1) else buf[..., d-D/2].
 *
 * \param buf           Buffer to load from.
 * \param base_indices  All indices except the last (d) dimension.
 * \param d_expr        The dimension index expression (e.g. tx*VEC_SIZE + vec).
 * \param D             Head dimension (static).
 * \param pos_expr      Rope position (int32 PrimExpr).
 * \param rope_scale    rope_scale kernel parameter.
 * \param rope_theta    rope_theta kernel parameter.
 * \param rotary_mode   rotary_mode kernel parameter.
 * \param dtype         Element dtype string.
 * \param rope_scaling  RoPE scaling config map.
 * \return              PrimExpr in `dtype` for the (possibly rotated) element.
 */
PrimExpr BuildDecodeRopeExpr(tir::Buffer buf, ffi::Array<PrimExpr> base_indices, PrimExpr d_expr,
                             int64_t D, PrimExpr pos_expr, tir::Var rope_scale, tir::Var rope_theta,
                             tir::Var rotary_mode, const std::string& dtype,
                             const ffi::Map<ffi::String, ffi::Any>& rope_scaling);

// ============================================================================
// KV-load body builder
// ============================================================================

/*!
 * \brief Build the if/else body for loading one KV tile row into shared memory.
 *
 * Produces:
 *   if row_g < kv_chunk_len[0]:
 *     seq_offset = ...
 *     page_no    = page_values[cur_begin + seq_offset // page_size]
 *     page_offset = seq_offset % page_size
 *     for vec in vectorized(VEC_SIZE):
 *       K_smem[tile_start_s + j, tx*VEC_SIZE+vec] = rope_or_raw(pages[page_no,0,...])
 *       V_smem[tile_start_s + j, tx*VEC_SIZE+vec] = pages[page_no,1,...]
 *   else:
 *     for vec in vectorized(VEC_SIZE):
 *       K_smem[...] = 0;  V_smem[...] = 0
 *
 * \param cfg            Kernel configuration.
 * \param pages_buf      Paged KV buffer.
 * \param page_values_buf Page-table values buffer.
 * \param length_info_buf Length-info buffer (shape depends on sliding_window).
 * \param k_rope_pos_offset_buf  k_rope_pos_offset buffer.
 * \param K_smem         K shared-memory buffer.
 * \param V_smem         V shared-memory buffer.
 * \param kv_chunk_len_buf  kv_chunk_len local buffer.
 * \param cur_begin      cur_page_indptr_begin let-var.
 * \param batch_idx      batch index let-var.
 * \param by_var         by (KV head index) let-var.
 * \param tx             threadIdx.x var.
 * \param tile_start_s   tile_start_s let-var.
 * \param tile_start_g   tile_start_g let-var.
 * \param j_var          inner tile-row loop var.
 * \param rope_scale     rope_scale kernel param.
 * \param rope_theta     rope_theta kernel param.
 * \param rotary_mode    rotary_mode kernel param.
 * \param rope_scaling   RoPE scaling config.
 * \return               Stmt for the KV_load sblock body.
 */
Stmt BuildDecodeKVLoadBody(const DecodeGpuConfig& cfg, tir::Buffer pages_buf,
                           tir::Buffer page_values_buf, tir::Buffer length_info_buf,
                           tir::Buffer k_rope_pos_offset_buf, tir::Buffer K_smem,
                           tir::Buffer V_smem, tir::Buffer kv_chunk_len_buf, tir::Var cur_begin,
                           tir::Var batch_idx, tir::Var by_var, tir::Var tx, PrimExpr tile_start_s,
                           PrimExpr tile_start_g, tir::Var j_var, tir::Var rope_scale,
                           tir::Var rope_theta, tir::Var rotary_mode,
                           const ffi::Map<ffi::String, ffi::Any>& rope_scaling);

// ============================================================================
// bdz allreduce block builder
// ============================================================================

/*!
 * \brief Build the bdz>1 allreduce block that merges partial results across tz.
 *
 * Produces the block:
 *   store O_local → O_allreduce[tz, ty, :]
 *   store st_m, st_d → md_allreduce[tz, ty, :]
 *   sync
 *   reset st_m=-5e4, st_d=1, O_local=0
 *   for j in bdz:
 *     load other_m, other_d, other_o from allreduce buffers
 *     merge into st_m, st_d, O_local
 *
 * \param cfg         Kernel configuration.
 * \param O_allreduce Shared O allreduce buffer.
 * \param md_allreduce Shared m/d allreduce buffer.
 * \param O_local     Local O buffer.
 * \param st_m        Local st_m buffer.
 * \param st_d        Local st_d buffer.
 * \param m_prev      Local m_prev buffer.
 * \param d_prev      Local d_prev buffer.
 * \param other_m     Local other_m buffer.
 * \param other_d     Local other_d buffer.
 * \param other_o     Local other_o buffer.
 * \param exp_mprev   Local exp_mprev buffer.
 * \param exp_otherm  Local exp_otherm buffer.
 * \param tz          threadIdx.z var.
 * \param ty          threadIdx.y var.
 * \param tx          threadIdx.x var.
 * \return            Stmt for the allreduce block (only emitted when bdz > 1).
 */
Stmt BuildDecodeAllreduceBlock(const DecodeGpuConfig& cfg, tir::Buffer O_allreduce,
                               tir::Buffer md_allreduce, tir::Buffer O_local, tir::Buffer st_m,
                               tir::Buffer st_d, tir::Buffer m_prev, tir::Buffer d_prev,
                               tir::Buffer other_m, tir::Buffer other_d, tir::Buffer other_o,
                               tir::Buffer exp_mprev, tir::Buffer exp_otherm, tir::Var tz,
                               tir::Var ty, tir::Var tx);

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_ATTENTION_DECODE_GPU_HELPERS_H_
