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
 * \file src/relax/frontend/nn/llm/kv_cache_attention_prefill_gpu_helpers.h
 * \brief Shared helpers for GPU prefill attention kernels (no scheduling).
 *
 * Provides:
 *   - PrefillKernelConfig  : thread/tile configuration (mirrors _get_prefill_kernel_config).
 *   - ComputePrefillKernelConfig : compute config from (h_kv, h_q, d, dtype, target).
 *   - BuildPrefillRopeExpr  : build the RoPE-applied element expression for prefill kernels.
 *   - GetKvChunkLen / GetSeqOffset : paged KV helpers.
 *   - DeclLengthInfo        : declare the length_info buffer (sliding window or not).
 */

#ifndef TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_ATTENTION_PREFILL_GPU_HELPERS_H_
#define TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_ATTENTION_PREFILL_GPU_HELPERS_H_

#include <tvm/target/target.h>
#include <tvm/tir/function.h>
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
// Thread / tile configuration  (mirrors _get_prefill_kernel_config)
// ============================================================================

struct PrefillKernelConfig {
  int64_t NUM_BLKS;
  int64_t LOAD_VEC;
  int64_t group_size;
  int64_t bdx;        // threadIdx.x = 32
  int64_t num_warps;  // threadIdx.y = 4
  int64_t tile_x;
  int64_t tile_y;
  int64_t tile_z;
};

/*!
 * \brief Compute PrefillKernelConfig from kernel parameters.
 * Mirrors Python _get_prefill_kernel_config().
 */
PrefillKernelConfig ComputePrefillKernelConfig(int64_t h_kv, int64_t h_q, int64_t d,
                                               const std::string& dtype, Target target);

// ============================================================================
// RoPE expression builder for prefill kernels
// ============================================================================

/*!
 * \brief Build the RoPE-applied element expression for a prefill kernel.
 *
 * Produces:
 *   T.if_then_else(rotary_mode == 1,
 *     T.Let(cos(freq)*elem + sin(freq)*partner, where={freq: ...}),
 *     elem)
 *
 * For float32: result is float32.
 * For float16: result is float16 (cast applied inside).
 */
PrimExpr BuildPrefillRopeExpr(tir::Buffer buf, ffi::Array<PrimExpr> base_indices, tir::Var d_idx,
                              int64_t d, PrimExpr pos_expr, tir::Var rope_scale,
                              tir::Var rope_theta, tir::Var rotary_mode, const std::string& dtype,
                              const ffi::Map<ffi::String, ffi::Any>& rope_scaling);

// ============================================================================
// Paged KV helpers  (mirrors _get_kv_chunk_len / _get_seq_offset / _declare_length_info)
// ============================================================================

/*!
 * \brief Compute kv_chunk_len from paged KV indptr and length_info.
 *
 * Non-sliding-window: (num_pages - 1) * page_size + length_info[seq_id]
 * Sliding-window:     (num_pages - 1) * page_size + length_info[0,seq_id]
 *                       - length_info[1,seq_id] + length_info[2,seq_id]
 */
PrimExpr GetKvChunkLen(PrimExpr num_pages, int64_t page_size, PrimExpr seq_id,
                       tir::Buffer length_info, bool sliding_window);

/*!
 * \brief Compute the actual page-table offset for a given position.
 *
 * Non-sliding-window: pos
 * Sliding-window: if pos < sink_size: pos  else: pos - sink_size + sw_offset
 */
PrimExpr GetSeqOffset(PrimExpr pos, PrimExpr seq_id, tir::Buffer length_info, bool sliding_window);

/*!
 * \brief Declare the length_info buffer.
 * Non-sliding-window: shape = (batch_size,)
 * Sliding-window:     shape = (3, batch_size)
 */
tir::Buffer DeclLengthInfo(tir::Var var_length_info, PrimExpr batch_size, bool sliding_window,
                           tir::Var elem_offset_var);

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_ATTENTION_PREFILL_GPU_HELPERS_H_
