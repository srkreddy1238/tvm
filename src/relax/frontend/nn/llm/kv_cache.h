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
 * \file src/relax/frontend/nn/llm/kv_cache.h
 * \brief KV cache TIR kernel generators for LLM paged attention.
 *
 * Provides C++ implementations of the paged KV cache TIR kernel generators
 * that were previously implemented in Python using TVMScript.
 *
 * Supported kernels:
 *   Transpose-append:
 *     - KVCacheTransposeAppend
 *     - KVCacheTransposeAppendMLA
 *   Debug get-KV:
 *     - KVCacheDebugGetKV
 *     - KVCacheDebugGetKVMLA
 *   Copy single page:
 *     - CopySinglePageCpu
 *     - CopySinglePage (GPU)
 *     - CopySinglePageMLA (GPU)
 *   Compact KV copy:
 *     - CompactKVCopyCpu
 *     - CompactKVCopy (GPU)
 *   Merge state inplace:
 *     - MergeStateInplaceCpu
 *     - MergeStateInplace (GPU)
 *   Attention prefill:
 *     - AttentionPrefillCpu
 *     - AttentionPrefill (GPU)
 *     - AttentionPrefillRagged (GPU)
 *     - AttentionPrefillRaggedCpu
 *     - AttentionSequencePrefill (GPU)
 *     - AttentionPrefillMLA (GPU)
 *   Attention decode:
 *     - AttentionDecodeCpu
 *     - AttentionDecode (GPU)
 */

#ifndef TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_H_
#define TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>

#include <string>

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

// ---------------------------------------------------------------------------
// Transpose-append kernels
// ---------------------------------------------------------------------------

/*!
 * \brief Generate TIR function that appends new k/v data to PagedKVCache.
 * \param num_key_value_heads  Number of KV heads.
 * \param head_dim             Head dimension.
 * \param dtype                Data type string (e.g. "float16").
 * \param page_size            Page size (default 16).
 * \return TIR PrimFunc.
 */
tir::PrimFunc KVCacheTransposeAppend(int64_t num_key_value_heads, int64_t head_dim,
                                     const std::string& dtype, int64_t page_size = 16);

/*!
 * \brief Generate TIR function that appends compressed KV data to PagedKVCache for MLA.
 * \param d_qk      QK head dimension.
 * \param dtype     Data type string.
 * \param page_size Page size (default 16).
 * \return TIR PrimFunc.
 */
tir::PrimFunc KVCacheTransposeAppendMLA(int64_t d_qk, const std::string& dtype,
                                        int64_t page_size = 16);

// ---------------------------------------------------------------------------
// Debug get-KV kernels
// ---------------------------------------------------------------------------

/*!
 * \brief Generate TIR function that fetches k/v data on given positions and layer.
 * \param num_hidden_layers    Number of hidden layers.
 * \param num_key_value_heads  Number of KV heads.
 * \param head_dim             Head dimension.
 * \param dtype                Data type string.
 * \return TIR PrimFunc.
 */
tir::PrimFunc KVCacheDebugGetKV(int64_t num_hidden_layers, int64_t num_key_value_heads,
                                int64_t head_dim, const std::string& dtype);

/*!
 * \brief Generate TIR function that fetches MLA k/v data on given positions and layer.
 * \param num_hidden_layers  Number of hidden layers.
 * \param d_qk               QK head dimension.
 * \param dtype              Data type string.
 * \return TIR PrimFunc.
 */
tir::PrimFunc KVCacheDebugGetKVMLA(int64_t num_hidden_layers, int64_t d_qk,
                                   const std::string& dtype);

// ---------------------------------------------------------------------------
// Copy single page kernels
// ---------------------------------------------------------------------------

/*!
 * \brief Generate CPU TIR function that copies a single page.
 * \param num_key_value_heads  Number of KV heads.
 * \param page_size            Page size.
 * \param head_dim             Head dimension.
 * \param dtype                Data type string.
 * \return TIR PrimFunc.
 */
tir::PrimFunc CopySinglePageCpu(int64_t num_key_value_heads, int64_t page_size, int64_t head_dim,
                                const std::string& dtype);

/*!
 * \brief Generate GPU TIR function that copies a single page.
 * \param num_key_value_heads  Number of KV heads.
 * \param page_size            Page size.
 * \param head_dim             Head dimension.
 * \param dtype                Data type string.
 * \param target               Target device.
 * \return TIR PrimFunc.
 */
tir::PrimFunc CopySinglePage(int64_t num_key_value_heads, int64_t page_size, int64_t head_dim,
                             const std::string& dtype, Target target);

/*!
 * \brief Generate GPU TIR function that copies a single MLA page.
 * \param page_size  Page size.
 * \param d_qk       QK head dimension.
 * \param dtype      Data type string.
 * \param target     Target device.
 * \return TIR PrimFunc.
 */
tir::PrimFunc CopySinglePageMLA(int64_t page_size, int64_t d_qk, const std::string& dtype,
                                Target target);

// ---------------------------------------------------------------------------
// Compact KV copy kernels
// ---------------------------------------------------------------------------

/*!
 * \brief Generate CPU TIR function for compact KV copy.
 * \param num_key_value_heads  Number of KV heads.
 * \param head_dim             Head dimension.
 * \param dtype                Data type string.
 * \return TIR PrimFunc.
 */
tir::PrimFunc CompactKVCopyCpu(int64_t num_key_value_heads, int64_t head_dim,
                               const std::string& dtype);

/*!
 * \brief Generate GPU TIR function for compact KV copy.
 * \param num_key_value_heads  Number of KV heads.
 * \param head_dim             Head dimension.
 * \param dtype                Data type string.
 * \param target               Target device.
 * \return TIR PrimFunc.
 */
tir::PrimFunc CompactKVCopy(int64_t num_key_value_heads, int64_t head_dim,
                            const std::string& dtype, Target target);

// ---------------------------------------------------------------------------
// Merge state inplace kernels
// ---------------------------------------------------------------------------

/*!
 * \brief Generate CPU TIR function for merge state inplace.
 * \param dtype  Data type string.
 * \return TIR PrimFunc.
 */
tir::PrimFunc MergeStateInplaceCpu(const std::string& dtype);

/*!
 * \brief Generate GPU TIR function for merge state inplace.
 * \param num_attention_heads  Number of attention heads.
 * \param v_head_dim           V head dimension.
 * \param dtype                Data type string.
 * \param target               Target device.
 * \param global_symbol        Global symbol name for the function.
 * \return TIR PrimFunc.
 */
tir::PrimFunc MergeStateInplace(int64_t num_attention_heads, int64_t v_head_dim,
                                const std::string& dtype, Target target,
                                const std::string& global_symbol);

// ---------------------------------------------------------------------------
// Attention prefill kernels
// ---------------------------------------------------------------------------

/*!
 * \brief Generate CPU TIR function for batched prefill with paged KV cache.
 * \param h_kv          Number of KV heads.
 * \param h_q           Number of query heads.
 * \param d             Head dimension.
 * \param dtype         Data type string.
 * \param sliding_window Whether to use sliding window attention.
 * \param rope_scaling  RoPE scaling configuration.
 * \param page_size     Page size (default 16).
 * \return TIR PrimFunc.
 */
tir::PrimFunc AttentionPrefillCpu(int64_t h_kv, int64_t h_q, int64_t d, const std::string& dtype,
                                  bool sliding_window,
                                  const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
                                  int64_t page_size = 16);

/*!
 * \brief Generate GPU TIR function for batched prefill with paged KV cache.
 * \param h_kv          Number of KV heads.
 * \param h_q           Number of query heads.
 * \param d             Head dimension.
 * \param dtype         Data type string.
 * \param sliding_window Whether to use sliding window attention.
 * \param rope_scaling  RoPE scaling configuration.
 * \param target        Target device.
 * \param page_size     Page size (default 16).
 * \return TIR PrimFunc.
 */
tir::PrimFunc AttentionPrefill(int64_t h_kv, int64_t h_q, int64_t d, const std::string& dtype,
                               bool sliding_window,
                               const ffi::Map<ffi::String, ffi::Any>& rope_scaling, Target target,
                               int64_t page_size = 16);

/*!
 * \brief Generate GPU TIR function for ragged prefill.
 * \param h_kv          Number of KV heads.
 * \param h_q           Number of query heads.
 * \param d_qk          QK head dimension.
 * \param d_v           V head dimension.
 * \param dtype         Data type string.
 * \param rope_scaling  RoPE scaling configuration.
 * \param target        Target device.
 * \return TIR PrimFunc.
 */
tir::PrimFunc AttentionPrefillRagged(int64_t h_kv, int64_t h_q, int64_t d_qk, int64_t d_v,
                                     const std::string& dtype,
                                     const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
                                     Target target);

/*!
 * \brief Generate CPU TIR function for ragged prefill.
 * \param h_kv          Number of KV heads.
 * \param h_q           Number of query heads.
 * \param d_qk          QK head dimension.
 * \param d_v           V head dimension.
 * \param dtype         Data type string.
 * \param rope_scaling  RoPE scaling configuration.
 * \return TIR PrimFunc.
 */
tir::PrimFunc AttentionPrefillRaggedCpu(int64_t h_kv, int64_t h_q, int64_t d_qk, int64_t d_v,
                                        const std::string& dtype,
                                        const ffi::Map<ffi::String, ffi::Any>& rope_scaling);

/*!
 * \brief Generate GPU TIR function for sequence prefill (non-ragged).
 * \param h_kv          Number of KV heads.
 * \param h_q           Number of query heads.
 * \param d_qk          QK head dimension.
 * \param d_v           V head dimension.
 * \param dtype         Data type string.
 * \param rope_scaling  RoPE scaling configuration.
 * \param target        Target device.
 * \return TIR PrimFunc.
 */
tir::PrimFunc AttentionSequencePrefill(int64_t h_kv, int64_t h_q, int64_t d_qk, int64_t d_v,
                                       const std::string& dtype,
                                       const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
                                       Target target);

/*!
 * \brief Generate GPU TIR function for MLA prefill.
 * \param num_attention_heads  Number of attention heads.
 * \param v_head_dim           V head dimension.
 * \param qk_nope_head_dim     QK nope head dimension.
 * \param dtype                Data type string.
 * \param causal               Whether to use causal masking.
 * \param target               Target device.
 * \return TIR PrimFunc.
 */
tir::PrimFunc AttentionPrefillMLA(int64_t num_attention_heads, int64_t v_head_dim,
                                  int64_t qk_nope_head_dim, const std::string& dtype, bool causal,
                                  Target target);

// ---------------------------------------------------------------------------
// Attention decode kernels
// ---------------------------------------------------------------------------

/*!
 * \brief Generate CPU TIR function for batched decode with paged KV cache.
 * \param num_kv_heads   Number of KV heads.
 * \param num_qo_heads   Number of QO heads.
 * \param head_dim       Head dimension.
 * \param qkv_dtype      Data type string.
 * \param sliding_window Whether to use sliding window attention.
 * \param rope_scaling   RoPE scaling configuration.
 * \param page_size      Page size (default 16).
 * \return TIR PrimFunc.
 */
tir::PrimFunc AttentionDecodeCpu(int64_t num_kv_heads, int64_t num_qo_heads, int64_t head_dim,
                                 const std::string& qkv_dtype, bool sliding_window,
                                 const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
                                 int64_t page_size = 16);

/*!
 * \brief Generate GPU TIR function for batched decode with paged KV cache.
 * \param num_kv_heads   Number of KV heads.
 * \param num_qo_heads   Number of QO heads.
 * \param head_dim       Head dimension.
 * \param qkv_dtype      Data type string.
 * \param sliding_window Whether to use sliding window attention.
 * \param rope_scaling   RoPE scaling configuration.
 * \param target         Target device.
 * \param page_size      Page size (default 16).
 * \return TIR PrimFunc.
 */
tir::PrimFunc AttentionDecode(int64_t num_kv_heads, int64_t num_qo_heads, int64_t head_dim,
                              const std::string& qkv_dtype, bool sliding_window,
                              const ffi::Map<ffi::String, ffi::Any>& rope_scaling, Target target,
                              int64_t page_size = 16);

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_H_
