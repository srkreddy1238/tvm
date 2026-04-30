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
 * \file src/relax/frontend/nn/llm/tree_attn.h
 * \brief Tree attention kernel generators for LLM batched tree attention.
 *
 * Provides C++ implementations of the tree attention TIR kernel generators
 * that were previously implemented in Python using TVMScript.
 *
 * Supported kernels:
 *   - TreeAttnCpu:                    CPU batched tree attention
 *   - TreeAttn:                       GPU batched tree attention
 *   - TreeAttnWithPagedKVCacheCpu:    CPU tree attention with paged KV cache
 *   - TreeAttnWithPagedKVCache:       GPU tree attention with paged KV cache
 */

#ifndef TVM_RELAX_FRONTEND_NN_LLM_TREE_ATTN_H_
#define TVM_RELAX_FRONTEND_NN_LLM_TREE_ATTN_H_

#include <tvm/target/target.h>
#include <tvm/tir/function.h>

#include <string>

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

/*!
 * \brief Generate CPU tree attention kernel for batched tree attention.
 *
 * \param h_kv         Number of key/value heads.
 * \param h_q          Number of query heads.
 * \param d            Head dimension.
 * \param dtype        Data type string (e.g. "float16").
 * \param rope_scaling RoPE scaling configuration dictionary.
 * \return             TIR PrimFunc implementing the CPU tree attention kernel.
 */
tir::PrimFunc TreeAttnCpu(int64_t h_kv, int64_t h_q, int64_t d, const std::string& dtype,
                          const ffi::Map<ffi::String, ffi::Any>& rope_scaling);

/*!
 * \brief Generate GPU tree attention kernel for batched tree attention.
 *
 * \param h_kv         Number of key/value heads.
 * \param h_q          Number of query heads.
 * \param d            Head dimension.
 * \param dtype        Data type string (e.g. "float16").
 * \param rope_scaling RoPE scaling configuration dictionary.
 * \param target       The target device.
 * \return             TIR PrimFunc implementing the GPU tree attention kernel.
 */
tir::PrimFunc TreeAttn(int64_t h_kv, int64_t h_q, int64_t d, const std::string& dtype,
                       const ffi::Map<ffi::String, ffi::Any>& rope_scaling, Target target);

/*!
 * \brief Generate CPU tree attention kernel with paged KV cache.
 *
 * \param h_kv         Number of key/value heads.
 * \param h_q          Number of query heads.
 * \param d            Head dimension.
 * \param dtype        Data type string (e.g. "float16").
 * \param rope_scaling RoPE scaling configuration dictionary.
 * \return             TIR PrimFunc implementing the CPU paged KV tree attention kernel.
 */
tir::PrimFunc TreeAttnWithPagedKVCacheCpu(int64_t h_kv, int64_t h_q, int64_t d,
                                          const std::string& dtype,
                                          const ffi::Map<ffi::String, ffi::Any>& rope_scaling);

/*!
 * \brief Generate GPU tree attention kernel with paged KV cache.
 *
 * \param h_kv         Number of key/value heads.
 * \param h_q          Number of query heads.
 * \param d            Head dimension.
 * \param dtype        Data type string (e.g. "float16").
 * \param rope_scaling RoPE scaling configuration dictionary.
 * \param target       The target device.
 * \return             TIR PrimFunc implementing the GPU paged KV tree attention kernel.
 */
tir::PrimFunc TreeAttnWithPagedKVCache(int64_t h_kv, int64_t h_q, int64_t d,
                                       const std::string& dtype,
                                       const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
                                       Target target);

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_LLM_TREE_ATTN_H_
