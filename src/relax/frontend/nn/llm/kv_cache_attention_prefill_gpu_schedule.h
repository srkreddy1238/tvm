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
 * \file src/relax/frontend/nn/llm/kv_cache_attention_prefill_gpu_schedule.h
 * \brief Schedule pass for GPU prefill attention kernels.
 *
 * Provides SchedulePrefillKernel() which applies the standard tiling /
 * vectorization / thread-binding schedule to an unscheduled prefill TIR
 * function.  Mirrors Python _schedule_prefill_kernel().
 */

#ifndef TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_ATTENTION_PREFILL_GPU_SCHEDULE_H_
#define TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_ATTENTION_PREFILL_GPU_SCHEDULE_H_

#include "kv_cache_attention_prefill_gpu_helpers.h"

#include <tvm/tir/function.h>

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

/*!
 * \brief Apply the standard prefill schedule to an unscheduled TIR function.
 *
 * Mirrors Python _schedule_prefill_kernel().
 *
 * \param func             The unscheduled TIR PrimFunc (must have "main" in an IRModule).
 * \param cfg              The kernel configuration (tile sizes, thread counts).
 * \param transform_k_load Whether to transpose K_load write layout (ragged variant).
 * \param merged_qk_load   Whether K and V are merged into KV_load (MLA variant).
 * \return Scheduled PrimFunc with tir.is_scheduled=True attribute.
 */
tir::PrimFunc SchedulePrefillKernel(tir::PrimFunc func, const PrefillKernelConfig& cfg,
                                    bool transform_k_load, bool merged_qk_load);

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_LLM_KV_CACHE_ATTENTION_PREFILL_GPU_SCHEDULE_H_
