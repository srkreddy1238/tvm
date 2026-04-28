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
 * \file src/relax/frontend/nn/llm/position_embedding.h
 * \brief Rotary position embedding (RoPE) implementations for LLM attention.
 *
 * This module provides various RoPE frequency computation functions and
 * the main llama_rope operator that applies rotary embeddings to fused QKV tensors.
 *
 * Supported RoPE scaling types:
 *   - default: Standard RoPE without scaling
 *   - gptj: GPT-J style RoPE with different rotation pattern
 *   - llama3: LLaMA-3 style linear interpolation scaling
 *   - llama4: LLaMA-4 style smooth interpolation scaling
 *   - longrope: Long-context RoPE with extension factors
 *   - yarn: YaRN (Yet another RoPE extensioN) scaling
 */

#ifndef TVM_RELAX_FRONTEND_NN_LLM_POSITION_EMBEDDING_H_
#define TVM_RELAX_FRONTEND_NN_LLM_POSITION_EMBEDDING_H_

#include <tvm/relax/expr.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/var.h>

#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// Forward declarations
class NNTensor;

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

// Import TVM types into this namespace for convenience
using tvm::PrimExpr;
using tvm::tir::Var;

/*!
 * \brief Result of RoPE frequency computation.
 *
 * Contains the cosine and sine of the frequency, plus a map of intermediate
 * variables that should be bound via tir::Let when constructing the final expression.
 */
struct RopeFreqResult {
  /*! \brief Cosine of the frequency. */
  PrimExpr cos_freq;
  /*! \brief Sine of the frequency. */
  PrimExpr sin_freq;
  /*! \brief Map of intermediate variables to their definitions. */
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> var_map;
};

/*!
 * \brief Signature for RoPE frequency computation functions.
 *
 * \param s      Position index (tir::Var).
 * \param d      Dimension index (tir::Var).
 * \param d_range  Maximum dimension index (rotary_dim).
 * \param theta  Base frequency (typically 10000.0).
 * \param dtype  Output data type string.
 * \param extra_args  Additional arguments specific to the scaling type.
 * \return       RopeFreqResult containing cos/sin and intermediate variables.
 */
using RopeFreqFunc = std::function<RopeFreqResult(
    Var s, Var d, int64_t d_range, double theta, const std::string& dtype,
    const ffi::Map<ffi::String, ffi::Any>& extra_args)>;

/*!
 * \brief Default RoPE frequency computation (no scaling).
 *
 * Computes: freq = s / (theta ^ (d * 2 % d_range / d_range))
 *
 * \param s      Position index.
 * \param d      Dimension index.
 * \param d_range  Rotary dimension.
 * \param theta  Base frequency.
 * \param dtype  Output data type.
 * \param extra_args  Unused for default mode.
 * \return       Cosine and sine of the frequency.
 */
RopeFreqResult RopeFreqDefault(Var s, Var d, int64_t d_range, double theta,
                               const std::string& dtype,
                               const ffi::Map<ffi::String, ffi::Any>& extra_args);

/*!
 * \brief GPT-J style RoPE frequency computation.
 *
 * Uses a different rotation pattern: freq = s / (theta ^ (2 * (d // 2) % d_range / d_range))
 *
 * \param s      Position index.
 * \param d      Dimension index.
 * \param d_range  Rotary dimension.
 * \param theta  Base frequency.
 * \param dtype  Output data type.
 * \param extra_args  Unused for gptj mode.
 * \return       Cosine and sine of the frequency.
 */
RopeFreqResult RopeFreqGptj(Var s, Var d, int64_t d_range, double theta,
                            const std::string& dtype,
                            const ffi::Map<ffi::String, ffi::Any>& extra_args);

/*!
 * \brief LLaMA-3 style RoPE frequency computation with linear interpolation.
 *
 * Applies smooth interpolation between scaled and unscaled frequencies based on
 * the original max position embeddings.
 *
 * Required extra_args:
 *   - factor: Scaling factor
 *   - low_freq_factor: Low frequency threshold
 *   - high_freq_factor: High frequency threshold
 *   - original_max_position_embeddings: Original context length
 *
 * \param s      Position index.
 * \param d      Dimension index.
 * \param d_range  Rotary dimension.
 * \param theta  Base frequency.
 * \param dtype  Output data type.
 * \param extra_args  Scaling configuration.
 * \return       Cosine and sine of the frequency.
 */
RopeFreqResult RopeFreqLlama3(Var s, Var d, int64_t d_range, double theta,
                              const std::string& dtype,
                              const ffi::Map<ffi::String, ffi::Any>& extra_args);

/*!
 * \brief LLaMA-4 style RoPE frequency computation with smooth interpolation.
 *
 * Similar to LLaMA-3 but with a different interpolation formula and support
 * for uniform scaling when low_freq_factor == high_freq_factor.
 *
 * Required extra_args: Same as RopeFreqLlama3.
 *
 * \param s      Position index.
 * \param d      Dimension index.
 * \param d_range  Rotary dimension.
 * \param theta  Base frequency.
 * \param dtype  Output data type.
 * \param extra_args  Scaling configuration.
 * \return       Cosine and sine of the frequency.
 */
RopeFreqResult RopeFreqLlama4(Var s, Var d, int64_t d_range, double theta,
                              const std::string& dtype,
                              const ffi::Map<ffi::String, ffi::Any>& extra_args);

/*!
 * \brief Long-context RoPE with extension factors.
 *
 * Applies per-dimension extension factors to support longer contexts.
 *
 * Required extra_args:
 *   - max_position_embeddings: Target context length
 *   - original_max_position_embeddings: Original context length
 *   - ext_factors: Optional buffer of extension factors (one per rotary dimension)
 *
 * \param s      Position index.
 * \param d      Dimension index.
 * \param d_range  Rotary dimension.
 * \param theta  Base frequency.
 * \param dtype  Output data type.
 * \param extra_args  Scaling configuration.
 * \return       Cosine and sine of the frequency.
 */
RopeFreqResult RopeFreqLongrope(Var s, Var d, int64_t d_range, double theta,
                                const std::string& dtype,
                                const ffi::Map<ffi::String, ffi::Any>& extra_args);

/*!
 * \brief YaRN (Yet another RoPE extensioN) scaling.
 *
 * Applies frequency-dependent interpolation with correction ranges.
 *
 * Required extra_args:
 *   - original_max_position_embeddings: Original context length
 *   - scaling_factor: Overall scaling factor
 *   - beta_fast: Fast correction threshold
 *   - beta_slow: Slow correction threshold
 *   - inv_theta_log_scale: Precomputed 1 / (2 * log(theta))
 *
 * \param s      Position index.
 * \param d      Dimension index.
 * \param d_range  Rotary dimension.
 * \param theta  Base frequency.
 * \param dtype  Output data type.
 * \param extra_args  Scaling configuration.
 * \return       Cosine and sine of the frequency.
 */
RopeFreqResult RopeFreqYarn(Var s, Var d, int64_t d_range, double theta,
                            const std::string& dtype,
                            const ffi::Map<ffi::String, ffi::Any>& extra_args);

/*!
 * \brief Select the appropriate RoPE frequency function based on rope_scaling config.
 *
 * \param rope_scaling  Configuration dictionary with "rope_type" key.
 * \return              Function pointer to the selected RoPE frequency computation.
 */
RopeFreqFunc SwitchRopeFreqFunc(const ffi::Map<ffi::String, ffi::Any>& rope_scaling);

/*!
 * \brief Apply Llama-style RoPE to a fused QKV tensor.
 *
 * Given a fused QKV tensor of shape [batch, seq_len, num_q_heads + num_kv_heads * 2, head_dim],
 * this function:
 *   1. Splits it into Q, K, V tensors
 *   2. Applies rotary position embeddings to Q and K
 *   3. Returns the three separate tensors
 *
 * The rotation is applied to the first rotary_dim dimensions of each head.
 * If rotary_dim is not specified, it defaults to head_dim.
 *
 * \param qkv              Fused QKV tensor.
 * \param total_seq_len    Total sequence length (for computing position offsets).
 * \param theta            Base frequency (typically 10000.0).
 * \param scale            Position scaling factor.
 * \param num_q_heads      Number of query heads.
 * \param num_kv_heads     Number of key/value heads (for GQA/MQA).
 * \param rope_scaling     RoPE scaling configuration dictionary.
 * \param rotary_dim       Number of dimensions to apply RoPE to (default: head_dim).
 * \return                 Tuple of (Q, K, V) tensors.
 */
std::tuple<NNTensor, NNTensor, NNTensor> LlamaRope(
    NNTensor qkv, Var total_seq_len, double theta, double scale, int64_t num_q_heads,
    int64_t num_kv_heads, const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
    ffi::Optional<int64_t> rotary_dim);

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_LLM_POSITION_EMBEDDING_H_
