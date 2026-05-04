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
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/var.h>

#include <functional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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
  /*! \brief Ordered list of (var, definition) pairs for intermediate variables.
   *  Insertion order is preserved so callers can bind them with tir::Let in
   *  the correct dependency order (innermost binding last). */
  std::vector<std::pair<Var, PrimExpr>> var_map;
};

/*!
 * \brief Signature for RoPE frequency computation functions.
 *
 * \param s      Position expression (float32 PrimExpr, e.g. Cast(f32, Cast(dtype, loop_s +
 * offset))).
 * \param d      Dimension index (PrimExpr; may be a plain tir::Var or a compound
 *               expression such as tx*4+vec for vectorized GPU kernels).
 * \param d_range  Maximum dimension index (rotary_dim).
 * \param theta  Base frequency (typically 10000.0).
 * \param dtype  Output data type string.
 * \param extra_args  Additional arguments specific to the scaling type.
 * \return       RopeFreqResult containing cos/sin and intermediate variables.
 */
using RopeFreqFunc = std::function<RopeFreqResult(
    PrimExpr s, PrimExpr d, int64_t d_range, PrimExpr theta, const std::string& dtype,
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
RopeFreqResult RopeFreqDefault(PrimExpr s, PrimExpr d, int64_t d_range, PrimExpr theta,
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
RopeFreqResult RopeFreqGptj(PrimExpr s, PrimExpr d, int64_t d_range, PrimExpr theta,
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
RopeFreqResult RopeFreqLlama3(PrimExpr s, PrimExpr d, int64_t d_range, PrimExpr theta,
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
RopeFreqResult RopeFreqLlama4(PrimExpr s, PrimExpr d, int64_t d_range, PrimExpr theta,
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
RopeFreqResult RopeFreqLongrope(PrimExpr s, PrimExpr d, int64_t d_range, PrimExpr theta,
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
RopeFreqResult RopeFreqYarn(PrimExpr s, PrimExpr d, int64_t d_range, PrimExpr theta,
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

/*!
 * \brief Return the TIR PrimFunc for Llama-style RoPE with a position map.
 *
 * Corresponds to the Python `llama_rope_with_position_map` function.
 * The returned PrimFunc accepts:
 *   (var_qkv, var_position_map, var_q, var_k, var_v, apply_rope)
 * and writes the split + rotated tensors into q, k, v.
 *
 * For longrope scaling the caller should use the longrope variant directly;
 * this function handles all other scaling types.
 *
 * \param theta         Base frequency.
 * \param scale         Position scaling factor.
 * \param head_dim      Head dimension (static).
 * \param num_q_heads   Number of query heads.
 * \param num_kv_heads  Number of key/value heads.
 * \param dtype         Element dtype string.
 * \param rope_scaling  RoPE scaling configuration dictionary.
 * \param rotary_dim    Dimensions to rotate; defaults to head_dim when nullopt.
 * \return              TIR PrimFunc ready for use with tensor_ir_op.
 */
tir::PrimFunc LlamaRopeWithPositionMap(double theta, double scale, int64_t head_dim,
                                       int64_t num_q_heads, int64_t num_kv_heads,
                                       const std::string& dtype,
                                       const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
                                       ffi::Optional<int64_t> rotary_dim);

/*!
 * \brief Return the TIR PrimFunc for Llama-4-style RoPE with a position map.
 *
 * Identical in structure to LlamaRopeWithPositionMap but uses the Llama-4
 * adjacent-pair (gptj-style) rotation pattern together with the llama4
 * frequency scaling formula.
 *
 * \param theta         Base frequency.
 * \param scale         Position scaling factor.
 * \param head_dim      Head dimension (static).
 * \param num_q_heads   Number of query heads.
 * \param num_kv_heads  Number of key/value heads.
 * \param dtype         Element dtype string.
 * \param rope_scaling  RoPE scaling configuration dictionary.
 * \param rotary_dim    Dimensions to rotate; defaults to head_dim when nullopt.
 * \return              TIR PrimFunc ready for use with tensor_ir_op.
 */
tir::PrimFunc Llama4RopeWithPositionMap(double theta, double scale, int64_t head_dim,
                                        int64_t num_q_heads, int64_t num_kv_heads,
                                        const std::string& dtype,
                                        const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
                                        ffi::Optional<int64_t> rotary_dim);

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_LLM_POSITION_EMBEDDING_H_
