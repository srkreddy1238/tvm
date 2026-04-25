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
 * \file src/relax/frontend/nn/op.h
 * \brief C++ declarations for all nn.Tensor operators.
 *
 * These functions are implemented in op.cc and emit relax IR bindings into
 * the thread-local BlockBuilder (installed by the exporter before calling
 * forward()).  They return Var directly so that modules.cc can call them
 * without going through the FFI global registry.
 *
 * Tuple-producing ops (split, chunk, topk, tensor_expr_op, tensor_ir_op,
 * tensor_ir_inplace_op, extern) return ffi::Any (Array<Any> of element Vars)
 * because their arity is not statically known.
 */

#ifndef TVM_RELAX_FRONTEND_NN_OP_H_
#define TVM_RELAX_FRONTEND_NN_OP_H_

#include <tvm/ffi/function.h>
#include <tvm/relax/expr.h>
#include <tvm/tir/function.h>

#include <string>

#include "core.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ---------------------------------------------------------------------------
// Unary element-wise ops
// ---------------------------------------------------------------------------

Var NNRelu(Var x, ffi::String name = "relu");
Var NNRelu6(Var x, ffi::String name = "relu6");
Var NNSilu(Var x, ffi::String name = "silu");
Var NNGelu(Var x, ffi::Optional<ffi::String> approximate, ffi::String name = "gelu");
Var NNSigmoid(Var x, ffi::String name = "sigmoid");
Var NNTanh(Var x, ffi::String name = "tanh");
Var NNExp(Var x, ffi::String name = "exp");
Var NNLog(Var x, ffi::String name = "log");
Var NNFloor(Var x, ffi::String name = "floor");
Var NNSqrt(Var x, ffi::String name = "sqrt");
Var NNSquare(Var x, ffi::String name = "square");
Var NNNegative(Var x, ffi::String name = "negative");
Var NNSoftmax(Var x, int axis, ffi::String name = "softmax");
Var NNSoftplus(Var x, double beta, double threshold, ffi::String name = "softplus");
Var NNPrelu(Var x, Var alpha, ffi::String name = "prelu");

// ---------------------------------------------------------------------------
// Binary element-wise ops
// ---------------------------------------------------------------------------

Var NNAdd(Expr a, Expr b, ffi::String name = "add");
Var NNSubtract(Expr a, Expr b, ffi::String name = "subtract");
Var NNMultiply(Expr a, Expr b, ffi::String name = "multiply");
Var NNDivide(Expr a, Expr b, ffi::String name = "divide");
Var NNMaximum(Expr a, Expr b, ffi::String name = "maximum");
Var NNMinimum(Expr a, Expr b, ffi::String name = "minimum");
Var NNLess(Expr a, Expr b, ffi::String name = "less");
Var NNLessEqual(Expr a, Expr b, ffi::String name = "less_equal");
Var NNGreater(Expr a, Expr b, ffi::String name = "greater");
Var NNGreaterEqual(Expr a, Expr b, ffi::String name = "greater_equal");
Var NNEqual(Expr a, Expr b, ffi::String name = "equal");
Var NNNotEqual(Expr a, Expr b, ffi::String name = "not_equal");
Var NNWhere(Var condition, Var x1, Var x2, ffi::String name = "where");

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

Var NNUnsqueeze(Var x, int dim, ffi::String name = "unsqueeze");
Var NNSqueeze(Var x, int axis, ffi::String name = "squeeze");
Var NNReshape(Var x, ffi::Array<ffi::Any> shape, ffi::String name = "reshape");
Var NNPermuteDims(Var x, ffi::Optional<ffi::Array<Integer>> axes,
                  ffi::Optional<ffi::String> name = std::nullopt);
Var NNBroadcastTo(Var x, ffi::Array<ffi::Any> shape, ffi::String name = "broadcast_to");
Var NNRepeat(Var x, int repeats, ffi::Optional<Integer> axis, ffi::String name = "repeat");
Var NNConcat(ffi::Array<Var> tensors, int dim, ffi::String name = "concat");
// split / chunk produce tuples → return ffi::Any (Array<Any> of element Vars)
ffi::Any NNSplit(Var x, ffi::Any indices_or_sections, int axis, ffi::String name = "split");
ffi::Any NNChunk(Var x, int chunks, int dim, ffi::String name = "chunk");
Var NNTriu(Var x, int diagonal, ffi::String name = "triu");

// ---------------------------------------------------------------------------
// Reduction ops
// ---------------------------------------------------------------------------

Var NNSum(Var x, ffi::Optional<ffi::Array<Integer>> axis, bool keepdims,
          ffi::String name = "sum");
Var NNMax(Var x, ffi::Optional<ffi::Array<Integer>> axis, bool keepdims,
          ffi::String name = "max");
Var NNMin(Var x, ffi::Optional<ffi::Array<Integer>> axis, bool keepdims,
          ffi::String name = "min");
Var NNCumsum(Var x, ffi::Optional<Integer> axis, ffi::Optional<ffi::String> dtype,
             ffi::Optional<Bool> exclusive, ffi::String name = "cumsum");

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

Var NNMatmul(Var a, Var b, ffi::Optional<ffi::String> out_dtype = std::nullopt,
             ffi::String name = "matmul");

// ---------------------------------------------------------------------------
// Type casting
// ---------------------------------------------------------------------------

Var NNAstype(Var x, ffi::String dtype, ffi::String name = "astype");

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

Var NNTake(Var x, Var indices, ffi::Optional<Integer> axis, ffi::String name = "take");

// ---------------------------------------------------------------------------
// Creation ops
// ---------------------------------------------------------------------------

Var NNArange(ffi::Any start, ffi::Any end, ffi::Any step, ffi::Optional<ffi::String> dtype,
             ffi::String name = "arange");
Var NNFull(ffi::Array<ffi::Any> shape, Expr fill_value, ffi::String dtype,
           ffi::String name = "full");
Var NNZeros(ffi::Array<ffi::Any> shape, ffi::String dtype, ffi::String name = "zeros");
Var NNOnes(ffi::Array<ffi::Any> shape, ffi::String dtype, ffi::String name = "ones");

// ---------------------------------------------------------------------------
// Normalization ops
// ---------------------------------------------------------------------------

Var NNLayerNorm(Var x, ffi::Array<Integer> axes, Expr gamma, Expr beta, double epsilon,
                ffi::String name = "layer_norm");
Var NNRmsNorm(Var x, Var weight, ffi::Array<Integer> axes, double epsilon,
              ffi::String name = "rms_norm");
Var NNGroupNorm(Var x, ffi::Optional<Var> weight, ffi::Optional<Var> bias, int num_groups,
                int channel_axis, ffi::Array<Integer> axes, double epsilon,
                ffi::String name = "group_norm");

// ---------------------------------------------------------------------------
// Convolution ops
// ---------------------------------------------------------------------------

Var NNConv1d(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides, ffi::Any padding,
             ffi::Any dilation, int groups, ffi::String name = "conv1d");
Var NNConv2d(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides, ffi::Any padding,
             ffi::Any dilation, int groups, ffi::String data_layout,
             ffi::String name = "conv2d");
Var NNConv3d(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides, ffi::Any padding,
             ffi::Any dilation, int groups, ffi::String data_layout,
             ffi::String name = "conv3d");
Var NNConv1dTranspose(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides,
                      ffi::Any padding, ffi::Any output_padding, ffi::Any dilation, int groups,
                      ffi::String name = "conv1d_transpose");

// ---------------------------------------------------------------------------
// Padding
// ---------------------------------------------------------------------------

Var NNPad(Var x, ffi::Array<Integer> pad_width, ffi::String mode, double value,
          ffi::String name = "pad");

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

Var NNScaledDotProductAttention(Var query, Var key, Var value,
                                ffi::Optional<ffi::String> causal_mask = std::nullopt,
                                ffi::Optional<double> scale = std::nullopt,
                                ffi::String name = "scaled_dot_product_attention");

// ---------------------------------------------------------------------------
// Sorting / searching
// ---------------------------------------------------------------------------

Var NNSort(Var x, int axis, bool descending, ffi::String name = "sort");
Var NNArgsort(Var x, int axis, bool descending, ffi::String dtype, ffi::String name = "argsort");
// topk produces a tuple (values, indices) → returns ffi::Any (Array<Any>)
ffi::Any NNTopk(Var x, int k, int axis, ffi::String ret_type, bool largest, ffi::String dtype,
                ffi::String name = "topk");

// ---------------------------------------------------------------------------
// CCL ops
// ---------------------------------------------------------------------------

Var NNCclAllreduce(Var x, ffi::String op_type, bool in_group, ffi::String name = "allreduce");
Var NNCclAllgather(Var x, int num_workers, ffi::String name = "allgather");
Var NNCclBroadcastFromWorker0(Var x, ffi::String name = "broadcast_from_worker0");

// ---------------------------------------------------------------------------
// Multinomial sampling
// ---------------------------------------------------------------------------

Var NNMultinomialFromUniform(Var prob, Var uniform_sample, Var sample_indices, ffi::String dtype,
                             ffi::String name = "multinomial_from_uniform");

// ---------------------------------------------------------------------------
// Clip
// ---------------------------------------------------------------------------

Var NNClip(Var x, double min_val, double max_val, ffi::String name = "clip");

// ---------------------------------------------------------------------------
// TE / TIR ops  (return ffi::Any — output arity is not statically known)
// ---------------------------------------------------------------------------

ffi::Any NNTensorExprOp(ffi::Function tensor_expr_func, ffi::String name_hint,
                        ffi::Array<Expr> args,
                        ffi::Optional<ffi::Map<ffi::String, ffi::Any>> primfunc_attrs);
ffi::Any NNTensorIrOp(tir::PrimFunc func, ffi::String name_hint, ffi::Array<Expr> args,
                      ffi::Array<ffi::Any> out);
ffi::Any NNTensorIrInplaceOp(tir::PrimFunc func, ffi::String name_hint, ffi::Array<Expr> args,
                             ffi::Array<Integer> inplace_indices, ffi::Array<ffi::Any> out);
ffi::Any NNExtern(ffi::String name, ffi::Array<ffi::Any> args, ffi::Array<ffi::Any> out);

// ---------------------------------------------------------------------------
// debug_func
// ---------------------------------------------------------------------------

Var NNDebugFunc(ffi::String name, ffi::Array<ffi::Any> args, Var io_effect,
                ffi::String line_info);

// ---------------------------------------------------------------------------
// Timestep embedding
// ---------------------------------------------------------------------------

/*!
 * \brief Emit the full timestep-embedding computation into the current BB.
 *
 * Returns the final output Var (astype(emb, out_dtype)).
 */
Var NNGetTimestepEmbedding(Var x, int64_t embedding_dim, bool flip_sin_to_cos,
                           double downscale_freq_shift, double scale, int64_t max_period,
                           ffi::String out_dtype, ffi::String name = "get_timestep_embedding");

// ---------------------------------------------------------------------------
// Image / resize
// ---------------------------------------------------------------------------

Var NNResize2d(Var x, ffi::Array<Integer> size, ffi::String layout, ffi::String method,
               ffi::String coord_trans, ffi::String name = "resize2d");
Var NNInterpolate(Var x, ffi::Array<Integer> size, ffi::String data_layout, ffi::String method,
                  ffi::String coord_trans, ffi::String name = "interpolate");

// ---------------------------------------------------------------------------
// Sampling helpers (high-level, multi-step)
// ---------------------------------------------------------------------------

ffi::Any NNRenormalizeTopPTopKProb(Var prob, Var sorted_prob, Var top_p, Var top_k);
ffi::Any NNSampleTopPTopKFromSortedProb(Var sorted_prob, Var sorted_index, Var top_p, Var top_k,
                                        Var uniform_sample,
                                        ffi::Optional<Var> sample_indices_opt);

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_OP_H_
