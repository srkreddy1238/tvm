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
 * forward()).  Single-tensor ops return Var directly so that modules.cc can
 * call them without going through the FFI global registry.
 *
 * Tuple-producing ops (split, chunk, topk) always return
 * ffi::Any (Array<Any> of element Vars).  Variable-arity ops
 * (tensor_expr_op, tensor_ir_op, tensor_ir_inplace_op, extern) also return
 * ffi::Any because their output count is determined at runtime.
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

/*!
 * \brief Apply ReLU element-wise: output = max(0, x).
 * \param x     Input tensor.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor with negative values clamped to zero.
 */
Var NNRelu(Var x, ffi::String name = "relu");

/*!
 * \brief Apply ReLU6 element-wise: output = clip(x, 0, 6).
 * \param x     Input tensor.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor clipped to [0, 6].
 */
Var NNRelu6(Var x, ffi::String name = "relu6");

/*!
 * \brief Apply SiLU element-wise: output = x * sigmoid(x).
 * \param x     Input tensor.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor after applying x * sigmoid(x).
 */
Var NNSilu(Var x, ffi::String name = "silu");

/*!
 * \brief Apply GELU element-wise.
 *
 * \param x            Input tensor.
 * \param approximate  Approximation mode: nullopt or "" for exact erf-based
 *                     GELU; "tanh" for the fast tanh approximation.
 * \param name         Name hint for the emitted binding.
 * \return             Output tensor after applying the GELU activation.
 */
Var NNGelu(Var x, ffi::Optional<ffi::String> approximate, ffi::String name = "gelu");

/*!
 * \brief Apply sigmoid element-wise: output = 1 / (1 + exp(-x)).
 * \param x     Input tensor.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor with values in (0, 1).
 */
Var NNSigmoid(Var x, ffi::String name = "sigmoid");

/*!
 * \brief Apply tanh element-wise.
 * \param x     Input tensor.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor with values in (-1, 1).
 */
Var NNTanh(Var x, ffi::String name = "tanh");

/*!
 * \brief Apply natural exponential element-wise: output = exp(x).
 * \param x     Input tensor.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor.
 */
Var NNExp(Var x, ffi::String name = "exp");

/*!
 * \brief Apply natural logarithm element-wise: output = log(x).
 * \param x     Input tensor.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor.
 */
Var NNLog(Var x, ffi::String name = "log");

/*!
 * \brief Apply floor element-wise.
 * \param x     Input tensor.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor.
 */
Var NNFloor(Var x, ffi::String name = "floor");

/*!
 * \brief Apply square root element-wise.
 * \param x     Input tensor.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor.
 */
Var NNSqrt(Var x, ffi::String name = "sqrt");

/*!
 * \brief Apply element-wise square: output = x * x.
 * \param x     Input tensor.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor.
 */
Var NNSquare(Var x, ffi::String name = "square");

/*!
 * \brief Negate element-wise: output = -x.
 * \param x     Input tensor.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor.
 */
Var NNNegative(Var x, ffi::String name = "negative");

/*!
 * \brief Apply softmax along the given axis.
 * \param x     Input tensor.
 * \param axis  Axis along which softmax is computed.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor with values summing to 1 along \p axis.
 */
Var NNSoftmax(Var x, int axis, ffi::String name = "softmax");

/*!
 * \brief Apply softplus element-wise: output = (1/beta) * log(1 + exp(beta * x)).
 *
 * When x * beta > threshold the function falls back to the linear identity.
 *
 * \param x          Input tensor.
 * \param beta       Scaling factor (default 1.0).
 * \param threshold  Threshold above which the linear approximation is used.
 * \param name       Name hint for the emitted binding.
 * \return           Output tensor.
 */
Var NNSoftplus(Var x, double beta, double threshold, ffi::String name = "softplus");

/*!
 * \brief Apply parametric ReLU: output = max(0, x) + alpha * min(0, x).
 * \param x      Input tensor.
 * \param alpha  Learnable slope for negative values; shape must broadcast with x.
 * \param name   Name hint for the emitted binding.
 * \return       Output tensor.
 */
Var NNPrelu(Var x, Var alpha, ffi::String name = "prelu");

// ---------------------------------------------------------------------------
// Binary element-wise ops
// ---------------------------------------------------------------------------

/*!
 * \brief Element-wise addition: output = a + b.
 * \param a     First operand (broadcast-compatible with b).
 * \param b     Second operand.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor.
 */
Var NNAdd(Expr a, Expr b, ffi::String name = "add");

/*!
 * \brief Element-wise subtraction: output = a - b.
 * \param a     Minuend (broadcast-compatible with b).
 * \param b     Subtrahend.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor.
 */
Var NNSubtract(Expr a, Expr b, ffi::String name = "subtract");

/*!
 * \brief Element-wise multiplication: output = a * b.
 * \param a     First operand (broadcast-compatible with b).
 * \param b     Second operand.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor.
 */
Var NNMultiply(Expr a, Expr b, ffi::String name = "multiply");

/*!
 * \brief Element-wise division: output = a / b.
 * \param a     Dividend (broadcast-compatible with b).
 * \param b     Divisor.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor.
 */
Var NNDivide(Expr a, Expr b, ffi::String name = "divide");

/*!
 * \brief Element-wise maximum: output = max(a, b).
 * \param a     First operand (broadcast-compatible with b).
 * \param b     Second operand.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor.
 */
Var NNMaximum(Expr a, Expr b, ffi::String name = "maximum");

/*!
 * \brief Element-wise minimum: output = min(a, b).
 * \param a     First operand (broadcast-compatible with b).
 * \param b     Second operand.
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor.
 */
Var NNMinimum(Expr a, Expr b, ffi::String name = "minimum");

/*!
 * \brief Element-wise less-than comparison: output = (a < b).
 * \param a     First operand.
 * \param b     Second operand.
 * \param name  Name hint for the emitted binding.
 * \return      Boolean output tensor.
 */
Var NNLess(Expr a, Expr b, ffi::String name = "less");

/*!
 * \brief Element-wise less-or-equal comparison: output = (a <= b).
 * \param a     First operand.
 * \param b     Second operand.
 * \param name  Name hint for the emitted binding.
 * \return      Boolean output tensor.
 */
Var NNLessEqual(Expr a, Expr b, ffi::String name = "less_equal");

/*!
 * \brief Element-wise greater-than comparison: output = (a > b).
 * \param a     First operand.
 * \param b     Second operand.
 * \param name  Name hint for the emitted binding.
 * \return      Boolean output tensor.
 */
Var NNGreater(Expr a, Expr b, ffi::String name = "greater");

/*!
 * \brief Element-wise greater-or-equal comparison: output = (a >= b).
 * \param a     First operand.
 * \param b     Second operand.
 * \param name  Name hint for the emitted binding.
 * \return      Boolean output tensor.
 */
Var NNGreaterEqual(Expr a, Expr b, ffi::String name = "greater_equal");

/*!
 * \brief Element-wise equality comparison: output = (a == b).
 * \param a     First operand.
 * \param b     Second operand.
 * \param name  Name hint for the emitted binding.
 * \return      Boolean output tensor.
 */
Var NNEqual(Expr a, Expr b, ffi::String name = "equal");

/*!
 * \brief Element-wise inequality comparison: output = (a != b).
 * \param a     First operand.
 * \param b     Second operand.
 * \param name  Name hint for the emitted binding.
 * \return      Boolean output tensor.
 */
Var NNNotEqual(Expr a, Expr b, ffi::String name = "not_equal");

/*!
 * \brief Select elements from \p x1 or \p x2 based on a boolean condition.
 *
 * The condition is cast to bool before selection.
 *
 * \param condition  Boolean (or castable-to-bool) selector tensor.
 * \param x1         Values selected where condition is true.
 * \param x2         Values selected where condition is false.
 * \param name       Name hint for the emitted binding.
 * \return           Output tensor with elements drawn from \p x1 or \p x2.
 */
Var NNWhere(Var condition, Var x1, Var x2, ffi::String name = "where");

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

/*!
 * \brief Insert a size-1 dimension at position \p dim.
 * \param x     Input tensor.
 * \param dim   Axis at which to insert the new dimension (may be negative).
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor with one extra dimension.
 */
Var NNUnsqueeze(Var x, int dim, ffi::String name = "unsqueeze");

/*!
 * \brief Remove the size-1 dimension at position \p axis.
 * \param x     Input tensor.
 * \param axis  Axis to remove (must have size 1).
 * \param name  Name hint for the emitted binding.
 * \return      Output tensor with one fewer dimension.
 */
Var NNSqueeze(Var x, int axis, ffi::String name = "squeeze");

/*!
 * \brief Reshape \p x to the given shape.
 *
 * \param x     Input tensor.
 * \param shape Target shape; each element may be int64 (static) or PrimExpr
 *              (symbolic).  A value of -1 infers that dimension.
 * \param name  Name hint for the emitted binding.
 * \return      Reshaped output tensor.
 */
Var NNReshape(Var x, ffi::Array<ffi::Any> shape, ffi::String name = "reshape");

/*!
 * \brief Permute the dimensions of \p x.
 *
 * When \p name is nullopt the binding name is derived from the input
 * variable's name hint (replacing "linear" with "matmul" if present).
 *
 * \param x     Input tensor.
 * \param axes  New axis order; nullopt reverses all axes.
 * \param name  Name hint for the emitted binding, or nullopt to auto-derive.
 * \return      Transposed output tensor.
 */
Var NNPermuteDims(Var x, ffi::Optional<ffi::Array<Integer>> axes,
                  ffi::Optional<ffi::String> name = std::nullopt);

/*!
 * \brief Broadcast \p x to the given shape.
 * \param x     Input tensor.
 * \param shape Target broadcast shape; each element may be int64 or PrimExpr.
 * \param name  Name hint for the emitted binding.
 * \return      Broadcast output tensor.
 */
Var NNBroadcastTo(Var x, ffi::Array<ffi::Any> shape, ffi::String name = "broadcast_to");

/*!
 * \brief Repeat elements of \p x along \p axis.
 * \param x        Input tensor.
 * \param repeats  Number of repetitions.
 * \param axis     Axis along which to repeat; nullopt flattens first.
 * \param name     Name hint for the emitted binding.
 * \return         Output tensor with the repeated elements.
 */
Var NNRepeat(Var x, int repeats, ffi::Optional<Integer> axis, ffi::String name = "repeat");

/*!
 * \brief Concatenate a list of tensors along \p dim.
 * \param tensors  List of tensors to concatenate (must share all dims except \p dim).
 * \param dim      Axis along which to concatenate.
 * \param name     Name hint for the emitted binding.
 * \return         Concatenated output tensor.
 */
Var NNConcat(ffi::Array<Var> tensors, int dim, ffi::String name = "concat");

/*!
 * \brief Split \p x into sub-tensors along \p axis.
 *
 * \param x                    Input tensor.
 * \param indices_or_sections  int64 number of equal sections, or Array<int64>
 *                             of split points.
 * \param axis                 Axis along which to split.
 * \param name                 Name hint prefix for the emitted bindings.
 * \return                     Array<Any> of output Vars, one per section.
 */
ffi::Any NNSplit(Var x, ffi::Any indices_or_sections, int axis, ffi::String name = "split");

/*!
 * \brief Split \p x into \p chunks equal pieces along \p dim.
 * \param x      Input tensor.
 * \param chunks Number of chunks.
 * \param dim    Axis along which to split.
 * \param name   Name hint prefix for the emitted bindings.
 * \return       Array<Any> of output Vars, one per chunk.
 */
ffi::Any NNChunk(Var x, int chunks, int dim, ffi::String name = "chunk");

/*!
 * \brief Return the upper triangular part of \p x.
 * \param x         Input 2-D (or batched) tensor.
 * \param diagonal  Diagonal offset: 0 = main diagonal, >0 above, <0 below.
 * \param name      Name hint for the emitted binding.
 * \return          Output tensor with elements below the diagonal zeroed.
 */
Var NNTriu(Var x, int diagonal, ffi::String name = "triu");

// ---------------------------------------------------------------------------
// Reduction ops
// ---------------------------------------------------------------------------

/*!
 * \brief Sum elements of \p x along the given axes.
 * \param x         Input tensor.
 * \param axis      Axes to reduce; nullopt reduces all axes.
 * \param keepdims  If true, retain reduced axes with size 1.
 * \param name      Name hint for the emitted binding.
 * \return          Reduced output tensor.
 */
Var NNSum(Var x, ffi::Optional<ffi::Array<Integer>> axis, bool keepdims, ffi::String name = "sum");

/*!
 * \brief Compute the maximum of \p x along the given axes.
 * \param x         Input tensor.
 * \param axis      Axes to reduce; nullopt reduces all axes.
 * \param keepdims  If true, retain reduced axes with size 1.
 * \param name      Name hint for the emitted binding.
 * \return          Reduced output tensor.
 */
Var NNMax(Var x, ffi::Optional<ffi::Array<Integer>> axis, bool keepdims, ffi::String name = "max");

/*!
 * \brief Compute the minimum of \p x along the given axes.
 * \param x         Input tensor.
 * \param axis      Axes to reduce; nullopt reduces all axes.
 * \param keepdims  If true, retain reduced axes with size 1.
 * \param name      Name hint for the emitted binding.
 * \return          Reduced output tensor.
 */
Var NNMin(Var x, ffi::Optional<ffi::Array<Integer>> axis, bool keepdims, ffi::String name = "min");

/*!
 * \brief Compute the cumulative sum of \p x along \p axis.
 * \param x          Input tensor.
 * \param axis       Axis along which to accumulate; nullopt flattens first.
 * \param dtype      Output dtype override; nullopt keeps the input dtype.
 * \param exclusive  If true, each output element excludes the current input.
 * \param name       Name hint for the emitted binding.
 * \return           Cumulative-sum output tensor.
 */
Var NNCumsum(Var x, ffi::Optional<Integer> axis, ffi::Optional<ffi::String> dtype,
             ffi::Optional<Bool> exclusive, ffi::String name = "cumsum");

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

/*!
 * \brief General matrix multiplication: output = a @ b.
 *
 * \param a          Left operand; shape [..., M, K].
 * \param b          Right operand; shape [..., K, N].
 * \param out_dtype  Optional accumulator dtype override (e.g. "float32" when
 *                   inputs are "float16").
 * \param name       Name hint for the emitted binding.
 * \return           Output tensor of shape [..., M, N].
 */
Var NNMatmul(Var a, Var b, ffi::Optional<ffi::String> out_dtype = std::nullopt,
             ffi::String name = "matmul");

// ---------------------------------------------------------------------------
// Type casting
// ---------------------------------------------------------------------------

/*!
 * \brief Cast \p x to \p dtype.
 *
 * Returns \p x unchanged (without emitting a binding) when the input dtype
 * already matches the target.
 *
 * \param x      Input tensor.
 * \param dtype  Target dtype string (e.g. "float16").
 * \param name   Name hint for the emitted binding.
 * \return       Output tensor with the requested dtype.
 */
Var NNAstype(Var x, ffi::String dtype, ffi::String name = "astype");

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

/*!
 * \brief Gather slices from \p x along \p axis using \p indices.
 * \param x        Source tensor.
 * \param indices  Integer index tensor.
 * \param axis     Axis along which to index; nullopt uses axis 0.
 * \param name     Name hint for the emitted binding.
 * \return         Gathered output tensor.
 */
Var NNTake(Var x, Var indices, ffi::Optional<Integer> axis, ffi::String name = "take");

// ---------------------------------------------------------------------------
// Creation ops
// ---------------------------------------------------------------------------

/*!
 * \brief Create a 1-D tensor with evenly spaced values in [start, end).
 *
 * When \p end is null the range is [0, start) with step 1.
 *
 * \param start  Start of the range (int64, double, or PrimExpr).
 * \param end    End of the range (exclusive); null means use start as end.
 * \param step   Step size (int64, double, or PrimExpr).
 * \param dtype  Output dtype; defaults to "float32" when nullopt.
 * \param name   Name hint for the emitted binding.
 * \return       1-D output tensor.
 */
Var NNArange(ffi::Any start, ffi::Any end, ffi::Any step, ffi::Optional<ffi::String> dtype,
             ffi::String name = "arange");

/*!
 * \brief Create a tensor filled with \p fill_value.
 * \param shape       Target shape; each element may be int64 or PrimExpr.
 * \param fill_value  Scalar fill expression.
 * \param dtype       Output dtype string.
 * \param name        Name hint for the emitted binding.
 * \return            Output tensor filled with \p fill_value.
 */
Var NNFull(ffi::Array<ffi::Any> shape, Expr fill_value, ffi::String dtype,
           ffi::String name = "full");

/*!
 * \brief Create a zero-filled tensor of the given shape and dtype.
 * \param shape  Target shape; each element may be int64 or PrimExpr.
 * \param dtype  Output dtype string.
 * \param name   Name hint for the emitted binding.
 * \return       Zero-filled output tensor.
 */
Var NNZeros(ffi::Array<ffi::Any> shape, ffi::String dtype, ffi::String name = "zeros");

/*!
 * \brief Create a one-filled tensor of the given shape and dtype.
 * \param shape  Target shape; each element may be int64 or PrimExpr.
 * \param dtype  Output dtype string.
 * \param name   Name hint for the emitted binding.
 * \return       One-filled output tensor.
 */
Var NNOnes(ffi::Array<ffi::Any> shape, ffi::String dtype, ffi::String name = "ones");

// ---------------------------------------------------------------------------
// Normalization ops
// ---------------------------------------------------------------------------

/*!
 * \brief Apply layer normalisation over the given axes.
 * \param x        Input tensor.
 * \param axes     Axes to normalise over.
 * \param gamma    Scale parameter (must have shape matching the normalised dims).
 * \param beta     Shift parameter (must have shape matching the normalised dims).
 * \param epsilon  Small constant for numerical stability.
 * \param name     Name hint for the emitted binding.
 * \return         Normalised output tensor.
 */
Var NNLayerNorm(Var x, ffi::Array<Integer> axes, Expr gamma, Expr beta, double epsilon,
                ffi::String name = "layer_norm");

/*!
 * \brief Apply RMS normalisation over the given axes.
 * \param x        Input tensor.
 * \param weight   Learnable scale parameter.
 * \param axes     Axes to normalise over.
 * \param epsilon  Small constant for numerical stability.
 * \param name     Name hint for the emitted binding.
 * \return         RMS-normalised output tensor.
 */
Var NNRmsNorm(Var x, Var weight, ffi::Array<Integer> axes, double epsilon,
              ffi::String name = "rms_norm");

/*!
 * \brief Apply group normalisation.
 * \param x             Input tensor.
 * \param weight        Optional learnable scale parameter.
 * \param bias          Optional learnable bias parameter.
 * \param num_groups    Number of channel groups.
 * \param channel_axis  Axis index of the channel dimension.
 * \param axes          Axes over which to compute group statistics.
 * \param epsilon       Small constant for numerical stability.
 * \param name          Name hint for the emitted binding.
 * \return              Normalised output tensor.
 */
Var NNGroupNorm(Var x, ffi::Optional<Var> weight, ffi::Optional<Var> bias, int num_groups,
                int channel_axis, ffi::Array<Integer> axes, double epsilon,
                ffi::String name = "group_norm");

// ---------------------------------------------------------------------------
// Convolution ops
// ---------------------------------------------------------------------------

/*!
 * \brief Apply 1-D convolution (NCW layout).
 * \param x            Input tensor of shape [N, C_in, L].
 * \param weight       Kernel of shape [C_out, C_in/groups, kW].
 * \param bias         Optional bias of shape [C_out].
 * \param strides      Stride (int64 scalar or Array).
 * \param padding      Zero-padding (int64 scalar or Array).
 * \param dilation     Dilation factor (int64 scalar or Array).
 * \param groups       Number of blocked channel connections.
 * \param name         Name hint for the emitted binding.
 * \return             Output tensor of shape [N, C_out, L_out].
 */
Var NNConv1d(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides, ffi::Any padding,
             ffi::Any dilation, int groups, ffi::String name = "conv1d");

/*!
 * \brief Apply 2-D convolution.
 * \param x            Input tensor (NCHW or NHWC depending on \p data_layout).
 * \param weight       Convolution kernel.
 * \param bias         Optional bias of shape [C_out].
 * \param strides      Stride (int64 scalar or Array).
 * \param padding      Zero-padding (int64 scalar or Array).
 * \param dilation     Dilation factor (int64 scalar or Array).
 * \param groups       Number of blocked channel connections.
 * \param data_layout  Data layout string: "NCHW" or "NHWC".
 * \param name         Name hint for the emitted binding.
 * \return             Output tensor in the same layout as the input.
 */
Var NNConv2d(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides, ffi::Any padding,
             ffi::Any dilation, int groups, ffi::String data_layout, ffi::String name = "conv2d");

/*!
 * \brief Apply 3-D convolution.
 * \param x            Input tensor (NCDHW or NDHWC depending on \p data_layout).
 * \param weight       Convolution kernel.
 * \param bias         Optional bias of shape [C_out].
 * \param strides      Stride (int64 scalar or Array).
 * \param padding      Zero-padding (int64 scalar or Array).
 * \param dilation     Dilation factor (int64 scalar or Array).
 * \param groups       Number of blocked channel connections.
 * \param data_layout  Data layout string: "NCDHW" or "NDHWC".
 * \param name         Name hint for the emitted binding.
 * \return             Output tensor in the same layout as the input.
 */
Var NNConv3d(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides, ffi::Any padding,
             ffi::Any dilation, int groups, ffi::String data_layout, ffi::String name = "conv3d");

/*!
 * \brief Apply 1-D transposed convolution (NCW layout, IOW kernel layout).
 * \param x              Input tensor of shape [N, C_in, L].
 * \param weight         Kernel of shape [C_in, C_out/groups, kW].
 * \param bias           Optional bias of shape [C_out].
 * \param strides        Stride / upsampling factor.
 * \param padding        Input zero-padding removed from the output.
 * \param output_padding Additional output padding to resolve size ambiguity.
 * \param dilation       Kernel dilation factor.
 * \param groups         Number of blocked channel connections.
 * \param name           Name hint for the emitted binding.
 * \return               Output tensor of shape [N, C_out, L_out].
 */
Var NNConv1dTranspose(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides,
                      ffi::Any padding, ffi::Any output_padding, ffi::Any dilation, int groups,
                      ffi::String name = "conv1d_transpose");

// ---------------------------------------------------------------------------
// Padding
// ---------------------------------------------------------------------------

/*!
 * \brief Pad \p x with the given widths and mode.
 * \param x          Input tensor.
 * \param pad_width  Flat array of (before, after) pairs for each axis,
 *                   ordered from the outermost to the innermost axis.
 * \param mode       Padding mode: "constant", "reflect", or "edge".
 * \param value      Fill value used when mode is "constant".
 * \param name       Name hint for the emitted binding.
 * \return           Padded output tensor.
 */
Var NNPad(Var x, ffi::Array<Integer> pad_width, ffi::String mode, double value,
          ffi::String name = "pad");

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/*!
 * \brief Compute scaled dot-product attention.
 *
 * Implements: softmax(Q @ K^T / sqrt(d_k) + mask) @ V.
 *
 * \param query       Query tensor of shape [batch, seq_q, heads, head_dim].
 * \param key         Key tensor of shape [batch, seq_k, heads, head_dim].
 * \param value       Value tensor of shape [batch, seq_k, heads, head_dim].
 * \param causal_mask Optional causal mask type string (e.g. "TopLeft");
 *                    nullopt for no mask.
 * \param scale       Optional attention scale; defaults to 1/sqrt(head_dim).
 * \param name        Name hint for the emitted binding.
 * \return            Attention output tensor of shape [batch, seq_q, heads, head_dim].
 */
Var NNScaledDotProductAttention(Var query, Var key, Var value,
                                ffi::Optional<ffi::String> causal_mask = std::nullopt,
                                ffi::Optional<double> scale = std::nullopt,
                                ffi::String name = "scaled_dot_product_attention");

// ---------------------------------------------------------------------------
// Sorting / searching
// ---------------------------------------------------------------------------

/*!
 * \brief Sort \p x along \p axis.
 * \param x            Input tensor.
 * \param axis         Axis along which to sort.
 * \param descending   If true, sort in descending order.
 * \param name         Name hint for the emitted binding.
 * \return             Sorted output tensor (same shape as \p x).
 */
Var NNSort(Var x, int axis, bool descending, ffi::String name = "sort");

/*!
 * \brief Return the indices that would sort \p x along \p axis.
 * \param x            Input tensor.
 * \param axis         Axis along which to sort.
 * \param descending   If true, sort in descending order.
 * \param dtype        Output index dtype string (e.g. "int32").
 * \param name         Name hint for the emitted binding.
 * \return             Integer index tensor (same shape as \p x).
 */
Var NNArgsort(Var x, int axis, bool descending, ffi::String dtype, ffi::String name = "argsort");

/*!
 * \brief Return the top-k values and their indices along \p axis.
 * \param x         Input tensor.
 * \param k         Number of top elements to return.
 * \param axis      Axis along which to select.
 * \param ret_type  Return type: "both", "values", or "indices".
 * \param largest   If true, return the largest k values; otherwise the smallest.
 * \param dtype     Output index dtype string.
 * \param name      Name hint prefix for the emitted bindings.
 * \return          Array<Any> of [values_Var, indices_Var] (or just one if ret_type != "both").
 */
ffi::Any NNTopk(Var x, int k, int axis, ffi::String ret_type, bool largest, ffi::String dtype,
                ffi::String name = "topk");

// ---------------------------------------------------------------------------
// CCL ops
// ---------------------------------------------------------------------------

/*!
 * \brief All-reduce across workers.
 * \param x        Input tensor.
 * \param op_type  Reduction operation string (e.g. "sum", "max").
 * \param in_group If true, reduce within the current worker group.
 * \param name     Name hint for the emitted binding.
 * \return         Reduced output tensor (same shape as \p x).
 */
Var NNCclAllreduce(Var x, ffi::String op_type, bool in_group, ffi::String name = "allreduce");

/*!
 * \brief All-gather across \p num_workers workers.
 * \param x            Input tensor (local shard).
 * \param num_workers  Total number of workers participating.
 * \param name         Name hint for the emitted binding.
 * \return             Gathered output tensor with the first axis scaled by \p num_workers.
 */
Var NNCclAllgather(Var x, int num_workers, ffi::String name = "allgather");

/*!
 * \brief Broadcast the tensor from worker 0 to all other workers.
 * \param x     Input tensor (only the value on worker 0 is used).
 * \param name  Name hint for the emitted binding.
 * \return      Broadcast output tensor (same shape as \p x).
 */
Var NNCclBroadcastFromWorker0(Var x, ffi::String name = "broadcast_from_worker0");

// ---------------------------------------------------------------------------
// Multinomial sampling
// ---------------------------------------------------------------------------

/*!
 * \brief Sample from a categorical distribution using a pre-drawn uniform sample.
 *
 * Implements the inverse-CDF method: finds the index where the cumulative
 * probability first exceeds \p uniform_sample.
 *
 * \param prob            2-D probability tensor of shape [batch, vocab].
 * \param uniform_sample  2-D uniform sample tensor of shape [batch, 1].
 * \param sample_indices  2-D index tensor of shape [batch, 1] selecting which
 *                        distribution to sample from.
 * \param dtype           Output index dtype string (e.g. "int32").
 * \param name            Name hint for the emitted binding.
 * \return                Integer sample tensor of shape [batch, 1].
 */
Var NNMultinomialFromUniform(Var prob, Var uniform_sample, Var sample_indices, ffi::String dtype,
                             ffi::String name = "multinomial_from_uniform");

// ---------------------------------------------------------------------------
// Clip
// ---------------------------------------------------------------------------

/*!
 * \brief Clamp all elements of \p x to the range [min_val, max_val].
 * \param x        Input tensor.
 * \param min_val  Lower bound (inclusive).
 * \param max_val  Upper bound (inclusive).
 * \param name     Name hint for the emitted binding.
 * \return         Clamped output tensor (same shape and dtype as \p x).
 */
Var NNClip(Var x, double min_val, double max_val, ffi::String name = "clip");

// ---------------------------------------------------------------------------
// TE / TIR ops  (return ffi::Any — output arity is not statically known)
// ---------------------------------------------------------------------------

/*!
 * \brief Emit a call_tir node for a TE compute function.
 *
 * Calls \p tensor_expr_func with the TE tensor inputs, creates a PrimFunc
 * via CreatePrimFunc, registers it in the IRModule, and emits a call_tir
 * node.  Returns a single Var when there is one output, or Array<Any> of
 * Vars when there are multiple outputs.
 *
 * \param tensor_expr_func  TE compute function: (Array<te.Tensor>) -> Array<te.Tensor>.
 * \param name_hint         Name hint for the GlobalVar and emitted binding.
 * \param args              Input tensor Vars.
 * \param primfunc_attrs    Optional extra attributes to attach to the PrimFunc.
 * \return                  Var (single output) or Array<Any> of Vars (multiple outputs).
 */
ffi::Any NNTensorExprOp(ffi::Function tensor_expr_func, ffi::String name_hint,
                        ffi::Array<Expr> args,
                        ffi::Optional<ffi::Map<ffi::String, ffi::Any>> primfunc_attrs);

/*!
 * \brief Emit a call_tir node for a pre-built TIR PrimFunc.
 *
 * Registers \p func in the IRModule and emits a call_tir node.  TIR
 * variable arguments (ShapeStructInfo or PrimStructInfo) are collected
 * into a trailing ShapeExpr.
 *
 * \param func       The TIR PrimFunc to call.
 * \param name_hint  Name hint for the GlobalVar and emitted binding.
 * \param args       Input Vars (tensor or tir-var).
 * \param out        Output placeholder Vars whose StructInfo describes the outputs.
 * \return           Var (single output) or Array<Any> of Vars (multiple outputs).
 */
ffi::Any NNTensorIrOp(tir::PrimFunc func, ffi::String name_hint, ffi::Array<Expr> args,
                      ffi::Array<ffi::Any> out);

/*!
 * \brief Emit a call_tir_inplace node for a pre-built TIR PrimFunc.
 *
 * Like NNTensorIrOp but uses call_tir_inplace semantics, allowing the
 * kernel to write results back into the input buffers at the given indices.
 *
 * \param func             The TIR PrimFunc to call.
 * \param name_hint        Name hint for the GlobalVar and emitted binding.
 * \param args             Input Vars (tensor or tir-var).
 * \param inplace_indices  Indices into \p args that are written in-place.
 * \param out              Output placeholder Vars.
 * \return                 Var (single output) or Array<Any> of Vars (multiple outputs).
 */
ffi::Any NNTensorIrInplaceOp(tir::PrimFunc func, ffi::String name_hint, ffi::Array<Expr> args,
                             ffi::Array<Integer> inplace_indices, ffi::Array<ffi::Any> out);

/*!
 * \brief Emit a call_dps_packed node for an external C function.
 *
 * \param name  Symbol name of the external function.
 * \param args  Input arguments (Var, int64, double, String, or PrimExpr).
 * \param out   Output placeholder Vars whose StructInfo describes the outputs.
 * \return      Var (single output) or Array<Any> of Vars (multiple outputs).
 */
ffi::Any NNExtern(ffi::String name, ffi::Array<ffi::Any> args, ffi::Array<ffi::Any> out);

// ---------------------------------------------------------------------------
// debug_func
// ---------------------------------------------------------------------------

/*!
 * \brief Emit a call_pure_packed node for a debug side-effect function.
 *
 * The packed function receives (line_info_string, user_arg0, user_arg1, ...).
 * After emitting, the thread-local IO Var is updated so that subsequent
 * debug_func calls chain the effect token correctly.
 *
 * \param name       Symbol name of the debug packed function.
 * \param args       User arguments forwarded after the line-info string.
 * \param io_effect  Current IO effect Var (used as the name hint base).
 * \param line_info  Source location string passed as the first argument.
 * \return           Updated IO effect Var.
 */
Var NNDebugFunc(ffi::String name, ffi::Array<ffi::Any> args, Var io_effect, ffi::String line_info);

// ---------------------------------------------------------------------------
// Timestep embedding
// ---------------------------------------------------------------------------

/*!
 * \brief Emit the full sinusoidal timestep-embedding computation.
 *
 * Converts a 1-D integer timestep tensor into a 2-D sinusoidal embedding
 * following the DDPM formulation:
 *   emb = [sin(t * exp(-log(max_period) * i / (half_dim - shift))),
 *          cos(t * exp(-log(max_period) * i / (half_dim - shift)))]
 * where i ranges over [0, half_dim).
 *
 * \param x                   1-D timestep tensor of shape [N].
 * \param embedding_dim        Output embedding dimension.
 * \param flip_sin_to_cos      If true, concatenate [cos, sin] instead of [sin, cos].
 * \param downscale_freq_shift Frequency downscale shift subtracted from half_dim.
 * \param scale                Multiplicative scale applied before sin/cos.
 * \param max_period            Maximum period for the frequency bands.
 * \param out_dtype            Output dtype string.
 * \param name                 Name hint for the final emitted binding.
 * \return                     Embedding tensor of shape [N, embedding_dim].
 */
Var NNGetTimestepEmbedding(Var x, int64_t embedding_dim, bool flip_sin_to_cos,
                           double downscale_freq_shift, double scale, int64_t max_period,
                           ffi::String out_dtype, ffi::String name = "get_timestep_embedding");

// ---------------------------------------------------------------------------
// Image / resize
// ---------------------------------------------------------------------------

/*!
 * \brief Resize a 2-D feature map to the given spatial size.
 * \param x            Input tensor (layout determined by \p layout).
 * \param size         Target spatial size [H_out, W_out].
 * \param layout       Data layout string (e.g. "NCHW" or "NHWC").
 * \param method       Interpolation method: "nearest_neighbor", "linear", or "cubic".
 * \param coord_trans  Coordinate transformation mode (e.g. "half_pixel").
 * \param name         Name hint for the emitted binding.
 * \return             Resized output tensor.
 */
Var NNResize2d(Var x, ffi::Array<Integer> size, ffi::String layout, ffi::String method,
               ffi::String coord_trans, ffi::String name = "resize2d");

/*!
 * \brief Interpolate a 2-D feature map to the given spatial size.
 *
 * Thin wrapper around NNResize2d with the same parameter semantics.
 *
 * \param x            Input tensor.
 * \param size         Target spatial size [H_out, W_out].
 * \param data_layout  Data layout string.
 * \param method       Interpolation method.
 * \param coord_trans  Coordinate transformation mode.
 * \param name         Name hint for the emitted binding.
 * \return             Interpolated output tensor.
 */
Var NNInterpolate(Var x, ffi::Array<Integer> size, ffi::String data_layout, ffi::String method,
                  ffi::String coord_trans, ffi::String name = "interpolate");

// ---------------------------------------------------------------------------
// Sampling helpers (high-level, multi-step)
// ---------------------------------------------------------------------------

/*!
 * \brief Renormalise probabilities after top-p / top-k filtering.
 *
 * Steps:
 *   1. cumsum_sorted = cumsum(sorted_prob, axis=1)
 *   2. renorm_cutoff = _get_renorm_cutoff(sorted_prob, cumsum_sorted, top_p, top_k)
 *   3. filtered_prob = where(prob >= renorm_cutoff, prob, 0)
 *   4. output        = filtered_prob / sum(filtered_prob, axis=1, keepdims=True)
 *
 * \param prob         2-D probability tensor of shape [batch, vocab].
 * \param sorted_prob  2-D probabilities sorted in descending order.
 * \param top_p        2-D cumulative probability threshold of shape [batch, 1].
 * \param top_k        2-D top-k count of shape [batch, 1].
 * \return             Renormalised probability tensor of shape [batch, vocab].
 */
ffi::Any NNRenormalizeTopPTopKProb(Var prob, Var sorted_prob, Var top_p, Var top_k);

/*!
 * \brief Sample token indices from a sorted probability tensor.
 *
 * Steps:
 *   1. cumsum_sorted = cumsum(sorted_prob, axis=1)
 *   2. renorm_prob   = _get_renorm_prob(cumsum_sorted, top_p, top_k)
 *   3. out_index     = _get_index_from_sorted(cumsum_sorted, sorted_index,
 *                                             renorm_prob, uniform_sample,
 *                                             sample_indices)
 *
 * \param sorted_prob     2-D probabilities sorted descending; shape [batch, vocab].
 * \param sorted_index    2-D argsort indices; shape [batch, vocab].
 * \param top_p           2-D cumulative probability threshold; shape [batch, 1].
 * \param top_k           2-D top-k count; shape [batch, 1].
 * \param uniform_sample  2-D uniform samples; shape [n, 1].
 * \param sample_indices  Optional 2-D distribution selector; shape [n, 1].
 *                        Defaults to arange(n) reshaped to [n, 1] when nullopt.
 * \return                2-D integer index tensor of shape [n, 1].
 */
ffi::Any NNSampleTopPTopKFromSortedProb(Var sorted_prob, Var sorted_index, Var top_p, Var top_k,
                                        Var uniform_sample, ffi::Optional<Var> sample_indices_opt);

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_OP_H_
