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
 * \file src/relax/frontend/nn/op.cc
 * \brief FFI registrations for all nn.Tensor operators.
 *
 * Every function here:
 *   1. Accepts relax.Var arguments (the underlying _expr of nn.Tensor).
 *   2. Builds the corresponding relax op call.
 *   3. Emits it via WrapNested (relax.frontend.nn.WrapNested) and returns
 *      the bound Var (or Array of Vars for tuple outputs).
 *
 * The Python side calls these via tvm.get_global_func and wraps the
 * returned Var back into nn.Tensor.
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/function.h>

#include <cmath>
#include <string>
#include <vector>

#include "../../op/nn/attention.h"
#include "../../op/nn/convolution.h"
#include "../../op/nn/nn.h"
#include "../../op/tensor/binary.h"
#include "../../op/tensor/create.h"
#include "../../op/tensor/datatype.h"
#include "../../op/tensor/index.h"
#include "../../op/tensor/linear_algebra.h"
#include "../../op/tensor/manipulate.h"
#include "../../op/tensor/sampling.h"
#include "../../op/tensor/search.h"
#include "../../op/tensor/sorting.h"
#include "../../op/tensor/statistical.h"
#include "../../op/tensor/unary.h"
#include "../../op/tensor/ternary.h"
#include "../../op/ccl/ccl.h"
#include "../../op/image/resize.h"
#include "../../../relax/ir/emit_te.h"
#include "../../../te/operation/create_primfunc.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ---------------------------------------------------------------------------
// Internal helper: emit expr and return bound Var (or Array for tuples)
// ---------------------------------------------------------------------------

static ffi::Any WrapNested(Expr expr, const std::string& name) {
  BlockBuilder bb = BlockBuilder::Current();
  TVM_FFI_ICHECK(bb.defined()) << "nn op called outside of a BlockBuilder scope";
  Var v = bb->Emit(expr, name);
  StructInfo sinfo = GetStructInfo(v);
  if (sinfo->IsInstance<TensorStructInfoNode>()) {
    return ffi::Any(v);
  }
  if (const auto* ts = sinfo.as<TupleStructInfoNode>()) {
    ffi::Array<ffi::Any> results;
    for (int i = 0; i < static_cast<int>(ts->fields.size()); ++i) {
      Var vi = bb->Emit(TupleGetItem(v, i), name + "." + std::to_string(i));
      results.push_back(ffi::Any(vi));
    }
    return ffi::Any(results);
  }
  TVM_FFI_THROW(TypeError) << "WrapNested: unsupported struct_info: " << sinfo->GetTypeKey();
}

// ---------------------------------------------------------------------------
// Unary element-wise ops
// ---------------------------------------------------------------------------

#define NN_UNARY_OP(func_name, relax_op)                                                    \
  static ffi::Any func_name(Var x, ffi::String name) {                                     \
    return WrapNested(relax_op(x), std::string(name));                                      \
  }

NN_UNARY_OP(NNRelu, relax::relu)

// relu6 is not declared in nn.h (defined via macro in nn.cc), so implement directly
static ffi::Any NNRelu6(Var x, ffi::String name) {
  static const Op& op = Op::Get("relax.nn.relu6");
  return WrapNested(Call(op, {x}, Attrs(), {}), std::string(name));
}

NN_UNARY_OP(NNSilu, relax::silu)
NN_UNARY_OP(NNSigmoid, relax::sigmoid)
NN_UNARY_OP(NNTanh, relax::tanh)
NN_UNARY_OP(NNExp, relax::exp)
NN_UNARY_OP(NNLog, relax::log)
NN_UNARY_OP(NNFloor, relax::floor)
NN_UNARY_OP(NNSqrt, relax::sqrt)
NN_UNARY_OP(NNSquare, relax::square)
NN_UNARY_OP(NNNegative, relax::negative)

#undef NN_UNARY_OP

// ---------------------------------------------------------------------------
// GeLU (with optional tanh approximation)
// ---------------------------------------------------------------------------

static ffi::Any NNGelu(Var x, ffi::Optional<ffi::String> approximate, ffi::String name) {
  Expr out;
  if (approximate.defined() && approximate.value() == "tanh") {
    out = relax::gelu_tanh(x);
  } else {
    out = relax::gelu(x);
  }
  return WrapNested(out, std::string(name));
}

// ---------------------------------------------------------------------------
// Softmax / Softplus / PReLU
// ---------------------------------------------------------------------------

static ffi::Any NNSoftmax(Var x, int axis, ffi::String name) {
  return WrapNested(relax::softmax(x, axis), std::string(name));
}

static ffi::Any NNSoftplus(Var x, double beta, double threshold, ffi::String name) {
  return WrapNested(relax::softplus(x, beta, threshold), std::string(name));
}

static ffi::Any NNPrelu(Var x, Var alpha, ffi::String name) {
  return WrapNested(relax::prelu(x, alpha), std::string(name));
}

// ---------------------------------------------------------------------------
// Binary element-wise ops
// ---------------------------------------------------------------------------

#define NN_BINARY_OP(func_name, relax_op)                                                   \
  static ffi::Any func_name(Var a, Var b, ffi::String name) {                              \
    return WrapNested(relax_op(a, b), std::string(name));                                   \
  }

NN_BINARY_OP(NNAdd, relax::add)
NN_BINARY_OP(NNSubtract, relax::subtract)
NN_BINARY_OP(NNMultiply, relax::multiply)
NN_BINARY_OP(NNDivide, relax::divide)
NN_BINARY_OP(NNMaximum, relax::maximum)
NN_BINARY_OP(NNMinimum, relax::minimum)
NN_BINARY_OP(NNLess, relax::less)
NN_BINARY_OP(NNLessEqual, relax::less_equal)
NN_BINARY_OP(NNGreater, relax::greater)
NN_BINARY_OP(NNGreaterEqual, relax::greater_equal)
NN_BINARY_OP(NNEqual, relax::equal)
NN_BINARY_OP(NNNotEqual, relax::not_equal)

#undef NN_BINARY_OP

// ---------------------------------------------------------------------------
// Ternary
// ---------------------------------------------------------------------------

static ffi::Any NNWhere(Var condition, Var x1, Var x2, ffi::String name) {
  // Cast condition to bool first
  Expr cond_bool = relax::astype(condition, DataType::Bool());
  return WrapNested(relax::where(cond_bool, x1, x2), std::string(name));
}

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

static ffi::Any NNUnsqueeze(Var x, int dim, ffi::String name) {
  return WrapNested(relax::expand_dims(x, {dim}), std::string(name));
}

static ffi::Any NNSqueeze(Var x, int axis, ffi::String name) {
  return WrapNested(relax::squeeze(x, ffi::Optional<ffi::Array<Integer>>({Integer(axis)})),
                    std::string(name));
}

static ffi::Any NNReshape(Var x, ffi::Array<ffi::Any> shape, ffi::String name) {
  ffi::Array<PrimExpr> new_shape;
  for (const ffi::Any& s : shape) {
    if (auto opt_int = s.TryAs<int64_t>()) {
      new_shape.push_back(tir::IntImm(DataType::Int(64), opt_int.value()));
    } else if (auto opt_prim = s.TryAs<PrimExpr>()) {
      new_shape.push_back(opt_prim.value());
    } else {
      TVM_FFI_THROW(TypeError) << "NNReshape: invalid shape element type: " << s.GetTypeKey();
    }
  }
  return WrapNested(relax::reshape(x, ShapeExpr(new_shape)), std::string(name));
}

static ffi::Any NNPermuteDims(Var x, ffi::Optional<ffi::Array<Integer>> axes, ffi::String name) {
  ffi::Optional<ffi::Array<Integer>> ax = axes;
  return WrapNested(relax::permute_dims(x, ax), std::string(name));
}

static ffi::Any NNBroadcastTo(Var x, ffi::Array<ffi::Any> shape, ffi::String name) {
  ffi::Array<PrimExpr> new_shape;
  for (const ffi::Any& s : shape) {
    if (auto opt_int = s.TryAs<int64_t>()) {
      new_shape.push_back(tir::IntImm(DataType::Int(64), opt_int.value()));
    } else if (auto opt_prim = s.TryAs<PrimExpr>()) {
      new_shape.push_back(opt_prim.value());
    } else {
      TVM_FFI_THROW(TypeError) << "NNBroadcastTo: invalid shape element: " << s.GetTypeKey();
    }
  }
  return WrapNested(relax::broadcast_to(x, ShapeExpr(new_shape)), std::string(name));
}

static ffi::Any NNRepeat(Var x, int repeats, ffi::Optional<Integer> axis, ffi::String name) {
  ffi::Optional<int64_t> ax = axis.defined() ? ffi::Optional<int64_t>(axis.value()->value) : std::nullopt;
  return WrapNested(relax::repeat(x, repeats, ax), std::string(name));
}

static ffi::Any NNConcat(ffi::Array<Var> tensors, int dim, ffi::String name) {
  ffi::Array<Expr> exprs;
  for (const Var& v : tensors) exprs.push_back(v);
  return WrapNested(relax::concat(Tuple(exprs), ffi::Optional<int64_t>(dim)), std::string(name));
}

static ffi::Any NNSplit(Var x, ffi::Any indices_or_sections, int axis, ffi::String name) {
  // indices_or_sections is either int64 (number of sections) or Array<int64>
  if (auto opt_int = indices_or_sections.TryAs<int64_t>()) {
    return WrapNested(
        relax::split(x, IntImm(DataType::Int(64), opt_int.value()), axis),
        std::string(name));
  }
  if (auto opt_arr = indices_or_sections.TryAs<ffi::Array<ffi::Any>>()) {
    ffi::Array<IntImm> indices;
    for (const ffi::Any& idx : opt_arr.value()) {
      indices.push_back(IntImm(DataType::Int(64), idx.operator int64_t()));
    }
    return WrapNested(relax::split(x, indices, axis), std::string(name));
  }
  TVM_FFI_THROW(TypeError) << "NNSplit: indices_or_sections must be int or Array[int]";
}

static ffi::Any NNChunk(Var x, int chunks, int dim, ffi::String name) {
  return WrapNested(relax::split(x, IntImm(DataType::Int(64), chunks), dim), std::string(name));
}

static ffi::Any NNTriu(Var x, int diagonal, ffi::String name) {
  return WrapNested(relax::triu(x, diagonal), std::string(name));
}

// ---------------------------------------------------------------------------
// Reduction ops
// ---------------------------------------------------------------------------

// Helper: convert an Optional<Array<Integer>> where empty array means "all axes" (None)
static ffi::Optional<ffi::Array<Integer>> NormalizeAxis(
    ffi::Optional<ffi::Array<Integer>> axis) {
  if (!axis.defined() || axis.value().empty()) return std::nullopt;
  return axis;
}

static ffi::Any NNSum(Var x, ffi::Optional<ffi::Array<Integer>> axis, bool keepdims,
                      ffi::String name) {
  return WrapNested(relax::sum(x, NormalizeAxis(axis), keepdims), std::string(name));
}

static ffi::Any NNMax(Var x, ffi::Optional<ffi::Array<Integer>> axis, bool keepdims,
                      ffi::String name) {
  return WrapNested(relax::max(x, NormalizeAxis(axis), keepdims), std::string(name));
}

static ffi::Any NNMin(Var x, ffi::Optional<ffi::Array<Integer>> axis, bool keepdims,
                      ffi::String name) {
  return WrapNested(relax::min(x, NormalizeAxis(axis), keepdims), std::string(name));
}

static ffi::Any NNCumsum(Var x, ffi::Optional<Integer> axis, ffi::Optional<ffi::String> dtype,
                         ffi::Optional<Bool> exclusive, ffi::String name) {
  ffi::Optional<int64_t> ax = axis.defined() ? ffi::Optional<int64_t>(axis.value()->value) : std::nullopt;
  ffi::Optional<DataType> dt = dtype.defined()
      ? ffi::Optional<DataType>(DataType(runtime::String2DLDataType(dtype.value())))
      : std::nullopt;
  Bool excl = exclusive.defined() ? exclusive.value() : Bool(false);
  return WrapNested(relax::cumsum(x, ax, dt, excl), std::string(name));
}

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

static ffi::Any NNMatmul(Var a, Var b, ffi::Optional<ffi::String> out_dtype, ffi::String name) {
  ffi::Optional<DataType> dt = out_dtype.defined()
      ? ffi::Optional<DataType>(DataType(runtime::String2DLDataType(out_dtype.value())))
      : std::nullopt;
  return WrapNested(relax::matmul(a, b, dt), std::string(name));
}

// ---------------------------------------------------------------------------
// Type casting
// ---------------------------------------------------------------------------

static ffi::Any NNAstype(Var x, ffi::String dtype, ffi::String name) {
  DataType target_dtype = DataType(runtime::String2DLDataType(dtype));
  // Skip cast if same dtype
  const auto* sinfo = x->struct_info_.as<TensorStructInfoNode>();
  if (sinfo && sinfo->dtype == target_dtype) {
    return ffi::Any(x);
  }
  return WrapNested(relax::astype(x, target_dtype), std::string(name));
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

static ffi::Any NNTake(Var x, Var indices, ffi::Optional<Integer> axis, ffi::String name) {
  ffi::Optional<int64_t> ax = axis.defined() ? ffi::Optional<int64_t>(axis.value()->value) : std::nullopt;
  return WrapNested(relax::take(x, indices, ax), std::string(name));
}

// ---------------------------------------------------------------------------
// Creation ops
// ---------------------------------------------------------------------------

static ffi::Any NNArange(ffi::Any start, ffi::Optional<ffi::Any> end, ffi::Any step,
                         ffi::Optional<ffi::String> dtype, ffi::String name) {
  // Convert start/end/step to PrimValue
  auto to_prim_val = [](const ffi::Any& v) -> PrimValue {
    if (auto opt = v.TryAs<int64_t>()) return PrimValue(tir::IntImm(DataType::Int(64), opt.value()));
    if (auto opt = v.TryAs<double>()) return PrimValue(tir::FloatImm(DataType::Float(64), opt.value()));
    if (auto opt = v.TryAs<PrimExpr>()) return PrimValue(opt.value());
    TVM_FFI_THROW(TypeError) << "NNArange: invalid argument type: " << v.GetTypeKey();
  };
  PrimValue s = to_prim_val(start);
  PrimValue e = end.defined() ? to_prim_val(end.value()) : s;
  PrimValue st = to_prim_val(step);
  if (!end.defined()) {
    // arange(end) -> start=0, end=start, step=1
    e = s;
    s = PrimValue(tir::IntImm(DataType::Int(64), 0));
    st = PrimValue(tir::IntImm(DataType::Int(64), 1));
  }
  DataType dt = DataType(runtime::String2DLDataType(dtype.value_or("float32")));
  return WrapNested(relax::arange(s, e, st, dt), std::string(name));
}

static ffi::Any NNFull(ffi::Array<ffi::Any> shape, Var fill_value, ffi::String dtype,
                       ffi::String name) {
  ffi::Array<PrimExpr> new_shape;
  for (const ffi::Any& s : shape) {
    if (auto opt = s.TryAs<int64_t>())
      new_shape.push_back(tir::IntImm(DataType::Int(64), opt.value()));
    else if (auto opt = s.TryAs<PrimExpr>())
      new_shape.push_back(opt.value());
    else
      TVM_FFI_THROW(TypeError) << "NNFull: invalid shape element: " << s.GetTypeKey();
  }
  DataType dt = DataType(runtime::String2DLDataType(dtype));
  return WrapNested(relax::full(ShapeExpr(new_shape), fill_value, dt), std::string(name));
}

static ffi::Any NNZeros(ffi::Array<ffi::Any> shape, ffi::String dtype, ffi::String name) {
  ffi::Array<PrimExpr> new_shape;
  for (const ffi::Any& s : shape) {
    if (auto opt = s.TryAs<int64_t>())
      new_shape.push_back(tir::IntImm(DataType::Int(64), opt.value()));
    else if (auto opt = s.TryAs<PrimExpr>())
      new_shape.push_back(opt.value());
    else
      TVM_FFI_THROW(TypeError) << "NNZeros: invalid shape element: " << s.GetTypeKey();
  }
  DataType dt = DataType(runtime::String2DLDataType(dtype));
  return WrapNested(relax::zeros(ShapeExpr(new_shape), dt), std::string(name));
}

static ffi::Any NNOnes(ffi::Array<ffi::Any> shape, ffi::String dtype, ffi::String name) {
  ffi::Array<PrimExpr> new_shape;
  for (const ffi::Any& s : shape) {
    if (auto opt = s.TryAs<int64_t>())
      new_shape.push_back(tir::IntImm(DataType::Int(64), opt.value()));
    else if (auto opt = s.TryAs<PrimExpr>())
      new_shape.push_back(opt.value());
    else
      TVM_FFI_THROW(TypeError) << "NNOnes: invalid shape element: " << s.GetTypeKey();
  }
  DataType dt = DataType(runtime::String2DLDataType(dtype));
  return WrapNested(relax::ones(ShapeExpr(new_shape), dt), std::string(name));
}

// ---------------------------------------------------------------------------
// Normalization ops
// ---------------------------------------------------------------------------

static ffi::Any NNLayerNorm(Var x, ffi::Array<Integer> axes, Var gamma, Var beta, double epsilon,
                            ffi::String name) {
  return WrapNested(relax::layer_norm(x, gamma, beta, axes, epsilon, /*center=*/true, /*scale=*/true),
                    std::string(name));
}

static ffi::Any NNRmsNorm(Var x, Var weight, ffi::Array<Integer> axes, double epsilon,
                          ffi::String name) {
  return WrapNested(relax::rms_norm(x, weight, axes, epsilon), std::string(name));
}

static ffi::Any NNGroupNorm(Var x, ffi::Optional<Var> weight, ffi::Optional<Var> bias,
                            int num_groups, int channel_axis, ffi::Array<Integer> axes,
                            double epsilon, ffi::String name) {
  TVM_FFI_ICHECK(weight.defined() && bias.defined())
      << "NNGroupNorm: weight and bias must be provided";
  return WrapNested(
      relax::group_norm(x, weight.value(), bias.value(), num_groups, channel_axis, axes, epsilon,
                        /*center=*/true, /*scale=*/true),
      std::string(name));
}

// ---------------------------------------------------------------------------
// Convolution ops
// ---------------------------------------------------------------------------

static ffi::Any NNConv1d(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides,
                         ffi::Any padding, ffi::Any dilation, int groups, ffi::String name) {
  auto to_arr = [](const ffi::Any& v) -> ffi::Array<int64_t> {
    if (auto opt = v.TryAs<int64_t>()) return {opt.value()};
    if (auto opt = v.TryAs<ffi::Array<ffi::Any>>()) {
      ffi::Array<int64_t> r;
      for (auto& e : opt.value()) r.push_back(e.operator int64_t());
      return r;
    }
    TVM_FFI_THROW(TypeError) << "conv1d: invalid stride/padding/dilation: " << v.GetTypeKey();
  };
  Expr out = relax::conv1d(x, weight, to_arr(strides), to_arr(padding), to_arr(dilation),
                           groups, "NCW", "OIW", std::nullopt, std::nullopt);
  if (bias.defined()) {
    // bias shape [O] -> reshape to [1, O, 1]
    Expr b = relax::reshape(bias.value(), ShapeExpr({tir::IntImm(DataType::Int(64), 1),
                                                     tir::IntImm(DataType::Int(64), -1),
                                                     tir::IntImm(DataType::Int(64), 1)}));
    out = relax::add(out, b);
  }
  return WrapNested(out, std::string(name));
}

static ffi::Any NNConv2d(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides,
                         ffi::Any padding, ffi::Any dilation, int groups, ffi::String data_layout,
                         ffi::String name) {
  auto to_arr = [](const ffi::Any& v) -> ffi::Array<int64_t> {
    if (auto opt = v.TryAs<int64_t>()) return {opt.value()};
    if (auto opt = v.TryAs<ffi::Array<ffi::Any>>()) {
      ffi::Array<int64_t> r;
      for (auto& e : opt.value()) r.push_back(e.operator int64_t());
      return r;
    }
    TVM_FFI_THROW(TypeError) << "conv2d: invalid stride/padding/dilation: " << v.GetTypeKey();
  };
  std::string dl = std::string(data_layout);
  std::string kl = (dl == "NCHW") ? "OIHW" : "HWIO";
  Expr out = relax::conv2d(x, weight, to_arr(strides), to_arr(padding), to_arr(dilation),
                           groups, dl, kl, std::nullopt, std::nullopt);
  if (bias.defined()) {
    Expr b;
    if (dl == "NCHW") {
      b = relax::reshape(bias.value(),
                         ShapeExpr({tir::IntImm(DataType::Int(64), 1),
                                    tir::IntImm(DataType::Int(64), -1),
                                    tir::IntImm(DataType::Int(64), 1),
                                    tir::IntImm(DataType::Int(64), 1)}));
    } else {
      b = relax::reshape(bias.value(),
                         ShapeExpr({tir::IntImm(DataType::Int(64), 1),
                                    tir::IntImm(DataType::Int(64), 1),
                                    tir::IntImm(DataType::Int(64), 1),
                                    tir::IntImm(DataType::Int(64), -1)}));
    }
    out = relax::add(out, b);
  }
  return WrapNested(out, std::string(name));
}

static ffi::Any NNConv3d(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides,
                         ffi::Any padding, ffi::Any dilation, int groups, ffi::String data_layout,
                         ffi::String name) {
  auto to_arr = [](const ffi::Any& v) -> ffi::Array<int64_t> {
    if (auto opt = v.TryAs<int64_t>()) return {opt.value()};
    if (auto opt = v.TryAs<ffi::Array<ffi::Any>>()) {
      ffi::Array<int64_t> r;
      for (auto& e : opt.value()) r.push_back(e.operator int64_t());
      return r;
    }
    TVM_FFI_THROW(TypeError) << "conv3d: invalid stride/padding/dilation: " << v.GetTypeKey();
  };
  std::string dl = std::string(data_layout);
  Expr out = relax::conv3d(x, weight, to_arr(strides), to_arr(padding), to_arr(dilation),
                           groups, dl, "OIDHW", std::nullopt, std::nullopt);
  if (bias.defined()) {
    Expr b;
    if (dl == "NCDHW") {
      b = relax::reshape(bias.value(),
                         ShapeExpr({tir::IntImm(DataType::Int(64), 1),
                                    tir::IntImm(DataType::Int(64), -1),
                                    tir::IntImm(DataType::Int(64), 1),
                                    tir::IntImm(DataType::Int(64), 1),
                                    tir::IntImm(DataType::Int(64), 1)}));
    } else {
      b = relax::reshape(bias.value(),
                         ShapeExpr({tir::IntImm(DataType::Int(64), 1),
                                    tir::IntImm(DataType::Int(64), 1),
                                    tir::IntImm(DataType::Int(64), 1),
                                    tir::IntImm(DataType::Int(64), 1),
                                    tir::IntImm(DataType::Int(64), -1)}));
    }
    out = relax::add(out, b);
  }
  return WrapNested(out, std::string(name));
}

static ffi::Any NNConv1dTranspose(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides,
                                  ffi::Any padding, ffi::Any output_padding, ffi::Any dilation,
                                  int groups, ffi::String name) {
  auto to_arr = [](const ffi::Any& v) -> ffi::Array<int64_t> {
    if (auto opt = v.TryAs<int64_t>()) return {opt.value()};
    if (auto opt = v.TryAs<ffi::Array<ffi::Any>>()) {
      ffi::Array<int64_t> r;
      for (auto& e : opt.value()) r.push_back(e.operator int64_t());
      return r;
    }
    TVM_FFI_THROW(TypeError) << "conv1d_transpose: invalid arg: " << v.GetTypeKey();
  };
  Expr out =
      relax::conv1d_transpose(x, weight, to_arr(strides), to_arr(padding),
                              to_arr(output_padding), to_arr(dilation), groups, "NCW", "IOW",
                              std::nullopt, std::nullopt);
  if (bias.defined()) {
    Expr b = relax::reshape(bias.value(), ShapeExpr({tir::IntImm(DataType::Int(64), 1),
                                                     tir::IntImm(DataType::Int(64), -1),
                                                     tir::IntImm(DataType::Int(64), 1)}));
    out = relax::add(out, b);
  }
  return WrapNested(out, std::string(name));
}

// ---------------------------------------------------------------------------
// Padding
// ---------------------------------------------------------------------------

static ffi::Any NNPad(Var x, ffi::Array<Integer> pad_width, ffi::String mode, double value,
                      ffi::String name) {
  return WrapNested(relax::pad(x, pad_width, mode, value), std::string(name));
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

static ffi::Any NNScaledDotProductAttention(Var query, Var key, Var value,
                                            ffi::Optional<ffi::String> causal_mask,
                                            ffi::Optional<double> scale, ffi::String name) {
  ffi::Optional<FloatImm> scale_imm = scale.defined()
      ? ffi::Optional<FloatImm>(FloatImm(DataType::Float(64), scale.value()))
      : std::nullopt;
  return WrapNested(
      relax::attention(query, key, value, std::nullopt, scale_imm, causal_mask, std::nullopt),
      std::string(name));
}

// ---------------------------------------------------------------------------
// Sorting / searching
// ---------------------------------------------------------------------------

static ffi::Any NNSort(Var x, int axis, bool descending, ffi::String name) {
  return WrapNested(relax::sort(x, axis, descending), std::string(name));
}

static ffi::Any NNArgsort(Var x, int axis, bool descending, ffi::String dtype, ffi::String name) {
  return WrapNested(relax::argsort(x, axis, descending,
                                   DataType(runtime::String2DLDataType(dtype))),
                    std::string(name));
}

static ffi::Any NNTopk(Var x, int k, int axis, ffi::String ret_type, bool largest,
                       ffi::String dtype, ffi::String name) {
  return WrapNested(relax::topk(x, k, axis, ret_type, largest,
                                DataType(runtime::String2DLDataType(dtype))),
                    std::string(name));
}

// ---------------------------------------------------------------------------
// CCL ops
// ---------------------------------------------------------------------------

static ffi::Any NNCclAllreduce(Var x, ffi::String op_type, bool in_group, ffi::String name) {
  return WrapNested(relax::allreduce(x, op_type, in_group), std::string(name));
}

static ffi::Any NNCclAllgather(Var x, int num_workers, ffi::String name) {
  return WrapNested(relax::allgather(x, num_workers, /*in_group=*/true), std::string(name));
}

static ffi::Any NNCclBroadcastFromWorker0(Var x, ffi::String name) {
  return WrapNested(relax::broadcast_from_worker0(x), std::string(name));
}

// ---------------------------------------------------------------------------
// Multinomial sampling
// ---------------------------------------------------------------------------

static ffi::Any NNMultinomialFromUniform(Var prob, Var uniform_sample, Var sample_indices,
                                         ffi::String dtype, ffi::String name) {
  return WrapNested(
      relax::multinomial_from_uniform(prob, uniform_sample, sample_indices,
                                      DataType(runtime::String2DLDataType(dtype))),
      std::string(name));
}

// ---------------------------------------------------------------------------
// Clip
// ---------------------------------------------------------------------------

static ffi::Any NNClip(Var x, double min_val, double max_val, ffi::String name) {
  // clip takes PrimValue min/max
  const auto* sinfo = x->struct_info_.as<TensorStructInfoNode>();
  DataType dtype = sinfo ? sinfo->dtype : DataType::Float(32);
  Expr mn, mx;
  if (dtype.is_float()) {
    mn = PrimValue(tir::FloatImm(dtype, min_val));
    mx = PrimValue(tir::FloatImm(dtype, max_val));
  } else {
    mn = PrimValue(tir::IntImm(dtype, static_cast<int64_t>(min_val)));
    mx = PrimValue(tir::IntImm(dtype, static_cast<int64_t>(max_val)));
  }
  return WrapNested(relax::clip(x, mn, mx), std::string(name));
}

// ---------------------------------------------------------------------------
// tensor_expr_op
//
// Signature (C++ FFI):
//   tensor_expr_op(
//       tensor_expr_func : ffi::Function(Array<te::Tensor>) -> Array<te::Tensor>,
//       name_hint        : String,
//       args             : Array<Expr>,          // relax Vars (tensor inputs)
//       primfunc_attrs   : Optional<Map<String,Any>>
//   ) -> Any   (Var or Array<Var>)
//
// Mirrors Python's BlockBuilder.emit_te path:
//   1. Wrap each relax Var in a TE placeholder via TETensor.
//   2. Call tensor_expr_func(te_inputs) to get te_outputs.
//   3. Lower {te_inputs..., te_outputs...} to a TIR PrimFunc via CreatePrimFunc.
//   4. Optionally attach primfunc_attrs.
//   5. Register the PrimFunc in the BlockBuilder and emit call_tir.
// ---------------------------------------------------------------------------

static ffi::Any NNTensorExprOp(
    ffi::Function tensor_expr_func,
    ffi::String name_hint,
    ffi::Array<Expr> args,
    ffi::Optional<ffi::Map<ffi::String, ffi::Any>> primfunc_attrs) {
  BlockBuilder bb = BlockBuilder::Current();
  TVM_FFI_ICHECK(bb.defined()) << "tensor_expr_op called outside of a BlockBuilder scope";

  // Step 1: wrap each relax Var in a TE placeholder
  ffi::Map<tir::Var, PrimExpr> empty_map;
  ffi::Array<te::Tensor> te_inputs;
  for (size_t i = 0; i < args.size(); ++i) {
    te_inputs.push_back(TETensor(args[i], empty_map, "input_" + std::to_string(i)));
  }

  // Step 2: call the user-supplied TE function
  ffi::Array<te::Tensor> te_outputs = tensor_expr_func(te_inputs);

  // Step 3: lower to TIR PrimFunc (inputs first, then outputs)
  ffi::Array<te::Tensor> all_tensors;
  for (const te::Tensor& t : te_inputs)  all_tensors.push_back(t);
  for (const te::Tensor& t : te_outputs) all_tensors.push_back(t);
  tir::PrimFunc prim_func = tir::CreatePrimFunc(all_tensors);

  // Step 4: attach optional primfunc_attrs
  if (primfunc_attrs.defined()) {
    for (const auto& [k, v] : primfunc_attrs.value()) {
      prim_func = WithAttr(prim_func, std::string(k), v);
    }
  }
  // Always mark private
  prim_func = WithAttr(prim_func, tvm::attr::kIsPrivateFunc, tvm::Bool(true));

  // Step 5: register and emit call_tir
  GlobalVar gv = bb->AddFunction(prim_func, std::string(name_hint));
  static const Op& call_tir_op = Op::Get("relax.call_tir");

  // Build out_sinfo from te_outputs
  ffi::Array<StructInfo> out_sinfo_list;
  for (const te::Tensor& t : te_outputs) {
    out_sinfo_list.push_back(TensorStructInfo(ShapeExpr(t->shape), t->dtype));
  }
  StructInfo out_sinfo = (out_sinfo_list.size() == 1)
      ? out_sinfo_list[0]
      : StructInfo(TupleStructInfo(out_sinfo_list));

  Expr call = Call(call_tir_op, {gv, relax::Tuple(args)}, tvm::Attrs(), {out_sinfo});
  return WrapNested(call, std::string(name_hint));
}

// ---------------------------------------------------------------------------
// tensor_ir_op
//
// Signature (C++ FFI):
//   tensor_ir_op(
//       func       : tir::PrimFunc,
//       name_hint  : String,
//       args       : Array<Expr>,   // Tensor Vars + optional ShapeExpr tir_vars
//       out        : Array<Any>     // NNTensor placeholders (carry struct_info)
//   ) -> Any
//
// Partitions args into:
//   - call_tir_args : Vars whose struct_info is TensorStructInfo
//   - tir_vars      : Vars whose struct_info is ShapeStructInfo (from SpecInt)
// ---------------------------------------------------------------------------

static ffi::Any NNTensorIrOp(
    tir::PrimFunc func,
    ffi::String name_hint,
    ffi::Array<Expr> args,
    ffi::Array<ffi::Any> out) {
  BlockBuilder bb = BlockBuilder::Current();
  TVM_FFI_ICHECK(bb.defined()) << "tensor_ir_op called outside of a BlockBuilder scope";

  // Partition args into tensor inputs and tir_vars
  ffi::Array<Expr> call_tir_args;
  ffi::Array<PrimExpr> tir_vars;
  for (const Expr& arg : args) {
    StructInfo sinfo = GetStructInfo(arg);
    if (sinfo->IsInstance<TensorStructInfoNode>()) {
      call_tir_args.push_back(arg);
    } else if (const auto* shape_sinfo = sinfo.as<ShapeStructInfoNode>()) {
      // ShapeExpr with one value → extract the PrimExpr
      TVM_FFI_ICHECK(shape_sinfo->values.defined() && shape_sinfo->values.value().size() == 1)
          << "tensor_ir_op: tir_var arg must be a Shape with exactly one value";
      tir_vars.push_back(shape_sinfo->values.value()[0]);
    } else {
      TVM_FFI_THROW(TypeError)
          << "tensor_ir_op: unsupported arg struct_info: " << sinfo->GetTypeKey();
    }
  }

  // Collect out_sinfo from the placeholder tensors
  ffi::Array<StructInfo> out_sinfo_list;
  for (const ffi::Any& ph : out) {
    Var ph_var = ph.cast<Var>();
    out_sinfo_list.push_back(GetStructInfo(ph_var));
  }
  StructInfo out_sinfo = (out_sinfo_list.size() == 1)
      ? out_sinfo_list[0]
      : StructInfo(TupleStructInfo(out_sinfo_list));

  // Mark private and register
  func = WithAttr(func, tvm::attr::kIsPrivateFunc, tvm::Bool(true));
  GlobalVar gv = bb->AddFunction(func, std::string(name_hint));

  static const Op& call_tir_op = Op::Get("relax.call_tir");
  ffi::Array<Expr> call_args{gv, relax::Tuple(call_tir_args)};
  if (!tir_vars.empty()) call_args.push_back(ShapeExpr(tir_vars));

  Expr call = Call(call_tir_op, call_args, tvm::Attrs(), {out_sinfo});
  return WrapNested(call, std::string(name_hint));
}

// ---------------------------------------------------------------------------
// tensor_ir_inplace_op
//
// Signature (C++ FFI):
//   tensor_ir_inplace_op(
//       func             : tir::PrimFunc,
//       name_hint        : String,
//       args             : Array<Expr>,
//       inplace_indices  : Array<Integer>,
//       out              : Array<Any>     // NNTensor placeholders
//   ) -> Any
// ---------------------------------------------------------------------------

static ffi::Any NNTensorIrInplaceOp(
    tir::PrimFunc func,
    ffi::String name_hint,
    ffi::Array<Expr> args,
    ffi::Array<Integer> inplace_indices,
    ffi::Array<ffi::Any> out) {
  BlockBuilder bb = BlockBuilder::Current();
  TVM_FFI_ICHECK(bb.defined()) << "tensor_ir_inplace_op called outside of a BlockBuilder scope";

  // Partition args (same logic as tensor_ir_op)
  ffi::Array<Expr> call_tir_args;
  ffi::Array<PrimExpr> tir_vars;
  for (const Expr& arg : args) {
    StructInfo sinfo = GetStructInfo(arg);
    if (sinfo->IsInstance<TensorStructInfoNode>()) {
      call_tir_args.push_back(arg);
    } else if (const auto* shape_sinfo = sinfo.as<ShapeStructInfoNode>()) {
      TVM_FFI_ICHECK(shape_sinfo->values.defined() && shape_sinfo->values.value().size() == 1)
          << "tensor_ir_inplace_op: tir_var arg must be a Shape with exactly one value";
      tir_vars.push_back(shape_sinfo->values.value()[0]);
    } else {
      TVM_FFI_THROW(TypeError)
          << "tensor_ir_inplace_op: unsupported arg struct_info: " << sinfo->GetTypeKey();
    }
  }

  // Collect out_sinfo
  ffi::Array<StructInfo> out_sinfo_list;
  for (const ffi::Any& ph : out) {
    Var ph_var = ph.cast<Var>();
    out_sinfo_list.push_back(GetStructInfo(ph_var));
  }
  StructInfo out_sinfo = (out_sinfo_list.size() == 1)
      ? out_sinfo_list[0]
      : StructInfo(TupleStructInfo(out_sinfo_list));

  // Build CallTIRInplaceAttrs
  ObjectPtr<CallTIRInplaceAttrs> attrs = ffi::make_object<CallTIRInplaceAttrs>();
  attrs->inplace_indices = inplace_indices;

  // Mark private and register
  func = WithAttr(func, tvm::attr::kIsPrivateFunc, tvm::Bool(true));
  GlobalVar gv = bb->AddFunction(func, std::string(name_hint));

  static const Op& call_tir_inplace_op = Op::Get("relax.call_tir_inplace");
  ffi::Array<Expr> call_args{gv, relax::Tuple(call_tir_args)};
  if (!tir_vars.empty()) call_args.push_back(ShapeExpr(tir_vars));

  Expr call = Call(call_tir_inplace_op, call_args, tvm::Attrs(attrs), {out_sinfo});
  return WrapNested(call, std::string(name_hint));
}

// ---------------------------------------------------------------------------
// extern
//
// Signature (C++ FFI):
//   extern_op(
//       name : String,                  // packed-func name
//       args : Array<Any>,              // Var | int64 | double | String | PrimExpr
//       out  : Array<Any>               // NNTensor placeholders
//   ) -> Any
//
// Lowers to R.call_dps_packed(name, Tuple(rx_args), out_sinfo).
// ---------------------------------------------------------------------------

// Convert a single extern arg to a relax Expr.
static Expr ConvertExternArg(const ffi::Any& arg) {
  if (auto opt = arg.TryAs<Var>())       return opt.value();
  if (auto opt = arg.TryAs<Expr>())      return opt.value();
  if (auto opt = arg.TryAs<int64_t>())
    return PrimValue(tir::IntImm(DataType::Int(64), opt.value()));
  if (auto opt = arg.TryAs<double>())
    return PrimValue(tir::FloatImm(DataType::Float(64), opt.value()));
  if (auto opt = arg.TryAs<ffi::String>())
    return StringImm(opt.value());
  if (auto opt = arg.TryAs<PrimExpr>())
    return PrimValue(opt.value());
  TVM_FFI_THROW(TypeError) << "extern: unsupported arg type: " << arg.GetTypeKey();
}

static ffi::Any NNExtern(
    ffi::String name,
    ffi::Array<ffi::Any> args,
    ffi::Array<ffi::Any> out) {
  BlockBuilder bb = BlockBuilder::Current();
  TVM_FFI_ICHECK(bb.defined()) << "extern called outside of a BlockBuilder scope";

  // Convert args to relax Exprs
  ffi::Array<Expr> rx_args;
  for (const ffi::Any& a : args) rx_args.push_back(ConvertExternArg(a));

  // Collect out_sinfo
  ffi::Array<TensorStructInfo> out_sinfo_list;
  for (const ffi::Any& ph : out) {
    Var ph_var = ph.cast<Var>();
    const auto* ts = GetStructInfo(ph_var).as<TensorStructInfoNode>();
    TVM_FFI_ICHECK(ts) << "extern: output placeholder must have TensorStructInfo";
    out_sinfo_list.push_back(GetRef<TensorStructInfo>(ts));
  }
  StructInfo out_sinfo = (out_sinfo_list.size() == 1)
      ? StructInfo(out_sinfo_list[0])
      : StructInfo(TupleStructInfo(
            ffi::Array<StructInfo>(out_sinfo_list.begin(), out_sinfo_list.end())));

  // call_dps_packed(name, Tuple(rx_args), out_sinfo)
  static const Op& call_dps_op = Op::Get("relax.call_dps_packed");
  Expr call = Call(call_dps_op,
                   {StringImm(name), relax::Tuple(rx_args)},
                   tvm::Attrs(),
                   {out_sinfo});
  return WrapNested(call, std::string(name));
}

// ---------------------------------------------------------------------------
// debug_func
//
// Signature (C++ FFI):
//   debug_func(
//       name       : String,
//       args       : Array<Any>,   // Var | int64 | double | String | PrimExpr
//       io_effect  : Var,          // the current _io effect Var
//       line_info  : String        // "filename:lineno"
//   ) -> Var   (the updated _io effect Var)
//
// Emits:
//   io = R.call_pure_packed(
//       "vm.builtin.invoke_debug_func",
//       io_effect, StringImm(name), StringImm(line_info),
//       ...converted_args...,
//       sinfo_args=[R.ObjectStructInfo()])
// and returns the new io Var.
// ---------------------------------------------------------------------------

static Var NNDebugFunc(
    ffi::String name,
    ffi::Array<ffi::Any> args,
    Var io_effect,
    ffi::String line_info) {
  BlockBuilder bb = BlockBuilder::Current();
  TVM_FFI_ICHECK(bb.defined()) << "debug_func called outside of a BlockBuilder scope";

  // Build the argument list for call_pure_packed:
  //   io_effect, StringImm(name), StringImm(line_info), ...converted_args...
  ffi::Array<Expr> call_args;
  call_args.push_back(io_effect);
  call_args.push_back(StringImm(name));
  call_args.push_back(StringImm(line_info));
  for (const ffi::Any& a : args) call_args.push_back(ConvertExternArg(a));

  static const Op& call_pure_packed_op = Op::Get("relax.call_pure_packed");
  Expr call = Call(call_pure_packed_op, call_args, tvm::Attrs(), {ObjectStructInfo()});
  return bb->Emit(call, std::string(io_effect->name_hint));
}

// ---------------------------------------------------------------------------
// get_timestep_embedding
//
// Signature (C++ FFI):
//   get_timestep_embedding(
//       x                   : Var,     // 1-D tensor of timesteps
//       embedding_dim       : int64,
//       flip_sin_to_cos     : bool,
//       downscale_freq_shift: double,
//       scale               : double,
//       max_period          : int64,
//       out_dtype           : String,  // default dtype for the output
//       name                : String
//   ) -> Any
//
// Mirrors the Python implementation exactly:
//   timesteps = astype(x, "float32")
//   half_dim  = embedding_dim // 2
//   exponent  = const(-log(max_period)) * arange(0, half_dim, "float32")
//   exponent  = exponent / const(half_dim - downscale_freq_shift)
//   emb       = exp(exponent)
//   emb       = expand_dims(timesteps, 1) * expand_dims(emb, 0)
//   if scale != 1: emb = const(scale) * emb
//   if flip_sin_to_cos: emb = concat([cos(emb), sin(emb)], axis=-1)
//   else:               emb = concat([sin(emb), cos(emb)], axis=-1)
//   if embedding_dim % 2 == 1: emb = pad(emb, (0,1,0,0))
//   emb = astype(emb, out_dtype)
// ---------------------------------------------------------------------------

static ffi::Any NNGetTimestepEmbedding(
    Var x,
    int64_t embedding_dim,
    bool flip_sin_to_cos,
    double downscale_freq_shift,
    double scale,
    int64_t max_period,
    ffi::String out_dtype,
    ffi::String name) {
  BlockBuilder bb = BlockBuilder::Current();
  TVM_FFI_ICHECK(bb.defined()) << "get_timestep_embedding called outside of a BlockBuilder scope";

  DataType f32 = DataType::Float(32);
  auto I64 = [](int64_t v) { return tir::IntImm(DataType::Int(64), v); };
  auto F32 = [](double v)  { return relax::const_(static_cast<float>(v)); };

  // timesteps = astype(x, "float32")
  Var timesteps = bb->Emit(relax::astype(x, f32), "timesteps");

  int64_t half_dim = embedding_dim / 2;

  // exponent = const(-log(max_period)) * arange(0, half_dim, "float32")
  double log_val = -std::log(static_cast<double>(max_period));
  Expr log_const = F32(log_val);
  Var arange_var = bb->Emit(
      relax::arange(PrimValue(I64(0)), PrimValue(I64(half_dim)),
                    PrimValue(I64(1)), f32),
      "arange");
  Var exponent = bb->Emit(relax::multiply(log_const, arange_var), "exponent");

  // exponent = exponent / const(half_dim - downscale_freq_shift)
  double denom = static_cast<double>(half_dim) - downscale_freq_shift;
  Var exponent2 = bb->Emit(relax::divide(exponent, F32(denom)), "exponent");

  // emb = exp(exponent)
  Var emb = bb->Emit(relax::exp(exponent2), "emb");

  // emb = expand_dims(timesteps, 1) * expand_dims(emb, 0)
  Var ts_exp  = bb->Emit(relax::expand_dims(timesteps, {1}), "timesteps");
  Var emb_exp = bb->Emit(relax::expand_dims(emb, {0}),       "emb");
  Var emb2    = bb->Emit(relax::multiply(ts_exp, emb_exp),   "emb");

  // if scale != 1: emb = const(scale) * emb
  Var emb3 = emb2;
  if (scale != 1.0) {
    emb3 = bb->Emit(relax::multiply(F32(scale), emb2), "emb");
  }

  // sin/cos concat
  Var sin_emb = bb->Emit(relax::sin(emb3), "sin");
  Var cos_emb = bb->Emit(relax::cos(emb3), "cos");
  ffi::Array<Expr> concat_args = flip_sin_to_cos
      ? ffi::Array<Expr>{cos_emb, sin_emb}
      : ffi::Array<Expr>{sin_emb, cos_emb};
  Var emb4 = bb->Emit(
      relax::concat(relax::Tuple(concat_args), ffi::Optional<int64_t>(-1)),
      "emb");

  // if embedding_dim % 2 == 1: pad
  Var emb5 = emb4;
  if (embedding_dim % 2 == 1) {
    emb5 = bb->Emit(
        relax::pad(emb4, {0, 1, 0, 0}, "constant", 0.0),
        "emb");
  }

  // astype to out_dtype
  DataType target_dt = DataType(runtime::String2DLDataType(out_dtype));
  Expr final_emb = (target_dt == f32)
      ? Expr(emb5)
      : relax::astype(emb5, target_dt);

  return WrapNested(final_emb, std::string(name));
}

// ---------------------------------------------------------------------------
// interpolate  (thin wrapper over NNResize2d with PyTorch-style API)
//
// Signature (C++ FFI):
//   interpolate(
//       x                      : Var,
//       size                   : Array<Integer>,   // already-resolved output HW
//       data_layout            : String,
//       method                 : String,           // "nearest_neighbor"|"linear"|"bicubic"
//       coord_trans            : String,           // "asymmetric"|"align_corners"|"half_pixel"
//       name                   : String
//   ) -> Any
//
// The Python wrapper resolves size/scale_factor and mode strings before
// calling this FFI entry, keeping all the PyTorch-compat logic in Python.
// ---------------------------------------------------------------------------

static ffi::Any NNInterpolate(
    Var x,
    ffi::Array<Integer> size,
    ffi::String data_layout,
    ffi::String method,
    ffi::String coord_trans,
    ffi::String name) {
  return NNResize2d(x, size, data_layout, method, coord_trans, name);
}

// ---------------------------------------------------------------------------
// Image resize
// ---------------------------------------------------------------------------

static ffi::Any NNResize2d(Var x, ffi::Array<Integer> size, ffi::String layout, ffi::String method,
                           ffi::String coord_trans, ffi::String name) {
  ffi::Array<PrimExpr> size_prim;
  for (const Integer& s : size) size_prim.push_back(s);
  // roi is empty for non-crop resize
  ffi::Array<FloatImm> roi;
  return WrapNested(
      relax::resize2d(x, ShapeExpr(size_prim), roi, layout, method, coord_trans,
                      "round", -0.5, 0, 0.0, std::nullopt),
      std::string(name));
}

// ---------------------------------------------------------------------------
// FFI registrations
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      // Unary
      .def("relax.frontend.nn.op.relu", NNRelu)
      .def("relax.frontend.nn.op.relu6", NNRelu6)
      .def("relax.frontend.nn.op.silu", NNSilu)
      .def("relax.frontend.nn.op.gelu", NNGelu)
      .def("relax.frontend.nn.op.sigmoid", NNSigmoid)
      .def("relax.frontend.nn.op.tanh", NNTanh)
      .def("relax.frontend.nn.op.exp", NNExp)
      .def("relax.frontend.nn.op.log", NNLog)
      .def("relax.frontend.nn.op.floor", NNFloor)
      .def("relax.frontend.nn.op.sqrt", NNSqrt)
      .def("relax.frontend.nn.op.square", NNSquare)
      .def("relax.frontend.nn.op.negative", NNNegative)
      .def("relax.frontend.nn.op.softmax", NNSoftmax)
      .def("relax.frontend.nn.op.softplus", NNSoftplus)
      .def("relax.frontend.nn.op.prelu", NNPrelu)
      // Binary
      .def("relax.frontend.nn.op.add", NNAdd)
      .def("relax.frontend.nn.op.subtract", NNSubtract)
      .def("relax.frontend.nn.op.multiply", NNMultiply)
      .def("relax.frontend.nn.op.divide", NNDivide)
      .def("relax.frontend.nn.op.maximum", NNMaximum)
      .def("relax.frontend.nn.op.minimum", NNMinimum)
      .def("relax.frontend.nn.op.less", NNLess)
      .def("relax.frontend.nn.op.less_equal", NNLessEqual)
      .def("relax.frontend.nn.op.greater", NNGreater)
      .def("relax.frontend.nn.op.greater_equal", NNGreaterEqual)
      .def("relax.frontend.nn.op.equal", NNEqual)
      .def("relax.frontend.nn.op.not_equal", NNNotEqual)
      .def("relax.frontend.nn.op.where", NNWhere)
      // Shape manipulation
      .def("relax.frontend.nn.op.unsqueeze", NNUnsqueeze)
      .def("relax.frontend.nn.op.squeeze", NNSqueeze)
      .def("relax.frontend.nn.op.reshape", NNReshape)
      .def("relax.frontend.nn.op.permute_dims", NNPermuteDims)
      .def("relax.frontend.nn.op.broadcast_to", NNBroadcastTo)
      .def("relax.frontend.nn.op.repeat", NNRepeat)
      .def("relax.frontend.nn.op.concat", NNConcat)
      .def("relax.frontend.nn.op.split", NNSplit)
      .def("relax.frontend.nn.op.chunk", NNChunk)
      .def("relax.frontend.nn.op.triu", NNTriu)
      // Reduction
      .def("relax.frontend.nn.op.sum", NNSum)
      .def("relax.frontend.nn.op.max", NNMax)
      .def("relax.frontend.nn.op.min", NNMin)
      .def("relax.frontend.nn.op.cumsum", NNCumsum)
      // Linear algebra
      .def("relax.frontend.nn.op.matmul", NNMatmul)
      // Type casting
      .def("relax.frontend.nn.op.astype", NNAstype)
      // Indexing
      .def("relax.frontend.nn.op.take", NNTake)
      // Creation
      .def("relax.frontend.nn.op.arange", NNArange)
      .def("relax.frontend.nn.op.full", NNFull)
      .def("relax.frontend.nn.op.zeros", NNZeros)
      .def("relax.frontend.nn.op.ones", NNOnes)
      // Normalization
      .def("relax.frontend.nn.op.layer_norm", NNLayerNorm)
      .def("relax.frontend.nn.op.rms_norm", NNRmsNorm)
      .def("relax.frontend.nn.op.group_norm", NNGroupNorm)
      // Convolution
      .def("relax.frontend.nn.op.conv1d", NNConv1d)
      .def("relax.frontend.nn.op.conv2d", NNConv2d)
      .def("relax.frontend.nn.op.conv3d", NNConv3d)
      .def("relax.frontend.nn.op.conv1d_transpose", NNConv1dTranspose)
      // Padding
      .def("relax.frontend.nn.op.pad", NNPad)
      // Attention
      .def("relax.frontend.nn.op.scaled_dot_product_attention", NNScaledDotProductAttention)
      // Sorting
      .def("relax.frontend.nn.op.sort", NNSort)
      .def("relax.frontend.nn.op.argsort", NNArgsort)
      .def("relax.frontend.nn.op.topk", NNTopk)
      // CCL
      .def("relax.frontend.nn.op.ccl_allreduce", NNCclAllreduce)
      .def("relax.frontend.nn.op.ccl_allgather", NNCclAllgather)
      .def("relax.frontend.nn.op.ccl_broadcast_from_worker0", NNCclBroadcastFromWorker0)
      // Sampling
      .def("relax.frontend.nn.op.multinomial_from_uniform", NNMultinomialFromUniform)
                  // Image
      .def("relax.frontend.nn.op.resize2d", NNResize2d)
      // Clip
      .def("relax.frontend.nn.op.clip", NNClip)
      // TE / TIR ops
      .def("relax.frontend.nn.op.tensor_expr_op",     NNTensorExprOp)
      .def("relax.frontend.nn.op.tensor_ir_op",       NNTensorIrOp)
      .def("relax.frontend.nn.op.tensor_ir_inplace_op", NNTensorIrInplaceOp)
      .def("relax.frontend.nn.op.extern",             NNExtern)
      .def("relax.frontend.nn.op.debug_func",         NNDebugFunc)
      // Timestep embedding
      .def("relax.frontend.nn.op.get_timestep_embedding", NNGetTimestepEmbedding)
      // Interpolate
      .def("relax.frontend.nn.op.interpolate",        NNInterpolate);
}

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
