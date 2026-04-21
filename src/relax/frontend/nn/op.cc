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
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/tensor.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/var.h>

#include <cmath>
#include <string>
#include <vector>

#include "../../../relax/ir/emit_te.h"
#include "../../../te/operation/create_primfunc.h"
#include "../../../tir/ir/script/script_complete.h"
#include "../../op/ccl/ccl.h"
#include "../../op/image/resize.h"
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
#include "../../op/tensor/ternary.h"
#include "../../op/tensor/unary.h"
#include "core.h"
#include "exporter.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ---------------------------------------------------------------------------
// Internal helper: emit expr and return bound Var (or Array for tuples)
// ---------------------------------------------------------------------------

static ffi::Any WrapNested(Expr expr, const std::string& name) {
  BlockBuilder bb = BlockBuilder_Current();
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
  TVM_FFI_UNREACHABLE();
}

// ---------------------------------------------------------------------------
// Unary element-wise ops
// ---------------------------------------------------------------------------

#define NN_UNARY_OP(func_name, relax_op)               \
  static ffi::Any func_name(Var x, ffi::String name) { \
    return WrapNested(relax_op(x), std::string(name)); \
  }

NN_UNARY_OP(NNRelu, relax::relu)

static ffi::Any NNRelu6(Var x, ffi::String name) {
  // relu6 = clip(x, 0, 6) -- matches relax.op.nn.relu6 Python implementation
  PrimValue zero = PrimValue(IntImm(DataType::Int(64), 0));
  PrimValue six = PrimValue(IntImm(DataType::Int(64), 6));
  return WrapNested(relax::clip(x, zero, six), std::string(name));
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
// GeLU
// ---------------------------------------------------------------------------

static ffi::Any NNGelu(Var x, ffi::Optional<ffi::String> approximate, ffi::String name) {
  Expr out;
  if (approximate.has_value() && approximate.value() == "tanh") {
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
  return WrapNested(relax::prelu(x, alpha, 1), std::string(name));
}

// ---------------------------------------------------------------------------
// Binary element-wise ops
// ---------------------------------------------------------------------------

#define NN_BINARY_OP(func_name, relax_op)                     \
  static ffi::Any func_name(Var a, Var b, ffi::String name) { \
    return WrapNested(relax_op(a, b), std::string(name));     \
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
    if (auto opt = s.try_cast<int64_t>()) {
      new_shape.push_back(IntImm(DataType::Int(64), opt.value()));
    } else if (auto opt = s.try_cast<PrimExpr>()) {
      new_shape.push_back(opt.value());
    } else {
      TVM_FFI_THROW(TypeError) << "NNReshape: invalid shape element type: " << s.GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }
  }
  return WrapNested(relax::reshape(x, ShapeExpr(new_shape)), std::string(name));
}

static ffi::Any NNPermuteDims(Var x, ffi::Optional<ffi::Array<Integer>> axes, ffi::String name) {
  return WrapNested(relax::permute_dims(x, axes), std::string(name));
}

static ffi::Any NNBroadcastTo(Var x, ffi::Array<ffi::Any> shape, ffi::String name) {
  ffi::Array<PrimExpr> new_shape;
  for (const ffi::Any& s : shape) {
    if (auto opt = s.try_cast<int64_t>()) {
      new_shape.push_back(IntImm(DataType::Int(64), opt.value()));
    } else if (auto opt = s.try_cast<PrimExpr>()) {
      new_shape.push_back(opt.value());
    } else {
      TVM_FFI_THROW(TypeError) << "NNBroadcastTo: invalid shape element: " << s.GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }
  }
  return WrapNested(relax::broadcast_to(x, ShapeExpr(new_shape)), std::string(name));
}

static ffi::Any NNRepeat(Var x, int repeats, ffi::Optional<Integer> axis, ffi::String name) {
  ffi::Optional<int64_t> ax =
      axis.has_value() ? ffi::Optional<int64_t>(axis.value()->value) : std::nullopt;
  return WrapNested(relax::repeat(x, repeats, ax), std::string(name));
}

static ffi::Any NNConcat(ffi::Array<Var> tensors, int dim, ffi::String name) {
  ffi::Array<Expr> exprs;
  for (const Var& v : tensors) exprs.push_back(v);
  return WrapNested(relax::concat(Tuple(exprs), ffi::Optional<int64_t>(dim)), std::string(name));
}

static ffi::Any NNSplit(Var x, ffi::Any indices_or_sections, int axis, ffi::String name) {
  if (auto opt = indices_or_sections.try_cast<int64_t>()) {
    return WrapNested(relax::split(x, IntImm(DataType::Int(64), opt.value()), axis),
                      std::string(name));
  }
  if (auto opt = indices_or_sections.try_cast<ffi::Array<ffi::Any>>()) {
    ffi::Array<IntImm> indices;
    for (const ffi::Any& idx : opt.value()) {
      indices.push_back(IntImm(DataType::Int(64), idx.cast<int64_t>()));
    }
    return WrapNested(relax::split(x, indices, axis), std::string(name));
  }
  TVM_FFI_THROW(TypeError) << "NNSplit: indices_or_sections must be int or Array[int]";
  TVM_FFI_UNREACHABLE();
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

static ffi::Optional<ffi::Array<Integer>> NormalizeAxis(ffi::Optional<ffi::Array<Integer>> axis) {
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
  ffi::Optional<int64_t> ax =
      axis.has_value() ? ffi::Optional<int64_t>(axis.value()->value) : std::nullopt;
  ffi::Optional<DataType> dt =
      dtype.has_value() ? ffi::Optional<DataType>(DataType(ffi::StringToDLDataType(dtype.value())))
                        : std::nullopt;
  Bool excl = exclusive.has_value() ? exclusive.value() : Bool(false);
  return WrapNested(relax::cumsum(x, ax, dt, excl), std::string(name));
}

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

static ffi::Any NNMatmul(Var a, Var b, ffi::Optional<ffi::String> out_dtype, ffi::String name) {
  ffi::Optional<DataType> dt =
      out_dtype.has_value()
          ? ffi::Optional<DataType>(DataType(ffi::StringToDLDataType(out_dtype.value())))
          : std::nullopt;
  return WrapNested(relax::matmul(a, b, dt), std::string(name));
}

// ---------------------------------------------------------------------------
// Type casting
// ---------------------------------------------------------------------------

static ffi::Any NNAstype(Var x, ffi::String dtype, ffi::String name) {
  DataType target_dtype = DataType(ffi::StringToDLDataType(dtype));
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
  ffi::Optional<int64_t> ax =
      axis.has_value() ? ffi::Optional<int64_t>(axis.value()->value) : std::nullopt;
  return WrapNested(relax::take(x, indices, ax), std::string(name));
}

// ---------------------------------------------------------------------------
// Creation ops
// ---------------------------------------------------------------------------

static ffi::Any NNArange(ffi::Any start, ffi::Any end, ffi::Any step,
                         ffi::Optional<ffi::String> dtype, ffi::String name) {
  auto to_prim_val = [](const ffi::Any& v) -> PrimValue {
    if (auto opt = v.try_cast<int64_t>()) return PrimValue(IntImm(DataType::Int(64), opt.value()));
    if (auto opt = v.try_cast<double>())
      return PrimValue(FloatImm(DataType::Float(64), opt.value()));
    if (auto opt = v.try_cast<PrimExpr>()) return PrimValue(opt.value());
    TVM_FFI_THROW(TypeError) << "NNArange: invalid argument type: " << v.GetTypeKey();
    TVM_FFI_UNREACHABLE();
  };
  // If end is null/None (represented as a null Any), treat as arange(end=start, start=0, step=1)
  bool end_is_null = (end == nullptr);
  PrimValue s = end_is_null ? PrimValue(IntImm(DataType::Int(64), 0)) : to_prim_val(start);
  PrimValue e = end_is_null ? to_prim_val(start) : to_prim_val(end);
  PrimValue st = end_is_null ? PrimValue(IntImm(DataType::Int(64), 1)) : to_prim_val(step);
  DataType dt = DataType(ffi::StringToDLDataType(dtype.value_or("float32")));
  return WrapNested(relax::arange(s, e, st, dt), std::string(name));
}

static ffi::Any NNFull(ffi::Array<ffi::Any> shape, Expr fill_value, ffi::String dtype,
                       ffi::String name) {
  ffi::Array<PrimExpr> new_shape;
  for (const ffi::Any& s : shape) {
    if (auto opt = s.try_cast<int64_t>())
      new_shape.push_back(IntImm(DataType::Int(64), opt.value()));
    else if (auto opt = s.try_cast<PrimExpr>())
      new_shape.push_back(opt.value());
    else {
      TVM_FFI_THROW(TypeError) << "NNFull: invalid shape element: " << s.GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }
  }
  DataType dt = DataType(ffi::StringToDLDataType(dtype));
  return WrapNested(relax::full(ShapeExpr(new_shape), fill_value, dt), std::string(name));
}

static ffi::Any NNZeros(ffi::Array<ffi::Any> shape, ffi::String dtype, ffi::String name) {
  ffi::Array<PrimExpr> new_shape;
  for (const ffi::Any& s : shape) {
    if (auto opt = s.try_cast<int64_t>())
      new_shape.push_back(IntImm(DataType::Int(64), opt.value()));
    else if (auto opt = s.try_cast<PrimExpr>())
      new_shape.push_back(opt.value());
    else {
      TVM_FFI_THROW(TypeError) << "NNZeros: invalid shape element: " << s.GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }
  }
  DataType dt = DataType(ffi::StringToDLDataType(dtype));
  return WrapNested(relax::zeros(ShapeExpr(new_shape), dt), std::string(name));
}

static ffi::Any NNOnes(ffi::Array<ffi::Any> shape, ffi::String dtype, ffi::String name) {
  ffi::Array<PrimExpr> new_shape;
  for (const ffi::Any& s : shape) {
    if (auto opt = s.try_cast<int64_t>())
      new_shape.push_back(IntImm(DataType::Int(64), opt.value()));
    else if (auto opt = s.try_cast<PrimExpr>())
      new_shape.push_back(opt.value());
    else {
      TVM_FFI_THROW(TypeError) << "NNOnes: invalid shape element: " << s.GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }
  }
  DataType dt = DataType(ffi::StringToDLDataType(dtype));
  return WrapNested(relax::ones(ShapeExpr(new_shape), dt), std::string(name));
}

// ---------------------------------------------------------------------------
// Normalization ops
// ---------------------------------------------------------------------------

static ffi::Any NNLayerNorm(Var x, ffi::Array<Integer> axes, Expr gamma, Expr beta, double epsilon,
                            ffi::String name) {
  return WrapNested(
      relax::layer_norm(x, gamma, beta, axes, epsilon, /*center=*/true, /*scale=*/true),
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
    if (auto opt = v.try_cast<int64_t>()) return ffi::Array<int64_t>{opt.value()};
    if (auto opt = v.try_cast<ffi::Array<ffi::Any>>()) {
      ffi::Array<int64_t> r;
      for (auto& e : opt.value()) r.push_back(e.cast<int64_t>());
      return r;
    }
    TVM_FFI_THROW(TypeError) << "conv1d: invalid stride/padding/dilation: " << v.GetTypeKey();
    TVM_FFI_UNREACHABLE();
  };
  Expr out = relax::conv1d(x, weight, to_arr(strides), to_arr(padding), to_arr(dilation), groups,
                           "NCW", "OIW", std::nullopt, std::nullopt);
  if (bias.has_value()) {
    Expr b = relax::reshape(
        bias.value(),
        ShapeExpr(ffi::Array<PrimExpr>{IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), -1),
                                       IntImm(DataType::Int(64), 1)}));
    out = relax::add(out, b);
  }
  return WrapNested(out, std::string(name));
}

static ffi::Any NNConv2d(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides,
                         ffi::Any padding, ffi::Any dilation, int groups, ffi::String data_layout,
                         ffi::String name) {
  auto to_arr = [](const ffi::Any& v) -> ffi::Array<int64_t> {
    if (auto opt = v.try_cast<int64_t>()) return ffi::Array<int64_t>{opt.value()};
    if (auto opt = v.try_cast<ffi::Array<ffi::Any>>()) {
      ffi::Array<int64_t> r;
      for (auto& e : opt.value()) r.push_back(e.cast<int64_t>());
      return r;
    }
    TVM_FFI_THROW(TypeError) << "conv2d: invalid stride/padding/dilation: " << v.GetTypeKey();
    TVM_FFI_UNREACHABLE();
  };
  std::string dl = std::string(data_layout);
  std::string kl = (dl == "NCHW") ? "OIHW" : "HWIO";
  Expr out = relax::conv2d(x, weight, to_arr(strides), to_arr(padding), to_arr(dilation), groups,
                           dl, kl, std::nullopt, std::nullopt);
  if (bias.has_value()) {
    Expr b;
    if (dl == "NCHW") {
      b = relax::reshape(bias.value(),
                         ShapeExpr(ffi::Array<PrimExpr>{
                             IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), -1),
                             IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 1)}));
    } else {
      b = relax::reshape(bias.value(),
                         ShapeExpr(ffi::Array<PrimExpr>{
                             IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 1),
                             IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), -1)}));
    }
    out = relax::add(out, b);
  }
  return WrapNested(out, std::string(name));
}

static ffi::Any NNConv3d(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides,
                         ffi::Any padding, ffi::Any dilation, int groups, ffi::String data_layout,
                         ffi::String name) {
  auto to_arr = [](const ffi::Any& v) -> ffi::Array<int64_t> {
    if (auto opt = v.try_cast<int64_t>()) return ffi::Array<int64_t>{opt.value()};
    if (auto opt = v.try_cast<ffi::Array<ffi::Any>>()) {
      ffi::Array<int64_t> r;
      for (auto& e : opt.value()) r.push_back(e.cast<int64_t>());
      return r;
    }
    TVM_FFI_THROW(TypeError) << "conv3d: invalid stride/padding/dilation: " << v.GetTypeKey();
    TVM_FFI_UNREACHABLE();
  };
  std::string dl = std::string(data_layout);
  Expr out = relax::conv3d(x, weight, to_arr(strides), to_arr(padding), to_arr(dilation), groups,
                           dl, "OIDHW", std::nullopt, std::nullopt);
  if (bias.has_value()) {
    Expr b;
    if (dl == "NCDHW") {
      b = relax::reshape(bias.value(),
                         ShapeExpr(ffi::Array<PrimExpr>{
                             IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), -1),
                             IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 1),
                             IntImm(DataType::Int(64), 1)}));
    } else {
      b = relax::reshape(
          bias.value(),
          ShapeExpr(ffi::Array<PrimExpr>{IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 1),
                                         IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 1),
                                         IntImm(DataType::Int(64), -1)}));
    }
    out = relax::add(out, b);
  }
  return WrapNested(out, std::string(name));
}

static ffi::Any NNConv1dTranspose(Var x, Var weight, ffi::Optional<Var> bias, ffi::Any strides,
                                  ffi::Any padding, ffi::Any output_padding, ffi::Any dilation,
                                  int groups, ffi::String name) {
  auto to_arr = [](const ffi::Any& v) -> ffi::Array<int64_t> {
    if (auto opt = v.try_cast<int64_t>()) return ffi::Array<int64_t>{opt.value()};
    if (auto opt = v.try_cast<ffi::Array<ffi::Any>>()) {
      ffi::Array<int64_t> r;
      for (auto& e : opt.value()) r.push_back(e.cast<int64_t>());
      return r;
    }
    TVM_FFI_THROW(TypeError) << "conv1d_transpose: invalid arg: " << v.GetTypeKey();
    TVM_FFI_UNREACHABLE();
  };
  Expr out =
      relax::conv1d_transpose(x, weight, to_arr(strides), to_arr(padding), to_arr(output_padding),
                              to_arr(dilation), groups, "NCW", "IOW", std::nullopt, std::nullopt);
  if (bias.has_value()) {
    Expr b = relax::reshape(
        bias.value(),
        ShapeExpr(ffi::Array<PrimExpr>{IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), -1),
                                       IntImm(DataType::Int(64), 1)}));
    out = relax::add(out, b);
  }
  return WrapNested(out, std::string(name));
}

// ---------------------------------------------------------------------------
// Padding
// ---------------------------------------------------------------------------

static ffi::Any NNPad(Var x, ffi::Array<Integer> pad_width, ffi::String mode, double value,
                      ffi::String name) {
  static const Op& pad_op = Op::Get("relax.nn.pad");
  auto attrs = ffi::make_object<tvm::relax::PadAttrs>();
  attrs->pad_width = pad_width;
  attrs->pad_mode = std::string(mode);
  attrs->pad_value = value;
  return WrapNested(Call(pad_op, {x}, Attrs(attrs), {}), std::string(name));
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

static ffi::Any NNScaledDotProductAttention(Var query, Var key, Var value,
                                            ffi::Optional<ffi::String> causal_mask,
                                            ffi::Optional<double> scale, ffi::String name) {
  ffi::Optional<FloatImm> scale_imm =
      scale.has_value() ? ffi::Optional<FloatImm>(FloatImm(DataType::Float(64), scale.value()))
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
  return WrapNested(relax::argsort(x, axis, descending, DataType(ffi::StringToDLDataType(dtype))),
                    std::string(name));
}

static ffi::Any NNTopk(Var x, int k, int axis, ffi::String ret_type, bool largest,
                       ffi::String dtype, ffi::String name) {
  return WrapNested(
      relax::topk(x, k, axis, ret_type, largest, DataType(ffi::StringToDLDataType(dtype))),
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
  return WrapNested(relax::multinomial_from_uniform(prob, uniform_sample, sample_indices,
                                                    DataType(ffi::StringToDLDataType(dtype))),
                    std::string(name));
}

// ---------------------------------------------------------------------------
// Clip
// ---------------------------------------------------------------------------

static ffi::Any NNClip(Var x, double min_val, double max_val, ffi::String name) {
  const auto* sinfo = x->struct_info_.as<TensorStructInfoNode>();
  DataType dtype = sinfo ? sinfo->dtype : DataType::Float(32);
  Expr mn, mx;
  if (dtype.is_float()) {
    mn = PrimValue(FloatImm(dtype, min_val));
    mx = PrimValue(FloatImm(dtype, max_val));
  } else {
    mn = PrimValue(IntImm(dtype, static_cast<int64_t>(min_val)));
    mx = PrimValue(IntImm(dtype, static_cast<int64_t>(max_val)));
  }
  return WrapNested(relax::clip(x, mn, mx), std::string(name));
}

// ---------------------------------------------------------------------------
// tensor_expr_op
// ---------------------------------------------------------------------------

static ffi::Any NNTensorExprOp(ffi::Function tensor_expr_func, ffi::String name_hint,
                               ffi::Array<Expr> args,
                               ffi::Optional<ffi::Map<ffi::String, ffi::Any>> primfunc_attrs) {
  BlockBuilder bb = BlockBuilder_Current();
  TVM_FFI_ICHECK(bb.defined()) << "tensor_expr_op called outside of a BlockBuilder scope";

  ffi::Map<tir::Var, PrimExpr> empty_map;
  ffi::Array<te::Tensor> te_inputs;
  for (size_t i = 0; i < args.size(); ++i) {
    te_inputs.push_back(TETensor(args[i], empty_map, "input_" + std::to_string(i)));
  }

  ffi::Array<te::Tensor> te_outputs = tensor_expr_func(te_inputs).cast<ffi::Array<te::Tensor>>();

  ffi::Array<te::Tensor> all_tensors;
  for (const te::Tensor& t : te_inputs) all_tensors.push_back(t);
  for (const te::Tensor& t : te_outputs) all_tensors.push_back(t);
  tir::PrimFunc prim_func = tir::CreatePrimFunc(all_tensors, DataType::Int(64));
  // CreatePrimFunc always sets global_symbol="main"; remove it so the
  // function is private (no global_symbol) and gets named by name_hint.
  prim_func = WithoutAttr(prim_func, "global_symbol");

  if (primfunc_attrs.defined()) {
    for (const auto& [k, v] : primfunc_attrs.value()) {
      prim_func = WithAttr(prim_func, std::string(k), v);
    }
  }
  GlobalVar gv = bb->AddFunction(prim_func, std::string(name_hint));
  static const Op& call_tir_op = Op::Get("relax.call_tir");

  ffi::Array<StructInfo> out_sinfo_list;
  for (const te::Tensor& t : te_outputs) {
    out_sinfo_list.push_back(TensorStructInfo(ShapeExpr(t->shape), t->dtype));
  }
  StructInfo out_sinfo = (out_sinfo_list.size() == 1) ? out_sinfo_list[0]
                                                      : StructInfo(TupleStructInfo(out_sinfo_list));

  Expr call = Call(call_tir_op, {gv, relax::Tuple(args)}, tvm::Attrs(), {out_sinfo});
  return WrapNested(call, std::string(name_hint));
}

// ---------------------------------------------------------------------------
// tensor_ir_op
// ---------------------------------------------------------------------------

static ffi::Any NNTensorIrOp(tir::PrimFunc func, ffi::String name_hint, ffi::Array<Expr> args,
                             ffi::Array<ffi::Any> out) {
  BlockBuilder bb = BlockBuilder_Current();
  TVM_FFI_ICHECK(bb.defined()) << "tensor_ir_op called outside of a BlockBuilder scope";

  ffi::Array<Expr> call_tir_args;
  ffi::Array<PrimExpr> tir_vars;
  for (const Expr& arg : args) {
    StructInfo sinfo = GetStructInfo(arg);
    if (sinfo->IsInstance<TensorStructInfoNode>()) {
      call_tir_args.push_back(arg);
    } else if (const auto* shape_sinfo = sinfo.as<ShapeStructInfoNode>()) {
      TVM_FFI_ICHECK(shape_sinfo->values.defined() && shape_sinfo->values.value().size() == 1)
          << "tensor_ir_op: tir_var arg must be a Shape with exactly one value";
      tir_vars.push_back(shape_sinfo->values.value()[0]);
    } else if (const auto* prim_sinfo = sinfo.as<PrimStructInfoNode>()) {
      tir_vars.push_back(prim_sinfo->value.value());
    } else {
      TVM_FFI_THROW(TypeError) << "tensor_ir_op: unsupported arg struct_info: "
                               << sinfo->GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }
  }

  ffi::Array<StructInfo> out_sinfo_list;
  for (const ffi::Any& ph : out) {
    Var ph_var = ph.cast<Var>();
    out_sinfo_list.push_back(GetStructInfo(ph_var));
  }
  StructInfo out_sinfo = (out_sinfo_list.size() == 1) ? out_sinfo_list[0]
                                                      : StructInfo(TupleStructInfo(out_sinfo_list));

  GlobalVar gv = bb->AddFunction(func, std::string(name_hint));

  static const Op& call_tir_op = Op::Get("relax.call_tir");
  ffi::Array<Expr> call_args{gv, relax::Tuple(call_tir_args)};
  if (!tir_vars.empty()) call_args.push_back(ShapeExpr(tir_vars));

  Expr call = Call(call_tir_op, call_args, tvm::Attrs(), {out_sinfo});
  return WrapNested(call, std::string(name_hint));
}

// ---------------------------------------------------------------------------
// tensor_ir_inplace_op
// ---------------------------------------------------------------------------

static ffi::Any NNTensorIrInplaceOp(tir::PrimFunc func, ffi::String name_hint,
                                    ffi::Array<Expr> args, ffi::Array<Integer> inplace_indices,
                                    ffi::Array<ffi::Any> out) {
  BlockBuilder bb = BlockBuilder_Current();
  TVM_FFI_ICHECK(bb.defined()) << "tensor_ir_inplace_op called outside of a BlockBuilder scope";

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
    } else if (const auto* prim_sinfo = sinfo.as<PrimStructInfoNode>()) {
      tir_vars.push_back(prim_sinfo->value.value());
    } else {
      TVM_FFI_THROW(TypeError) << "tensor_ir_inplace_op: unsupported arg struct_info: "
                               << sinfo->GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }
  }

  ffi::Array<StructInfo> out_sinfo_list;
  for (const ffi::Any& ph : out) {
    Var ph_var = ph.cast<Var>();
    out_sinfo_list.push_back(GetStructInfo(ph_var));
  }
  StructInfo out_sinfo = (out_sinfo_list.size() == 1) ? out_sinfo_list[0]
                                                      : StructInfo(TupleStructInfo(out_sinfo_list));

  ObjectPtr<CallTIRInplaceAttrs> attrs = ffi::make_object<CallTIRInplaceAttrs>();
  attrs->inplace_indices = inplace_indices;

  GlobalVar gv = bb->AddFunction(func, std::string(name_hint));

  static const Op& call_tir_inplace_op = Op::Get("relax.call_tir_inplace");
  ffi::Array<Expr> call_args{gv, relax::Tuple(call_tir_args)};
  if (!tir_vars.empty()) call_args.push_back(ShapeExpr(tir_vars));

  Expr call = Call(call_tir_inplace_op, call_args, tvm::Attrs(attrs), {out_sinfo});
  return WrapNested(call, std::string(name_hint));
}

// ---------------------------------------------------------------------------
// extern
// ---------------------------------------------------------------------------

static Expr ConvertExternArg(const ffi::Any& arg) {
  if (auto opt = arg.try_cast<Var>()) return opt.value();
  if (auto opt = arg.try_cast<Expr>()) return opt.value();
  if (auto opt = arg.try_cast<int64_t>()) return PrimValue(IntImm(DataType::Int(64), opt.value()));
  if (auto opt = arg.try_cast<double>())
    return PrimValue(FloatImm(DataType::Float(64), opt.value()));
  if (auto opt = arg.try_cast<ffi::String>()) return StringImm(opt.value());
  if (auto opt = arg.try_cast<PrimExpr>()) return PrimValue(opt.value());
  TVM_FFI_THROW(TypeError) << "extern: unsupported arg type: " << arg.GetTypeKey();
  TVM_FFI_UNREACHABLE();
}

static ffi::Any NNExtern(ffi::String name, ffi::Array<ffi::Any> args, ffi::Array<ffi::Any> out) {
  BlockBuilder bb = BlockBuilder_Current();
  TVM_FFI_ICHECK(bb.defined()) << "extern called outside of a BlockBuilder scope";

  ffi::Array<Expr> rx_args;
  for (const ffi::Any& a : args) rx_args.push_back(ConvertExternArg(a));

  ffi::Array<TensorStructInfo> out_sinfo_list;
  for (const ffi::Any& ph : out) {
    Var ph_var = ph.cast<Var>();
    const auto* ts = GetStructInfo(ph_var).as<TensorStructInfoNode>();
    TVM_FFI_ICHECK(ts) << "extern: output placeholder must have TensorStructInfo";
    out_sinfo_list.push_back(ffi::GetRef<TensorStructInfo>(ts));
  }
  StructInfo out_sinfo = (out_sinfo_list.size() == 1)
                             ? StructInfo(out_sinfo_list[0])
                             : StructInfo(TupleStructInfo(ffi::Array<StructInfo>(
                                   out_sinfo_list.begin(), out_sinfo_list.end())));

  static const Op& call_dps_op = Op::Get("relax.call_dps_packed");
  Expr call =
      Call(call_dps_op, {ExternFunc(name), relax::Tuple(rx_args)}, tvm::Attrs(), {out_sinfo});
  return WrapNested(call, std::string(name));
}

// ---------------------------------------------------------------------------
// debug_func
// ---------------------------------------------------------------------------

static Var NNDebugFunc(ffi::String name, ffi::Array<ffi::Any> args, Var io_effect,
                       ffi::String line_info) {
  BlockBuilder bb = BlockBuilder_Current();
  TVM_FFI_ICHECK(bb.defined()) << "debug_func called outside of a BlockBuilder scope";

  // call_pure_packed(ExternFunc(name), StringImm(line_info), ...user_args)
  // The packed function receives: (line_info, user_arg0, user_arg1, ...)
  // matching the convention: def my_debug_func(_lineo, arg0, arg1, ...)
  // io_effect is used only to chain the effect binding, not passed to the function.
  ffi::Array<Expr> call_args;
  call_args.push_back(ExternFunc(name));
  call_args.push_back(StringImm(line_info));
  for (const ffi::Any& a : args) call_args.push_back(ConvertExternArg(a));

  static const Op& call_pure_packed_op = Op::Get("relax.call_pure_packed");
  Expr call = Call(call_pure_packed_op, call_args, tvm::Attrs(), {ObjectStructInfo()});
  Var new_io = bb->Emit(call, std::string(io_effect->name_hint()));
  // Update the thread-local io var so subsequent debug_func calls chain correctly
  // and EmitMethod can read the final updated io var after forward() returns.
  SetCurrentIOVar(new_io);
  return new_io;
}

// ---------------------------------------------------------------------------
// get_timestep_embedding
// ---------------------------------------------------------------------------

static ffi::Any NNGetTimestepEmbedding(Var x, int64_t embedding_dim, bool flip_sin_to_cos,
                                       double downscale_freq_shift, double scale,
                                       int64_t max_period, ffi::String out_dtype,
                                       ffi::String name) {
  BlockBuilder bb = BlockBuilder_Current();
  TVM_FFI_ICHECK(bb.defined()) << "get_timestep_embedding called outside of a BlockBuilder scope";

  DataType f32 = DataType::Float(32);
  auto I64 = [](int64_t v) { return IntImm(DataType::Int(64), v); };
  // Build a scalar float32 constant: equivalent to Python's relax.const(v, "float32")
  auto MakeF32Const = [](float v) -> Expr {
    auto tensor = runtime::Tensor::Empty(ffi::Shape({}), DLDataType{kDLFloat, 32, 1},
                                         DLDevice{kDLCPU, 0}, std::nullopt);
    tensor.CopyFromBytes(&v, sizeof(float));
    return relax::Constant(tensor, std::nullopt);
  };
  auto F32 = [&MakeF32Const](double v) { return MakeF32Const(static_cast<float>(v)); };

  Var timesteps = bb->Emit(relax::astype(x, f32), "timesteps");
  Var ts_exp = bb->Emit(relax::expand_dims(timesteps, {1}), "timesteps");
  int64_t half_dim = embedding_dim / 2;

  double log_val = -std::log(static_cast<double>(max_period));
  Var arange_var = bb->Emit(
      relax::arange(PrimValue(I64(0)), PrimValue(I64(half_dim)), PrimValue(I64(1)), f32), "arange");
  Var exponent = bb->Emit(relax::multiply(F32(log_val), arange_var), "exponent");

  double denom = static_cast<double>(half_dim) - downscale_freq_shift;
  Var exponent2 = bb->Emit(relax::divide(exponent, F32(denom)), "exponent");
  Var emb = bb->Emit(relax::exp(exponent2), "emb");

  Var emb_exp = bb->Emit(relax::expand_dims(emb, {0}), "emb");
  Var emb2 = bb->Emit(relax::multiply(ts_exp, emb_exp), "emb");

  Var emb3 = emb2;
  if (scale != 1.0) {
    emb3 = bb->Emit(relax::multiply(F32(scale), emb2), "emb");
  }

  Var sin_emb = bb->Emit(relax::sin(emb3), "sin");
  Var cos_emb = bb->Emit(relax::cos(emb3), "cos");
  ffi::Array<Expr> concat_args =
      flip_sin_to_cos ? ffi::Array<Expr>{cos_emb, sin_emb} : ffi::Array<Expr>{sin_emb, cos_emb};
  Var emb4 = bb->Emit(relax::concat(relax::Tuple(concat_args), ffi::Optional<int64_t>(-1)), "emb");

  Var emb5 = emb4;
  if (embedding_dim % 2 == 1) {
    emb5 = bb->Emit(
        [&]() -> Expr {
          static const Op& pad_op = Op::Get("relax.nn.pad");
          auto attrs = ffi::make_object<tvm::relax::PadAttrs>();
          attrs->pad_width = ffi::Array<Integer>{0, 1, 0, 0};
          attrs->pad_mode = "constant";
          attrs->pad_value = 0.0;
          return Call(pad_op, {emb4}, Attrs(attrs), {});
        }(),
        "emb");
  }

  DataType target_dt = DataType(ffi::StringToDLDataType(out_dtype));
  Expr final_emb = relax::astype(emb5, target_dt);
  return WrapNested(final_emb, std::string(name));
}

// ---------------------------------------------------------------------------
// TIR PrimFunc builder helpers (shared by sampling ops)
// ---------------------------------------------------------------------------

// Convenience: int64 constant
static PrimExpr I64(int64_t v) { return IntImm(DataType::Int(64), v); }

/*!
 * \brief Build a 2-D nested For loop (serial) wrapping an inner Stmt.
 *
 *   for ax0 in range(ext0):
 *     for ax1 in range(ext1):
 *       body
 */
static tir::Stmt NestedFor2D(const tir::Var& ax0, const tir::Var& ax1, const PrimExpr& ext0,
                             const PrimExpr& ext1, tir::Stmt body) {
  tir::Stmt inner = tir::For(ax1, I64(0), ext1, tir::ForKind::kSerial, std::move(body));
  return tir::For(ax0, I64(0), ext0, tir::ForKind::kSerial, std::move(inner));
}

/*!
 * \brief Declare a 2-D buffer backed by a handle Var (for match_buffer semantics).
 *
 * The buffer is contiguous (no explicit strides) and uses the default scope.
 * This mirrors `T.match_buffer(handle, (d0, d1), dtype)` in TVMScript.
 */
static tir::Buffer DeclMatchBuffer2D(const std::string& name, const PrimExpr& d0,
                                     const PrimExpr& d1, const DataType& dtype) {
  return tir::decl_buffer({d0, d1}, dtype, name);
}

// Convenience: int32 constant (for static buffer dimensions like the "1" in (batch, 1))
static PrimExpr I32(int32_t v) { return IntImm(DataType::Int(32), v); }

/*!
 * \brief Declare a 2-D buffer with a static second dimension of 1.
 *
 * Uses int32 `1` for the static dimension to match the TVMScript convention
 * `T.match_buffer(h, (batch, 1), dtype)` where `1` is an int32 literal.
 */
static tir::Buffer DeclMatchBuffer2D1(const std::string& name, const PrimExpr& d0,
                                      const DataType& dtype) {
  return tir::decl_buffer({d0, I32(1)}, dtype, name);
}

// ---------------------------------------------------------------------------
// _get_renorm_cutoff  (used by renormalize_top_p_top_k_prob)
// ---------------------------------------------------------------------------

/*!
 * \brief Build the `_get_renorm_cutoff` TIR PrimFunc.
 *
 * Equivalent Python TVMScript:
 * \code
 *   @T.prim_func(private=True)
 *   def _get_renorm_cutoff(A, B, C, D, E):
 *       batch, vocab_size = T.int64(is_size_var=True), T.int64(is_size_var=True)
 *       sorted_prob    = T.match_buffer(A, (batch, vocab_size), prob_dtype)
 *       cumsum_sorted  = T.match_buffer(B, (batch, vocab_size), prob_dtype)
 *       top_p          = T.match_buffer(C, (batch, 1),          prob_dtype)
 *       top_k          = T.match_buffer(D, (batch, 1),          top_k_dtype)
 *       cutoff         = T.match_buffer(E, (batch, 1),          prob_dtype)
 *       for ax0, ax1 in T.grid(batch, vocab_size):
 *           with T.sblock("T_get_renorm_cutoff"):
 *               v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
 *               if not _cumsum_mask(cumsum_sorted, top_p, top_k, v_ax0, 0):
 *                   cutoff[v_ax0, 0] = sorted_prob[v_ax0, 0]
 *               elif _cumsum_mask(cumsum_sorted, top_p, top_k, v_ax0, v_ax1):
 *                   if v_ax1 + 1 == vocab_size:
 *                       cutoff[v_ax0, 0] = sorted_prob[v_ax0, v_ax1]
 *                   elif not _cumsum_mask(cumsum_sorted, top_p, top_k, v_ax0, v_ax1 + 1):
 *                       cutoff[v_ax0, 0] = sorted_prob[v_ax0, v_ax1 + 1]
 * \endcode
 *
 * where `_cumsum_mask(cs, tp, tk, i, j)` is:
 *   `cs[i,j] < tp[i,0]  AND  j+1 < tk[i,0]`
 *
 * \param prob_dtype   DataType for probability buffers.
 * \param top_k_dtype  DataType for the top_k buffer.
 * \return The constructed private PrimFunc.
 */
static tir::PrimFunc BuildGetRenormCutoff(const DataType& prob_dtype, const DataType& top_k_dtype) {
  // Plain T.int64() vars (no is_size_var), matching the Python reference
  tir::Var batch("batch", DataType::Int(64));
  tir::Var vocab_size("vocab_size", DataType::Int(64));

  // Handle parameters
  tir::Var h_A("A", DataType::Handle());
  tir::Var h_B("B", DataType::Handle());
  tir::Var h_C("C", DataType::Handle());
  tir::Var h_D("D", DataType::Handle());
  tir::Var h_E("E", DataType::Handle());

  // Buffers (match_buffer semantics)
  tir::Buffer sorted_prob = DeclMatchBuffer2D("sorted_prob", batch, vocab_size, prob_dtype);
  tir::Buffer cumsum_sorted = DeclMatchBuffer2D("cumsum_sorted", batch, vocab_size, prob_dtype);
  tir::Buffer top_p = DeclMatchBuffer2D1("top_p", batch, prob_dtype);
  tir::Buffer top_k = DeclMatchBuffer2D1("top_k", batch, top_k_dtype);
  tir::Buffer cutoff = DeclMatchBuffer2D1("cutoff", batch, prob_dtype);

  // Loop variables
  tir::Var ax0("ax0", DataType::Int(64));
  tir::Var ax1("ax1", DataType::Int(64));

  // Block iter vars (T.axis.remap("SS", ...))
  tir::IterVar iv0(Range(I64(0), batch), tir::Var("v_ax0", DataType::Int(64)), tir::kDataPar);
  tir::IterVar iv1(Range(I64(0), vocab_size), tir::Var("v_ax1", DataType::Int(64)), tir::kDataPar);
  const tir::Var& v_ax0 = iv0->var;
  const tir::Var& v_ax1 = iv1->var;

  // Helper: _cumsum_mask(cs, tp, tk, i, j)
  //   cs[i,j] < tp[i,0]  AND  j+1 < tk[i,0]
  // When j is a constant (e.g. 0), Python folds j+1 and writes tk > (j+1),
  // i.e. GT(tk, j+1). When j is a variable, Python writes (j+1) < tk,
  // i.e. LT(j+1, tk). Match this exactly for structural equality.
  auto cumsum_mask = [&](const tir::Var& i, const PrimExpr& j) -> PrimExpr {
    PrimExpr cs_val = tir::BufferLoad(cumsum_sorted, {i, j});
    PrimExpr tp_val = tir::BufferLoad(top_p, {i, I32(0)});
    PrimExpr tk_val = tir::BufferLoad(top_k, {i, I32(0)});
    PrimExpr jp1 = j + I64(1);
    PrimExpr tk_cmp = j->IsInstance<IntImmNode>() ? PrimExpr(tir::GT(tk_val, jp1))
                                                  : PrimExpr(tir::LT(jp1, tk_val));
    return tir::And(tir::LT(cs_val, tp_val), tk_cmp);
  };

  // Build the conditional body:
  //   if not mask(v_ax0, 0):
  //       cutoff[v_ax0, 0] = sorted_prob[v_ax0, 0]
  //   elif mask(v_ax0, v_ax1):
  //       if v_ax1 + 1 == vocab_size:
  //           cutoff[v_ax0, 0] = sorted_prob[v_ax0, v_ax1]
  //       elif not mask(v_ax0, v_ax1 + 1):
  //           cutoff[v_ax0, 0] = sorted_prob[v_ax0, v_ax1 + 1]
  PrimExpr mask_at_0 = cumsum_mask(v_ax0, I32(0));
  PrimExpr mask_at_j = cumsum_mask(v_ax0, v_ax1);
  PrimExpr mask_at_jp1 = cumsum_mask(v_ax0, v_ax1 + I64(1));

  tir::Stmt store_sp_0 =
      tir::BufferStore(cutoff, tir::BufferLoad(sorted_prob, {v_ax0, I32(0)}), {v_ax0, I32(0)});
  tir::Stmt store_sp_j =
      tir::BufferStore(cutoff, tir::BufferLoad(sorted_prob, {v_ax0, v_ax1}), {v_ax0, I32(0)});
  tir::Stmt store_sp_jp1 = tir::BufferStore(
      cutoff, tir::BufferLoad(sorted_prob, {v_ax0, v_ax1 + I64(1)}), {v_ax0, I32(0)});

  // Inner elif branch: if v_ax1+1==vocab_size: store_sp_j  elif not mask_jp1: store_sp_jp1
  tir::Stmt inner_elif =
      tir::IfThenElse(tir::EQ(v_ax1 + I64(1), PrimExpr(vocab_size)), store_sp_j,
                      tir::IfThenElse(tir::EQ(mask_at_jp1, Bool(false)), store_sp_jp1));

  // Top-level if/elif
  tir::Stmt block_body =
      tir::IfThenElse(tir::EQ(mask_at_0, Bool(false)), store_sp_0,
                      tir::IfThenElse(tir::EQ(mask_at_j, Bool(true)), inner_elif));

  // Annotate the inner block so ScriptComplete infers reads/writes,
  // matching the TVMScript-generated IR.
  ffi::Map<ffi::String, ffi::Any> inner_annots;
  inner_annots.Set("tir.script_parsing_detect_access", IntImm(DataType::Int(32), 3));
  tir::Stmt sblock = tir::SBlockRealize(
      /*iter_values=*/{PrimExpr(ax0), PrimExpr(ax1)},
      /*predicate=*/tir::const_true(),
      tir::SBlock(
          /*iter_vars=*/{iv0, iv1},
          /*reads=*/{},
          /*writes=*/{},
          /*name_hint=*/"T_get_renorm_prob",
          /*body=*/block_body,
          /*init=*/std::nullopt,
          /*alloc_buffers=*/{},
          /*match_buffers=*/{},
          /*annotations=*/inner_annots));

  tir::Stmt loops = NestedFor2D(ax0, ax1, batch, vocab_size, sblock);

  // Wrap the loop nest in a root SBlock, matching the TVMScript-generated IR
  // (TVMScript automatically adds `with T.sblock("root")` around the body).
  tir::Stmt root_block = tir::SBlockRealize(
      /*iter_values=*/{},
      /*predicate=*/tir::const_true(),
      tir::SBlock(
          /*iter_vars=*/{},
          /*reads=*/{},
          /*writes=*/{},
          /*name_hint=*/"root",
          /*body=*/loops));

  // Buffer map: handle -> buffer
  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_A, sorted_prob);
  buf_map.Set(h_B, cumsum_sorted);
  buf_map.Set(h_C, top_p);
  buf_map.Set(h_D, top_k);
  buf_map.Set(h_E, cutoff);

  // Run ScriptComplete to infer reads/writes from the detect_access annotation.
  return tir::ScriptComplete(
      tir::PrimFunc({h_A, h_B, h_C, h_D, h_E}, root_block, VoidType(), buf_map), {});
}

// ---------------------------------------------------------------------------
// _get_renorm_prob  (used by sample_top_p_top_k_from_sorted_prob)
// ---------------------------------------------------------------------------

/*!
 * \brief Build the `_get_renorm_prob` TIR PrimFunc.
 *
 * Equivalent Python TVMScript:
 * \code
 *   @T.prim_func(private=True)
 *   def _get_renorm_prob(A, B, C, D):
 *       batch, vocab_size = T.int64(is_size_var=True), T.int64(is_size_var=True)
 *       cumsum_sorted = T.match_buffer(A, (batch, vocab_size), prob_dtype)
 *       top_p         = T.match_buffer(B, (batch, 1),          prob_dtype)
 *       top_k         = T.match_buffer(C, (batch, 1),          index_dtype)
 *       renorm_prob   = T.match_buffer(D, (batch, 1),          prob_dtype)
 *       for ax0, ax1 in T.grid(batch, vocab_size):
 *           with T.sblock("T_get_renorm_prob"):
 *               v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
 *               if not mask(v_ax0, 0):
 *                   renorm_prob[v_ax0, 0] = cumsum_sorted[v_ax0, 0]
 *               elif mask(v_ax0, v_ax1):
 *                   if v_ax1 + 1 == vocab_size:
 *                       renorm_prob[v_ax0, 0] = cumsum_sorted[v_ax0, v_ax1]
 *                   elif not mask(v_ax0, v_ax1 + 1):
 *                       renorm_prob[v_ax0, 0] = cumsum_sorted[v_ax0, v_ax1 + 1]
 * \endcode
 *
 * \param prob_dtype   DataType for probability buffers.
 * \param index_dtype  DataType for the top_k buffer.
 * \return The constructed private PrimFunc.
 */
static tir::PrimFunc BuildGetRenormProb(const DataType& prob_dtype, const DataType& index_dtype) {
  tir::SizeVar batch("batch", DataType::Int(64));
  tir::SizeVar vocab_size("vocab_size", DataType::Int(64));

  tir::Var h_A("A", DataType::Handle());
  tir::Var h_B("B", DataType::Handle());
  tir::Var h_C("C", DataType::Handle());
  tir::Var h_D("D", DataType::Handle());

  tir::Buffer cumsum_sorted = DeclMatchBuffer2D("cumsum_sorted", batch, vocab_size, prob_dtype);
  tir::Buffer top_p = DeclMatchBuffer2D1("top_p", batch, prob_dtype);
  tir::Buffer top_k = DeclMatchBuffer2D1("top_k", batch, index_dtype);
  tir::Buffer renorm_prob = DeclMatchBuffer2D1("renorm_prob", batch, prob_dtype);

  tir::Var ax0("ax0", DataType::Int(64));
  tir::Var ax1("ax1", DataType::Int(64));

  tir::IterVar iv0(Range(I64(0), batch), tir::Var("v_ax0", DataType::Int(64)), tir::kDataPar);
  tir::IterVar iv1(Range(I64(0), vocab_size), tir::Var("v_ax1", DataType::Int(64)), tir::kDataPar);
  const tir::Var& v_ax0 = iv0->var;
  const tir::Var& v_ax1 = iv1->var;

  // _cumsum_mask: cs[i,j] < tp[i,0]  AND  j+1 < tk[i,0]
  auto cumsum_mask = [&](const tir::Var& i, const PrimExpr& j) -> PrimExpr {
    PrimExpr cs_val = tir::BufferLoad(cumsum_sorted, {i, j});
    PrimExpr tp_val = tir::BufferLoad(top_p, {i, I32(0)});
    PrimExpr tk_val = tir::BufferLoad(top_k, {i, I32(0)});
    PrimExpr jp1 = j + I64(1);
    PrimExpr tk_cmp = j->IsInstance<IntImmNode>() ? PrimExpr(tir::GT(tk_val, jp1))
                                                  : PrimExpr(tir::LT(jp1, tk_val));
    return tir::And(tir::LT(cs_val, tp_val), tk_cmp);
  };

  PrimExpr mask_at_0 = cumsum_mask(v_ax0, I32(0));
  PrimExpr mask_at_j = cumsum_mask(v_ax0, v_ax1);
  PrimExpr mask_at_jp1 = cumsum_mask(v_ax0, v_ax1 + I64(1));

  tir::Stmt store_cs_0 = tir::BufferStore(
      renorm_prob, tir::BufferLoad(cumsum_sorted, {v_ax0, I32(0)}), {v_ax0, I32(0)});
  tir::Stmt store_cs_j = tir::BufferStore(
      renorm_prob, tir::BufferLoad(cumsum_sorted, {v_ax0, v_ax1}), {v_ax0, I32(0)});
  tir::Stmt store_cs_jp1 = tir::BufferStore(
      renorm_prob, tir::BufferLoad(cumsum_sorted, {v_ax0, v_ax1 + I64(1)}), {v_ax0, I32(0)});

  tir::Stmt inner_elif = tir::IfThenElse(tir::EQ(v_ax1 + I64(1), PrimExpr(vocab_size)), store_cs_j,
                                         tir::IfThenElse(tir::Not(mask_at_jp1), store_cs_jp1));

  tir::Stmt block_body =
      tir::IfThenElse(tir::Not(mask_at_0), store_cs_0, tir::IfThenElse(mask_at_j, inner_elif));

  // Annotate the inner block so ScriptComplete infers reads/writes.
  ffi::Map<ffi::String, ffi::Any> inner_annots;
  inner_annots.Set("tir.script_parsing_detect_access", IntImm(DataType::Int(32), 3));
  tir::Stmt sblock =
      tir::SBlockRealize({PrimExpr(ax0), PrimExpr(ax1)}, tir::const_true(),
                         tir::SBlock({iv0, iv1}, {}, {}, "T_get_renorm_prob", block_body,
                                     std::nullopt, {}, {}, inner_annots));

  tir::Stmt loops = NestedFor2D(ax0, ax1, batch, vocab_size, sblock);

  // Wrap in a root SBlock to match TVMScript-generated IR.
  tir::Stmt root_block =
      tir::SBlockRealize({}, tir::const_true(), tir::SBlock({}, {}, {}, "root", loops));

  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_A, cumsum_sorted);
  buf_map.Set(h_B, top_p);
  buf_map.Set(h_C, top_k);
  buf_map.Set(h_D, renorm_prob);

  return tir::ScriptComplete(tir::PrimFunc({h_A, h_B, h_C, h_D}, root_block, VoidType(), buf_map),
                             {});
}

// ---------------------------------------------------------------------------
// _get_index_from_sorted  (used by sample_top_p_top_k_from_sorted_prob)
// ---------------------------------------------------------------------------

/*!
 * \brief Build the `_get_index_from_sorted` TIR PrimFunc.
 *
 * Equivalent Python TVMScript:
 * \code
 *   @T.prim_func(private=True)
 *   def _get_index_from_sorted(A, B, C, D, E, F):
 *       batch, vocab_size = T.int64(is_size_var=True), T.int64(is_size_var=True)
 *       out_batch         = T.int64(is_size_var=True)
 *       cumsum_sorted  = T.match_buffer(A, (batch, vocab_size),  prob_dtype)
 *       indices        = T.match_buffer(B, (batch, vocab_size),  index_dtype)
 *       renorm_prob    = T.match_buffer(C, (batch, 1),           prob_dtype)
 *       usample        = T.match_buffer(D, (out_batch, 1),       prob_dtype)
 *       sample_indices = T.match_buffer(E, (out_batch, 1),       sample_indices_dtype)
 *       output_index   = T.match_buffer(F, (out_batch, 1),       index_dtype)
 *       for ax0, ax1 in T.grid(out_batch, vocab_size):
 *           with T.sblock("T_get_index_from_sorted"):
 *               v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
 *               T.writes(output_index[v_ax0, 0])
 *               si = sample_indices[v_ax0, 0]
 *               cond_enter = (usample[v_ax0,0] < cumsum_sorted[si,v_ax1]
 *                                               / renorm_prob[si,0])
 *                            OR (v_ax1 + 1 == vocab_size)
 *               if cond_enter:
 *                   if v_ax1 == 0:
 *                       output_index[v_ax0, 0] = indices[si, 0]
 *                   elif usample[v_ax0,0] >= cumsum_sorted[si,v_ax1-1]
 *                                             / renorm_prob[si,0]:
 *                       output_index[v_ax0, 0] = indices[si, v_ax1]
 * \endcode
 *
 * \param prob_dtype            DataType for probability buffers.
 * \param index_dtype           DataType for index buffers.
 * \param sample_indices_dtype  DataType for the sample_indices buffer.
 * \return The constructed private PrimFunc.
 */
static tir::PrimFunc BuildGetIndexFromSorted(const DataType& prob_dtype,
                                             const DataType& index_dtype,
                                             const DataType& sample_indices_dtype) {
  tir::SizeVar batch("batch", DataType::Int(64));
  tir::SizeVar vocab_size("vocab_size", DataType::Int(64));
  tir::SizeVar out_batch("out_batch", DataType::Int(64));

  tir::Var h_A("A", DataType::Handle());
  tir::Var h_B("B", DataType::Handle());
  tir::Var h_C("C", DataType::Handle());
  tir::Var h_D("D", DataType::Handle());
  tir::Var h_E("E", DataType::Handle());
  tir::Var h_F("F", DataType::Handle());

  tir::Buffer cumsum_sorted = DeclMatchBuffer2D("cumsum_sorted", batch, vocab_size, prob_dtype);
  tir::Buffer indices = DeclMatchBuffer2D("indices", batch, vocab_size, index_dtype);
  tir::Buffer renorm_prob = DeclMatchBuffer2D1("renorm_prob", batch, prob_dtype);
  tir::Buffer usample = DeclMatchBuffer2D1("usample", out_batch, prob_dtype);
  tir::Buffer sample_indices =
      DeclMatchBuffer2D1("sample_indices", out_batch, sample_indices_dtype);
  tir::Buffer output_index = DeclMatchBuffer2D1("output_index", out_batch, index_dtype);

  tir::Var ax0("ax0", DataType::Int(64));
  tir::Var ax1("ax1", DataType::Int(64));

  tir::IterVar iv0(Range(I64(0), out_batch), tir::Var("v_ax0", DataType::Int(64)), tir::kDataPar);
  tir::IterVar iv1(Range(I64(0), vocab_size), tir::Var("v_ax1", DataType::Int(64)), tir::kDataPar);
  const tir::Var& v_ax0 = iv0->var;
  const tir::Var& v_ax1 = iv1->var;

  // si = sample_indices[v_ax0, 0]  (cast to int64 for buffer indexing)
  // Use I64(0) for out_batch-indexed buffers (usample, sample_indices) to match
  // TVMScript which promotes the column index to int64 when the row is int64.
  PrimExpr si = tvm::cast(DataType::Int(64), tir::BufferLoad(sample_indices, {v_ax0, I64(0)}));

  // cumsum_sorted[si, v_ax1] / renorm_prob[si, 0]
  PrimExpr cs_j = tir::BufferLoad(cumsum_sorted, {si, v_ax1});
  PrimExpr rp = tir::BufferLoad(renorm_prob, {si, I32(0)});
  PrimExpr ratio_j = cs_j / rp;

  // cumsum_sorted[si, v_ax1 - 1] / renorm_prob[si, 0]
  PrimExpr cs_jm1 = tir::BufferLoad(cumsum_sorted, {si, v_ax1 - I64(1)});
  PrimExpr ratio_jm1 = cs_jm1 / rp;

  PrimExpr u_val = tir::BufferLoad(usample, {v_ax0, I64(0)});

  // cond_enter: u < ratio_j  OR  v_ax1+1 == vocab_size
  PrimExpr cond_enter =
      tir::Or(tir::LT(u_val, ratio_j), tir::EQ(v_ax1 + I64(1), PrimExpr(vocab_size)));

  // Inner body when cond_enter is true:
  //   if v_ax1 == 0: output_index[v_ax0,0] = indices[si, 0]
  //   elif u >= ratio_jm1: output_index[v_ax0,0] = indices[si, v_ax1]
  tir::Stmt store_idx_0 =
      tir::BufferStore(output_index, tir::BufferLoad(indices, {si, I32(0)}), {v_ax0, I32(0)});
  tir::Stmt store_idx_j =
      tir::BufferStore(output_index, tir::BufferLoad(indices, {si, v_ax1}), {v_ax0, I32(0)});

  tir::Stmt inner_body = tir::IfThenElse(tir::EQ(v_ax1, I64(0)), store_idx_0,
                                         tir::IfThenElse(tir::GE(u_val, ratio_jm1), store_idx_j));

  tir::Stmt block_body = tir::IfThenElse(cond_enter, inner_body);

  // Annotate the inner block so ScriptComplete infers reads/writes.
  ffi::Map<ffi::String, ffi::Any> inner_annots;
  inner_annots.Set("tir.script_parsing_detect_access", IntImm(DataType::Int(32), 3));
  tir::Stmt sblock =
      tir::SBlockRealize({PrimExpr(ax0), PrimExpr(ax1)}, tir::const_true(),
                         tir::SBlock({iv0, iv1}, {}, {}, "T_get_index_from_sorted", block_body,
                                     std::nullopt, {}, {}, inner_annots));

  tir::Stmt loops = NestedFor2D(ax0, ax1, out_batch, vocab_size, sblock);

  // Wrap in a root SBlock to match TVMScript-generated IR.
  tir::Stmt root_block =
      tir::SBlockRealize({}, tir::const_true(), tir::SBlock({}, {}, {}, "root", loops));

  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_A, cumsum_sorted);
  buf_map.Set(h_B, indices);
  buf_map.Set(h_C, renorm_prob);
  buf_map.Set(h_D, usample);
  buf_map.Set(h_E, sample_indices);
  buf_map.Set(h_F, output_index);

  return tir::ScriptComplete(
      tir::PrimFunc({h_A, h_B, h_C, h_D, h_E, h_F}, root_block, VoidType(), buf_map), {});
}

// ---------------------------------------------------------------------------
// Helper: emit a tensor_ir_op call directly into the current BlockBuilder
// ---------------------------------------------------------------------------

/*!
 * \brief Emit a call_tir node for a pre-built PrimFunc and return the bound Var.
 *
 * This is the C++ equivalent of the Python `tensor_ir_op(func, name, args, out)`
 * helper, but operates directly on Vars (already emitted) rather than nn.Tensors.
 *
 * \param bb         The current BlockBuilder.
 * \param func       The TIR PrimFunc to call.
 * \param name_hint  Name hint for the GlobalVar and the emitted binding.
 * \param args       Input tensor Vars.
 * \param out_sinfo  StructInfo describing the output tensor(s).
 * \return The emitted output Var.
 */
static Var EmitTensorIrOp(BlockBuilder& bb, const tir::PrimFunc& func, const std::string& name_hint,
                          const ffi::Array<Expr>& args, const StructInfo& out_sinfo) {
  GlobalVar gv = bb->AddFunction(func, name_hint);
  static const Op& call_tir_op = Op::Get("relax.call_tir");
  Expr call = Call(call_tir_op, {gv, relax::Tuple(args)}, tvm::Attrs(), {out_sinfo});
  return bb->Emit(call, name_hint);
}

// ---------------------------------------------------------------------------
// renormalize_top_p_top_k_prob
// ---------------------------------------------------------------------------

/*!
 * \brief Renormalizes probabilities after filtering with top_p and top_k.
 *
 * C++ port of the Python `renormalize_top_p_top_k_prob` function.
 *
 * Steps:
 *   1. cumsum_sorted = cumsum(sorted_prob, axis=1)
 *   2. renorm_cutoff = tensor_ir_op(_get_renorm_cutoff, ...)
 *   3. filtered_prob = where(prob >= renorm_cutoff, prob, 0)
 *   4. renorm_prob   = filtered_prob / sum(filtered_prob, axis=1, keepdims=True)
 *
 * \param prob        2-D tensor (batch, vocab_size) of probabilities.
 * \param sorted_prob 2-D tensor (batch, vocab_size) sorted in descending order.
 * \param top_p       2-D tensor (batch, 1) cumulative probability threshold.
 * \param top_k       2-D tensor (batch, 1) top-k count.
 * \return Filtered and renormalized probability tensor, same shape as prob.
 */
static ffi::Any NNRenormalizeTopPTopKProb(Var prob, Var sorted_prob, Var top_p, Var top_k) {
  BlockBuilder bb = BlockBuilder_Current();
  TVM_FFI_ICHECK(bb.defined())
      << "renormalize_top_p_top_k_prob called outside of a BlockBuilder scope";

  // Extract dtypes from struct_info
  const auto* prob_sinfo = prob->struct_info_.as<TensorStructInfoNode>();
  const auto* top_k_sinfo = top_k->struct_info_.as<TensorStructInfoNode>();
  TVM_FFI_ICHECK(prob_sinfo) << "renormalize_top_p_top_k_prob: prob must have TensorStructInfo";
  TVM_FFI_ICHECK(top_k_sinfo) << "renormalize_top_p_top_k_prob: top_k must have TensorStructInfo";

  DataType prob_dtype = prob_sinfo->dtype;
  DataType top_k_dtype = top_k_sinfo->dtype;

  // Extract batch dimension from sorted_prob shape
  const auto* sp_sinfo = sorted_prob->struct_info_.as<TensorStructInfoNode>();
  TVM_FFI_ICHECK(sp_sinfo && sp_sinfo->shape.defined())
      << "renormalize_top_p_top_k_prob: sorted_prob must have a known shape";
  const auto* sp_shape = sp_sinfo->shape.as<ShapeExprNode>();
  TVM_FFI_ICHECK(sp_shape && sp_shape->values.size() == 2)
      << "renormalize_top_p_top_k_prob: sorted_prob must be 2-D";
  PrimExpr batch = sp_shape->values[0];
  PrimExpr vocab_size = sp_shape->values[1];

  // Step 1: cumsum_sorted = cumsum(sorted_prob, axis=1)
  Var cumsum_sorted = bb->Emit(relax::cumsum(sorted_prob, /*axis=*/ffi::Optional<int64_t>(1),
                                             /*dtype=*/std::nullopt, /*exclusive=*/Bool(false)),
                               "cumsum");

  // Step 2: renorm_cutoff = tensor_ir_op(_get_renorm_cutoff, ...)
  tir::PrimFunc get_renorm_cutoff_func = BuildGetRenormCutoff(prob_dtype, top_k_dtype);
  StructInfo cutoff_sinfo = TensorStructInfo(ShapeExpr({batch, I64(1)}), prob_dtype);
  Var renorm_cutoff = EmitTensorIrOp(bb, get_renorm_cutoff_func, "get_renorm_cutoff",
                                     {sorted_prob, cumsum_sorted, top_p, top_k}, cutoff_sinfo);

  // Step 3: filtered_prob = tensor_expr_op(filter_with_top_p_top_k, [prob, renorm_cutoff])
  // Uses TE compute with T.Select(cutoff[i,0] <= prob[i,j], prob[i,j], 0.0),
  // matching the Python reference which uses tensor_expr_op.
  ffi::Map<tir::Var, PrimExpr> empty_tir_map;
  ffi::Array<te::Tensor> te_filter_inputs = {
      TETensor(prob, empty_tir_map, "input_0"),
      TETensor(renorm_cutoff, empty_tir_map, "input_1"),
  };
  te::Tensor te_prob = te_filter_inputs[0];
  te::Tensor te_cutoff = te_filter_inputs[1];
  te::Tensor te_filtered = te::compute(
      te_prob->shape,
      [&](const ffi::Array<tir::Var>& indices) -> PrimExpr {
        PrimExpr prob_val = te_prob(ffi::Array<PrimExpr>(indices.begin(), indices.end()));
        PrimExpr cutoff_val =
            te_cutoff(ffi::Array<PrimExpr>{PrimExpr(indices[0]), IntImm(DataType::Int(64), 0)});
        return tir::Select(tir::LE(cutoff_val, prob_val), prob_val,
                           tir::make_const(prob_dtype, 0.0));
      },
      "filter_with_top_p_top_k");
  ffi::Array<te::Tensor> all_filter_tensors;
  for (const te::Tensor& t : te_filter_inputs) all_filter_tensors.push_back(t);
  all_filter_tensors.push_back(te_filtered);
  tir::PrimFunc filter_func = tir::CreatePrimFunc(all_filter_tensors, DataType::Int(64));
  filter_func = WithoutAttr(filter_func, "global_symbol");
  GlobalVar filter_gv = bb->AddFunction(filter_func, "filter_with_top_p_top_k");
  static const Op& call_tir_op = Op::Get("relax.call_tir");
  StructInfo filter_sinfo = TensorStructInfo(ShapeExpr({batch, vocab_size}), prob_dtype);
  Var filtered_prob =
      bb->Emit(Call(call_tir_op, {filter_gv, relax::Tuple(ffi::Array<Expr>{prob, renorm_cutoff})},
                    tvm::Attrs(), {filter_sinfo}),
               "filter_with_top_p_top_k");

  // Step 4: renorm_prob = filtered_prob / sum(filtered_prob, axis=1, keepdims=True)
  Var sum_filtered = bb->Emit(
      relax::sum(filtered_prob, ffi::Array<Integer>{Integer(1)}, /*keepdims=*/true), "sum");
  Expr renorm_prob_expr = relax::divide(filtered_prob, sum_filtered);
  return WrapNested(renorm_prob_expr, std::string("renorm_prob"));
}

// ---------------------------------------------------------------------------
// sample_top_p_top_k_from_sorted_prob
// ---------------------------------------------------------------------------

/*!
 * \brief Samples indices from a sorted probability tensor using top_p / top_k.
 *
 * C++ port of the Python `sample_top_p_top_k_from_sorted_prob` function.
 *
 * Steps:
 *   1. cumsum_sorted    = cumsum(sorted_prob, axis=1)
 *   2. renorm_prob      = tensor_ir_op(_get_renorm_prob, ...)
 *   3. out_index        = tensor_ir_op(_get_index_from_sorted, ...)
 *
 * \param sorted_prob     2-D (batch, vocab_size) probabilities sorted descending.
 * \param sorted_index    2-D (batch, vocab_size) argsort indices.
 * \param top_p           2-D (batch, 1) cumulative probability threshold.
 * \param top_k           2-D (batch, 1) top-k count.
 * \param uniform_sample  2-D (n, 1) uniform samples.
 * \param sample_indices  Optional 2-D (n, 1) distribution selector; if null,
 *                        defaults to arange(n) reshaped to (n, 1).
 * \return 2-D (n, 1) selected token indices.
 */
static ffi::Any NNSampleTopPTopKFromSortedProb(Var sorted_prob, Var sorted_index, Var top_p,
                                               Var top_k, Var uniform_sample,
                                               ffi::Optional<Var> sample_indices_opt) {
  BlockBuilder bb = BlockBuilder_Current();
  TVM_FFI_ICHECK(bb.defined())
      << "sample_top_p_top_k_from_sorted_prob called outside of a BlockBuilder scope";

  // Extract dtypes
  const auto* sp_sinfo = sorted_prob->struct_info_.as<TensorStructInfoNode>();
  const auto* si_sinfo = sorted_index->struct_info_.as<TensorStructInfoNode>();
  const auto* us_sinfo = uniform_sample->struct_info_.as<TensorStructInfoNode>();
  TVM_FFI_ICHECK(sp_sinfo) << "sample_top_p_top_k_from_sorted_prob: sorted_prob needs sinfo";
  TVM_FFI_ICHECK(si_sinfo) << "sample_top_p_top_k_from_sorted_prob: sorted_index needs sinfo";
  TVM_FFI_ICHECK(us_sinfo) << "sample_top_p_top_k_from_sorted_prob: uniform_sample needs sinfo";

  DataType prob_dtype = sp_sinfo->dtype;
  DataType index_dtype = si_sinfo->dtype;

  // Extract shape dimensions
  TVM_FFI_ICHECK(sp_sinfo->shape.defined()) << "sorted_prob must have a known shape";
  const auto* sp_shape = sp_sinfo->shape.as<ShapeExprNode>();
  TVM_FFI_ICHECK(sp_shape && sp_shape->values.size() == 2) << "sorted_prob must be 2-D";
  PrimExpr prob_batch = sp_shape->values[0];
  PrimExpr vocab_size = sp_shape->values[1];

  TVM_FFI_ICHECK(us_sinfo->shape.defined()) << "uniform_sample must have a known shape";
  const auto* us_shape = us_sinfo->shape.as<ShapeExprNode>();
  TVM_FFI_ICHECK(us_shape && us_shape->values.size() == 2) << "uniform_sample must be 2-D";
  PrimExpr out_batch = us_shape->values[0];

  // Resolve sample_indices: if not provided, build arange(out_batch) reshaped to (out_batch, 1)
  Var sample_indices;
  DataType sample_indices_dtype;
  if (sample_indices_opt.defined()) {
    sample_indices = sample_indices_opt.value();
    const auto* samp_sinfo = sample_indices->struct_info_.as<TensorStructInfoNode>();
    TVM_FFI_ICHECK(samp_sinfo) << "sample_indices must have TensorStructInfo";
    sample_indices_dtype = samp_sinfo->dtype;
  } else {
    // Default: arange(out_batch, dtype=int64) reshaped to (out_batch, 1)
    sample_indices_dtype = DataType::Int(64);
    Var arange_var = bb->Emit(relax::arange(PrimValue(I64(0)), PrimValue(out_batch),
                                            PrimValue(I64(1)), sample_indices_dtype),
                              "sample_indices_arange");
    sample_indices =
        bb->Emit(relax::reshape(arange_var, ShapeExpr({out_batch, I64(1)})), "sample_indices");
  }

  // Step 1: cumsum_sorted = cumsum(sorted_prob, axis=1)
  Var cumsum_sorted = bb->Emit(
      relax::cumsum(sorted_prob, ffi::Optional<int64_t>(1), std::nullopt, Bool(false)), "cumsum");

  // Step 2: renorm_prob = tensor_ir_op(_get_renorm_prob, ...)
  tir::PrimFunc get_renorm_prob_func = BuildGetRenormProb(prob_dtype, index_dtype);
  StructInfo renorm_sinfo = TensorStructInfo(ShapeExpr({prob_batch, I64(1)}), prob_dtype);
  Var renorm_prob = EmitTensorIrOp(bb, get_renorm_prob_func, "get_renorm_prob",
                                   {cumsum_sorted, top_p, top_k}, renorm_sinfo);

  // Step 3: out_index = tensor_ir_op(_get_index_from_sorted, ...)
  tir::PrimFunc get_index_func =
      BuildGetIndexFromSorted(prob_dtype, index_dtype, sample_indices_dtype);
  StructInfo out_index_sinfo = TensorStructInfo(ShapeExpr({out_batch, I64(1)}), index_dtype);
  Var out_index = EmitTensorIrOp(
      bb, get_index_func, "get_index_from_sorted",
      {cumsum_sorted, sorted_index, renorm_prob, uniform_sample, sample_indices}, out_index_sinfo);

  return ffi::Any(out_index);
}

// ---------------------------------------------------------------------------
// interpolate / resize2d
// ---------------------------------------------------------------------------

static ffi::Any NNResize2d(Var x, ffi::Array<Integer> size, ffi::String layout, ffi::String method,
                           ffi::String coord_trans, ffi::String name) {
  ffi::Array<PrimExpr> size_prim;
  for (const Integer& s : size) size_prim.push_back(s);
  ffi::Array<FloatImm> roi = {
      FloatImm(DataType::Float(32), 0.0), FloatImm(DataType::Float(32), 0.0),
      FloatImm(DataType::Float(32), 0.0), FloatImm(DataType::Float(32), 0.0)};
  return WrapNested(relax::resize2d(x, ShapeExpr(size_prim), roi, layout, method, coord_trans,
                                    "round", -0.75, 0, 0.0, std::nullopt),
                    std::string(name));
}

static ffi::Any NNInterpolate(Var x, ffi::Array<Integer> size, ffi::String data_layout,
                              ffi::String method, ffi::String coord_trans, ffi::String name) {
  return NNResize2d(x, size, data_layout, method, coord_trans, name);
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
      .def("relax.frontend.nn.op.renormalize_top_p_top_k_prob", NNRenormalizeTopPTopKProb)
      .def("relax.frontend.nn.op.sample_top_p_top_k_from_sorted_prob",
           NNSampleTopPTopKFromSortedProb)
      // Image
      .def("relax.frontend.nn.op.resize2d", NNResize2d)
      // Clip
      .def("relax.frontend.nn.op.clip", NNClip)
      // TE / TIR ops
      .def("relax.frontend.nn.op.tensor_expr_op", NNTensorExprOp)
      .def("relax.frontend.nn.op.tensor_ir_op", NNTensorIrOp)
      .def("relax.frontend.nn.op.tensor_ir_inplace_op", NNTensorIrInplaceOp)
      .def("relax.frontend.nn.op.extern", NNExtern)
      .def("relax.frontend.nn.op.debug_func", NNDebugFunc)
      // Timestep embedding
      .def("relax.frontend.nn.op.get_timestep_embedding", NNGetTimestepEmbedding)
      // Interpolate
      .def("relax.frontend.nn.op.interpolate", NNInterpolate);
}

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
