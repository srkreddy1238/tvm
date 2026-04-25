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
 * \file src/relax/frontend/nn/modules.cc
 * \brief Implementations for built-in nn module types.
 */

#include "modules.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/op.h>

#include <string>

#include "../../op/tensor/create.h"
#include "../../op/tensor/manipulate.h"
#include "core.h"
#include "op.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Build a PrimExpr dimension from a mixed ffi::Any element.
// Accepts: int64, ffi::String (symbolic var name), tir::Var, or PrimExpr.
static PrimExpr AnyToDim(const ffi::Any& v) {
  if (auto opt = v.try_cast<int64_t>()) return IntImm(DataType::Int(64), opt.value());
  if (auto opt = v.try_cast<ffi::String>()) return tir::Var(opt.value(), DataType::Int(64));
  if (auto opt = v.try_cast<tir::Var>()) {
    TVM_FFI_ICHECK(opt.value()->dtype == DataType::Int(64))
        << "Symbolic shape var must have dtype int64";
    return opt.value();
  }
  if (auto opt = v.try_cast<PrimExpr>()) {
    TVM_FFI_ICHECK(opt.value()->dtype == DataType::Int(64))
        << "PrimExpr shape must have dtype int64";
    return opt.value();
  }
  TVM_FFI_THROW(TypeError) << "Expected int64, string, tir::Var, or PrimExpr, got "
                           << v.GetTypeKey();
  TVM_FFI_UNREACHABLE();
}

// ---------------------------------------------------------------------------
// Internal helper: populate NNModuleNode::attrs with NNParameter fields.
// Called by every Make* factory after constructing the node so that
// NNModuleNode::NamedParameters() can discover parameters.
// ---------------------------------------------------------------------------
static void PopulateAttrs(const runtime::ObjectRef& mod_ref, const ffi::String& name,
                          const NNParameter& param) {
  const_cast<NNModuleNode*>(mod_ref.as<NNModuleNode>())->attrs.Set(name, ffi::Any(param));
}
static void PopulateAttrs(const runtime::ObjectRef& mod_ref, const ffi::String& name,
                          const ffi::Optional<NNParameter>& param) {
  if (param.has_value())
    const_cast<NNModuleNode*>(mod_ref.as<NNModuleNode>())->attrs.Set(name, ffi::Any(param.value()));
}

NNParameter MakeParam(ffi::Array<ffi::Any> shape, ffi::String dtype) {
  ffi::Array<PrimExpr> dims;
  for (const auto& s : shape) dims.push_back(AnyToDim(s));
  Var v("param", TensorStructInfo(ShapeExpr(dims), DataType(ffi::StringToDLDataType(dtype))));
  return NNParameter(v);
}

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------
ReLUModule::ReLUModule() { data_ = ffi::make_object<ReLUModuleNode>(); }
Var ReLUModuleNode::Forward(Var x) const { return NNRelu(x); }

// ---------------------------------------------------------------------------
// SiLU
// ---------------------------------------------------------------------------
SiLUModule::SiLUModule() { data_ = ffi::make_object<SiLUModuleNode>(); }
Var SiLUModuleNode::Forward(Var x) const { return NNSilu(x); }

// ---------------------------------------------------------------------------
// GELU
// ---------------------------------------------------------------------------
GELUModule::GELUModule(ffi::String approximate) {
  data_ = ffi::make_object<GELUModuleNode>(std::move(approximate));
}
Var GELUModuleNode::Forward(Var x) const {
  ffi::Optional<ffi::String> approx =
      approximate.empty() ? std::nullopt : ffi::Optional<ffi::String>(approximate);
  return NNGelu(x, approx);
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------
LinearModule::LinearModule(NNParameter weight, ffi::Optional<NNParameter> bias,
                           ffi::Optional<ffi::String> out_dtype) {
  data_ =
      ffi::make_object<LinearModuleNode>(std::move(weight), std::move(bias), std::move(out_dtype));
}

Var LinearModuleNode::Forward(Var x) const {
  // permute_dims transposes the weight; matmul computes x @ w^T.
  // With bias the add result uses hint "linear".
  Var w = NNPermuteDims(weight->expr, std::nullopt);
  Var mm = NNMatmul(x, w, out_dtype, "matmul");
  if (bias.has_value()) {
    return NNAdd(mm, bias.value()->expr, "linear");
  }
  return mm;
}

LinearModule MakeLinear(ffi::Any in_features, ffi::Any out_features, bool has_bias,
                        ffi::Optional<ffi::String> dtype, ffi::Optional<ffi::String> out_dtype) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  ffi::String bias_dt = out_dtype.has_value() ? out_dtype.value() : dt;
  NNParameter w = MakeParam({out_features, in_features}, dt);
  ffi::Optional<NNParameter> b =
      has_bias ? ffi::Optional<NNParameter>(MakeParam({out_features}, bias_dt)) : std::nullopt;
  LinearModule mod(w, b, out_dtype);
  PopulateAttrs(mod, "weight", w);
  PopulateAttrs(mod, "bias", b);
  return mod;
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------
EmbeddingModule::EmbeddingModule(NNParameter weight) {
  data_ = ffi::make_object<EmbeddingModuleNode>(std::move(weight));
}

Var EmbeddingModuleNode::Forward(Var x, ffi::Array<ffi::Any> out_shape_if_nd) const {
  if (out_shape_if_nd.empty()) {
    return NNTake(weight->expr, x, ffi::Optional<Integer>(Integer(0)), "embedding");
  }
  // ND path: flatten -> take -> reshape to out_shape.
  Var flat = NNReshape(x, {ffi::Any(int64_t(-1))}, "reshape");
  Var taken = NNTake(weight->expr, flat, ffi::Optional<Integer>(Integer(0)), "take");
  return NNReshape(taken, out_shape_if_nd, "embedding");
}

EmbeddingModule MakeEmbedding(ffi::Any num, ffi::Any dim, ffi::Optional<ffi::String> dtype) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  NNParameter w = MakeParam({num, dim}, dt);
  EmbeddingModule mod(w);
  PopulateAttrs(mod, "weight", w);
  return mod;
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------
LayerNormModule::LayerNormModule(ffi::Optional<NNParameter> weight, ffi::Optional<NNParameter> bias,
                                 ffi::Array<Integer> axes, double epsilon,
                                 bool elementwise_affine) {
  data_ = ffi::make_object<LayerNormModuleNode>(std::move(weight), std::move(bias), std::move(axes),
                                                epsilon, elementwise_affine);
}

Var LayerNormModuleNode::Forward(Var x) const {
  TVM_FFI_ICHECK(weight.defined() && bias.defined());
  return NNLayerNorm(x, axes, weight.value()->expr, bias.value()->expr, epsilon);
}

LayerNormModule MakeLayerNorm(ffi::Any normalized_shape, double eps, bool elementwise_affine,
                              ffi::Optional<ffi::String> dtype) {
  ffi::String dt = dtype.value_or(ffi::String(elementwise_affine ? GetDefaultDtype() : "float32"));
  int64_t dim_num = 1;
  ffi::Array<ffi::Any> shape_arr;
  if (auto opt = normalized_shape.try_cast<int64_t>()) {
    shape_arr.push_back(normalized_shape);
  } else if (auto opt = normalized_shape.try_cast<ffi::Array<ffi::Any>>()) {
    shape_arr = opt.value();
    dim_num = static_cast<int64_t>(shape_arr.size());
  }
  ffi::Array<Integer> axes;
  for (int64_t i = -dim_num; i < 0; ++i) axes.push_back(Integer(i));

  ffi::Optional<NNParameter> w, b;
  if (elementwise_affine) {
    w = MakeParam(shape_arr, dt);
    b = MakeParam(shape_arr, dt);
  }
  LayerNormModule mod(w, b, axes, eps, elementwise_affine);
  PopulateAttrs(mod, "weight", w);
  PopulateAttrs(mod, "bias", b);
  return mod;
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------
RMSNormModule::RMSNormModule(NNParameter weight, ffi::Optional<NNParameter> bias,
                             ffi::Array<Integer> axes, double epsilon) {
  data_ = ffi::make_object<RMSNormModuleNode>(std::move(weight), std::move(bias), std::move(axes),
                                              epsilon);
}

Var RMSNormModuleNode::Forward(Var x) const {
  Var out = NNRmsNorm(x, weight->expr, axes, epsilon);
  if (bias.has_value()) return NNAdd(out, bias.value()->expr, "rms_norm");
  return out;
}

RMSNormModule MakeRMSNorm(ffi::Any hidden_size, ffi::Array<Integer> axes, double epsilon,
                          bool has_bias, ffi::Optional<ffi::String> dtype) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  NNParameter w = MakeParam({hidden_size}, dt);
  ffi::Optional<NNParameter> b =
      has_bias ? ffi::Optional<NNParameter>(MakeParam({hidden_size}, dt)) : std::nullopt;
  RMSNormModule mod(w, b, axes, epsilon);
  PopulateAttrs(mod, "weight", w);
  PopulateAttrs(mod, "bias", b);
  return mod;
}

// ---------------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------------
GroupNormModule::GroupNormModule(int64_t num_groups, ffi::Optional<NNParameter> weight,
                                 ffi::Optional<NNParameter> bias, double epsilon) {
  data_ = ffi::make_object<GroupNormModuleNode>(num_groups, std::move(weight), std::move(bias),
                                                epsilon);
}

Var GroupNormModuleNode::Forward(Var x, int64_t channel_axis, ffi::Array<Integer> axes) const {
  TVM_FFI_ICHECK(weight.defined() && bias.defined()) << "GroupNorm requires both weight and bias";
  return NNGroupNorm(x, ffi::Optional<Var>(weight.value()->expr),
                     ffi::Optional<Var>(bias.value()->expr), static_cast<int>(num_groups),
                     static_cast<int>(channel_axis), axes, epsilon);
}

GroupNormModule MakeGroupNorm(int64_t num_groups, ffi::Any num_channels, double eps, bool affine,
                              ffi::Optional<ffi::String> dtype) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  ffi::Optional<NNParameter> w, b;
  if (affine) {
    w = MakeParam({num_channels}, dt);
    b = MakeParam({num_channels}, dt);
  }
  GroupNormModule mod(num_groups, w, b, eps);
  PopulateAttrs(mod, "weight", w);
  PopulateAttrs(mod, "bias", b);
  return mod;
}

// ---------------------------------------------------------------------------
// Conv1D
// ---------------------------------------------------------------------------
Conv1DModule::Conv1DModule(NNParameter weight, ffi::Optional<NNParameter> bias, int64_t stride,
                           int64_t padding, int64_t dilation, int64_t groups) {
  data_ = ffi::make_object<Conv1DModuleNode>(std::move(weight), std::move(bias), stride, padding,
                                             dilation, groups);
}

Var Conv1DModuleNode::Forward(Var x) const {
  return NNConv1d(
      x, weight->expr, bias.has_value() ? ffi::Optional<Var>(bias.value()->expr) : std::nullopt,
      ffi::Any(stride), ffi::Any(padding), ffi::Any(dilation), static_cast<int>(groups));
}

Conv1DModule MakeConv1D(ffi::Any in_channels, ffi::Any out_channels, ffi::Any kernel_size,
                        int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                        bool has_bias, ffi::Optional<ffi::String> dtype) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  // in_per_group = in_channels / groups.
  PrimExpr in_per_group = arith::Analyzer().Simplify(
      tir::FloorDiv(AnyToDim(in_channels), IntImm(DataType::Int(64), groups)));
  NNParameter w = MakeParam({out_channels, ffi::Any(in_per_group), kernel_size}, dt);
  ffi::Optional<NNParameter> b =
      has_bias ? ffi::Optional<NNParameter>(MakeParam({out_channels}, dt)) : std::nullopt;
  Conv1DModule mod(w, b, stride, padding, dilation, groups);
  PopulateAttrs(mod, "weight", w);
  PopulateAttrs(mod, "bias", b);
  return mod;
}

// ---------------------------------------------------------------------------
// Conv2D
// ---------------------------------------------------------------------------
Conv2DModule::Conv2DModule(NNParameter weight, ffi::Optional<NNParameter> bias, int64_t stride,
                           int64_t padding, int64_t dilation, int64_t groups,
                           ffi::String data_layout) {
  data_ = ffi::make_object<Conv2DModuleNode>(std::move(weight), std::move(bias), stride, padding,
                                             dilation, groups, std::move(data_layout));
}

Var Conv2DModuleNode::Forward(Var x) const {
  return NNConv2d(x, weight->expr,
                  bias.has_value() ? ffi::Optional<Var>(bias.value()->expr) : std::nullopt,
                  ffi::Any(stride), ffi::Any(padding), ffi::Any(dilation), static_cast<int>(groups),
                  std::string(data_layout));
}

Conv2DModule MakeConv2D(ffi::Any in_channels, ffi::Any out_channels,
                        ffi::Array<Integer> kernel_size, int64_t stride, int64_t padding,
                        int64_t dilation, int64_t groups, bool has_bias,
                        ffi::Optional<ffi::String> dtype, ffi::String data_layout) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  PrimExpr in_per_group = arith::Analyzer().Simplify(
      tir::FloorDiv(AnyToDim(in_channels), IntImm(DataType::Int(64), groups)));
  int64_t kh = kernel_size[0]->value, kw = kernel_size[1]->value;
  NNParameter w = MakeParam({out_channels, ffi::Any(in_per_group), ffi::Any(kh), ffi::Any(kw)}, dt);
  ffi::Optional<NNParameter> b =
      has_bias ? ffi::Optional<NNParameter>(MakeParam({out_channels}, dt)) : std::nullopt;
  Conv2DModule mod(w, b, stride, padding, dilation, groups, data_layout);
  PopulateAttrs(mod, "weight", w);
  PopulateAttrs(mod, "bias", b);
  return mod;
}

// ---------------------------------------------------------------------------
// Conv3D
// ---------------------------------------------------------------------------
Conv3DModule::Conv3DModule(NNParameter weight, ffi::Optional<NNParameter> bias, int64_t stride,
                           int64_t padding, int64_t dilation, int64_t groups,
                           ffi::String data_layout) {
  data_ = ffi::make_object<Conv3DModuleNode>(std::move(weight), std::move(bias), stride, padding,
                                             dilation, groups, std::move(data_layout));
}

Var Conv3DModuleNode::Forward(Var x) const {
  return NNConv3d(x, weight->expr,
                  bias.has_value() ? ffi::Optional<Var>(bias.value()->expr) : std::nullopt,
                  ffi::Any(stride), ffi::Any(padding), ffi::Any(dilation), static_cast<int>(groups),
                  std::string(data_layout));
}

Conv3DModule MakeConv3D(ffi::Any in_channels, ffi::Any out_channels,
                        ffi::Array<Integer> kernel_size, int64_t stride, int64_t padding,
                        int64_t dilation, int64_t groups, bool has_bias,
                        ffi::Optional<ffi::String> dtype, ffi::String data_layout) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  int64_t kd = kernel_size[0]->value, kh = kernel_size[1]->value, kw = kernel_size[2]->value;
  NNParameter w =
      MakeParam({out_channels, in_channels, ffi::Any(kd), ffi::Any(kh), ffi::Any(kw)}, dt);
  ffi::Optional<NNParameter> b =
      has_bias ? ffi::Optional<NNParameter>(MakeParam({out_channels}, dt)) : std::nullopt;
  Conv3DModule mod(w, b, stride, padding, dilation, groups, data_layout);
  PopulateAttrs(mod, "weight", w);
  PopulateAttrs(mod, "bias", b);
  return mod;
}

// ---------------------------------------------------------------------------
// ConvTranspose1D
// ---------------------------------------------------------------------------
ConvTranspose1DModule::ConvTranspose1DModule(NNParameter weight, ffi::Optional<NNParameter> bias,
                                             int64_t stride, int64_t padding,
                                             int64_t output_padding, int64_t dilation,
                                             int64_t groups) {
  data_ = ffi::make_object<ConvTranspose1DModuleNode>(std::move(weight), std::move(bias), stride,
                                                      padding, output_padding, dilation, groups);
}

Var ConvTranspose1DModuleNode::Forward(Var x) const {
  return NNConv1dTranspose(x, weight->expr,
                           bias.has_value() ? ffi::Optional<Var>(bias.value()->expr) : std::nullopt,
                           ffi::Any(stride), ffi::Any(padding), ffi::Any(output_padding),
                           ffi::Any(dilation), static_cast<int>(groups));
}

ConvTranspose1DModule MakeConvTranspose1D(ffi::Any in_channels, ffi::Any out_channels,
                                          ffi::Any kernel_size, int64_t stride, int64_t padding,
                                          int64_t output_padding, int64_t dilation, int64_t groups,
                                          bool has_bias, ffi::Optional<ffi::String> dtype) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  PrimExpr out_per_group = arith::Analyzer().Simplify(
      tir::FloorDiv(AnyToDim(out_channels), IntImm(DataType::Int(64), groups)));
  NNParameter w = MakeParam({in_channels, ffi::Any(out_per_group), kernel_size}, dt);
  ffi::Optional<NNParameter> b =
      has_bias ? ffi::Optional<NNParameter>(MakeParam({out_channels}, dt)) : std::nullopt;
  ConvTranspose1DModule mod(w, b, stride, padding, output_padding, dilation, groups);
  PopulateAttrs(mod, "weight", w);
  PopulateAttrs(mod, "bias", b);
  return mod;
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

IdentityModule::IdentityModule() { data_ = ffi::make_object<IdentityModuleNode>(); }

Var IdentityModuleNode::Forward(Var x) const { return x; }

// ---------------------------------------------------------------------------
// IOEffect
// ---------------------------------------------------------------------------

IOEffectModule::IOEffectModule() { data_ = ffi::make_object<IOEffectModuleNode>(); }

ffi::Array<Var> IOEffectModuleNode::EmitInit(ffi::String name_hint, BlockBuilder bb) const {
  static const Op& null_value_op = Op::Get("relax.null_value");
  Var io = bb->Emit(Call(null_value_op, {}, {}, {}), std::string(name_hint) + ".io");
  return {io};
}

ffi::Array<Var> IOEffectModuleNode::Create(ffi::String name_hint) {
  Var v{std::string(name_hint) + ".io", ObjectStructInfo()};
  effect = v;
  return {v};
}

void IOEffectModuleNode::SetState(ffi::Array<Var> state_vars) {
  TVM_FFI_ICHECK_EQ(state_vars.size(), 1) << "IOEffect::SetState expects exactly 1 var";
  effect = state_vars[0];
}

ffi::Array<Var> IOEffectModuleNode::Finalize() {
  TVM_FFI_ICHECK(effect.has_value()) << "IOEffect::Finalize called with no active effect";
  Var result = effect.value();
  effect = std::nullopt;
  return {result};
}

// ---------------------------------------------------------------------------
// KVCache
// ---------------------------------------------------------------------------

KVCacheModule::KVCacheModule(int64_t init_seq_len, ffi::Array<Integer> unit_shape,
                             ffi::String dtype) {
  data_ =
      ffi::make_object<KVCacheModuleNode>(init_seq_len, std::move(unit_shape), std::move(dtype));
}

ffi::Array<Var> KVCacheModuleNode::EmitInit(ffi::String name_hint, BlockBuilder bb) const {
  // Build init_shape = [init_seq_len, *unit_shape].
  ffi::Array<PrimExpr> shape_dims;
  shape_dims.push_back(IntImm(DataType::Int(64), init_seq_len));
  for (const Integer& d : unit_shape) shape_dims.push_back(d);
  ShapeExpr init_shape(shape_dims);

  DataType dt = DataType(ffi::StringToDLDataType(std::string(dtype)));
  Expr zeros_val = relax::zeros(init_shape, dt);

  // call_pure_packed("vm.builtin.attention_kv_cache_create", zeros, init_shape, PrimValue(0)).
  static const Op& cpp_op = Op::Get("relax.call_pure_packed");
  Expr call = Call(cpp_op,
                   {ExternFunc("vm.builtin.attention_kv_cache_create"), zeros_val, init_shape,
                    PrimValue(IntImm(DataType::Int(64), 0))},
                   {}, {ObjectStructInfo()});
  Var v = bb->Emit(call, std::string(name_hint));
  return {v};
}

ffi::Array<Var> KVCacheModuleNode::Create(ffi::String name_hint) {
  Var v{std::string(name_hint), ObjectStructInfo()};
  cache = v;
  return {v};
}

void KVCacheModuleNode::SetState(ffi::Array<Var> state_vars) {
  TVM_FFI_ICHECK_EQ(state_vars.size(), 1) << "KVCache::SetState expects exactly 1 var";
  cache = state_vars[0];
}

ffi::Array<Var> KVCacheModuleNode::Finalize() {
  TVM_FFI_ICHECK(cache.has_value()) << "KVCache::Finalize called with no active cache";
  Var result = cache.value();
  cache = std::nullopt;
  return {result};
}

void KVCacheModuleNode::To(ffi::String new_dtype) { dtype = std::move(new_dtype); }

NNTensor KVCacheModuleNode::View(PrimExpr seq_len) const {
  TVM_FFI_ICHECK(cache.has_value()) << "KVCache::View called with no active cache";
  BlockBuilder bb = BlockBuilder_Current();
  TVM_FFI_ICHECK(bb.defined()) << "KVCache::View called outside BlockBuilder scope";

  ffi::Array<PrimExpr> shape_dims;
  shape_dims.push_back(seq_len);  // symbolic or concrete
  for (const Integer& d : unit_shape) shape_dims.push_back(d);
  ShapeExpr shape(shape_dims);

  DataType dt = DataType(ffi::StringToDLDataType(std::string(dtype)));
  static const Op& cpp_op = Op::Get("relax.call_pure_packed");
  Expr call = Call(cpp_op, {ExternFunc("vm.builtin.attention_kv_cache_view"), cache.value(), shape},
                   {}, {TensorStructInfo(shape, dt)});
  Var v = bb->Emit(call, "kv_cache_view");
  return NNTensor(v);
}

void KVCacheModuleNode::Append(NNTensor new_element) {
  TVM_FFI_ICHECK(cache.has_value()) << "KVCache::Append called with no active cache";
  BlockBuilder bb = BlockBuilder_Current();
  TVM_FFI_ICHECK(bb.defined()) << "KVCache::Append called outside BlockBuilder scope";

  // call_inplace_packed("vm.builtin.attention_kv_cache_append", cache, new_element,
  //                     inplace_indices=[0]).
  ObjectPtr<CallInplacePackedAttrs> attrs = ffi::make_object<CallInplacePackedAttrs>();
  attrs->inplace_indices = {Integer(0)};
  static const Op& cpp_op = Op::Get("relax.call_inplace_packed");
  Expr call =
      Call(cpp_op,
           {ExternFunc("vm.builtin.attention_kv_cache_append"), cache.value(), new_element->expr},
           Attrs(attrs), {ObjectStructInfo()});
  cache = bb->Emit(call, "kv_cache_append");
}

// ---------------------------------------------------------------------------
// Timesteps
// ---------------------------------------------------------------------------

TimestepsModule::TimestepsModule(int64_t num_channels, bool flip_sin_to_cos,
                                 double downscale_freq_shift) {
  data_ =
      ffi::make_object<TimestepsModuleNode>(num_channels, flip_sin_to_cos, downscale_freq_shift);
}

Var TimestepsModuleNode::Forward(Var x) const {
  ffi::String dt = ffi::String(GetDefaultDtype());
  return NNGetTimestepEmbedding(x, num_channels, flip_sin_to_cos, downscale_freq_shift,
                                /*scale=*/1.0, /*max_period=*/10000, dt, "get_timestep_embedding");
}

// ---------------------------------------------------------------------------
// TimestepEmbedding
// ---------------------------------------------------------------------------

TimestepEmbeddingModule::TimestepEmbeddingModule(LinearModule linear_1,
                                                 ffi::Optional<LinearModule> cond_proj,
                                                 SiLUModule act, LinearModule linear_2,
                                                 ffi::Optional<SiLUModule> post_act) {
  data_ = ffi::make_object<TimestepEmbeddingModuleNode>(std::move(linear_1), std::move(cond_proj),
                                                        std::move(act), std::move(linear_2),
                                                        std::move(post_act));
}

Var TimestepEmbeddingModuleNode::Forward(Var sample, ffi::Optional<Var> condition) const {
  Var s = sample;
  if (condition.has_value()) {
    // sample = sample + cond_proj(condition)
    TVM_FFI_ICHECK(cond_proj.has_value()) << "TimestepEmbedding: condition given but no cond_proj";
    Var proj = cond_proj.value().get()->Forward(condition.value());
    s = NNAdd(s, proj, "cond_add");
  }
  // act(linear_1(sample)).
  Var l1 = linear_1.get()->Forward(s);
  Var a = act.get()->Forward(l1);
  // linear_2(act_out).
  Var l2 = linear_2.get()->Forward(a);
  if (post_act.has_value()) {
    l2 = post_act.value().get()->Forward(l2);
  }
  return l2;
}

TimestepEmbeddingModule MakeTimestepEmbedding(int64_t in_channels, int64_t time_embed_dim,
                                              ffi::String act_fn, ffi::Optional<int64_t> out_dim,
                                              ffi::Optional<ffi::String> post_act_fn,
                                              ffi::Optional<int64_t> cond_proj_dim) {
  TVM_FFI_ICHECK(act_fn == "silu") << "TimestepEmbedding: only act_fn='silu' is supported";
  int64_t out = out_dim.value_or(time_embed_dim);
  LinearModule l1 = MakeLinear(ffi::Any(in_channels), ffi::Any(time_embed_dim), /*bias=*/true,
                               std::nullopt, std::nullopt);
  ffi::Optional<LinearModule> cp;
  if (cond_proj_dim.has_value()) {
    cp = MakeLinear(ffi::Any(cond_proj_dim.value()), ffi::Any(in_channels), /*bias=*/false,
                    std::nullopt, std::nullopt);
  }
  SiLUModule act;
  LinearModule l2 = MakeLinear(ffi::Any(time_embed_dim), ffi::Any(out), /*bias=*/true, std::nullopt,
                               std::nullopt);
  ffi::Optional<SiLUModule> post;
  if (post_act_fn.has_value() && post_act_fn.value() == "silu") post = SiLUModule();

  TimestepEmbeddingModule mod(l1, cp, act, l2, post);
  // Populate attrs for named_parameters traversal.
  auto* node = const_cast<TimestepEmbeddingModuleNode*>(mod.get());
  node->attrs.Set("linear_1", ffi::Any(l1));
  if (cp.has_value()) node->attrs.Set("cond_proj", ffi::Any(cp.value()));
  node->attrs.Set("linear_2", ffi::Any(l2));
  return mod;
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

AttentionModule::AttentionModule(int64_t heads, int64_t inner_dim, LinearModule to_q,
                                 LinearModule to_k, LinearModule to_v,
                                 ffi::Optional<GroupNormModule> group_norm, ModuleList to_out) {
  data_ = ffi::make_object<AttentionModuleNode>(heads, inner_dim, std::move(to_q), std::move(to_k),
                                                std::move(to_v), std::move(group_norm),
                                                std::move(to_out));
}

Var AttentionModuleNode::Forward(Var hidden_states,
                                 ffi::Optional<Var> encoder_hidden_states) const {
  Var hs = hidden_states;
  if (group_norm.has_value()) {
    // group_norm(hidden_states, channel_axis=2, axes=[1]).
    hs = group_norm.value().get()->Forward(hs, 2, {Integer(1)});
  }
  Var q = to_q.get()->Forward(hs);
  Var enc = encoder_hidden_states.value_or(hs);
  Var k = to_k.get()->Forward(enc);
  Var v = to_v.get()->Forward(enc);

  int64_t head_dim = inner_dim / heads;
  // reshape to [batch, seq, heads, head_dim].
  auto reshape_4d = [&](Var t, const std::string& name) -> Var {
    return NNReshape(
        t, {ffi::Any(int64_t(0)), ffi::Any(int64_t(-1)), ffi::Any(heads), ffi::Any(head_dim)},
        name);
  };
  q = reshape_4d(q, "q");
  k = reshape_4d(k, "k");
  v = reshape_4d(v, "v");

  // scaled_dot_product_attention (no causal mask).
  Var out = NNScaledDotProductAttention(q, k, v, std::nullopt, std::nullopt, "attn_out");

  // reshape back to [batch, seq, heads*head_dim].
  out = NNReshape(out, {ffi::Any(int64_t(0)), ffi::Any(int64_t(-1)), ffi::Any(heads * head_dim)},
                  "attn_reshape");

  // to_out[0](out).
  const auto* list_node = to_out.get();
  TVM_FFI_ICHECK(!list_node->modules.empty()) << "Attention: to_out is empty";
  auto opt = list_node->modules[0].try_cast<LinearModule>();
  TVM_FFI_ICHECK(opt.has_value()) << "Attention: to_out[0] is not a LinearModule";
  return opt.value().get()->Forward(out);
}

AttentionModule MakeAttention(int64_t query_dim, ffi::Optional<int64_t> cross_attention_dim,
                              int64_t heads, int64_t dim_head, bool bias,
                              ffi::Optional<int64_t> norm_num_groups, bool out_bias) {
  int64_t inner_dim = dim_head * heads;
  int64_t cross_dim = cross_attention_dim.value_or(query_dim);
  LinearModule to_q =
      MakeLinear(ffi::Any(query_dim), ffi::Any(inner_dim), bias, std::nullopt, std::nullopt);
  LinearModule to_k =
      MakeLinear(ffi::Any(cross_dim), ffi::Any(inner_dim), bias, std::nullopt, std::nullopt);
  LinearModule to_v =
      MakeLinear(ffi::Any(cross_dim), ffi::Any(inner_dim), bias, std::nullopt, std::nullopt);
  ffi::Optional<GroupNormModule> gn;
  if (norm_num_groups.has_value()) {
    gn = MakeGroupNorm(norm_num_groups.value(), ffi::Any(query_dim), 1e-5, /*affine=*/true,
                       std::nullopt);
  }
  LinearModule to_out_linear =
      MakeLinear(ffi::Any(inner_dim), ffi::Any(query_dim), out_bias, std::nullopt, std::nullopt);
  ModuleList to_out({ffi::Any(to_out_linear)});

  AttentionModule mod(heads, inner_dim, to_q, to_k, to_v, gn, to_out);
  // Populate attrs for named_parameters traversal.
  auto* node = const_cast<AttentionModuleNode*>(mod.get());
  node->attrs.Set("to_q", ffi::Any(to_q));
  node->attrs.Set("to_k", ffi::Any(to_k));
  node->attrs.Set("to_v", ffi::Any(to_v));
  if (gn.has_value()) node->attrs.Set("group_norm", ffi::Any(gn.value()));
  node->attrs.Set("to_out", ffi::Any(to_out));
  return mod;
}

// ---------------------------------------------------------------------------
// Register all reflection + factory globals
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  ReLUModuleNode::RegisterReflection();
  SiLUModuleNode::RegisterReflection();
  GELUModuleNode::RegisterReflection();
  LinearModuleNode::RegisterReflection();
  EmbeddingModuleNode::RegisterReflection();
  LayerNormModuleNode::RegisterReflection();
  RMSNormModuleNode::RegisterReflection();
  GroupNormModuleNode::RegisterReflection();
  Conv1DModuleNode::RegisterReflection();
  Conv2DModuleNode::RegisterReflection();
  Conv3DModuleNode::RegisterReflection();
  ConvTranspose1DModuleNode::RegisterReflection();
  IdentityModuleNode::RegisterReflection();
  EffectNode::RegisterReflection();
  IOEffectModuleNode::RegisterReflection();
  KVCacheModuleNode::RegisterReflection();
  TimestepsModuleNode::RegisterReflection();
  TimestepEmbeddingModuleNode::RegisterReflection();
  AttentionModuleNode::RegisterReflection();

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      // No-parameter constructors for activation modules.
      .def("relax.frontend.nn.ReLU", []() { return ReLUModule(); })
      .def("relax.frontend.nn.SiLU", []() { return SiLUModule(); })
      .def("relax.frontend.nn.GELU",
           [](ffi::String approximate) { return GELUModule(std::move(approximate)); })
      .def("relax.frontend.nn.MakeLinear",
           [](ffi::Any in_f, ffi::Any out_f, bool bias, ffi::Optional<ffi::String> dtype,
              ffi::Optional<ffi::String> out_dtype) {
             return MakeLinear(in_f, out_f, bias, dtype, out_dtype);
           })
      .def("relax.frontend.nn.MakeEmbedding",
           [](ffi::Any num, ffi::Any dim, ffi::Optional<ffi::String> dtype) {
             return MakeEmbedding(num, dim, dtype);
           })
      .def("relax.frontend.nn.MakeLayerNorm",
           [](ffi::Any normalized_shape, double eps, bool elementwise_affine,
              ffi::Optional<ffi::String> dtype) {
             return MakeLayerNorm(normalized_shape, eps, elementwise_affine, dtype);
           })
      .def("relax.frontend.nn.MakeRMSNorm",
           [](ffi::Any hidden_size, ffi::Array<Integer> axes, double epsilon, bool has_bias,
              ffi::Optional<ffi::String> dtype) {
             return MakeRMSNorm(hidden_size, axes, epsilon, has_bias, dtype);
           })
      .def("relax.frontend.nn.MakeGroupNorm",
           [](int64_t num_groups, ffi::Any num_channels, double eps, bool affine,
              ffi::Optional<ffi::String> dtype) {
             return MakeGroupNorm(num_groups, num_channels, eps, affine, dtype);
           })
      .def("relax.frontend.nn.MakeConv1D",
           [](ffi::Any in_ch, ffi::Any out_ch, ffi::Any ks, int64_t stride, int64_t padding,
              int64_t dilation, int64_t groups, bool bias, ffi::Optional<ffi::String> dtype) {
             return MakeConv1D(in_ch, out_ch, ks, stride, padding, dilation, groups, bias, dtype);
           })
      .def("relax.frontend.nn.MakeConv2D",
           [](ffi::Any in_ch, ffi::Any out_ch, ffi::Array<Integer> ks, int64_t stride,
              int64_t padding, int64_t dilation, int64_t groups, bool bias,
              ffi::Optional<ffi::String> dtype, ffi::String layout) {
             return MakeConv2D(in_ch, out_ch, ks, stride, padding, dilation, groups, bias, dtype,
                               layout);
           })
      .def("relax.frontend.nn.MakeConv3D",
           [](ffi::Any in_ch, ffi::Any out_ch, ffi::Array<Integer> ks, int64_t stride,
              int64_t padding, int64_t dilation, int64_t groups, bool bias,
              ffi::Optional<ffi::String> dtype, ffi::String layout) {
             return MakeConv3D(in_ch, out_ch, ks, stride, padding, dilation, groups, bias, dtype,
                               layout);
           })
      .def("relax.frontend.nn.MakeConvTranspose1D",
           [](ffi::Any in_ch, ffi::Any out_ch, ffi::Any ks, int64_t stride, int64_t padding,
              int64_t out_pad, int64_t dilation, int64_t groups, bool bias,
              ffi::Optional<ffi::String> dtype) {
             return MakeConvTranspose1D(in_ch, out_ch, ks, stride, padding, out_pad, dilation,
                                        groups, bias, dtype);
           })
      // New modules
      .def("relax.frontend.nn.Identity", []() { return IdentityModule(); })
      .def("relax.frontend.nn.IOEffect", []() { return IOEffectModule(); })
      .def("relax.frontend.nn.MakeKVCache",
           [](int64_t init_seq_len, ffi::Array<Integer> unit_shape,
              ffi::Optional<ffi::String> dtype) {
             ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
             return KVCacheModule(init_seq_len, unit_shape, dt);
           })
      .def("relax.frontend.nn.MakeTimesteps",
           [](int64_t num_channels, bool flip_sin_to_cos, double downscale_freq_shift) {
             return TimestepsModule(num_channels, flip_sin_to_cos, downscale_freq_shift);
           })
      .def("relax.frontend.nn.MakeTimestepEmbedding",
           [](int64_t in_channels, int64_t time_embed_dim, ffi::String act_fn,
              ffi::Optional<int64_t> out_dim, ffi::Optional<ffi::String> post_act_fn,
              ffi::Optional<int64_t> cond_proj_dim) {
             return MakeTimestepEmbedding(in_channels, time_embed_dim, act_fn, out_dim, post_act_fn,
                                          cond_proj_dim);
           })
      .def("relax.frontend.nn.MakeAttention",
           [](int64_t query_dim, ffi::Optional<int64_t> cross_attention_dim, int64_t heads,
              int64_t dim_head, bool bias, ffi::Optional<int64_t> norm_num_groups, bool out_bias) {
             return MakeAttention(query_dim, cross_attention_dim, heads, dim_head, bias,
                                  norm_num_groups, out_bias);
           });
}

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
