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
 */

#include "modules.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>

#include <string>

#include "../../op/nn/convolution.h"
#include "../../op/nn/nn.h"
#include "../../op/tensor/binary.h"
#include "../../op/tensor/index.h"
#include "../../op/tensor/linear_algebra.h"
#include "../../op/tensor/manipulate.h"
#include "core.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static Var Emit(Expr expr, const std::string& name) {
  BlockBuilder bb = BlockBuilder_Current();
  TVM_FFI_ICHECK(bb.defined()) << "forward() called outside a BlockBuilder scope";
  return bb->Emit(expr, name);
}

// Build a ShapeExpr from a mixed ffi::Any element (int64 or PrimExpr).
static PrimExpr AnyToDim(const ffi::Any& v) {
  if (auto opt = v.try_cast<int64_t>()) return IntImm(DataType::Int(64), opt.value());
  if (auto opt = v.try_cast<PrimExpr>()) return opt.value();
  TVM_FFI_THROW(TypeError) << "Expected int64 or PrimExpr, got " << v.GetTypeKey();
  TVM_FFI_UNREACHABLE();
}

// ---------------------------------------------------------------------------
// MakeParam helper – shared by all Make* factories
// ---------------------------------------------------------------------------

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
Var ReLUModuleNode::Forward(Var x) const { return Emit(relax::relu(x), "relu"); }

// ---------------------------------------------------------------------------
// SiLU
// ---------------------------------------------------------------------------
SiLUModule::SiLUModule() { data_ = ffi::make_object<SiLUModuleNode>(); }
Var SiLUModuleNode::Forward(Var x) const { return Emit(relax::silu(x), "silu"); }

// ---------------------------------------------------------------------------
// GELU
// ---------------------------------------------------------------------------
GELUModule::GELUModule(ffi::String approximate) {
  data_ = ffi::make_object<GELUModuleNode>(std::move(approximate));
}
Var GELUModuleNode::Forward(Var x) const {
  return Emit((approximate == "tanh") ? relax::gelu_tanh(x) : relax::gelu(x), "gelu");
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
  Expr w = relax::permute_dims(weight->expr, std::nullopt);
  ffi::Optional<DataType> dt =
      out_dtype.has_value()
          ? ffi::Optional<DataType>(DataType(ffi::StringToDLDataType(out_dtype.value())))
          : std::nullopt;
  Expr out = relax::matmul(x, w, dt);
  if (bias.has_value()) out = relax::add(out, bias.value()->expr);
  return Emit(out, "linear");
}

LinearModule MakeLinear(ffi::Any in_features, ffi::Any out_features, bool has_bias,
                        ffi::Optional<ffi::String> dtype, ffi::Optional<ffi::String> out_dtype) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  ffi::String bias_dt = out_dtype.has_value() ? out_dtype.value() : dt;
  NNParameter w = MakeParam({out_features, in_features}, dt);
  ffi::Optional<NNParameter> b =
      has_bias ? ffi::Optional<NNParameter>(MakeParam({out_features}, bias_dt)) : std::nullopt;
  return LinearModule(std::move(w), std::move(b), out_dtype);
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------
EmbeddingModule::EmbeddingModule(NNParameter weight) {
  data_ = ffi::make_object<EmbeddingModuleNode>(std::move(weight));
}

Var EmbeddingModuleNode::Forward(Var x, ffi::Array<ffi::Any> out_shape_if_nd) const {
  if (out_shape_if_nd.empty()) {
    return Emit(relax::take(weight->expr, x, ffi::Optional<int64_t>(0)), "embedding");
  }
  Expr flat = relax::reshape(x, ShapeExpr(ffi::Array<PrimExpr>{IntImm(DataType::Int(64), -1)}));
  Expr taken = relax::take(weight->expr, flat, ffi::Optional<int64_t>(0));
  ffi::Array<PrimExpr> new_shape;
  for (const auto& s : out_shape_if_nd) new_shape.push_back(AnyToDim(s));
  return Emit(relax::reshape(taken, ShapeExpr(new_shape)), "embedding");
}

EmbeddingModule MakeEmbedding(ffi::Any num, ffi::Any dim, ffi::Optional<ffi::String> dtype) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  return EmbeddingModule(MakeParam({num, dim}, dt));
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
  // When elementwise_affine=false we still need weight/bias exprs for the op.
  // The factory stores constant ones/zeros as ParameterNode with no data.
  TVM_FFI_ICHECK(weight.defined() && bias.defined());
  return Emit(
      relax::layer_norm(x, weight.value()->expr, bias.value()->expr, axes, epsilon, true, true),
      "layer_norm");
}

LayerNormModule MakeLayerNorm(ffi::Any normalized_shape, double eps, bool elementwise_affine,
                              ffi::Optional<ffi::String> dtype) {
  ffi::String dt = dtype.value_or(ffi::String(elementwise_affine ? GetDefaultDtype() : "float32"));
  // Build axes: [-dim_num, ..., -1]
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
  return LayerNormModule(w, b, axes, eps, elementwise_affine);
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
  Expr out = relax::rms_norm(x, weight->expr, axes, epsilon);
  if (bias.has_value()) out = relax::add(out, bias.value()->expr);
  return Emit(out, "rms_norm");
}

RMSNormModule MakeRMSNorm(int64_t hidden_size, ffi::Array<Integer> axes, double epsilon,
                          bool has_bias, ffi::Optional<ffi::String> dtype) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  NNParameter w = MakeParam({ffi::Any(hidden_size)}, dt);
  ffi::Optional<NNParameter> b =
      has_bias ? ffi::Optional<NNParameter>(MakeParam({ffi::Any(hidden_size)}, dt)) : std::nullopt;
  return RMSNormModule(std::move(w), std::move(b), axes, epsilon);
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
  return Emit(relax::group_norm(x, weight.value()->expr, bias.value()->expr, num_groups,
                                static_cast<int>(channel_axis), axes, epsilon, true, true),
              "group_norm");
}

GroupNormModule MakeGroupNorm(int64_t num_groups, int64_t num_channels, double eps, bool affine,
                              ffi::Optional<ffi::String> dtype) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  ffi::Optional<NNParameter> w, b;
  if (affine) {
    w = MakeParam({ffi::Any(num_channels)}, dt);
    b = MakeParam({ffi::Any(num_channels)}, dt);
  }
  return GroupNormModule(num_groups, std::move(w), std::move(b), eps);
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
  Expr out = relax::conv1d(x, weight->expr, {stride}, {padding}, {dilation},
                           static_cast<int>(groups), "NCW", "OIW", std::nullopt, std::nullopt);
  if (bias.has_value()) {
    Expr b = relax::reshape(
        bias.value()->expr,
        ShapeExpr(ffi::Array<PrimExpr>{IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), -1),
                                       IntImm(DataType::Int(64), 1)}));
    out = relax::add(out, b);
  }
  return Emit(out, "conv1d");
}

Conv1DModule MakeConv1D(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                        int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                        bool has_bias, ffi::Optional<ffi::String> dtype) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  int64_t in_per_group = in_channels / groups;
  NNParameter w =
      MakeParam({ffi::Any(out_channels), ffi::Any(in_per_group), ffi::Any(kernel_size)}, dt);
  ffi::Optional<NNParameter> b =
      has_bias ? ffi::Optional<NNParameter>(MakeParam({ffi::Any(out_channels)}, dt)) : std::nullopt;
  return Conv1DModule(std::move(w), std::move(b), stride, padding, dilation, groups);
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
  std::string dl = std::string(data_layout);
  std::string kl = (dl == "NCHW") ? "OIHW" : "HWIO";
  Expr out = relax::conv2d(x, weight->expr, {stride}, {padding}, {dilation},
                           static_cast<int>(groups), dl, kl, std::nullopt, std::nullopt);
  if (bias.has_value()) {
    Expr b =
        (dl == "NCHW")
            ? relax::reshape(bias.value()->expr,
                             ShapeExpr(ffi::Array<PrimExpr>{
                                 IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), -1),
                                 IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 1)}))
            : relax::reshape(bias.value()->expr,
                             ShapeExpr(ffi::Array<PrimExpr>{
                                 IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 1),
                                 IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), -1)}));
    out = relax::add(out, b);
  }
  return Emit(out, "conv2d");
}

Conv2DModule MakeConv2D(int64_t in_channels, int64_t out_channels, ffi::Array<Integer> kernel_size,
                        int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                        bool has_bias, ffi::Optional<ffi::String> dtype, ffi::String data_layout) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  int64_t in_per_group = in_channels / groups;
  int64_t kh = kernel_size[0]->value, kw = kernel_size[1]->value;
  NNParameter w =
      MakeParam({ffi::Any(out_channels), ffi::Any(in_per_group), ffi::Any(kh), ffi::Any(kw)}, dt);
  ffi::Optional<NNParameter> b =
      has_bias ? ffi::Optional<NNParameter>(MakeParam({ffi::Any(out_channels)}, dt)) : std::nullopt;
  return Conv2DModule(std::move(w), std::move(b), stride, padding, dilation, groups,
                      std::move(data_layout));
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
  std::string dl = std::string(data_layout);
  Expr out = relax::conv3d(x, weight->expr, {stride}, {padding}, {dilation},
                           static_cast<int>(groups), dl, "OIDHW", std::nullopt, std::nullopt);
  if (bias.has_value()) {
    bool nchw = (dl == "NCDHW");
    Expr b = nchw ? relax::reshape(bias.value()->expr,
                                   ShapeExpr(ffi::Array<PrimExpr>{
                                       IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), -1),
                                       IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 1),
                                       IntImm(DataType::Int(64), 1)}))
                  : relax::reshape(bias.value()->expr,
                                   ShapeExpr(ffi::Array<PrimExpr>{
                                       IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 1),
                                       IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 1),
                                       IntImm(DataType::Int(64), -1)}));
    out = relax::add(out, b);
  }
  return Emit(out, "conv3d");
}

Conv3DModule MakeConv3D(int64_t in_channels, int64_t out_channels, ffi::Array<Integer> kernel_size,
                        int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                        bool has_bias, ffi::Optional<ffi::String> dtype, ffi::String data_layout) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  int64_t kd = kernel_size[0]->value, kh = kernel_size[1]->value, kw = kernel_size[2]->value;
  NNParameter w = MakeParam(
      {ffi::Any(out_channels), ffi::Any(in_channels), ffi::Any(kd), ffi::Any(kh), ffi::Any(kw)},
      dt);
  ffi::Optional<NNParameter> b =
      has_bias ? ffi::Optional<NNParameter>(MakeParam({ffi::Any(out_channels)}, dt)) : std::nullopt;
  return Conv3DModule(std::move(w), std::move(b), stride, padding, dilation, groups,
                      std::move(data_layout));
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
  Expr out =
      relax::conv1d_transpose(x, weight->expr, {stride}, {padding}, {output_padding}, {dilation},
                              static_cast<int>(groups), "NCW", "IOW", std::nullopt, std::nullopt);
  if (bias.has_value()) {
    Expr b = relax::reshape(
        bias.value()->expr,
        ShapeExpr(ffi::Array<PrimExpr>{IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), -1),
                                       IntImm(DataType::Int(64), 1)}));
    out = relax::add(out, b);
  }
  return Emit(out, "conv1d_transpose");
}

ConvTranspose1DModule MakeConvTranspose1D(int64_t in_channels, int64_t out_channels,
                                          int64_t kernel_size, int64_t stride, int64_t padding,
                                          int64_t output_padding, int64_t dilation, int64_t groups,
                                          bool has_bias, ffi::Optional<ffi::String> dtype) {
  ffi::String dt = dtype.value_or(ffi::String(GetDefaultDtype()));
  int64_t out_per_group = out_channels / groups;
  NNParameter w =
      MakeParam({ffi::Any(in_channels), ffi::Any(out_per_group), ffi::Any(kernel_size)}, dt);
  ffi::Optional<NNParameter> b =
      has_bias ? ffi::Optional<NNParameter>(MakeParam({ffi::Any(out_channels)}, dt)) : std::nullopt;
  return ConvTranspose1DModule(std::move(w), std::move(b), stride, padding, output_padding,
                               dilation, groups);
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

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
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
           [](int64_t hidden_size, ffi::Array<Integer> axes, double epsilon, bool has_bias,
              ffi::Optional<ffi::String> dtype) {
             return MakeRMSNorm(hidden_size, axes, epsilon, has_bias, dtype);
           })
      .def("relax.frontend.nn.MakeGroupNorm",
           [](int64_t num_groups, int64_t num_channels, double eps, bool affine,
              ffi::Optional<ffi::String> dtype) {
             return MakeGroupNorm(num_groups, num_channels, eps, affine, dtype);
           })
      .def("relax.frontend.nn.MakeConv1D",
           [](int64_t in_ch, int64_t out_ch, int64_t ks, int64_t stride, int64_t padding,
              int64_t dilation, int64_t groups, bool bias, ffi::Optional<ffi::String> dtype) {
             return MakeConv1D(in_ch, out_ch, ks, stride, padding, dilation, groups, bias, dtype);
           })
      .def("relax.frontend.nn.MakeConv2D",
           [](int64_t in_ch, int64_t out_ch, ffi::Array<Integer> ks, int64_t stride,
              int64_t padding, int64_t dilation, int64_t groups, bool bias,
              ffi::Optional<ffi::String> dtype, ffi::String layout) {
             return MakeConv2D(in_ch, out_ch, ks, stride, padding, dilation, groups, bias, dtype,
                               layout);
           })
      .def("relax.frontend.nn.MakeConv3D",
           [](int64_t in_ch, int64_t out_ch, ffi::Array<Integer> ks, int64_t stride,
              int64_t padding, int64_t dilation, int64_t groups, bool bias,
              ffi::Optional<ffi::String> dtype, ffi::String layout) {
             return MakeConv3D(in_ch, out_ch, ks, stride, padding, dilation, groups, bias, dtype,
                               layout);
           })
      .def("relax.frontend.nn.MakeConvTranspose1D",
           [](int64_t in_ch, int64_t out_ch, int64_t ks, int64_t stride, int64_t padding,
              int64_t out_pad, int64_t dilation, int64_t groups, bool bias,
              ffi::Optional<ffi::String> dtype) {
             return MakeConvTranspose1D(in_ch, out_ch, ks, stride, padding, out_pad, dilation,
                                        groups, bias, dtype);
           });
}

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
