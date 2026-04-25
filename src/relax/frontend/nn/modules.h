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
 * \file src/relax/frontend/nn/modules.h
 * \brief Built-in nn module definitions.
 *
 * Each XxxModuleNode owns all its members natively:
 *   - Trainable weights and biases are stored as NNParameter so that
 *     NNModuleNode::NamedParameters() can discover them via the attrs map.
 *   - Scalar hyper-parameters (epsilon, stride, etc.) are stored as their
 *     natural C++ types.
 *
 * Each node exposes:
 *   - def_ro / def_rw fields for direct attribute access.
 *   - refl::init<...>() for construction via the FFI.
 *   - A Forward() method that emits relax ops and returns a relax::Var.
 *
 * A companion Make* factory function per module handles shape arithmetic
 * (e.g. computing the weight shape for Conv2D) and returns the fully
 * constructed module with its attrs map populated.
 */

#ifndef TVM_RELAX_FRONTEND_NN_MODULES_H_
#define TVM_RELAX_FRONTEND_NN_MODULES_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>
#include <tvm/runtime/object.h>

#include <string>

#include "core.h"
#include "spec.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

/*!
 * \brief Build an NNParameter from a shape specification and dtype string.
 *
 * Used by all Make* factory functions to create weight and bias parameters.
 *
 * \param shape  Shape specification; each element is int64, String, or PrimExpr.
 * \param dtype  Data type string (e.g. "float32").
 * \return       A new unbound NNParameter.
 */
NNParameter MakeParam(ffi::Array<ffi::Any> shape, ffi::String dtype);

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

/*! \brief Rectified linear unit activation. */

/*! \brief Rectified linear unit activation. */
class ReLUModuleNode : public NNModuleNode {
 public:
  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReLUModuleNode>().def(refl::init<>()).def("_forward", &ReLUModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.ReLU", ReLUModuleNode, NNModuleNode);
};
class ReLUModule : public runtime::ObjectRef {
 public:
  explicit ReLUModule();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ReLUModule, runtime::ObjectRef, ReLUModuleNode);
};

// ---------------------------------------------------------------------------
// SiLU
// ---------------------------------------------------------------------------

/*! \brief Sigmoid linear unit activation. */
class SiLUModuleNode : public NNModuleNode {
 public:
  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SiLUModuleNode>().def(refl::init<>()).def("_forward", &SiLUModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.SiLU", SiLUModuleNode, NNModuleNode);
};
class SiLUModule : public runtime::ObjectRef {
 public:
  explicit SiLUModule();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SiLUModule, runtime::ObjectRef, SiLUModuleNode);
};

// ---------------------------------------------------------------------------
// GELU
// ---------------------------------------------------------------------------

/*! \brief Gaussian error linear unit activation. */
class GELUModuleNode : public NNModuleNode {
 public:
  /*! \brief Approximation method: "" for exact, "tanh" for tanh approximation. */
  ffi::String approximate;

  explicit GELUModuleNode(ffi::String approximate = "") : approximate(std::move(approximate)) {}
  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GELUModuleNode>()
        .def(refl::init<ffi::String>())
        .def_ro("approximate", &GELUModuleNode::approximate)
        .def("_forward", &GELUModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.GELU", GELUModuleNode, NNModuleNode);
};
class GELUModule : public runtime::ObjectRef {
 public:
  explicit GELUModule(ffi::String approximate = "");
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GELUModule, runtime::ObjectRef, GELUModuleNode);
};

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/*! \brief Fully-connected linear transformation. */
class LinearModuleNode : public NNModuleNode {
 public:
  /*! \brief Weight matrix; shape [out_features, in_features]. */
  NNParameter weight;
  /*! \brief Bias vector; shape [out_features], or nullopt when bias=False. */
  ffi::Optional<NNParameter> bias;
  /*! \brief Optional output dtype override. */
  ffi::Optional<ffi::String> out_dtype;

  LinearModuleNode(NNParameter weight, ffi::Optional<NNParameter> bias,
                   ffi::Optional<ffi::String> out_dtype)
      : weight(std::move(weight)), bias(std::move(bias)), out_dtype(std::move(out_dtype)) {}

  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LinearModuleNode>()
        .def(refl::init<NNParameter, ffi::Optional<NNParameter>, ffi::Optional<ffi::String>>())
        .def_ro("weight", &LinearModuleNode::weight)
        .def_ro("bias", &LinearModuleNode::bias)
        .def_ro("out_dtype", &LinearModuleNode::out_dtype)
        .def("_forward", &LinearModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Linear", LinearModuleNode, NNModuleNode);
};
class LinearModule : public runtime::ObjectRef {
 public:
  explicit LinearModule(NNParameter weight, ffi::Optional<NNParameter> bias,
                        ffi::Optional<ffi::String> out_dtype);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LinearModule, runtime::ObjectRef, LinearModuleNode);
};
/*!
 * \brief Factory: create a Linear module with the given dimensions.
 *
 * \param in_features   Input feature count (int64 or symbolic).
 * \param out_features  Output feature count (int64 or symbolic).
 * \param bias          Whether to include a bias parameter.
 * \param dtype         Weight dtype; defaults to the current default dtype.
 * \param out_dtype     Optional output dtype override.
 * \return              A fully constructed LinearModule.
 */
LinearModule MakeLinear(ffi::Any in_features, ffi::Any out_features, bool bias,
                        ffi::Optional<ffi::String> dtype, ffi::Optional<ffi::String> out_dtype);

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/*! \brief Lookup-table embedding layer. */
class EmbeddingModuleNode : public NNModuleNode {
 public:
  /*! \brief Embedding table; shape [num_embeddings, embedding_dim]. */
  NNParameter weight;

  explicit EmbeddingModuleNode(NNParameter weight) : weight(std::move(weight)) {}
  Var Forward(Var x, ffi::Array<ffi::Any> out_shape_if_nd) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<EmbeddingModuleNode>()
        .def(refl::init<NNParameter>())
        .def_ro("weight", &EmbeddingModuleNode::weight)
        .def("_forward", &EmbeddingModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Embedding", EmbeddingModuleNode,
                                    NNModuleNode);
};
class EmbeddingModule : public runtime::ObjectRef {
 public:
  explicit EmbeddingModule(NNParameter weight);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(EmbeddingModule, runtime::ObjectRef,
                                                EmbeddingModuleNode);
};
/*!
 * \brief Factory: create an Embedding module.
 *
 * \param num    Vocabulary size (int64 or symbolic).
 * \param dim    Embedding dimension (int64 or symbolic).
 * \param dtype  Weight dtype; defaults to the current default dtype.
 * \return       A fully constructed EmbeddingModule.
 */
EmbeddingModule MakeEmbedding(ffi::Any num, ffi::Any dim, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

/*! \brief Layer normalisation over the last N dimensions. */
class LayerNormModuleNode : public NNModuleNode {
 public:
  /*! \brief Scale parameter (gamma); nullopt when elementwise_affine=False. */
  ffi::Optional<NNParameter> weight;
  /*! \brief Shift parameter (beta); nullopt when elementwise_affine=False. */
  ffi::Optional<NNParameter> bias;
  /*! \brief Normalisation axes (negative indices). */
  ffi::Array<Integer> axes;
  double epsilon;
  bool elementwise_affine;

  LayerNormModuleNode(ffi::Optional<NNParameter> weight, ffi::Optional<NNParameter> bias,
                      ffi::Array<Integer> axes, double epsilon, bool elementwise_affine)
      : weight(std::move(weight)),
        bias(std::move(bias)),
        axes(std::move(axes)),
        epsilon(epsilon),
        elementwise_affine(elementwise_affine) {}

  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LayerNormModuleNode>()
        .def(refl::init<ffi::Optional<NNParameter>, ffi::Optional<NNParameter>, ffi::Array<Integer>,
                        double, bool>())
        .def_ro("weight", &LayerNormModuleNode::weight)
        .def_ro("bias", &LayerNormModuleNode::bias)
        .def_ro("axes", &LayerNormModuleNode::axes)
        .def_ro("epsilon", &LayerNormModuleNode::epsilon)
        .def_ro("elementwise_affine", &LayerNormModuleNode::elementwise_affine)
        .def("_forward", &LayerNormModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.LayerNorm", LayerNormModuleNode,
                                    NNModuleNode);
};
class LayerNormModule : public runtime::ObjectRef {
 public:
  explicit LayerNormModule(ffi::Optional<NNParameter> weight, ffi::Optional<NNParameter> bias,
                           ffi::Array<Integer> axes, double epsilon, bool elementwise_affine);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LayerNormModule, runtime::ObjectRef,
                                                LayerNormModuleNode);
};
/*!
 * \brief Factory: create a LayerNorm module.
 *
 * \param normalized_shape  int64 or Array<Any> of the normalised dimensions.
 * \param eps               Epsilon for numerical stability.
 * \param elementwise_affine  Whether to include learnable affine parameters.
 * \param dtype             Parameter dtype; defaults to the current default dtype.
 * \return                  A fully constructed LayerNormModule.
 */
LayerNormModule MakeLayerNorm(ffi::Any normalized_shape, double eps, bool elementwise_affine,
                              ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/*! \brief Root-mean-square layer normalisation. */
class RMSNormModuleNode : public NNModuleNode {
 public:
  /*! \brief Scale parameter. */
  NNParameter weight;
  /*! \brief Optional bias parameter. */
  ffi::Optional<NNParameter> bias;
  /*! \brief Normalisation axes. */
  ffi::Array<Integer> axes;
  double epsilon;

  RMSNormModuleNode(NNParameter weight, ffi::Optional<NNParameter> bias, ffi::Array<Integer> axes,
                    double epsilon)
      : weight(std::move(weight)), bias(std::move(bias)), axes(std::move(axes)), epsilon(epsilon) {}

  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RMSNormModuleNode>()
        .def(refl::init<NNParameter, ffi::Optional<NNParameter>, ffi::Array<Integer>, double>())
        .def_ro("weight", &RMSNormModuleNode::weight)
        .def_ro("bias", &RMSNormModuleNode::bias)
        .def_ro("axes", &RMSNormModuleNode::axes)
        .def_ro("epsilon", &RMSNormModuleNode::epsilon)
        .def("_forward", &RMSNormModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.RMSNorm", RMSNormModuleNode, NNModuleNode);
};
class RMSNormModule : public runtime::ObjectRef {
 public:
  explicit RMSNormModule(NNParameter weight, ffi::Optional<NNParameter> bias,
                         ffi::Array<Integer> axes, double epsilon);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RMSNormModule, runtime::ObjectRef,
                                                RMSNormModuleNode);
};
/*!
 * \brief Factory: create an RMSNorm module.
 *
 * \param hidden_size  Hidden dimension size (int64 or symbolic).
 * \param axes         Normalisation axes.
 * \param epsilon      Epsilon for numerical stability.
 * \param has_bias     Whether to include a bias parameter.
 * \param dtype        Parameter dtype; defaults to the current default dtype.
 * \return             A fully constructed RMSNormModule.
 */
RMSNormModule MakeRMSNorm(ffi::Any hidden_size, ffi::Array<Integer> axes, double epsilon,
                          bool has_bias, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------------

/*! \brief Group normalisation. */
class GroupNormModuleNode : public NNModuleNode {
 public:
  int64_t num_groups;
  /*! \brief Scale parameter; nullopt when affine=False. */
  ffi::Optional<NNParameter> weight;
  /*! \brief Bias parameter; nullopt when affine=False. */
  ffi::Optional<NNParameter> bias;
  double epsilon;

  GroupNormModuleNode(int64_t num_groups, ffi::Optional<NNParameter> weight,
                      ffi::Optional<NNParameter> bias, double epsilon)
      : num_groups(num_groups),
        weight(std::move(weight)),
        bias(std::move(bias)),
        epsilon(epsilon) {}

  Var Forward(Var x, int64_t channel_axis, ffi::Array<Integer> axes) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GroupNormModuleNode>()
        .def(refl::init<int64_t, ffi::Optional<NNParameter>, ffi::Optional<NNParameter>, double>())
        .def_ro("num_groups", &GroupNormModuleNode::num_groups)
        .def_ro("weight", &GroupNormModuleNode::weight)
        .def_ro("bias", &GroupNormModuleNode::bias)
        .def_ro("epsilon", &GroupNormModuleNode::epsilon)
        .def("_forward", &GroupNormModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.GroupNorm", GroupNormModuleNode,
                                    NNModuleNode);
};
class GroupNormModule : public runtime::ObjectRef {
 public:
  explicit GroupNormModule(int64_t num_groups, ffi::Optional<NNParameter> weight,
                           ffi::Optional<NNParameter> bias, double epsilon);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GroupNormModule, runtime::ObjectRef,
                                                GroupNormModuleNode);
};
/*!
 * \brief Factory: create a GroupNorm module.
 *
 * \param num_groups   Number of groups.
 * \param num_channels Channel count (int64 or symbolic).
 * \param eps          Epsilon for numerical stability.
 * \param affine       Whether to include learnable affine parameters.
 * \param dtype        Parameter dtype; defaults to the current default dtype.
 * \return             A fully constructed GroupNormModule.
 */
GroupNormModule MakeGroupNorm(int64_t num_groups, ffi::Any num_channels, double eps, bool affine,
                              ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// Conv1D
// ---------------------------------------------------------------------------

/*! \brief 1-D convolution. */
class Conv1DModuleNode : public NNModuleNode {
 public:
  NNParameter weight;
  ffi::Optional<NNParameter> bias;
  int64_t stride, padding, dilation, groups;

  Conv1DModuleNode(NNParameter weight, ffi::Optional<NNParameter> bias, int64_t stride,
                   int64_t padding, int64_t dilation, int64_t groups)
      : weight(std::move(weight)),
        bias(std::move(bias)),
        stride(stride),
        padding(padding),
        dilation(dilation),
        groups(groups) {}

  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<Conv1DModuleNode>()
        .def(refl::init<NNParameter, ffi::Optional<NNParameter>, int64_t, int64_t, int64_t,
                        int64_t>())
        .def_ro("weight", &Conv1DModuleNode::weight)
        .def_ro("bias", &Conv1DModuleNode::bias)
        .def_ro("stride", &Conv1DModuleNode::stride)
        .def_ro("padding", &Conv1DModuleNode::padding)
        .def_ro("dilation", &Conv1DModuleNode::dilation)
        .def_ro("groups", &Conv1DModuleNode::groups)
        .def("_forward", &Conv1DModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Conv1D", Conv1DModuleNode, NNModuleNode);
};
class Conv1DModule : public runtime::ObjectRef {
 public:
  explicit Conv1DModule(NNParameter weight, ffi::Optional<NNParameter> bias, int64_t stride,
                        int64_t padding, int64_t dilation, int64_t groups);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Conv1DModule, runtime::ObjectRef, Conv1DModuleNode);
};
/*! \brief Factory: create a Conv1D module with the given dimensions. */
Conv1DModule MakeConv1D(ffi::Any in_channels, ffi::Any out_channels, ffi::Any kernel_size,
                        int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                        bool has_bias, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// Conv2D
// ---------------------------------------------------------------------------

/*! \brief 2-D convolution. */
class Conv2DModuleNode : public NNModuleNode {
 public:
  NNParameter weight;
  ffi::Optional<NNParameter> bias;
  int64_t stride, padding, dilation, groups;
  ffi::String data_layout;

  Conv2DModuleNode(NNParameter weight, ffi::Optional<NNParameter> bias, int64_t stride,
                   int64_t padding, int64_t dilation, int64_t groups, ffi::String data_layout)
      : weight(std::move(weight)),
        bias(std::move(bias)),
        stride(stride),
        padding(padding),
        dilation(dilation),
        groups(groups),
        data_layout(std::move(data_layout)) {}

  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<Conv2DModuleNode>()
        .def(refl::init<NNParameter, ffi::Optional<NNParameter>, int64_t, int64_t, int64_t, int64_t,
                        ffi::String>())
        .def_ro("weight", &Conv2DModuleNode::weight)
        .def_ro("bias", &Conv2DModuleNode::bias)
        .def_ro("stride", &Conv2DModuleNode::stride)
        .def_ro("padding", &Conv2DModuleNode::padding)
        .def_ro("dilation", &Conv2DModuleNode::dilation)
        .def_ro("groups", &Conv2DModuleNode::groups)
        .def_ro("data_layout", &Conv2DModuleNode::data_layout)
        .def("_forward", &Conv2DModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Conv2D", Conv2DModuleNode, NNModuleNode);
};
class Conv2DModule : public runtime::ObjectRef {
 public:
  explicit Conv2DModule(NNParameter weight, ffi::Optional<NNParameter> bias, int64_t stride,
                        int64_t padding, int64_t dilation, int64_t groups, ffi::String data_layout);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Conv2DModule, runtime::ObjectRef, Conv2DModuleNode);
};
/*! \brief Factory: create a Conv2D module; expands scalar kernel_size to [kH, kW]. */
Conv2DModule MakeConv2D(ffi::Any in_channels, ffi::Any out_channels,
                        ffi::Array<Integer> kernel_size, int64_t stride, int64_t padding,
                        int64_t dilation, int64_t groups, bool has_bias,
                        ffi::Optional<ffi::String> dtype, ffi::String data_layout);

// ---------------------------------------------------------------------------
// Conv3D
// ---------------------------------------------------------------------------

/*! \brief 3-D convolution. */
class Conv3DModuleNode : public NNModuleNode {
 public:
  NNParameter weight;
  ffi::Optional<NNParameter> bias;
  int64_t stride, padding, dilation, groups;
  ffi::String data_layout;

  Conv3DModuleNode(NNParameter weight, ffi::Optional<NNParameter> bias, int64_t stride,
                   int64_t padding, int64_t dilation, int64_t groups, ffi::String data_layout)
      : weight(std::move(weight)),
        bias(std::move(bias)),
        stride(stride),
        padding(padding),
        dilation(dilation),
        groups(groups),
        data_layout(std::move(data_layout)) {}

  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<Conv3DModuleNode>()
        .def(refl::init<NNParameter, ffi::Optional<NNParameter>, int64_t, int64_t, int64_t, int64_t,
                        ffi::String>())
        .def_ro("weight", &Conv3DModuleNode::weight)
        .def_ro("bias", &Conv3DModuleNode::bias)
        .def_ro("stride", &Conv3DModuleNode::stride)
        .def_ro("padding", &Conv3DModuleNode::padding)
        .def_ro("dilation", &Conv3DModuleNode::dilation)
        .def_ro("groups", &Conv3DModuleNode::groups)
        .def_ro("data_layout", &Conv3DModuleNode::data_layout)
        .def("_forward", &Conv3DModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Conv3D", Conv3DModuleNode, NNModuleNode);
};
class Conv3DModule : public runtime::ObjectRef {
 public:
  explicit Conv3DModule(NNParameter weight, ffi::Optional<NNParameter> bias, int64_t stride,
                        int64_t padding, int64_t dilation, int64_t groups, ffi::String data_layout);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Conv3DModule, runtime::ObjectRef, Conv3DModuleNode);
};
/*! \brief Factory: create a Conv3D module; expands scalar kernel_size to [kD, kH, kW]. */
Conv3DModule MakeConv3D(ffi::Any in_channels, ffi::Any out_channels,
                        ffi::Array<Integer> kernel_size, int64_t stride, int64_t padding,
                        int64_t dilation, int64_t groups, bool has_bias,
                        ffi::Optional<ffi::String> dtype, ffi::String data_layout);

// ---------------------------------------------------------------------------
// ConvTranspose1D
// ---------------------------------------------------------------------------

/*! \brief 1-D transposed convolution. */
class ConvTranspose1DModuleNode : public NNModuleNode {
 public:
  NNParameter weight;
  ffi::Optional<NNParameter> bias;
  int64_t stride, padding, output_padding, dilation, groups;

  ConvTranspose1DModuleNode(NNParameter weight, ffi::Optional<NNParameter> bias, int64_t stride,
                            int64_t padding, int64_t output_padding, int64_t dilation,
                            int64_t groups)
      : weight(std::move(weight)),
        bias(std::move(bias)),
        stride(stride),
        padding(padding),
        output_padding(output_padding),
        dilation(dilation),
        groups(groups) {}

  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ConvTranspose1DModuleNode>()
        .def(refl::init<NNParameter, ffi::Optional<NNParameter>, int64_t, int64_t, int64_t, int64_t,
                        int64_t>())
        .def_ro("weight", &ConvTranspose1DModuleNode::weight)
        .def_ro("bias", &ConvTranspose1DModuleNode::bias)
        .def_ro("stride", &ConvTranspose1DModuleNode::stride)
        .def_ro("padding", &ConvTranspose1DModuleNode::padding)
        .def_ro("output_padding", &ConvTranspose1DModuleNode::output_padding)
        .def_ro("dilation", &ConvTranspose1DModuleNode::dilation)
        .def_ro("groups", &ConvTranspose1DModuleNode::groups)
        .def("_forward", &ConvTranspose1DModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.ConvTranspose1D", ConvTranspose1DModuleNode,
                                    NNModuleNode);
};
class ConvTranspose1DModule : public runtime::ObjectRef {
 public:
  explicit ConvTranspose1DModule(NNParameter weight, ffi::Optional<NNParameter> bias,
                                 int64_t stride, int64_t padding, int64_t output_padding,
                                 int64_t dilation, int64_t groups);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ConvTranspose1DModule, runtime::ObjectRef,
                                                ConvTranspose1DModuleNode);
};
/*! \brief Factory: create a ConvTranspose1D module. */
ConvTranspose1DModule MakeConvTranspose1D(ffi::Any in_channels, ffi::Any out_channels,
                                          ffi::Any kernel_size, int64_t stride, int64_t padding,
                                          int64_t output_padding, int64_t dilation, int64_t groups,
                                          bool has_bias, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/*! \brief Identity pass-through module. */
class IdentityModuleNode : public NNModuleNode {
 public:
  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IdentityModuleNode>()
        .def(refl::init<>())
        .def("_forward", &IdentityModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Identity", IdentityModuleNode, NNModuleNode);
};
class IdentityModule : public runtime::ObjectRef {
 public:
  explicit IdentityModule();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IdentityModule, runtime::ObjectRef,
                                                IdentityModuleNode);
};

// ---------------------------------------------------------------------------
// EffectNode
// ---------------------------------------------------------------------------

/*!
 * \brief Abstract base class for side-effecting state objects.
 *
 * Concrete subclasses (IOEffectModuleNode, KVCacheModuleNode) implement
 * the four protocol methods used by the exporter to manage effect state
 * across function boundaries.
 *
 * Inheriting from NNModuleNode allows effects to be stored in attrs and
 * passed through ModuleSpec as runtime::ObjectRef.  NamedParameters()
 * skips EffectNode subclasses since they carry no trainable parameters.
 *
 * The exporter detects effects via IsInstance<EffectNode>() and calls
 * the virtual protocol methods directly.
 */
class EffectNode : public NNModuleNode {
 public:
  /*! \brief Emit the initialisation expression into \p bb; return the state Vars. */
  virtual ffi::Array<Var> EmitInit(ffi::String name_hint, BlockBuilder bb) const = 0;
  /*! \brief Create placeholder state Vars and store them internally. */
  virtual ffi::Array<Var> Create(ffi::String name_hint) = 0;
  /*! \brief Restore internal state from previously created Vars. */
  virtual void SetState(ffi::Array<Var> state_vars) = 0;
  /*! \brief Return the current state Vars and clear internal state. */
  virtual ffi::Array<Var> Finalize() = 0;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<EffectNode>()
        .def("_cpp_emit_init", &EffectNode::EmitInit)
        .def("_cpp_create", &EffectNode::Create)
        .def("_cpp_set_state", &EffectNode::SetState)
        .def("_cpp_finalize", &EffectNode::Finalize);
  }
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.frontend.nn.Effect", EffectNode, NNModuleNode);
};
// EffectNode is abstract; concrete subclasses are held via IOEffectModule
// or KVCacheModule (or as runtime::ObjectRef).

// ---------------------------------------------------------------------------
// IOEffect
// ---------------------------------------------------------------------------

/*!
 * \brief Models the IO side-effect token.
 *
 * Tracks a single Var representing the IO effect state.  The exporter
 * threads this token through every method that carries side effects.
 */
class IOEffectModuleNode : public EffectNode {
 public:
  /*! \brief The current IO effect Var, or nullopt when not active. */
  ffi::Optional<Var> effect;

  IOEffectModuleNode() = default;

  ffi::Array<Var> EmitInit(ffi::String name_hint, BlockBuilder bb) const override;
  ffi::Array<Var> Create(ffi::String name_hint) override;
  void SetState(ffi::Array<Var> state_vars) override;
  ffi::Array<Var> Finalize() override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IOEffectModuleNode>()
        .def(refl::init<>())
        .def_rw("effect", &IOEffectModuleNode::effect);
    // _cpp_emit_init / _cpp_create / _cpp_set_state / _cpp_finalize
    // are inherited from EffectNode::RegisterReflection().
  }
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.IOEffect", IOEffectModuleNode, EffectNode);
};
class IOEffectModule : public runtime::ObjectRef {
 public:
  explicit IOEffectModule();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IOEffectModule, runtime::ObjectRef,
                                                IOEffectModuleNode);
};

// ---------------------------------------------------------------------------
// KVCache
// ---------------------------------------------------------------------------

/*!
 * \brief Attention KV-cache effect module.
 *
 * Holds the initial sequence length, per-token shape, and dtype for a
 * single KV-cache slot.  Implements the four EffectNode protocol methods
 * and provides View() and Append() for use inside Forward().
 */
class KVCacheModuleNode : public EffectNode {
 public:
  int64_t init_seq_len;
  /*! \brief Per-token shape dimensions. */
  ffi::Array<Integer> unit_shape;
  ffi::String dtype;
  /*! \brief Current cache Var (ObjectStructInfo), or nullopt when not active. */
  ffi::Optional<Var> cache;

  KVCacheModuleNode(int64_t init_seq_len, ffi::Array<Integer> unit_shape, ffi::String dtype)
      : init_seq_len(init_seq_len), unit_shape(std::move(unit_shape)), dtype(std::move(dtype)) {}

  ffi::Array<Var> EmitInit(ffi::String name_hint, BlockBuilder bb) const override;
  ffi::Array<Var> Create(ffi::String name_hint) override;
  void SetState(ffi::Array<Var> state_vars) override;
  ffi::Array<Var> Finalize() override;
  void To(ffi::String new_dtype);
  NNTensor View(PrimExpr seq_len) const;
  void Append(NNTensor new_element);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<KVCacheModuleNode>()
        .def(refl::init<int64_t, ffi::Array<Integer>, ffi::String>())
        .def_ro("init_seq_len", &KVCacheModuleNode::init_seq_len)
        .def_ro("unit_shape", &KVCacheModuleNode::unit_shape)
        .def_rw("dtype", &KVCacheModuleNode::dtype)
        .def_rw("cache", &KVCacheModuleNode::cache)
        .def("_cpp_to", &KVCacheModuleNode::To)
        .def("_view", &KVCacheModuleNode::View)
        .def("_append", &KVCacheModuleNode::Append);
    // _cpp_emit_init / _cpp_create / _cpp_set_state / _cpp_finalize
    // are inherited from EffectNode::RegisterReflection().
  }
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.KVCache", KVCacheModuleNode, EffectNode);
};
class KVCacheModule : public runtime::ObjectRef {
 public:
  explicit KVCacheModule(int64_t init_seq_len, ffi::Array<Integer> unit_shape, ffi::String dtype);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(KVCacheModule, runtime::ObjectRef,
                                                KVCacheModuleNode);
};

// ---------------------------------------------------------------------------
// Timesteps
// ---------------------------------------------------------------------------

/*! \brief Sinusoidal timestep embedding module. */
class TimestepsModuleNode : public NNModuleNode {
 public:
  int64_t num_channels;
  bool flip_sin_to_cos;
  double downscale_freq_shift;

  TimestepsModuleNode(int64_t num_channels, bool flip_sin_to_cos, double downscale_freq_shift)
      : num_channels(num_channels),
        flip_sin_to_cos(flip_sin_to_cos),
        downscale_freq_shift(downscale_freq_shift) {}

  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TimestepsModuleNode>()
        .def(refl::init<int64_t, bool, double>())
        .def_ro("num_channels", &TimestepsModuleNode::num_channels)
        .def_ro("flip_sin_to_cos", &TimestepsModuleNode::flip_sin_to_cos)
        .def_ro("downscale_freq_shift", &TimestepsModuleNode::downscale_freq_shift)
        .def("_forward", &TimestepsModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Timesteps", TimestepsModuleNode,
                                    NNModuleNode);
};
class TimestepsModule : public runtime::ObjectRef {
 public:
  explicit TimestepsModule(int64_t num_channels, bool flip_sin_to_cos, double downscale_freq_shift);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TimestepsModule, runtime::ObjectRef,
                                                TimestepsModuleNode);
};

// ---------------------------------------------------------------------------
// TimestepEmbedding
// ---------------------------------------------------------------------------

/*! \brief Two-layer MLP that projects timestep embeddings. */
class TimestepEmbeddingModuleNode : public NNModuleNode {
 public:
  LinearModule linear_1;
  ffi::Optional<LinearModule> cond_proj;
  SiLUModule act;
  LinearModule linear_2;
  ffi::Optional<SiLUModule> post_act;

  TimestepEmbeddingModuleNode(LinearModule linear_1, ffi::Optional<LinearModule> cond_proj,
                              SiLUModule act, LinearModule linear_2,
                              ffi::Optional<SiLUModule> post_act)
      : linear_1(std::move(linear_1)),
        cond_proj(std::move(cond_proj)),
        act(std::move(act)),
        linear_2(std::move(linear_2)),
        post_act(std::move(post_act)) {}

  Var Forward(Var sample, ffi::Optional<Var> condition) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TimestepEmbeddingModuleNode>()
        .def(refl::init<LinearModule, ffi::Optional<LinearModule>, SiLUModule, LinearModule,
                        ffi::Optional<SiLUModule>>())
        .def_ro("linear_1", &TimestepEmbeddingModuleNode::linear_1)
        .def_ro("cond_proj", &TimestepEmbeddingModuleNode::cond_proj)
        .def_ro("act", &TimestepEmbeddingModuleNode::act)
        .def_ro("linear_2", &TimestepEmbeddingModuleNode::linear_2)
        .def_ro("post_act", &TimestepEmbeddingModuleNode::post_act)
        .def("_forward", &TimestepEmbeddingModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.TimestepEmbedding",
                                    TimestepEmbeddingModuleNode, NNModuleNode);
};
class TimestepEmbeddingModule : public runtime::ObjectRef {
 public:
  explicit TimestepEmbeddingModule(LinearModule linear_1, ffi::Optional<LinearModule> cond_proj,
                                   SiLUModule act, LinearModule linear_2,
                                   ffi::Optional<SiLUModule> post_act);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TimestepEmbeddingModule, runtime::ObjectRef,
                                                TimestepEmbeddingModuleNode);
};
TimestepEmbeddingModule MakeTimestepEmbedding(int64_t in_channels, int64_t time_embed_dim,
                                              ffi::String act_fn, ffi::Optional<int64_t> out_dim,
                                              ffi::Optional<ffi::String> post_act_fn,
                                              ffi::Optional<int64_t> cond_proj_dim);

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/*!
 * \brief Multi-head attention module with optional cross-attention and
 *        group-norm pre-conditioning.
 */
class AttentionModuleNode : public NNModuleNode {
 public:
  int64_t heads;
  int64_t inner_dim;
  LinearModule to_q;
  LinearModule to_k;
  LinearModule to_v;
  ffi::Optional<GroupNormModule> group_norm;
  /*! \brief Output projection sub-modules; to_out[0] is a Linear layer. */
  ModuleList to_out;

  AttentionModuleNode(int64_t heads, int64_t inner_dim, LinearModule to_q, LinearModule to_k,
                      LinearModule to_v, ffi::Optional<GroupNormModule> group_norm,
                      ModuleList to_out)
      : heads(heads),
        inner_dim(inner_dim),
        to_q(std::move(to_q)),
        to_k(std::move(to_k)),
        to_v(std::move(to_v)),
        group_norm(std::move(group_norm)),
        to_out(std::move(to_out)) {}

  /*!
   * \brief Compute attention output.
   *
   * \param hidden_states          Query input tensor.
   * \param encoder_hidden_states  Key/value input for cross-attention,
   *                               or nullopt for self-attention.
   * \return                       Output tensor after projection.
   */
  Var Forward(Var hidden_states, ffi::Optional<Var> encoder_hidden_states) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AttentionModuleNode>()
        .def(refl::init<int64_t, int64_t, LinearModule, LinearModule, LinearModule,
                        ffi::Optional<GroupNormModule>, ModuleList>())
        .def_ro("heads", &AttentionModuleNode::heads)
        .def_ro("inner_dim", &AttentionModuleNode::inner_dim)
        .def_ro("to_q", &AttentionModuleNode::to_q)
        .def_ro("to_k", &AttentionModuleNode::to_k)
        .def_ro("to_v", &AttentionModuleNode::to_v)
        .def_ro("group_norm", &AttentionModuleNode::group_norm)
        .def_ro("to_out", &AttentionModuleNode::to_out)
        .def("_forward", &AttentionModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Attention", AttentionModuleNode,
                                    NNModuleNode);
};
class AttentionModule : public runtime::ObjectRef {
 public:
  explicit AttentionModule(int64_t heads, int64_t inner_dim, LinearModule to_q, LinearModule to_k,
                           LinearModule to_v, ffi::Optional<GroupNormModule> group_norm,
                           ModuleList to_out);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AttentionModule, runtime::ObjectRef,
                                                AttentionModuleNode);
};
AttentionModule MakeAttention(int64_t query_dim, ffi::Optional<int64_t> cross_attention_dim,
                              int64_t heads, int64_t dim_head, bool bias,
                              ffi::Optional<int64_t> norm_num_groups, bool out_bias);

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_MODULES_H_
