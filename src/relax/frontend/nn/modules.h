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
 *   - def_ro / def_rw fields for direct attribute access from Python.
 *   - refl::init<...>() for construction via the FFI.
 *   - A Forward() method that emits relax ops into the current BlockBuilder
 *     and returns the output relax::Var.
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
 * Each element of \p shape may be int64 (static dimension), String
 * (symbolic variable name), or PrimExpr.
 *
 * \param shape  Shape specification array.
 * \param dtype  Data type string (e.g. "float32").
 * \return       A new unbound NNParameter.
 */
NNParameter MakeParam(ffi::Array<ffi::Any> shape, ffi::String dtype);

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

/*! \brief Rectified linear unit activation: output = max(0, x). */
class ReLUModuleNode : public NNModuleNode {
 public:
  /*!
   * \brief Apply ReLU element-wise.
   * \param x  Input tensor.
   * \return   Output tensor with negative values clamped to zero.
   */
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

/*! \brief Sigmoid linear unit activation: output = x * sigmoid(x). */
class SiLUModuleNode : public NNModuleNode {
 public:
  /*!
   * \brief Apply SiLU element-wise.
   * \param x  Input tensor.
   * \return   Output tensor after applying x * sigmoid(x).
   */
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

/*!
 * \brief Gaussian error linear unit activation.
 *
 * Supports two approximation modes:
 *   - ""     (empty string): exact GELU using the error function.
 *   - "tanh": fast tanh approximation.
 */
class GELUModuleNode : public NNModuleNode {
 public:
  /*! \brief Approximation method: "" for exact erf-based GELU, "tanh" for tanh approximation. */
  ffi::String approximate;

  explicit GELUModuleNode(ffi::String approximate = "") : approximate(std::move(approximate)) {}

  /*!
   * \brief Apply GELU element-wise.
   * \param x  Input tensor.
   * \return   Output tensor after applying the GELU activation.
   */
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

/*!
 * \brief Fully-connected linear transformation: output = x @ weight^T + bias.
 *
 * Weight shape is [out_features, in_features]; bias shape is [out_features].
 * When out_dtype is set the matmul accumulates in that dtype.
 */
class LinearModuleNode : public NNModuleNode {
 public:
  /*! \brief Weight matrix; shape [out_features, in_features]. */
  NNParameter weight;
  /*! \brief Bias vector; shape [out_features], or nullopt when bias=False. */
  ffi::Optional<NNParameter> bias;
  /*! \brief Optional output dtype override for the matmul accumulator. */
  ffi::Optional<ffi::String> out_dtype;

  LinearModuleNode(NNParameter weight, ffi::Optional<NNParameter> bias,
                   ffi::Optional<ffi::String> out_dtype)
      : weight(std::move(weight)), bias(std::move(bias)), out_dtype(std::move(out_dtype)) {}

  /*!
   * \brief Apply the linear transformation.
   *
   * Emits permute_dims(weight) followed by matmul(x, w_T), then adds
   * the bias if present.
   *
   * \param x  Input tensor of shape [..., in_features].
   * \return   Output tensor of shape [..., out_features].
   */
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
 * \param in_features   Input feature count (int64 or symbolic PrimExpr).
 * \param out_features  Output feature count (int64 or symbolic PrimExpr).
 * \param bias          Whether to include a bias parameter.
 * \param dtype         Weight dtype; defaults to the current default dtype when nullopt.
 * \param out_dtype     Optional output dtype override for the matmul accumulator.
 * \return              A fully constructed LinearModule.
 */
LinearModule MakeLinear(ffi::Any in_features, ffi::Any out_features, bool bias,
                        ffi::Optional<ffi::String> dtype, ffi::Optional<ffi::String> out_dtype);

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/*!
 * \brief Lookup-table embedding layer.
 *
 * Maps integer indices to dense vectors by indexing into a weight table
 * of shape [num_embeddings, embedding_dim].
 */
class EmbeddingModuleNode : public NNModuleNode {
 public:
  /*! \brief Embedding weight table; shape [num_embeddings, embedding_dim]. */
  NNParameter weight;

  explicit EmbeddingModuleNode(NNParameter weight) : weight(std::move(weight)) {}

  /*!
   * \brief Look up embeddings for the given indices.
   *
   * When \p out_shape_if_nd is non-empty the input is first flattened,
   * looked up, then reshaped to \p out_shape_if_nd.
   *
   * \param x               Integer index tensor.
   * \param out_shape_if_nd Target output shape for N-D inputs; empty for 1-D.
   * \return                Embedding tensor of shape [..., embedding_dim].
   */
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
 * \param num    Vocabulary size (int64 or symbolic PrimExpr).
 * \param dim    Embedding dimension (int64 or symbolic PrimExpr).
 * \param dtype  Weight dtype; defaults to the current default dtype when nullopt.
 * \return       A fully constructed EmbeddingModule.
 */
EmbeddingModule MakeEmbedding(ffi::Any num, ffi::Any dim, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

/*!
 * \brief Layer normalisation over the last N dimensions.
 *
 * Normalises the input over the axes specified by \p axes, then applies
 * an optional learnable affine transformation (gamma scale + beta shift).
 */
class LayerNormModuleNode : public NNModuleNode {
 public:
  /*! \brief Scale parameter (gamma); nullopt when elementwise_affine=False. */
  ffi::Optional<NNParameter> weight;
  /*! \brief Shift parameter (beta); nullopt when elementwise_affine=False. */
  ffi::Optional<NNParameter> bias;
  /*! \brief Normalisation axes (typically negative indices, e.g. [-1]). */
  ffi::Array<Integer> axes;
  /*! \brief Small constant added to the denominator for numerical stability. */
  double epsilon;
  /*! \brief Whether learnable affine parameters are included. */
  bool elementwise_affine;

  LayerNormModuleNode(ffi::Optional<NNParameter> weight, ffi::Optional<NNParameter> bias,
                      ffi::Array<Integer> axes, double epsilon, bool elementwise_affine)
      : weight(std::move(weight)),
        bias(std::move(bias)),
        axes(std::move(axes)),
        epsilon(epsilon),
        elementwise_affine(elementwise_affine) {}

  /*!
   * \brief Apply layer normalisation.
   * \param x  Input tensor.
   * \return   Normalised (and optionally affine-transformed) output tensor.
   */
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
 * \param normalized_shape  int64 scalar or Array<Any> of the normalised dimensions.
 * \param eps               Small constant for numerical stability.
 * \param elementwise_affine  Whether to include learnable affine parameters.
 * \param dtype             Parameter dtype; defaults to the current default dtype when nullopt.
 * \return                  A fully constructed LayerNormModule.
 */
LayerNormModule MakeLayerNorm(ffi::Any normalized_shape, double eps, bool elementwise_affine,
                              ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/*!
 * \brief Root-mean-square layer normalisation.
 *
 * Normalises the input by its RMS over the specified axes, then scales
 * by a learnable weight.  Unlike LayerNorm there is no mean subtraction.
 */
class RMSNormModuleNode : public NNModuleNode {
 public:
  /*! \brief Learnable scale parameter. */
  NNParameter weight;
  /*! \brief Optional learnable bias parameter. */
  ffi::Optional<NNParameter> bias;
  /*! \brief Normalisation axes. */
  ffi::Array<Integer> axes;
  /*! \brief Small constant added to the denominator for numerical stability. */
  double epsilon;

  RMSNormModuleNode(NNParameter weight, ffi::Optional<NNParameter> bias, ffi::Array<Integer> axes,
                    double epsilon)
      : weight(std::move(weight)), bias(std::move(bias)), axes(std::move(axes)), epsilon(epsilon) {}

  /*!
   * \brief Apply RMS normalisation.
   * \param x  Input tensor.
   * \return   RMS-normalised and scaled (and optionally biased) output tensor.
   */
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
 * \param hidden_size  Hidden dimension size (int64 or symbolic PrimExpr).
 * \param axes         Normalisation axes.
 * \param epsilon      Small constant for numerical stability.
 * \param has_bias     Whether to include a learnable bias parameter.
 * \param dtype        Parameter dtype; defaults to the current default dtype when nullopt.
 * \return             A fully constructed RMSNormModule.
 */
RMSNormModule MakeRMSNorm(ffi::Any hidden_size, ffi::Array<Integer> axes, double epsilon,
                          bool has_bias, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------------

/*!
 * \brief Group normalisation.
 *
 * Divides the channels into \p num_groups groups and normalises each
 * group independently, then applies an optional learnable affine
 * transformation.
 */
class GroupNormModuleNode : public NNModuleNode {
 public:
  /*! \brief Number of channel groups. */
  int64_t num_groups;
  /*! \brief Learnable scale parameter; nullopt when affine=False. */
  ffi::Optional<NNParameter> weight;
  /*! \brief Learnable bias parameter; nullopt when affine=False. */
  ffi::Optional<NNParameter> bias;
  /*! \brief Small constant added to the denominator for numerical stability. */
  double epsilon;

  GroupNormModuleNode(int64_t num_groups, ffi::Optional<NNParameter> weight,
                      ffi::Optional<NNParameter> bias, double epsilon)
      : num_groups(num_groups),
        weight(std::move(weight)),
        bias(std::move(bias)),
        epsilon(epsilon) {}

  /*!
   * \brief Apply group normalisation.
   *
   * \param x             Input tensor.
   * \param channel_axis  Axis index of the channel dimension.
   * \param axes          Axes over which to compute the group statistics.
   * \return              Normalised (and optionally affine-transformed) output tensor.
   */
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
 * \param num_groups   Number of channel groups.
 * \param num_channels Channel count (int64 or symbolic PrimExpr).
 * \param eps          Small constant for numerical stability.
 * \param affine       Whether to include learnable affine parameters.
 * \param dtype        Parameter dtype; defaults to the current default dtype when nullopt.
 * \return             A fully constructed GroupNormModule.
 */
GroupNormModule MakeGroupNorm(int64_t num_groups, ffi::Any num_channels, double eps, bool affine,
                              ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// Conv1D
// ---------------------------------------------------------------------------

/*!
 * \brief 1-D convolution over a sequence of shape [N, C_in, L].
 *
 * Weight shape is [C_out, C_in/groups, kW].  An optional bias of shape
 * [C_out] is broadcast-added after the convolution.
 */
class Conv1DModuleNode : public NNModuleNode {
 public:
  /*! \brief Convolution weight; shape [C_out, C_in/groups, kW]. */
  NNParameter weight;
  /*! \brief Optional bias; shape [C_out], or nullopt when bias=False. */
  ffi::Optional<NNParameter> bias;
  /*! \brief Stride along the length dimension. */
  int64_t stride;
  /*! \brief Zero-padding added to both sides of the input. */
  int64_t padding;
  /*! \brief Dilation factor for the kernel. */
  int64_t dilation;
  /*! \brief Number of blocked connections from input to output channels. */
  int64_t groups;

  Conv1DModuleNode(NNParameter weight, ffi::Optional<NNParameter> bias, int64_t stride,
                   int64_t padding, int64_t dilation, int64_t groups)
      : weight(std::move(weight)),
        bias(std::move(bias)),
        stride(stride),
        padding(padding),
        dilation(dilation),
        groups(groups) {}

  /*!
   * \brief Apply 1-D convolution.
   * \param x  Input tensor of shape [N, C_in, L].
   * \return   Output tensor of shape [N, C_out, L_out].
   */
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
/*!
 * \brief Factory: create a Conv1D module with the given dimensions.
 *
 * \param in_channels   Number of input channels (int64 or symbolic).
 * \param out_channels  Number of output channels (int64 or symbolic).
 * \param kernel_size   Kernel width (int64 or symbolic).
 * \param stride        Stride along the length dimension.
 * \param padding       Zero-padding added to both sides.
 * \param dilation      Kernel dilation factor.
 * \param groups        Number of blocked channel connections.
 * \param has_bias      Whether to include a bias parameter.
 * \param dtype         Weight dtype; defaults to the current default dtype when nullopt.
 * \return              A fully constructed Conv1DModule.
 */
Conv1DModule MakeConv1D(ffi::Any in_channels, ffi::Any out_channels, ffi::Any kernel_size,
                        int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                        bool has_bias, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// Conv2D
// ---------------------------------------------------------------------------

/*!
 * \brief 2-D convolution over a spatial feature map.
 *
 * Supports both NCHW and NHWC data layouts.  Weight shape is
 * [C_out, C_in/groups, kH, kW] for NCHW (OIHW kernel layout) and
 * [C_out, C_in/groups, kH, kW] for NHWC (HWIO kernel layout).
 */
class Conv2DModuleNode : public NNModuleNode {
 public:
  /*! \brief Convolution weight. */
  NNParameter weight;
  /*! \brief Optional bias; shape [C_out], or nullopt when bias=False. */
  ffi::Optional<NNParameter> bias;
  /*! \brief Stride along the spatial dimensions. */
  int64_t stride;
  /*! \brief Zero-padding added to all spatial sides. */
  int64_t padding;
  /*! \brief Kernel dilation factor. */
  int64_t dilation;
  /*! \brief Number of blocked channel connections. */
  int64_t groups;
  /*! \brief Data layout string: "NCHW" or "NHWC". */
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

  /*!
   * \brief Apply 2-D convolution.
   * \param x  Input tensor of shape [N, C_in, H, W] (NCHW) or [N, H, W, C_in] (NHWC).
   * \return   Output tensor of shape [N, C_out, H_out, W_out] or [N, H_out, W_out, C_out].
   */
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
/*!
 * \brief Factory: create a Conv2D module; expands a scalar kernel_size to [kH, kW].
 *
 * \param in_channels   Number of input channels (int64 or symbolic).
 * \param out_channels  Number of output channels (int64 or symbolic).
 * \param kernel_size   Kernel spatial size as [kH, kW].
 * \param stride        Stride along the spatial dimensions.
 * \param padding       Zero-padding added to all spatial sides.
 * \param dilation      Kernel dilation factor.
 * \param groups        Number of blocked channel connections.
 * \param has_bias      Whether to include a bias parameter.
 * \param dtype         Weight dtype; defaults to the current default dtype when nullopt.
 * \param data_layout   Data layout string: "NCHW" or "NHWC".
 * \return              A fully constructed Conv2DModule.
 */
Conv2DModule MakeConv2D(ffi::Any in_channels, ffi::Any out_channels,
                        ffi::Array<Integer> kernel_size, int64_t stride, int64_t padding,
                        int64_t dilation, int64_t groups, bool has_bias,
                        ffi::Optional<ffi::String> dtype, ffi::String data_layout);

// ---------------------------------------------------------------------------
// Conv3D
// ---------------------------------------------------------------------------

/*!
 * \brief 3-D convolution over a volumetric feature map.
 *
 * Supports NCDHW and NDHWC data layouts.  Weight shape is
 * [C_out, C_in/groups, kD, kH, kW] (OIDHW kernel layout).
 */
class Conv3DModuleNode : public NNModuleNode {
 public:
  /*! \brief Convolution weight. */
  NNParameter weight;
  /*! \brief Optional bias; shape [C_out], or nullopt when bias=False. */
  ffi::Optional<NNParameter> bias;
  /*! \brief Stride along the volumetric dimensions. */
  int64_t stride;
  /*! \brief Zero-padding added to all volumetric sides. */
  int64_t padding;
  /*! \brief Kernel dilation factor. */
  int64_t dilation;
  /*! \brief Number of blocked channel connections. */
  int64_t groups;
  /*! \brief Data layout string: "NCDHW" or "NDHWC". */
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

  /*!
   * \brief Apply 3-D convolution.
   * \param x  Input tensor of shape [N, C_in, D, H, W] (NCDHW) or [N, D, H, W, C_in] (NDHWC).
   * \return   Output tensor of shape [N, C_out, D_out, H_out, W_out] or equivalent.
   */
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
/*!
 * \brief Factory: create a Conv3D module; expands a scalar kernel_size to [kD, kH, kW].
 *
 * \param in_channels   Number of input channels (int64 or symbolic).
 * \param out_channels  Number of output channels (int64 or symbolic).
 * \param kernel_size   Kernel volumetric size as [kD, kH, kW].
 * \param stride        Stride along the volumetric dimensions.
 * \param padding       Zero-padding added to all volumetric sides.
 * \param dilation      Kernel dilation factor.
 * \param groups        Number of blocked channel connections.
 * \param has_bias      Whether to include a bias parameter.
 * \param dtype         Weight dtype; defaults to the current default dtype when nullopt.
 * \param data_layout   Data layout string: "NCDHW" or "NDHWC".
 * \return              A fully constructed Conv3DModule.
 */
Conv3DModule MakeConv3D(ffi::Any in_channels, ffi::Any out_channels,
                        ffi::Array<Integer> kernel_size, int64_t stride, int64_t padding,
                        int64_t dilation, int64_t groups, bool has_bias,
                        ffi::Optional<ffi::String> dtype, ffi::String data_layout);

// ---------------------------------------------------------------------------
// ConvTranspose1D
// ---------------------------------------------------------------------------

/*!
 * \brief 1-D transposed convolution (fractionally-strided convolution).
 *
 * Computes the gradient of a Conv1D with respect to its input, effectively
 * upsampling the sequence.  Weight shape is [C_in, C_out/groups, kW] (IOW
 * kernel layout).
 */
class ConvTranspose1DModuleNode : public NNModuleNode {
 public:
  /*! \brief Transposed convolution weight; shape [C_in, C_out/groups, kW]. */
  NNParameter weight;
  /*! \brief Optional bias; shape [C_out], or nullopt when bias=False. */
  ffi::Optional<NNParameter> bias;
  /*! \brief Stride (upsampling factor). */
  int64_t stride;
  /*! \brief Input zero-padding (removed from output). */
  int64_t padding;
  /*! \brief Additional output padding to resolve output size ambiguity. */
  int64_t output_padding;
  /*! \brief Kernel dilation factor. */
  int64_t dilation;
  /*! \brief Number of blocked channel connections. */
  int64_t groups;

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

  /*!
   * \brief Apply 1-D transposed convolution.
   * \param x  Input tensor of shape [N, C_in, L].
   * \return   Output tensor of shape [N, C_out, L_out].
   */
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
/*!
 * \brief Factory: create a ConvTranspose1D module.
 *
 * \param in_channels    Number of input channels (int64 or symbolic).
 * \param out_channels   Number of output channels (int64 or symbolic).
 * \param kernel_size    Kernel width (int64 or symbolic).
 * \param stride         Stride (upsampling factor).
 * \param padding        Input zero-padding (removed from output).
 * \param output_padding Additional output padding to resolve size ambiguity.
 * \param dilation       Kernel dilation factor.
 * \param groups         Number of blocked channel connections.
 * \param has_bias       Whether to include a bias parameter.
 * \param dtype          Weight dtype; defaults to the current default dtype when nullopt.
 * \return               A fully constructed ConvTranspose1DModule.
 */
ConvTranspose1DModule MakeConvTranspose1D(ffi::Any in_channels, ffi::Any out_channels,
                                          ffi::Any kernel_size, int64_t stride, int64_t padding,
                                          int64_t output_padding, int64_t dilation, int64_t groups,
                                          bool has_bias, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/*! \brief Identity pass-through module: output = input, no parameters. */
class IdentityModuleNode : public NNModuleNode {
 public:
  /*!
   * \brief Return the input unchanged.
   * \param x  Input tensor.
   * \return   The same tensor \p x.
   */
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
  /*!
   * \brief Emit the initialisation expression into \p bb.
   *
   * Called once per export to create the initial effect state objects.
   *
   * \param name_hint  Name hint for the emitted binding.
   * \param bb         The active BlockBuilder.
   * \return           Array of Vars representing the initial effect state.
   */
  virtual ffi::Array<Var> EmitInit(ffi::String name_hint, BlockBuilder bb) const = 0;

  /*!
   * \brief Create placeholder state Vars and store them internally.
   *
   * Called by the exporter to allocate function-argument Vars for the
   * effect state before the dataflow block is opened.
   *
   * \param name_hint  Name hint for the created Vars.
   * \return           Array of newly created placeholder Vars.
   */
  virtual ffi::Array<Var> Create(ffi::String name_hint) = 0;

  /*!
   * \brief Restore internal state from previously created Vars.
   *
   * \param state_vars  Vars produced by a prior call to Create().
   */
  virtual void SetState(ffi::Array<Var> state_vars) = 0;

  /*!
   * \brief Return the current state Vars and clear internal state.
   *
   * Called after the dataflow block is closed to collect the final
   * effect outputs for the function return value.
   *
   * \return  Array of current state Vars.
   */
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

/*!
 * \brief Sinusoidal timestep embedding module.
 *
 * Converts a scalar timestep tensor into a sinusoidal positional embedding
 * of dimension \p num_channels, following the formulation used in DDPM and
 * related diffusion models.
 */
class TimestepsModuleNode : public NNModuleNode {
 public:
  /*! \brief Output embedding dimension (must be even unless padding is applied). */
  int64_t num_channels;
  /*! \brief If true, concatenate [cos, sin] instead of [sin, cos]. */
  bool flip_sin_to_cos;
  /*! \brief Frequency downscale shift applied before exponentiation. */
  double downscale_freq_shift;

  TimestepsModuleNode(int64_t num_channels, bool flip_sin_to_cos, double downscale_freq_shift)
      : num_channels(num_channels),
        flip_sin_to_cos(flip_sin_to_cos),
        downscale_freq_shift(downscale_freq_shift) {}

  /*!
   * \brief Compute the sinusoidal timestep embedding.
   * \param x  1-D integer timestep tensor of shape [N].
   * \return   Embedding tensor of shape [N, num_channels].
   */
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

/*!
 * \brief Two-layer MLP that projects sinusoidal timestep embeddings.
 *
 * Architecture: linear_1 → act → (optional cond_proj added) → linear_2 → (optional post_act).
 * Mirrors the HuggingFace Diffusers TimestepEmbedding class.
 */
class TimestepEmbeddingModuleNode : public NNModuleNode {
 public:
  /*! \brief First linear projection. */
  LinearModule linear_1;
  /*! \brief Optional conditioning projection added after linear_1. */
  ffi::Optional<LinearModule> cond_proj;
  /*! \brief Activation applied between the two linear layers. */
  SiLUModule act;
  /*! \brief Second linear projection. */
  LinearModule linear_2;
  /*! \brief Optional post-activation applied after linear_2. */
  ffi::Optional<SiLUModule> post_act;

  TimestepEmbeddingModuleNode(LinearModule linear_1, ffi::Optional<LinearModule> cond_proj,
                              SiLUModule act, LinearModule linear_2,
                              ffi::Optional<SiLUModule> post_act)
      : linear_1(std::move(linear_1)),
        cond_proj(std::move(cond_proj)),
        act(std::move(act)),
        linear_2(std::move(linear_2)),
        post_act(std::move(post_act)) {}

  /*!
   * \brief Project the timestep embedding.
   *
   * \param sample     Input embedding tensor of shape [N, in_channels].
   * \param condition  Optional conditioning tensor added after linear_1.
   * \return           Output tensor of shape [N, time_embed_dim] (or out_dim if set).
   */
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
/*!
 * \brief Factory: create a TimestepEmbedding module.
 *
 * \param in_channels      Input embedding dimension.
 * \param time_embed_dim   Hidden and default output dimension.
 * \param act_fn           Activation function name (currently only "silu" is supported).
 * \param out_dim          Optional output dimension override; defaults to time_embed_dim.
 * \param post_act_fn      Optional post-activation function name; nullopt for none.
 * \param cond_proj_dim    Optional conditioning projection input dimension; nullopt for none.
 * \return                 A fully constructed TimestepEmbeddingModule.
 */
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
/*!
 * \brief Factory: create an Attention module.
 *
 * \param query_dim            Dimension of the query input.
 * \param cross_attention_dim  Key/value input dimension for cross-attention;
 *                             nullopt for self-attention (uses query_dim).
 * \param heads                Number of attention heads.
 * \param dim_head             Dimension per attention head.
 * \param bias                 Whether to include bias in the Q/K/V projections.
 * \param norm_num_groups      If set, prepend a GroupNorm with this many groups.
 * \param out_bias             Whether to include bias in the output projection.
 * \return                     A fully constructed AttentionModule.
 */
AttentionModule MakeAttention(int64_t query_dim, ffi::Optional<int64_t> cross_attention_dim,
                              int64_t heads, int64_t dim_head, bool bias,
                              ffi::Optional<int64_t> norm_num_groups, bool out_bias);

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_MODULES_H_
