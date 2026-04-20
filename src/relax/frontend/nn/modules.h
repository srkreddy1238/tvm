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
 * \brief Native C++ Object definitions for the built-in nn.Module subclasses.
 *
 * Design:
 *   Every XxxModuleNode owns ALL its members natively in C++:
 *     - Trainable weights/biases are stored as NNParameter (ParameterNode*)
 *       so that Python's named_parameters() traversal can reach them via
 *       the FFI field accessors without any Python-side shadow copies.
 *     - Scalar hyper-parameters (epsilon, stride, …) are stored as their
 *       natural C++ types (double, int64_t, String).
 *
 *   Each node exposes:
 *     - def_ro / def_rw fields  → direct Python attribute access
 *     - refl::init<...>()       → __init_handle_by_constructor__ in Python
 *     - forward() method        → graph-building, returns relax.Var
 *
 *   A companion Make* global function per module handles all the
 *   shape-arithmetic that used to live in Python __init__ (e.g. computing
 *   kernel_shape for Conv2D) and returns the fully-constructed ObjectRef.
 *   Python __init__ is then a single FFI call.
 */

#ifndef TVM_RELAX_FRONTEND_NN_MODULES_H_
#define TVM_RELAX_FRONTEND_NN_MODULES_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>
#include <tvm/runtime/object.h>

#include <string>

#include "core.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ===========================================================================
// Helper: build a ParameterNode from shape + dtype, return as NNParameter.
// Used by all Make* factory functions below.
// ===========================================================================
NNParameter MakeParam(ffi::Array<ffi::Any> shape, ffi::String dtype);

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

class ReLUModuleNode : public runtime::Object {
 public:
  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReLUModuleNode>().def(refl::init<>()).def("forward", &ReLUModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.ReLU", ReLUModuleNode, runtime::Object);
};
class ReLUModule : public runtime::ObjectRef {
 public:
  explicit ReLUModule();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ReLUModule, runtime::ObjectRef, ReLUModuleNode);
};

// ---------------------------------------------------------------------------
// SiLU
// ---------------------------------------------------------------------------

class SiLUModuleNode : public runtime::Object {
 public:
  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SiLUModuleNode>().def(refl::init<>()).def("forward", &SiLUModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.SiLU", SiLUModuleNode, runtime::Object);
};
class SiLUModule : public runtime::ObjectRef {
 public:
  explicit SiLUModule();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SiLUModule, runtime::ObjectRef, SiLUModuleNode);
};

// ---------------------------------------------------------------------------
// GELU
// ---------------------------------------------------------------------------

class GELUModuleNode : public runtime::Object {
 public:
  ffi::String approximate;  //!< "" (exact) or "tanh"

  explicit GELUModuleNode(ffi::String approximate = "") : approximate(std::move(approximate)) {}
  Var Forward(Var x) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GELUModuleNode>()
        .def(refl::init<ffi::String>())
        .def_ro("approximate", &GELUModuleNode::approximate)
        .def("forward", &GELUModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.GELU", GELUModuleNode, runtime::Object);
};
class GELUModule : public runtime::ObjectRef {
 public:
  explicit GELUModule(ffi::String approximate = "");
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GELUModule, runtime::ObjectRef, GELUModuleNode);
};

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

class LinearModuleNode : public runtime::Object {
 public:
  NNParameter weight;               //!< shape [out_features, in_features]
  ffi::Optional<NNParameter> bias;  //!< shape [out_features], or nullopt
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
        .def("forward", &LinearModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Linear", LinearModuleNode, runtime::Object);
};
class LinearModule : public runtime::ObjectRef {
 public:
  explicit LinearModule(NNParameter weight, ffi::Optional<NNParameter> bias,
                        ffi::Optional<ffi::String> out_dtype);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LinearModule, runtime::ObjectRef, LinearModuleNode);
};
/*! \brief Factory: creates Parameters internally, returns LinearModule. */
LinearModule MakeLinear(ffi::Any in_features, ffi::Any out_features, bool bias,
                        ffi::Optional<ffi::String> dtype, ffi::Optional<ffi::String> out_dtype);

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

class EmbeddingModuleNode : public runtime::Object {
 public:
  NNParameter weight;  //!< shape [num_embeddings, embedding_dim]

  explicit EmbeddingModuleNode(NNParameter weight) : weight(std::move(weight)) {}
  Var Forward(Var x, ffi::Array<ffi::Any> out_shape_if_nd) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<EmbeddingModuleNode>()
        .def(refl::init<NNParameter>())
        .def_ro("weight", &EmbeddingModuleNode::weight)
        .def("forward", &EmbeddingModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Embedding", EmbeddingModuleNode,
                                    runtime::Object);
};
class EmbeddingModule : public runtime::ObjectRef {
 public:
  explicit EmbeddingModule(NNParameter weight);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(EmbeddingModule, runtime::ObjectRef,
                                                EmbeddingModuleNode);
};
/*! \brief Factory: creates the weight Parameter internally. */
EmbeddingModule MakeEmbedding(ffi::Any num, ffi::Any dim, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

class LayerNormModuleNode : public runtime::Object {
 public:
  ffi::Optional<NNParameter> weight;  //!< gamma, or nullopt when !elementwise_affine
  ffi::Optional<NNParameter> bias;    //!< beta,  or nullopt when !elementwise_affine
  ffi::Array<Integer> axes;           //!< normalisation axes (negative)
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
        .def("forward", &LayerNormModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.LayerNorm", LayerNormModuleNode,
                                    runtime::Object);
};
class LayerNormModule : public runtime::ObjectRef {
 public:
  explicit LayerNormModule(ffi::Optional<NNParameter> weight, ffi::Optional<NNParameter> bias,
                           ffi::Array<Integer> axes, double epsilon, bool elementwise_affine);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LayerNormModule, runtime::ObjectRef,
                                                LayerNormModuleNode);
};
/*! \brief Factory: creates Parameters internally. */
LayerNormModule MakeLayerNorm(ffi::Any normalized_shape, double eps, bool elementwise_affine,
                              ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

class RMSNormModuleNode : public runtime::Object {
 public:
  NNParameter weight;
  ffi::Optional<NNParameter> bias;
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
        .def("forward", &RMSNormModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.RMSNorm", RMSNormModuleNode,
                                    runtime::Object);
};
class RMSNormModule : public runtime::ObjectRef {
 public:
  explicit RMSNormModule(NNParameter weight, ffi::Optional<NNParameter> bias,
                         ffi::Array<Integer> axes, double epsilon);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RMSNormModule, runtime::ObjectRef,
                                                RMSNormModuleNode);
};
/*! \brief Factory: creates Parameters internally. */
RMSNormModule MakeRMSNorm(int64_t hidden_size, ffi::Array<Integer> axes, double epsilon,
                          bool has_bias, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------------

class GroupNormModuleNode : public runtime::Object {
 public:
  int64_t num_groups;
  ffi::Optional<NNParameter> weight;
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
        .def("forward", &GroupNormModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.GroupNorm", GroupNormModuleNode,
                                    runtime::Object);
};
class GroupNormModule : public runtime::ObjectRef {
 public:
  explicit GroupNormModule(int64_t num_groups, ffi::Optional<NNParameter> weight,
                           ffi::Optional<NNParameter> bias, double epsilon);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GroupNormModule, runtime::ObjectRef,
                                                GroupNormModuleNode);
};
/*! \brief Factory: creates Parameters internally. */
GroupNormModule MakeGroupNorm(int64_t num_groups, int64_t num_channels, double eps, bool affine,
                              ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// Conv1D
// ---------------------------------------------------------------------------

class Conv1DModuleNode : public runtime::Object {
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
        .def("forward", &Conv1DModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Conv1D", Conv1DModuleNode, runtime::Object);
};
class Conv1DModule : public runtime::ObjectRef {
 public:
  explicit Conv1DModule(NNParameter weight, ffi::Optional<NNParameter> bias, int64_t stride,
                        int64_t padding, int64_t dilation, int64_t groups);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Conv1DModule, runtime::ObjectRef, Conv1DModuleNode);
};
/*! \brief Factory: creates Parameters internally. */
Conv1DModule MakeConv1D(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                        int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                        bool has_bias, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// Conv2D
// ---------------------------------------------------------------------------

class Conv2DModuleNode : public runtime::Object {
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
        .def("forward", &Conv2DModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Conv2D", Conv2DModuleNode, runtime::Object);
};
class Conv2DModule : public runtime::ObjectRef {
 public:
  explicit Conv2DModule(NNParameter weight, ffi::Optional<NNParameter> bias, int64_t stride,
                        int64_t padding, int64_t dilation, int64_t groups, ffi::String data_layout);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Conv2DModule, runtime::ObjectRef, Conv2DModuleNode);
};
/*! \brief Factory: creates Parameters internally, handles kernel_size expansion. */
Conv2DModule MakeConv2D(int64_t in_channels, int64_t out_channels, ffi::Array<Integer> kernel_size,
                        int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                        bool has_bias, ffi::Optional<ffi::String> dtype, ffi::String data_layout);

// ---------------------------------------------------------------------------
// Conv3D
// ---------------------------------------------------------------------------

class Conv3DModuleNode : public runtime::Object {
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
        .def("forward", &Conv3DModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Conv3D", Conv3DModuleNode, runtime::Object);
};
class Conv3DModule : public runtime::ObjectRef {
 public:
  explicit Conv3DModule(NNParameter weight, ffi::Optional<NNParameter> bias, int64_t stride,
                        int64_t padding, int64_t dilation, int64_t groups, ffi::String data_layout);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Conv3DModule, runtime::ObjectRef, Conv3DModuleNode);
};
/*! \brief Factory: creates Parameters internally, handles kernel_size expansion. */
Conv3DModule MakeConv3D(int64_t in_channels, int64_t out_channels, ffi::Array<Integer> kernel_size,
                        int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                        bool has_bias, ffi::Optional<ffi::String> dtype, ffi::String data_layout);

// ---------------------------------------------------------------------------
// ConvTranspose1D
// ---------------------------------------------------------------------------

class ConvTranspose1DModuleNode : public runtime::Object {
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
        .def("forward", &ConvTranspose1DModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.ConvTranspose1D", ConvTranspose1DModuleNode,
                                    runtime::Object);
};
class ConvTranspose1DModule : public runtime::ObjectRef {
 public:
  explicit ConvTranspose1DModule(NNParameter weight, ffi::Optional<NNParameter> bias,
                                 int64_t stride, int64_t padding, int64_t output_padding,
                                 int64_t dilation, int64_t groups);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ConvTranspose1DModule, runtime::ObjectRef,
                                                ConvTranspose1DModuleNode);
};
/*! \brief Factory: creates Parameters internally. */
ConvTranspose1DModule MakeConvTranspose1D(int64_t in_channels, int64_t out_channels,
                                          int64_t kernel_size, int64_t stride, int64_t padding,
                                          int64_t output_padding, int64_t dilation, int64_t groups,
                                          bool has_bias, ffi::Optional<ffi::String> dtype);

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_MODULES_H_
