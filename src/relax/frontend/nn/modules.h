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
#include "spec.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ===========================================================================
// Helper: build a ParameterNode from shape + dtype, return as NNParameter.
// Used by all Make* factory functions below.
// ===========================================================================
NNParameter MakeParam(ffi::Array<ffi::Any> shape, ffi::String dtype);

// ===========================================================================
// Macro: declare a module node that inherits NNModuleNode.
//
// Each XxxModuleNode:
//   - Inherits NNModuleNode (gains named_parameters, state_dict,
//     load_state_dict, to, and the attrs map).
//   - Stores NNParameter fields AND scalar hyper-parameters as direct C++
//     members (for typed access in Forward()), AND mirrors every NNParameter
//     into attrs so that NNModuleNode::NamedParameters() finds them.
//   - The Make* factory populates attrs after construction.
// ===========================================================================

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

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

class GELUModuleNode : public NNModuleNode {
 public:
  ffi::String approximate;  //!< "" (exact) or "tanh"

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

class LinearModuleNode : public NNModuleNode {
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
/*! \brief Factory: creates Parameters internally, returns LinearModule. */
LinearModule MakeLinear(ffi::Any in_features, ffi::Any out_features, bool bias,
                        ffi::Optional<ffi::String> dtype, ffi::Optional<ffi::String> out_dtype);

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

class EmbeddingModuleNode : public NNModuleNode {
 public:
  NNParameter weight;  //!< shape [num_embeddings, embedding_dim]

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
/*! \brief Factory: creates the weight Parameter internally. */
EmbeddingModule MakeEmbedding(ffi::Any num, ffi::Any dim, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

class LayerNormModuleNode : public NNModuleNode {
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
/*! \brief Factory: creates Parameters internally. */
LayerNormModule MakeLayerNorm(ffi::Any normalized_shape, double eps, bool elementwise_affine,
                              ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

class RMSNormModuleNode : public NNModuleNode {
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
/*! \brief Factory: creates Parameters internally. */
RMSNormModule MakeRMSNorm(ffi::Any hidden_size, ffi::Array<Integer> axes, double epsilon,
                          bool has_bias, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------------

class GroupNormModuleNode : public NNModuleNode {
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
/*! \brief Factory: creates Parameters internally. */
GroupNormModule MakeGroupNorm(int64_t num_groups, ffi::Any num_channels, double eps, bool affine,
                              ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// Conv1D
// ---------------------------------------------------------------------------

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
/*! \brief Factory: creates Parameters internally. */
Conv1DModule MakeConv1D(ffi::Any in_channels, ffi::Any out_channels, ffi::Any kernel_size,
                        int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                        bool has_bias, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// Conv2D
// ---------------------------------------------------------------------------

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
/*! \brief Factory: creates Parameters internally, handles kernel_size expansion. */
Conv2DModule MakeConv2D(ffi::Any in_channels, ffi::Any out_channels,
                        ffi::Array<Integer> kernel_size, int64_t stride, int64_t padding,
                        int64_t dilation, int64_t groups, bool has_bias,
                        ffi::Optional<ffi::String> dtype, ffi::String data_layout);

// ---------------------------------------------------------------------------
// Conv3D
// ---------------------------------------------------------------------------

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
/*! \brief Factory: creates Parameters internally, handles kernel_size expansion. */
Conv3DModule MakeConv3D(ffi::Any in_channels, ffi::Any out_channels,
                        ffi::Array<Integer> kernel_size, int64_t stride, int64_t padding,
                        int64_t dilation, int64_t groups, bool has_bias,
                        ffi::Optional<ffi::String> dtype, ffi::String data_layout);

// ---------------------------------------------------------------------------
// ConvTranspose1D
// ---------------------------------------------------------------------------

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
/*! \brief Factory: creates Parameters internally. */
ConvTranspose1DModule MakeConvTranspose1D(ffi::Any in_channels, ffi::Any out_channels,
                                          ffi::Any kernel_size, int64_t stride, int64_t padding,
                                          int64_t output_padding, int64_t dilation, int64_t groups,
                                          bool has_bias, ffi::Optional<ffi::String> dtype);

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

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
// IOEffect
// ---------------------------------------------------------------------------

/*!
 * \brief Native C++ IOEffect: models the IO side-effect token (_io).
 *
 * Holds a single relax::Var (the current _io token).  The four Effect
 * protocol methods are exposed as FFI methods so Python can call them.
 */
class IOEffectModuleNode : public NNModuleNode {
 public:
  /*! \brief The current _io Var, or undefined when not active. */
  ffi::Optional<Var> effect;

  IOEffectModuleNode() = default;

  /*! \brief emit_init: emit null_value() into bb, return [io_var]. */
  ffi::Array<Var> EmitInit(ffi::String name_hint, BlockBuilder bb) const;
  /*! \brief create: create a placeholder Var for the _io token. */
  ffi::Array<Var> Create(ffi::String name_hint);
  /*! \brief set_state: update the stored _io Var from state_vars[0]. */
  void SetState(ffi::Array<Var> state_vars);
  /*! \brief finalize: return [effect] and clear it. */
  ffi::Array<Var> Finalize();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IOEffectModuleNode>()
        .def(refl::init<>())
        .def_rw("effect", &IOEffectModuleNode::effect)
        .def("_cpp_emit_init", &IOEffectModuleNode::EmitInit)
        .def("_cpp_create", &IOEffectModuleNode::Create)
        .def("_cpp_set_state", &IOEffectModuleNode::SetState)
        .def("_cpp_finalize", &IOEffectModuleNode::Finalize);
  }
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.IOEffect", IOEffectModuleNode, NNModuleNode);
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
 * \brief Native C++ KVCache effect module.
 *
 * Mirrors Python KVCache: holds init_seq_len, unit_shape, dtype, and the
 * current cache Var.  Provides emit_init / create / set_state / finalize
 * (Effect protocol) plus view() and append().
 */
class KVCacheModuleNode : public NNModuleNode {
 public:
  int64_t init_seq_len;
  ffi::Array<Integer> unit_shape;  //!< per-token shape dims
  ffi::String dtype;
  ffi::Optional<Var> cache;  //!< current cache Var (ObjectStructInfo)

  KVCacheModuleNode(int64_t init_seq_len, ffi::Array<Integer> unit_shape, ffi::String dtype)
      : init_seq_len(init_seq_len), unit_shape(std::move(unit_shape)), dtype(std::move(dtype)) {}

  ffi::Array<Var> EmitInit(ffi::String name_hint, BlockBuilder bb) const;
  ffi::Array<Var> Create(ffi::String name_hint);
  void SetState(ffi::Array<Var> state_vars);
  ffi::Array<Var> Finalize();
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
        .def("_cpp_emit_init", &KVCacheModuleNode::EmitInit)
        .def("_cpp_create", &KVCacheModuleNode::Create)
        .def("_cpp_set_state", &KVCacheModuleNode::SetState)
        .def("_cpp_finalize", &KVCacheModuleNode::Finalize)
        .def("_cpp_to", &KVCacheModuleNode::To)
        .def("_view", &KVCacheModuleNode::View)
        .def("_append", &KVCacheModuleNode::Append);
  }
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.KVCache", KVCacheModuleNode, NNModuleNode);
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

class AttentionModuleNode : public NNModuleNode {
 public:
  int64_t heads;
  int64_t inner_dim;
  LinearModule to_q;
  LinearModule to_k;
  LinearModule to_v;
  ffi::Optional<GroupNormModule> group_norm;
  ModuleList to_out;  //!< [Linear(inner_dim, query_dim)]

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

  /*! \brief forward(hidden_states, encoder_hidden_states=nullopt). */
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
