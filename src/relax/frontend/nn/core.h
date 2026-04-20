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
 * \file src/relax/frontend/nn/core.h
 * \brief Native C++ Object definitions for the nn frontend core types:
 *        Tensor, Parameter, and NNObject.
 *
 * Design:
 *   TensorNode  – holds a relax.Var whose struct_info is TensorStructInfo.
 *                 Exposes shape, ndim, dtype as read-only fields plus
 *                 a From* family of static constructors.
 *
 *   ParameterNode – subclass of TensorNode that additionally holds an
 *                   optional concrete data buffer (runtime::Tensor) and
 *                   a string→Any attribute map.
 *
 *   NNObjectNode  – holds a relax.Var whose struct_info is ObjectStructInfo.
 *                   Used for non-tensor frontend components (e.g. KVCache).
 *
 * Python side: each class is decorated with @tvm_ffi.register_object and
 * constructed via __init_handle_by_constructor__.  All field access and
 * method calls go through the FFI with zero Python overhead.
 */

#ifndef TVM_RELAX_FRONTEND_NN_CORE_H_
#define TVM_RELAX_FRONTEND_NN_CORE_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/tensor.h>

#include <string>

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ---------------------------------------------------------------------------
// TensorNode
// ---------------------------------------------------------------------------

/*!
 * \brief Native C++ representation of nn.Tensor.
 *
 * Wraps a relax.Var whose struct_info is TensorStructInfo.  All shape /
 * dtype information is derived directly from the Var's struct_info so there
 * is no duplication.
 */
class TensorNode : public runtime::Object {
 public:
  /*! \brief The underlying relax.Var (TensorStructInfo). */
  Var expr;

  // ---- Constructors -------------------------------------------------------

  /*! \brief Construct from an existing relax.Var (must have TensorStructInfo). */
  explicit TensorNode(Var expr);

  // ---- Accessors (exposed as read-only fields / methods) ------------------

  /*! \brief Return the shape as an Array of PrimExprs. */
  ffi::Array<PrimExpr> GetShape() const;

  /*! \brief Return the number of dimensions. */
  int64_t GetNdim() const;

  /*! \brief Return the dtype string. */
  ffi::String GetDtype() const;

  // ---- Static factory methods ---------------------------------------------

  /*!
   * \brief Create a placeholder Var with the given shape and dtype.
   * Shape elements may be int64, String (symbolic name), or PrimExpr.
   */
  static TensorNode* MakePlaceholder(ffi::Array<ffi::Any> shape, ffi::String dtype,
                                     ffi::String name);

  /*!
   * \brief Create a Tensor from an existing TensorStructInfo.
   */
  static TensorNode* MakeFromStructInfo(TensorStructInfo sinfo, ffi::String name);

  // ---- Reflection ---------------------------------------------------------

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorNode>()
        .def(refl::init<Var>())
        .def_ro("expr", &TensorNode::expr)
        .def("shape", &TensorNode::GetShape, "Shape as Array[PrimExpr].")
        .def("ndim", &TensorNode::GetNdim, "Number of dimensions.")
        .def("dtype", &TensorNode::GetDtype, "Data type string.");
  }

  static constexpr bool _type_mutable = false;
  // Reserve child slots for ParameterNode
  static constexpr uint32_t _type_child_slots = 1;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.frontend.nn.Tensor", TensorNode, runtime::Object);
};

class NNTensor : public runtime::ObjectRef {
 public:
  explicit NNTensor(Var expr);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(NNTensor, runtime::ObjectRef, TensorNode);
};

// ---------------------------------------------------------------------------
// ParameterNode
// ---------------------------------------------------------------------------

/*!
 * \brief Native C++ representation of nn.Parameter.
 *
 * Extends TensorNode with:
 *   - An optional concrete data buffer (runtime::Tensor / NDArray).
 *   - A string→Any attribute dictionary.
 *   - A to(dtype) method that re-creates the placeholder with a new dtype.
 */
class ParameterNode : public TensorNode {
 public:
  /*! \brief Concrete data, or nullopt when unbound. */
  ffi::Optional<runtime::Tensor> data;

  /*! \brief User-defined attributes (e.g. quantization metadata). */
  ffi::Map<ffi::String, ffi::Any> attrs;

  /*! \brief Construct an unbound parameter with the given shape and dtype. */
  ParameterNode(Var expr, ffi::Optional<runtime::Tensor> data,
                ffi::Map<ffi::String, ffi::Any> attrs);

  /*!
   * \brief Re-create the placeholder with a new dtype.
   * Only valid when data is not bound (data == nullopt).
   */
  void To(ffi::String dtype);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ParameterNode>()
        .def(refl::init<Var, ffi::Optional<runtime::Tensor>, ffi::Map<ffi::String, ffi::Any>>())
        .def_ro("expr", &TensorNode::expr)
        .def_rw("data", &ParameterNode::data)
        .def_rw("attrs", &ParameterNode::attrs)
        .def("shape", &TensorNode::GetShape)
        .def("ndim", &TensorNode::GetNdim)
        .def("dtype", &TensorNode::GetDtype)
        .def("to", &ParameterNode::To, "Re-create placeholder with new dtype.");
  }

  // Parameters are mutable (data and attrs can be set after construction).
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Parameter", ParameterNode, TensorNode);
};

class NNParameter : public runtime::ObjectRef {
 public:
  explicit NNParameter(Var expr, ffi::Optional<runtime::Tensor> data = std::nullopt,
                       ffi::Map<ffi::String, ffi::Any> attrs = {});
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(NNParameter, runtime::ObjectRef, ParameterNode);
};

// ---------------------------------------------------------------------------
// NNObjectNode
// ---------------------------------------------------------------------------

/*!
 * \brief Native C++ representation of nn.Object.
 *
 * Wraps a relax.Var whose struct_info is ObjectStructInfo.
 * Used for non-tensor frontend components such as KVCache handles.
 */
class NNObjectNode : public runtime::Object {
 public:
  /*! \brief The underlying relax.Var (ObjectStructInfo). */
  Var expr;

  explicit NNObjectNode(Var expr);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<NNObjectNode>().def(refl::init<Var>()).def_ro("expr", &NNObjectNode::expr);
  }

  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Object", NNObjectNode, runtime::Object);
};

class NNObject : public runtime::ObjectRef {
 public:
  explicit NNObject(Var expr);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(NNObject, runtime::ObjectRef, NNObjectNode);
};

// ===========================================================================
// NNModuleNode  -  C++ base for nn.Module
// ===========================================================================

/*!
 * \brief C++ base class for nn.Module.
 *
 * Holds the module's named sub-modules and parameters in a string-keyed map
 * (mirroring Python's __dict__) and provides:
 *   - named_parameters() / parameters()  via recursive _attribute_finder logic
 *   - state_dict() / load_state_dict()   parameter serialization
 *   - to(dtype)                          recursive dtype conversion
 *   - __call__ / forward dispatch        via a registered FFI forward function
 *
 * export_tvm() and jit() remain Python-only because they depend on
 * VirtualMachine, Target, Exporter and other Python-only infrastructure.
 */
class NNModuleNode : public runtime::Object {
 public:
  /*! \brief Named children: sub-modules, parameters, and other attributes. */
  ffi::Map<ffi::String, ffi::Any> attrs;

  NNModuleNode() = default;
  explicit NNModuleNode(ffi::Map<ffi::String, ffi::Any> attrs) : attrs(std::move(attrs)) {}

  // ---- parameter traversal -----------------------------------------------

  /*! \brief Return all (dotted_name, NNParameter) pairs in this module. */
  ffi::Map<ffi::String, NNParameter> NamedParameters(ffi::String prefix) const;

  // ---- state dict ---------------------------------------------------------

  /*! \brief Return an ordered map of all parameters keyed by dotted name. */
  ffi::Map<ffi::String, NNParameter> StateDict(ffi::String prefix) const;

  /*!
   * \brief Load parameters from state_dict into this module.
   * \param state_dict  Map of dotted-name -> NNParameter with bound data.
   * \param strict      If true, raise on missing or unexpected keys.
   * \return Pair (missing_keys, unexpected_keys) as Array<String>.
   */
  ffi::Array<ffi::Array<ffi::String>> LoadStateDict(ffi::Map<ffi::String, NNParameter> state_dict,
                                                    bool strict) const;

  // ---- dtype conversion --------------------------------------------------

  /*! \brief Recursively convert all parameters and sub-modules to dtype. */
  void To(ffi::String dtype) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<NNModuleNode>()
        .def(refl::init<ffi::Map<ffi::String, ffi::Any>>())
        .def_rw("attrs", &NNModuleNode::attrs)
        .def("named_parameters", &NNModuleNode::NamedParameters)
        .def("state_dict", &NNModuleNode::StateDict)
        .def("load_state_dict", &NNModuleNode::LoadStateDict)
        .def("to", &NNModuleNode::To);
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.frontend.nn.Module", NNModuleNode, runtime::Object);
};

class NNModule : public runtime::ObjectRef {
 public:
  NNModule() { data_ = ffi::make_object<NNModuleNode>(); }
  explicit NNModule(ffi::Map<ffi::String, ffi::Any> attrs) {
    data_ = ffi::make_object<NNModuleNode>(std::move(attrs));
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(NNModule, runtime::ObjectRef, NNModuleNode);
};

// ---------------------------------------------------------------------------
// ModuleListNode
// ---------------------------------------------------------------------------

/*!
 * \brief Native C++ representation of nn.ModuleList.
 *
 * Holds an ordered list of sub-module objects as ffi::Any so that both
 * native C++ module objects and pure-Python Module instances can be stored.
 * Python wrappers provide __iter__, __getitem__, __len__, append, etc.
 */
class ModuleListNode : public NNModuleNode {
 public:
  /*! \brief The ordered list of sub-modules (each element is a Module-like object). */
  ffi::Array<ffi::Any> modules;

  explicit ModuleListNode() = default;
  explicit ModuleListNode(ffi::Array<ffi::Any> modules) : modules(std::move(modules)) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ModuleListNode>()
        .def(refl::init<ffi::Array<ffi::Any>>())
        .def_rw("modules", &ModuleListNode::modules);
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.ModuleList", ModuleListNode, NNModuleNode);
};

class ModuleList : public runtime::ObjectRef {
 public:
  explicit ModuleList(ffi::Array<ffi::Any> modules);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ModuleList, runtime::ObjectRef, ModuleListNode);
};

// ---------------------------------------------------------------------------
// ModuleDictNode
// ---------------------------------------------------------------------------

/*!
 * \brief Native C++ representation of nn.ModuleDict.
 *
 * Holds an ordered string-keyed map of sub-module objects as ffi::Any.
 * Python wrappers provide dict-like access (__getitem__, keys, items, etc.).
 */
class ModuleDictNode : public NNModuleNode {
 public:
  /*! \brief The ordered map of sub-modules. */
  ffi::Map<ffi::String, ffi::Any> modules;

  explicit ModuleDictNode() = default;
  explicit ModuleDictNode(ffi::Map<ffi::String, ffi::Any> modules) : modules(std::move(modules)) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ModuleDictNode>()
        .def(refl::init<ffi::Map<ffi::String, ffi::Any>>())
        .def_rw("modules", &ModuleDictNode::modules);
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.ModuleDict", ModuleDictNode, NNModuleNode);
};

class ModuleDict : public runtime::ObjectRef {
 public:
  explicit ModuleDict(ffi::Map<ffi::String, ffi::Any> modules);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ModuleDict, runtime::ObjectRef, ModuleDictNode);
};

/*! \ brief Get the thread-local default dtype string. */
ffi::String GetDefaultDtype();

/*! \brief Set the thread-local default dtype string. */
void SetDefaultDtype(ffi::String dtype);

/*!
 * \brief Return the thread-local current BlockBuilder, or a null BlockBuilder
 *        if none is active.  Installed by exporter.cc's BBScope before
 *        calling forward() so that WrapNested / Emit helpers work.
 */
BlockBuilder BlockBuilder_Current();

/*! \brief Install (or clear) the thread-local current BlockBuilder. */
void BlockBuilder_SetCurrent(BlockBuilder* bb);

/*!
 * \brief Emit expr into the current BlockBuilder and return the bound Var.
 * For TupleStructInfo, recursively emits TupleGetItem and returns Array<Any>.
 */
ffi::Any WrapNested(Expr expr, ffi::String name);

/*!
 * \brief Return all NNParameter fields of a runtime::Object as a
 *        Map<String, NNParameter> by inspecting its registered field metadata.
 *
 * This is used by Python's _attribute_finder to discover parameters that
 * live inside native C++ module objects (Linear, Conv2D, etc.) without
 * requiring Python-side shadow copies.
 */
ffi::Map<ffi::String, NNParameter> GetNativeParameters(runtime::ObjectRef obj);

/*!
 * \brief Collect named parameters from a ModuleList or ModuleDict object.
 *
 * Recursively walks the list/dict and returns every NNParameter found,
 * keyed by its dotted path (e.g. "0.weight", "encoder.bias").
 * Used by Module.named_parameters() for native container types.
 */
ffi::Map<ffi::String, NNParameter> GetContainerParameters(runtime::ObjectRef container,
                                                          ffi::String prefix);

/*!
 * \brief Apply to(dtype) to every NNParameter inside a ModuleList or
 *        ModuleDict, recursing into nested containers.
 */
void ContainerApplyTo(runtime::ObjectRef container, ffi::String dtype);

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_CORE_H_
