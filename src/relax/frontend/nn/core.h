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
 * \brief Core object definitions for the nn module frontend.
 *
 * Defines the fundamental types used throughout the nn frontend:
 *
 *   - TensorNode     wraps a relax::Var with TensorStructInfo, exposing
 *                    shape, ndim, and dtype as read-only properties.
 *
 *   - ParameterNode  extends TensorNode with an optional concrete data
 *                    buffer (runtime::Tensor) and a string-keyed attribute
 *                    map for quantization metadata and similar annotations.
 *
 *   - NNObjectNode   wraps a relax::Var with ObjectStructInfo, used for
 *                    non-tensor handles such as KVCache.
 *
 *   - NNModuleNode   base class for all nn modules; owns named sub-modules
 *                    and parameters in a string-keyed attribute map and
 *                    provides parameter traversal, state-dict serialization,
 *                    dtype conversion, export, and JIT compilation.
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
 * \brief Wraps a relax::Var whose struct_info is TensorStructInfo.
 *
 * All shape and dtype information is derived directly from the Var's
 * struct_info; no duplication is maintained.
 */
class TensorNode : public runtime::Object {
 public:
  /*! \brief The underlying relax::Var (TensorStructInfo). */
  Var expr;

  /*! \brief Construct from an existing relax::Var (must have TensorStructInfo). */
  explicit TensorNode(Var expr);

  /*! \brief Return the shape as an Array of PrimExprs. */
  ffi::Array<PrimExpr> GetShape() const;

  /*! \brief Return the number of dimensions. */
  int64_t GetNdim() const;

  /*! \brief Return the dtype string. */
  ffi::String GetDtype() const;

  /*!
   * \brief Create a placeholder Var with the given shape and dtype.
   *
   * \param shape  Shape specification; each element is int64, String (symbolic), or PrimExpr.
   * \param dtype  Data type string (e.g. "float32").
   * \param name   Name hint for the created Var.
   * \return       A new TensorNode owning the placeholder Var.
   */
  static TensorNode* MakePlaceholder(ffi::Array<ffi::Any> shape, ffi::String dtype,
                                     ffi::String name);

  /*!
   * \brief Create a Tensor from an existing TensorStructInfo.
   *
   * \param sinfo  The struct info to use.
   * \param name   Name hint for the created Var.
   * \return       A new TensorNode owning the created Var.
   */
  static TensorNode* MakeFromStructInfo(TensorStructInfo sinfo, ffi::String name);

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
  /*! \brief Reserve child slots for ParameterNode. */
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
 * \brief Extends TensorNode with an optional concrete data buffer and
 *        a string-keyed attribute map.
 *
 * The data field holds the bound runtime tensor when the parameter has
 * been loaded from a checkpoint; it is nullopt for unbound parameters.
 * The attrs map carries user-defined annotations such as quantization
 * metadata.
 */
class ParameterNode : public TensorNode {
 public:
  /*! \brief Concrete data buffer, or nullopt when unbound. */
  ffi::Optional<runtime::Tensor> data;

  /*! \brief User-defined attribute map (e.g. quantization metadata). */
  ffi::Map<ffi::String, ffi::Any> attrs;

  /*! \brief Construct from a Var, optional data buffer, and attribute map. */
  ParameterNode(Var expr, ffi::Optional<runtime::Tensor> data,
                ffi::Map<ffi::String, ffi::Any> attrs);

  /*!
   * \brief Re-create the placeholder Var with a new dtype.
   *
   * Only valid when data is nullopt (parameter is unbound).
   *
   * \param dtype  Target dtype string.
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

  /*! \brief Parameters are mutable: data and attrs may be set after construction. */
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Parameter", ParameterNode, TensorNode);
};

class NNParameter : public runtime::ObjectRef {
 public:
  explicit NNParameter(Var expr, ffi::Optional<runtime::Tensor> data = std::nullopt,
                       ffi::Map<ffi::String, ffi::Any> attrs = {});
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(NNParameter, runtime::ObjectRef, ParameterNode);
};

// Forward declarations for types used by NNModuleNode::ExportTVM and Jit.
class ModuleSpecNode;
class ModuleSpec;

// ---------------------------------------------------------------------------
// NNObjectNode
// ---------------------------------------------------------------------------

/*!
 * \brief Wraps a relax::Var whose struct_info is ObjectStructInfo.
 *
 * Used for non-tensor handles such as KVCache objects.
 */
class NNObjectNode : public runtime::Object {
 public:
  /*! \brief The underlying relax::Var (ObjectStructInfo). */
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

/*!
 * \brief Base class for all nn modules.
 *
 * Owns named sub-modules and parameters in a string-keyed attribute map
 * and provides:
 *   - NamedParameters() / StateDict()  for parameter traversal.
 *   - LoadStateDict()                  for checkpoint loading.
 *   - To()                             for recursive dtype conversion.
 *   - ExportTVM() / Jit()              for compilation.
 */
class NNModuleNode : public runtime::Object {
 public:
  /*! \brief Named children: sub-modules, parameters, and scalar hyper-parameters. */
  ffi::Map<ffi::String, ffi::Any> attrs;

  NNModuleNode() = default;
  explicit NNModuleNode(ffi::Map<ffi::String, ffi::Any> attrs) : attrs(std::move(attrs)) {}

  /*!
   * \brief Return all (dotted_name, NNParameter) pairs in this module tree.
   *
   * \param prefix  Dotted prefix prepended to each name (pass "" at the root).
   * \return        Map from dotted parameter name to NNParameter.
   */
  ffi::Map<ffi::String, NNParameter> NamedParameters(ffi::String prefix) const;

  /*!
   * \brief Return an ordered map of all parameters keyed by dotted name.
   *
   * \param prefix  Dotted prefix prepended to each name (pass "" at the root).
   * \return        Map from dotted parameter name to NNParameter.
   */
  ffi::Map<ffi::String, NNParameter> StateDict(ffi::String prefix) const;

  /*!
   * \brief Load parameters from a state dict into this module.
   *
   * \param state_dict  Map of dotted-name to NNParameter with bound data.
   * \param strict      If true, raise on missing or unexpected keys.
   * \return            Array of two string arrays: [missing_keys, unexpected_keys].
   */
  ffi::Array<ffi::Array<ffi::String>> LoadStateDict(ffi::Map<ffi::String, NNParameter> state_dict,
                                                    bool strict) const;

  /*! \brief Recursively convert all parameters and sub-modules to \p dtype. */
  void To(ffi::String dtype) const;

  /*!
   * \brief Export this module to a TVM IRModule.
   *
   * \param spec         ModuleSpec describing methods and parameters.
   * \param debug        If true, add an IOEffect token to every method signature.
   * \param allow_extern If true, collect external modules via nn.add_extern.
   * \return             Array of [IRModule, named_params Map, extern_mods Array].
   */
  ffi::Array<ffi::Any> ExportTVM(ModuleSpec spec, bool debug, bool allow_extern = true) const;

  /*!
   * \brief JIT-compile this module to a CppModule ready for inference.
   *
   * \param spec      ModuleSpec describing methods and parameters.
   * \param device    Execution device.
   * \param pipeline  Relax compilation pipeline name.
   * \param debug     If true, add an IOEffect token to every method.
   * \return          A CppModule wrapping the compiled VM.
   */
  runtime::ObjectRef Jit(ModuleSpec spec, tvm::Device device, ffi::String pipeline,
                         bool debug) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<NNModuleNode>()
        .def(refl::init<ffi::Map<ffi::String, ffi::Any>>())
        .def_rw("attrs", &NNModuleNode::attrs)
        .def("named_parameters", &NNModuleNode::NamedParameters)
        .def("state_dict", &NNModuleNode::StateDict)
        .def("load_state_dict", &NNModuleNode::LoadStateDict)
        .def("to", &NNModuleNode::To)
        .def("export_tvm", &NNModuleNode::ExportTVM)
        .def("jit", &NNModuleNode::Jit);
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
 * \brief Holds an ordered list of sub-module objects.
 *
 * Each element is stored as ffi::Any so that both native C++ module objects
 * and Python Module instances can be held without a common base.
 */
class ModuleListNode : public NNModuleNode {
 public:
  /*! \brief Ordered list of sub-modules. */
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
 * \brief Holds a string-keyed ordered map of sub-module objects.
 *
 * Each value is stored as ffi::Any so that both native C++ module objects
 * and Python Module instances can be held without a common base.
 */
class ModuleDictNode : public NNModuleNode {
 public:
  /*! \brief String-keyed ordered map of sub-modules. */
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

/*! \brief Get the thread-local default dtype string. */
ffi::String GetDefaultDtype();

/*! \brief Set the thread-local default dtype string. */
void SetDefaultDtype(ffi::String dtype);

/*!
 * \brief Return the thread-local current BlockBuilder.
 *
 * Returns a null BlockBuilder when no export is in progress.
 * Installed by the exporter before calling forward() so that op helpers
 * can emit bindings into the active dataflow block.
 */
BlockBuilder BlockBuilder_Current();

/*!
 * \brief Install or clear the thread-local current BlockBuilder.
 *
 * \param bb  Pointer to the BlockBuilder to install, or nullptr to clear.
 */
void BlockBuilder_SetCurrent(BlockBuilder* bb);

/*!
 * \brief Emit \p expr into the current BlockBuilder and return the bound Var.
 *
 * For TupleStructInfo, recursively emits TupleGetItem bindings and returns
 * an Array<Any> of the extracted element Vars.
 *
 * \param expr  The expression to emit.
 * \param name  Name hint for the bound variable.
 * \return      The bound Var, or Array<Any> for tuple results.
 */
ffi::Any WrapNested(Expr expr, ffi::String name);

/*!
 * \brief Return all NNParameter fields of a runtime::Object.
 *
 * Inspects the registered field metadata of \p obj and returns every
 * field whose value is an NNParameter, keyed by field name.
 *
 * \param obj  The object to inspect.
 * \return     Map from field name to NNParameter.
 */
ffi::Map<ffi::String, NNParameter> GetNativeParameters(runtime::ObjectRef obj);

/*!
 * \brief Collect named parameters from a ModuleList or ModuleDict.
 *
 * Recursively walks the container and returns every NNParameter found,
 * keyed by its dotted path (e.g. "0.weight", "encoder.bias").
 *
 * \param container  A ModuleList or ModuleDict object.
 * \param prefix     Dotted prefix prepended to each name.
 * \return           Map from dotted name to NNParameter.
 */
ffi::Map<ffi::String, NNParameter> GetContainerParameters(runtime::ObjectRef container,
                                                          ffi::String prefix);

/*!
 * \brief Apply To(dtype) to every NNParameter inside a ModuleList or ModuleDict.
 *
 * Recurses into nested containers.
 *
 * \param container  A ModuleList or ModuleDict object.
 * \param dtype      Target dtype string.
 */
void ContainerApplyTo(runtime::ObjectRef container, ffi::String dtype);

/*!
 * \brief Apply dtype conversion to all Parameters in a module attribute map.
 *
 * \param py_dict  Attribute map (e.g. from NNModuleNode::attrs).
 * \param dtype    Target dtype string.
 */
void PythonModuleApplyTo(ffi::Map<ffi::String, ffi::Any> py_dict, ffi::String dtype);

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_CORE_H_
