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
 * \file src/relax/frontend/nn/spec.h
 * \brief Native C++ Object definitions for nn compilation specifications.
 *
 * All six spec types are now native C++ objects:
 *
 *   SpecInt      – type tag for a scalar integer input (no fields)
 *   SpecTensor   – shape (Array<Any>) + dtype (String)
 *   SpecTuple    – name (String) + elements (Array<Any>, recursive) + is_tuple (bool)
 *   MethodSpec   – forward: ffi::Function(Map<String,Any>) -> Any
 *                  arg_names: Array<String>
 *                  arg_specs: Array<Any>   (each is SpecInt/SpecTensor/SpecTuple)
 *                  param_mode / effect_mode: String
 *   ModuleSpec   – method_names: Array<String>
 *                  method_specs: Array<Any>  (each is MethodSpec)
 *                  named_params: Map<String, NNParameter>
 *
 * The key design decision for MethodSpec:
 *   The forward function is stored as ffi::Function with signature
 *       ffi::Any forward(ffi::Map<ffi::String, ffi::Any> named_args)
 *   where each value in named_args is a TensorNode (nn.Tensor).
 *   The implementation casts ffi::Any to TensorNode* as needed.
 *   This eliminates inspect.signature entirely.
 *
 * ModuleSpec stores the pre-collected named_params so the C++ Exporter
 * does not need to call back into Python for parameter discovery.
 *
 * Module-aware ModuleSpec construction
 * -------------------------------------
 * The preferred C++ pattern for simple NNModule subclasses is:
 *
 *   // 1. Create the module
 *   ReLUModule mod;
 *   // 2. Build the per-method argument spec
 *   ffi::Map<ffi::String, ffi::Any> forward_spec;
 *   forward_spec.Set("x", ffi::Any(MakeSpecTensor({3, 3}, "float32")));
 *   ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> spec;
 *   spec.Set("forward", forward_spec);
 *   // 3. Create ModuleSpec — derives ffi::Function for each method via
 *   //    reflection and builds MethodSpec objects internally.
 *   ModuleSpec mod_spec(mod, spec, false);  // debug=false
 *   // 4. Export
 *   ffi::Array<ffi::Any> result =
 *       mod->ExportTVM(mod_spec, false, false);  // debug=false, allow_extern=false
 *   IRModule ir = result[0].cast<IRModule>();
 *
 * This removes the need for MethodSpec to hold or receive an NNModule
 * object: ModuleSpec derives the ffi::Function for each method_name via
 * DeriveMethodFunction and passes it to MethodSpec's primary constructor.
 */

#ifndef TVM_RELAX_FRONTEND_NN_SPEC_H_
#define TVM_RELAX_FRONTEND_NN_SPEC_H_

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/object.h>

#include <string>

#include "core.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ---------------------------------------------------------------------------
// SpecIntNode
// ---------------------------------------------------------------------------

class SpecIntNode : public runtime::Object {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpecIntNode>().def(refl::init<>());
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.spec.Int", SpecIntNode, runtime::Object);
};
class SpecInt : public runtime::ObjectRef {
 public:
  explicit SpecInt();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SpecInt, runtime::ObjectRef, SpecIntNode);
};

// ---------------------------------------------------------------------------
// SpecTensorNode
// ---------------------------------------------------------------------------

class SpecTensorNode : public runtime::Object {
 public:
  ffi::Array<ffi::Any> shape;  //!< int64 (static) or String (symbolic)
  ffi::String dtype;

  SpecTensorNode(ffi::Array<ffi::Any> shape, ffi::String dtype)
      : shape(std::move(shape)), dtype(std::move(dtype)) {}

  ffi::String Repr() const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpecTensorNode>()
        .def(refl::init<ffi::Array<ffi::Any>, ffi::String>())
        .def_ro("shape", &SpecTensorNode::shape)
        .def_ro("dtype", &SpecTensorNode::dtype)
        .def("__repr__", &SpecTensorNode::Repr);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.spec.Tensor", SpecTensorNode,
                                    runtime::Object);
};
class SpecTensor : public runtime::ObjectRef {
 public:
  explicit SpecTensor(ffi::Array<ffi::Any> shape, ffi::String dtype);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SpecTensor, runtime::ObjectRef, SpecTensorNode);
};

// ---------------------------------------------------------------------------
// SpecTupleNode
// ---------------------------------------------------------------------------

class SpecTupleNode : public runtime::Object {
 public:
  ffi::String name;
  ffi::Array<ffi::Any> elements;  //!< each is SpecInt/SpecTensor/SpecTuple
  bool is_tuple;                  //!< true=tuple, false=list

  SpecTupleNode(ffi::String name, ffi::Array<ffi::Any> elements, bool is_tuple)
      : name(std::move(name)), elements(std::move(elements)), is_tuple(is_tuple) {}

  ffi::String Repr() const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpecTupleNode>()
        .def(refl::init<ffi::String, ffi::Array<ffi::Any>, bool>())
        .def_ro("name", &SpecTupleNode::name)
        .def_ro("elements", &SpecTupleNode::elements)
        .def_ro("is_tuple", &SpecTupleNode::is_tuple)
        .def("__repr__", &SpecTupleNode::Repr);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.spec.Tuple", SpecTupleNode, runtime::Object);
};
class SpecTuple : public runtime::ObjectRef {
 public:
  explicit SpecTuple(ffi::String name, ffi::Array<ffi::Any> elements, bool is_tuple);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SpecTuple, runtime::ObjectRef, SpecTupleNode);
};

// ---------------------------------------------------------------------------
// MethodSpecNode
// ---------------------------------------------------------------------------

/*!
 * \brief Spec for a single compiled method.
 *
 * forward is an ffi::Function with signature:
 *   ffi::Any forward(ffi::Map<ffi::String, ffi::Any> named_args)
 *
 * named_args maps each arg_name to its nn.Tensor (TensorNode) or tir.Var.
 * The implementation casts ffi::Any to the appropriate type.
 * No inspect.signature is needed: arg_names is provided explicitly.
 */
class MethodSpecNode : public runtime::Object {
 public:
  /*! \brief The forward function: (Map<String,Any>) -> Any */
  ffi::Function forward;
  /*! \brief Ordered argument names (matches arg_specs). */
  ffi::Array<ffi::String> arg_names;
  /*! \brief Spec for each argument (SpecInt, SpecTensor, or SpecTuple). */
  ffi::Array<ffi::Any> arg_specs;
  /*! \brief "plain", "packed", or "none". */
  ffi::String param_mode;
  /*! \brief "plain", "packed", or "none". */
  ffi::String effect_mode;

  MethodSpecNode(ffi::Function forward, ffi::Array<ffi::String> arg_names,
                 ffi::Array<ffi::Any> arg_specs, ffi::String param_mode, ffi::String effect_mode)
      : forward(std::move(forward)),
        arg_names(std::move(arg_names)),
        arg_specs(std::move(arg_specs)),
        param_mode(std::move(param_mode)),
        effect_mode(std::move(effect_mode)) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MethodSpecNode>()
        .def(refl::init<ffi::Function, ffi::Array<ffi::String>, ffi::Array<ffi::Any>, ffi::String,
                        ffi::String>())
        .def_ro("forward", &MethodSpecNode::forward)
        .def_ro("arg_names", &MethodSpecNode::arg_names)
        .def_ro("arg_specs", &MethodSpecNode::arg_specs)
        .def_ro("param_mode", &MethodSpecNode::param_mode)
        .def_ro("effect_mode", &MethodSpecNode::effect_mode);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.spec.MethodSpec", MethodSpecNode,
                                    runtime::Object);
};
class MethodSpec : public runtime::ObjectRef {
 public:
  /*! \brief Primary constructor: caller supplies the forward ffi::Function directly.
   *
   * Use DeriveMethodFunction() to obtain the ffi::Function from an NNModule
   * when the method is looked up via reflection.
   */
  explicit MethodSpec(ffi::Function forward, ffi::Array<ffi::String> arg_names,
                      ffi::Array<ffi::Any> arg_specs, ffi::String param_mode,
                      ffi::String effect_mode);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MethodSpec, runtime::ObjectRef, MethodSpecNode);
};

// ---------------------------------------------------------------------------
// ModuleSpecNode
// ---------------------------------------------------------------------------

/*!
 * \brief Spec for a complete nn.Module compilation.
 *
 * named_params is pre-collected by the Python Module.named_parameters()
 * call before constructing ModuleSpec, so the C++ Exporter never needs
 * to call back into Python for parameter discovery.
 */
class ModuleSpecNode : public runtime::Object {
 public:
  /*! \brief Ordered method names. */
  ffi::Array<ffi::String> method_names;
  /*! \brief Spec for each method (each is MethodSpec). */
  ffi::Array<ffi::Any> method_specs;
  /*! \brief Pre-collected named parameters: dotted_name -> NNParameter. */
  ffi::Map<ffi::String, NNParameter> named_params;
  /*! \brief Pre-collected named effects: dotted_name -> Effect object. */
  ffi::Map<ffi::String, runtime::ObjectRef> named_effects;

  ModuleSpecNode(ffi::Array<ffi::String> method_names, ffi::Array<ffi::Any> method_specs,
                 ffi::Map<ffi::String, NNParameter> named_params,
                 ffi::Map<ffi::String, runtime::ObjectRef> named_effects)
      : method_names(std::move(method_names)),
        method_specs(std::move(method_specs)),
        named_params(std::move(named_params)),
        named_effects(std::move(named_effects)) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ModuleSpecNode>()
        .def(refl::init<ffi::Array<ffi::String>, ffi::Array<ffi::Any>,
                        ffi::Map<ffi::String, NNParameter>,
                        ffi::Map<ffi::String, runtime::ObjectRef>>())
        .def_ro("method_names", &ModuleSpecNode::method_names)
        .def_ro("method_specs", &ModuleSpecNode::method_specs)
        .def_ro("named_params", &ModuleSpecNode::named_params)
        .def_ro("named_effects", &ModuleSpecNode::named_effects);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.spec.ModuleSpec", ModuleSpecNode,
                                    runtime::Object);
};
class ModuleSpec : public runtime::ObjectRef {
 public:
  /*! \brief Low-level constructor: caller supplies all fields explicitly. */
  explicit ModuleSpec(ffi::Array<ffi::String> method_names, ffi::Array<ffi::Any> method_specs,
                      ffi::Map<ffi::String, NNParameter> named_params,
                      ffi::Map<ffi::String, runtime::ObjectRef> named_effects);

  /*!
   * \brief Module-aware constructor (preferred for simple NNModule subclasses).
   *
   * Derives an ffi::Function for every method listed in \p spec by looking
   * up the module's "_forward" method via ffi::reflection::GetMethod, then
   * constructs a MethodSpec for each one.  Named parameters are collected
   * automatically from \p mod via NNModuleNode::NamedParameters("").
   *
   * The spec dictionary maps each method name to an ordered map of
   * argument-name → SpecTensor/SpecInt/SpecTuple:
   *
   *   ffi::Map<ffi::String, ffi::Any> fwd_spec;
   *   fwd_spec.Set("x", ffi::Any(SpecTensor({3,3}, "float32")));
   *   ffi::Map<ffi::String, ffi::Map<ffi::String,ffi::Any>> spec;
   *   spec.Set("forward", fwd_spec);
   *   ModuleSpec ms(mod, spec, false);  // debug=false
   *
   * \param mod    The NNModule whose "_forward" method is used for every
   *               method listed in \p spec.
   * \param spec   Map from method_name to (arg_name → arg_spec) map.
   * \param debug  When true the effect_mode for each MethodSpec is set to
   *               "plain"; when false it is set to "none".
   */
  explicit ModuleSpec(runtime::ObjectRef mod,
                      ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> spec, bool debug);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ModuleSpec, runtime::ObjectRef, ModuleSpecNode);
};

// ---------------------------------------------------------------------------
// DeriveMethodFunction
// ---------------------------------------------------------------------------

/*!
 * \brief Derive an ffi::Function for a single NNModule method.
 *
 * Resolves the reflection method name from \p method_name using the
 * following rule:
 *   - If \p method_name is "forward", the reflection lookup uses "_forward"
 *     (the C++ convention for the primary forward implementation).
 *   - Otherwise \p method_name is used directly as the reflection key.
 *
 * Returns a closure with signature:
 *   ffi::Any fn(ffi::Map<ffi::String, ffi::Any> named_args)
 * where each value in named_args is an NNTensor whose underlying Var is
 * extracted and passed positionally to the resolved method.
 *
 * This is the same logic that was previously embedded inline at each
 * MethodSpec construction site, now factored out so that ModuleSpec and
 * ExportDebug can call it without MethodSpec needing to hold a reference
 * to the module.
 *
 * \param mod_ref      The NNModuleNode-derived object.
 * \param method_name  The exported method name (e.g. "forward",
 *                     "encode", "decode").  "forward" is mapped to
 *                     "_forward" for the reflection lookup; all other
 *                     names are used verbatim.
 * \param arg_names    Ordered argument names (must match the arg_specs
 *                     that will be passed to MethodSpec).
 * \param extra_args   Optional trailing arguments appended to the method
 *                     call after the spec-driven inputs (default: empty).
 * \return             An ffi::Function suitable for MethodSpec's primary
 *                     constructor.
 */
ffi::Function DeriveMethodFunction(runtime::ObjectRef mod_ref, ffi::String method_name,
                                   ffi::Array<ffi::String> arg_names,
                                   ffi::Array<ffi::Any> extra_args = {});

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_SPEC_H_
