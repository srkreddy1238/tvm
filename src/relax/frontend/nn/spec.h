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
 * \brief Compilation specification types for the nn module frontend.
 *
 * Defines the types that describe how an nn module is compiled:
 *
 *   - SpecInt     type tag for a scalar integer input argument.
 *   - SpecTensor  shape and dtype specification for a tensor input.
 *   - SpecTuple   named, ordered collection of nested specs (tuple or list).
 *   - MethodSpec  specification for a single compiled method: forward
 *                 function, argument names and specs, and parameter/effect
 *                 handling modes.
 *   - ModuleSpec  specification for a complete module compilation: ordered
 *                 method names and specs, pre-collected named parameters,
 *                 and pre-collected named effects.
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

/*! \brief Type tag for a scalar integer input argument. */
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

/*! \brief Shape and dtype specification for a tensor input argument. */
class SpecTensorNode : public runtime::Object {
 public:
  /*! \brief Shape specification; each element is int64 (static) or String (symbolic). */
  ffi::Array<ffi::Any> shape;
  /*! \brief Data type string (e.g. "float32"). */
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

/*!
 * \brief Named, ordered collection of nested specs.
 *
 * Represents either a tuple (is_tuple=true) or a list (is_tuple=false)
 * of input arguments.  Elements may be SpecInt, SpecTensor, or nested
 * SpecTuple objects.
 */
class SpecTupleNode : public runtime::Object {
 public:
  /*! \brief Name used as a prefix for extracted element variables. */
  ffi::String name;
  /*! \brief Ordered elements; each is SpecInt, SpecTensor, or SpecTuple. */
  ffi::Array<ffi::Any> elements;
  /*! \brief True for tuple semantics, false for list semantics. */
  bool is_tuple;

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
 * \brief Specification for a single compiled method.
 *
 * The forward function has signature:
 *   ffi::Any forward(ffi::Map<ffi::String, ffi::Any> named_args)
 * where each value in named_args is an NNTensor (for SpecTensor arguments)
 * or a tir::Var (for SpecInt arguments).
 *
 * param_mode and effect_mode each take one of three values:
 *   - "plain"   individual parameters/effects as separate function arguments.
 *   - "packed"  all parameters/effects bundled into a single tuple argument.
 *   - "none"    parameters/effects omitted from the function signature.
 */
class MethodSpecNode : public runtime::Object {
 public:
  /*! \brief Forward function: (Map<String,Any>) -> Any. */
  ffi::Function forward;
  /*! \brief Ordered argument names, parallel to arg_specs. */
  ffi::Array<ffi::String> arg_names;
  /*! \brief Argument specifications; each is SpecInt, SpecTensor, or SpecTuple. */
  ffi::Array<ffi::Any> arg_specs;
  /*! \brief Parameter handling mode: "plain", "packed", or "none". */
  ffi::String param_mode;
  /*! \brief Effect handling mode: "plain", "packed", or "none". */
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
  /*!
   * \brief Construct a MethodSpec.
   *
   * Use DeriveMethodFunction() to obtain the forward ffi::Function from
   * an NNModule when the method is looked up via reflection.
   *
   * \param forward      Forward function with signature (Map<String,Any>)->Any.
   * \param arg_names    Ordered argument names, parallel to arg_specs.
   * \param arg_specs    Argument specifications (SpecInt/SpecTensor/SpecTuple).
   * \param param_mode   Parameter handling mode: "plain", "packed", or "none".
   * \param effect_mode  Effect handling mode: "plain", "packed", or "none".
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
 * \brief Specification for a complete module compilation.
 *
 * named_params is pre-collected before constructing ModuleSpec so that
 * the exporter never needs to call back into user code for parameter
 * discovery during IR generation.
 */
class ModuleSpecNode : public runtime::Object {
 public:
  /*! \brief Ordered method names, parallel to method_specs. */
  ffi::Array<ffi::String> method_names;
  /*! \brief Method specifications; each element is a MethodSpec. */
  ffi::Array<ffi::Any> method_specs;
  /*! \brief Pre-collected named parameters: dotted_name -> NNParameter. */
  ffi::Map<ffi::String, NNParameter> named_params;
  /*! \brief Pre-collected named effects: dotted_name -> EffectNode-derived object. */
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
  /*!
   * \brief Low-level constructor: caller supplies all fields explicitly.
   *
   * When named_effects is non-empty, any MethodSpec whose effect_mode is
   * "none" is automatically upgraded to "plain".
   *
   * \param method_names   Ordered method names.
   * \param method_specs   Method specifications (each is a MethodSpec).
   * \param named_params   Pre-collected named parameters.
   * \param named_effects  Pre-collected named effects.
   */
  explicit ModuleSpec(ffi::Array<ffi::String> method_names, ffi::Array<ffi::Any> method_specs,
                      ffi::Map<ffi::String, NNParameter> named_params,
                      ffi::Map<ffi::String, runtime::ObjectRef> named_effects);

  /*!
   * \brief Module-aware constructor.
   *
   * Derives a forward ffi::Function for every method listed in \p spec by
   * looking up the module's "_forward" method via reflection, then
   * constructs a MethodSpec for each one.  Named parameters and effects are
   * collected automatically from \p mod.
   *
   * \param mod    The NNModule to compile.
   * \param spec   Map from method name to (arg_name -> arg_spec) map.
   * \param debug  When true, effect_mode is set to "plain" for every method;
   *               when false, effect_mode is "none" unless effects are present.
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
 * Resolves the reflection method name from \p method_name:
 *   - "forward" maps to "_forward" (the C++ convention for the primary
 *     forward implementation).
 *   - All other names are used verbatim.
 *
 * Returns a closure with signature:
 *   ffi::Any fn(ffi::Map<ffi::String, ffi::Any> named_args)
 * where each value in named_args is an NNTensor whose underlying Var is
 * extracted and passed positionally to the resolved method.
 *
 * \param mod_ref      The NNModuleNode-derived object.
 * \param method_name  Exported method name (e.g. "forward", "encode").
 * \param arg_names    Ordered argument names matching the arg_specs that
 *                     will be passed to MethodSpec.
 * \param extra_args   Optional trailing arguments appended after the
 *                     spec-driven inputs (default: empty).
 * \return             An ffi::Function suitable for MethodSpec's constructor.
 */
ffi::Function DeriveMethodFunction(runtime::ObjectRef mod_ref, ffi::String method_name,
                                   ffi::Array<ffi::String> arg_names,
                                   ffi::Array<ffi::Any> extra_args = {});

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_SPEC_H_
