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
  explicit ModuleSpec(ffi::Array<ffi::String> method_names, ffi::Array<ffi::Any> method_specs,
                      ffi::Map<ffi::String, NNParameter> named_params,
                      ffi::Map<ffi::String, runtime::ObjectRef> named_effects);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ModuleSpec, runtime::ObjectRef, ModuleSpecNode);
};

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_SPEC_H_
