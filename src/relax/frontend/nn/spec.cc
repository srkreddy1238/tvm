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
 * \file src/relax/frontend/nn/spec.cc
 */

#include "spec.h"

#include <tvm/ffi/reflection/registry.h>

#include <sstream>
#include <string>

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ---------------------------------------------------------------------------
// SpecInt
// ---------------------------------------------------------------------------

SpecInt::SpecInt() { data_ = ffi::make_object<SpecIntNode>(); }

// ---------------------------------------------------------------------------
// SpecTensor
// ---------------------------------------------------------------------------

SpecTensor::SpecTensor(ffi::Array<ffi::Any> shape, ffi::String dtype) {
  data_ = ffi::make_object<SpecTensorNode>(std::move(shape), std::move(dtype));
}

ffi::String SpecTensorNode::Repr() const {
  std::ostringstream os;
  os << "Tensor([";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) os << ", ";
    if (auto opt = shape[i].try_cast<int64_t>()) {
      os << opt.value();
    } else if (auto opt = shape[i].try_cast<ffi::String>()) {
      os << std::string(opt.value());
    } else {
      os << "?";
    }
  }
  os << "], '" << std::string(dtype) << "')";
  return ffi::String(os.str());
}

// ---------------------------------------------------------------------------
// SpecTuple
// ---------------------------------------------------------------------------

SpecTuple::SpecTuple(ffi::String name, ffi::Array<ffi::Any> elements, bool is_tuple) {
  data_ = ffi::make_object<SpecTupleNode>(std::move(name), std::move(elements), is_tuple);
}

ffi::String SpecTupleNode::Repr() const {
  std::ostringstream os;
  os << (is_tuple ? "(" : "[");
  for (size_t i = 0; i < elements.size(); ++i) {
    if (i > 0) os << ", ";
    if (auto opt = elements[i].try_cast<runtime::ObjectRef>()) {
      runtime::ObjectRef elem = opt.value();
      if (elem.defined()) {
        if (const auto* t = elem.as<SpecTensorNode>()) {
          os << std::string(t->Repr());
        } else if (elem->IsInstance<SpecIntNode>()) {
          os << "int";
        } else if (const auto* tp = elem.as<SpecTupleNode>()) {
          os << std::string(tp->Repr());
        } else {
          os << elem->GetTypeKey();
        }
      } else {
        os << "None";
      }
    } else {
      os << "?";
    }
  }
  os << (is_tuple ? ")" : "]");
  return ffi::String(os.str());
}

// ---------------------------------------------------------------------------
// MethodSpec
// ---------------------------------------------------------------------------

MethodSpec::MethodSpec(ffi::Function forward, ffi::Array<ffi::String> arg_names,
                       ffi::Array<ffi::Any> arg_specs, ffi::String param_mode,
                       ffi::String effect_mode) {
  data_ = ffi::make_object<MethodSpecNode>(std::move(forward), std::move(arg_names),
                                           std::move(arg_specs), std::move(param_mode),
                                           std::move(effect_mode));
}

// ---------------------------------------------------------------------------
// ModuleSpec
// ---------------------------------------------------------------------------

ModuleSpec::ModuleSpec(ffi::Array<ffi::String> method_names, ffi::Array<ffi::Any> method_specs,
                       ffi::Map<ffi::String, NNParameter> named_params,
                       ffi::Map<ffi::String, runtime::ObjectRef> named_effects) {
  data_ = ffi::make_object<ModuleSpecNode>(std::move(method_names), std::move(method_specs),
                                           std::move(named_params), std::move(named_effects));
}

// ---------------------------------------------------------------------------
// DeriveMethodFunction
// ---------------------------------------------------------------------------

ffi::Function DeriveMethodFunction(runtime::ObjectRef mod_ref, ffi::String method_name,
                                   ffi::Array<ffi::String> arg_names,
                                   ffi::Array<ffi::Any> extra_args) {
  TVM_FFI_ICHECK(mod_ref.defined() && mod_ref->IsInstance<NNModuleNode>())
      << "DeriveMethodFunction: mod_ref must be an NNModuleNode, got "
      << (mod_ref.defined() ? mod_ref->GetTypeKey() : "null");

  // Resolve the reflection method name:
  //   "forward" -> "_forward"  (C++ convention for the primary forward impl)
  //   anything else -> use the given name verbatim
  std::string refl_method_name =
      (std::string(method_name) == "forward") ? "_forward" : std::string(method_name);

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward_fn =
      [mod_ref, refl_method_name, arg_names,
       extra_args](ffi::Map<ffi::String, ffi::Any> named_args) -> ffi::Any {
    namespace refl = tvm::ffi::reflection;
    std::string type_key = mod_ref->GetTypeKey();
    ffi::Function fwd = refl::GetMethod(type_key, refl_method_name.c_str());
    TVM_FFI_ICHECK(fwd.defined()) << "DeriveMethodFunction: module '" << type_key << "' has no '"
                                  << refl_method_name << "' method";

    std::vector<ffi::AnyView> call_args;
    call_args.push_back(ffi::AnyView(mod_ref));
    for (const ffi::String& name : arg_names) {
      ffi::Any val = named_args.at(name);
      // For SpecTensor args the exporter wraps the value as NNTensor; unwrap
      // to the underlying Var so the _forward method receives a relax::Var.
      // For SpecTuple args the exporter passes ffi::Array<ffi::Any>; pass
      // it through directly — the _forward method receives the array.
      if (auto opt = val.try_cast<NNTensor>()) {
        call_args.push_back(ffi::AnyView(opt.value()->expr));
      } else {
        call_args.push_back(ffi::AnyView(val));
      }
    }
    for (const ffi::Any& ea : extra_args) call_args.push_back(ffi::AnyView(ea));

    ffi::Any rv;
    fwd.CallPacked(ffi::PackedArgs(call_args.data(), call_args.size()), &rv);
    return rv;
  };

  return forward_fn.packed();
}

// Module-aware ModuleSpec constructor.
ModuleSpec::ModuleSpec(runtime::ObjectRef mod,
                       ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> spec, bool debug) {
  TVM_FFI_ICHECK(mod.defined() && mod->IsInstance<NNModuleNode>())
      << "ModuleSpec(mod, spec, debug): mod must be an NNModuleNode, got "
      << (mod.defined() ? mod->GetTypeKey() : "null");

  const NNModuleNode* mod_node = mod.as<NNModuleNode>();
  ffi::Map<ffi::String, NNParameter> named_params = mod_node->NamedParameters("");
  ffi::String effect_mode = debug ? ffi::String("plain") : ffi::String("none");

  ffi::Array<ffi::String> method_names;
  ffi::Array<ffi::Any> method_specs;

  for (const auto& [method_name, arg_spec_map] : spec) {
    // Collect ordered arg_names and arg_specs from the per-method map.
    ffi::Array<ffi::String> arg_names;
    ffi::Array<ffi::Any> arg_specs;
    for (const auto& [arg_name, arg_spec] : arg_spec_map) {
      arg_names.push_back(arg_name);
      arg_specs.push_back(arg_spec);
    }

    // Derive the ffi::Function for this method from the module.
    // "forward" maps to "_forward" in reflection; other names are used verbatim.
    ffi::Function forward_fn = DeriveMethodFunction(mod, method_name, arg_names);

    // Build the MethodSpec using the primary (ffi::Function) constructor.
    MethodSpec ms(forward_fn, arg_names, arg_specs, "plain", effect_mode);

    method_names.push_back(method_name);
    method_specs.push_back(ffi::Any(ms));
  }

  data_ = ffi::make_object<ModuleSpecNode>(std::move(method_names), std::move(method_specs),
                                           std::move(named_params),
                                           ffi::Map<ffi::String, runtime::ObjectRef>{});
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  SpecIntNode::RegisterReflection();
  SpecTensorNode::RegisterReflection();
  SpecTupleNode::RegisterReflection();
  MethodSpecNode::RegisterReflection();
  ModuleSpecNode::RegisterReflection();
}

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
