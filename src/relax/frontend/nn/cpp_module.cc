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
 * \file src/relax/frontend/nn/cpp_module.cc
 * \brief CppModule implementation: C++ port of Python's TorchModule + Module.jit().
 */

#include "cpp_module.h"

#include <tvm/driver/compile.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/vm.h>

#include <string>
#include <vector>

#include "exporter.h"
#include "spec.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ===========================================================================
// Internal helpers
// ===========================================================================

/*!
 * \brief Load and initialise a VM from a VMExecutable module.
 *
 * Mirrors the CompileToVM helper in test_nn_debug.cc:
 *   1. Cast to VMExecutable.
 *   2. Call VMLoadExecutable() to get the VM module.
 *   3. Call vm_initialization for the given device.
 */
static ffi::Module LoadAndInitVM(const ffi::Module& vm_mod, tvm::Device device) {
  auto vm_ex = vm_mod.as<runtime::vm::VMExecutable>();
  TVM_FFI_ICHECK(vm_ex) << "Jit: Compile did not return a VMExecutable";
  ffi::Module vm = vm_ex->VMLoadExecutable();

  // vm_initialization(dev_type, dev_id, allocator_type,  -- host device
  //                   dev_type, dev_id, allocator_type)  -- device
  std::vector<ffi::AnyView> init_args = {
      device.device_type,
      static_cast<int>(device.device_id),
      runtime::memory::AllocatorType::kPooled,
      device.device_type,
      static_cast<int>(device.device_id),
      runtime::memory::AllocatorType::kPooled,
  };
  ffi::Any rv;
  vm->GetFunction("vm_initialization")
      .value()
      .CallPacked(ffi::PackedArgs(init_args.data(), init_args.size()), &rv);
  return vm;
}

/*!
 * \brief Call _initialize_effect() on the VM and return the flat effect array.
 *
 * _initialize_effect() returns R.Tuple(effect0, effect1, ...).
 * We fetch each element individually via GetOutputRec and return them as a
 * flat ffi::Array<ffi::Any> so they can be spread into subsequent calls.
 *
 * Returns an empty array when the VM has no _initialize_effect function.
 */
static ffi::Array<ffi::Any> InitEffects(const ffi::Module& vm) {
  // Check whether _initialize_effect exists.
  auto set_input_fn = vm->GetFunction("set_input");
  auto invoke_fn = vm->GetFunction("invoke_stateful");
  auto get_arity_fn = vm->GetFunction("get_output_arity");
  if (!set_input_fn.has_value() || !invoke_fn.has_value() || !get_arity_fn.has_value()) {
    return {};
  }

  // Try to set_input for _initialize_effect; if it throws, there are no effects.
  try {
    std::vector<ffi::AnyView> set_args = {ffi::String("_initialize_effect")};
    ffi::Any rv;
    set_input_fn.value().CallPacked(ffi::PackedArgs(set_args.data(), set_args.size()), &rv);
  } catch (...) {
    return {};
  }

  invoke_fn.value()(ffi::String("_initialize_effect"));

  // Determine the arity of the returned tuple.
  std::vector<ffi::AnyView> arity_args = {ffi::String("_initialize_effect")};
  ffi::Any arity_rv;
  get_arity_fn.value().CallPacked(ffi::PackedArgs(arity_args.data(), arity_args.size()), &arity_rv);
  int64_t arity = arity_rv.cast<int64_t>();

  ffi::Array<ffi::Any> effects;
  if (arity == -1) {
    // Scalar return (shouldn't happen for _initialize_effect, but handle it).
    ffi::Function get_output = vm->GetFunction("get_output").value();
    ffi::Any out;
    get_output.CallPacked(ffi::PackedArgs(arity_args.data(), arity_args.size()), &out);
    effects.push_back(out);
  } else {
    for (int64_t i = 0; i < arity; ++i) {
      effects.push_back(
          CppModuleNode::GetOutputRec(vm, "_initialize_effect", {static_cast<int>(i)}));
    }
  }
  return effects;
}

// ===========================================================================
// CppModuleNode
// ===========================================================================

CppModuleNode::CppModuleNode(ffi::Module vm_, ModuleSpec spec_, ffi::Array<runtime::Tensor> params_,
                             tvm::Device device_)
    : vm(std::move(vm_)), spec(std::move(spec_)), params(std::move(params_)), device(device_) {
  effects = InitEffects(vm);
}

/* static */
ffi::Any CppModuleNode::GetOutputRec(const ffi::Module& vm, const std::string& func_name,
                                     const std::vector<int>& indices) {
  ffi::Function get_arity = vm->GetFunction("get_output_arity").value();
  ffi::Function get_output = vm->GetFunction("get_output").value();

  std::vector<ffi::AnyView> arity_args;
  arity_args.push_back(ffi::String(func_name));
  for (int idx : indices) arity_args.push_back(idx);

  ffi::Any arity_rv;
  get_arity.CallPacked(ffi::PackedArgs(arity_args.data(), arity_args.size()), &arity_rv);
  int64_t arity = arity_rv.cast<int64_t>();

  if (arity == -1) {
    // Leaf: fetch the value directly.
    ffi::Any out_rv;
    get_output.CallPacked(ffi::PackedArgs(arity_args.data(), arity_args.size()), &out_rv);
    return out_rv;
  }

  // Tuple node: recurse for each child.
  ffi::Array<ffi::Any> result;
  for (int64_t i = 0; i < arity; ++i) {
    std::vector<int> child_indices = indices;
    child_indices.push_back(static_cast<int>(i));
    result.push_back(GetOutputRec(vm, func_name, child_indices));
  }
  return ffi::Any(result);
}

ffi::Any CppModuleNode::RunMethod(const std::string& method_name,
                                  const std::vector<ffi::AnyView>& user_inputs) {
  // set_input(method_name, *user_inputs, *effects, *params)
  std::vector<ffi::AnyView> set_args;
  set_args.push_back(ffi::String(method_name));
  for (const auto& a : user_inputs) set_args.push_back(a);
  for (const auto& e : effects) set_args.push_back(ffi::AnyView(e));
  for (const auto& p : params) set_args.push_back(ffi::AnyView(p));

  ffi::Any rv;
  vm->GetFunction("set_input")
      .value()
      .CallPacked(ffi::PackedArgs(set_args.data(), set_args.size()), &rv);
  vm->GetFunction("invoke_stateful").value()(ffi::String(method_name));

  // Determine whether the output is (result, effect_tuple) or just result.
  // We check the top-level arity: if it is 2 and effects is non-empty, the
  // second element is the updated effect tuple.
  std::vector<ffi::AnyView> top_arity_args = {ffi::String(method_name)};
  ffi::Any top_arity_rv;
  vm->GetFunction("get_output_arity")
      .value()
      .CallPacked(ffi::PackedArgs(top_arity_args.data(), top_arity_args.size()), &top_arity_rv);
  int64_t top_arity = top_arity_rv.cast<int64_t>();

  bool has_effect_output = !effects.empty() && (top_arity == 2);

  if (has_effect_output) {
    // Output is Tuple(result, Tuple(new_effect0, new_effect1, ...)).
    // Fetch result from index 0.
    ffi::Any output = GetOutputRec(vm, method_name, {0});

    // Fetch updated effects from index 1.
    std::vector<ffi::AnyView> eff_arity_args = {ffi::String(method_name), 1};
    ffi::Any eff_arity_rv;
    vm->GetFunction("get_output_arity")
        .value()
        .CallPacked(ffi::PackedArgs(eff_arity_args.data(), eff_arity_args.size()), &eff_arity_rv);
    int64_t eff_arity = eff_arity_rv.cast<int64_t>();

    ffi::Array<ffi::Any> new_effects;
    if (eff_arity == -1) {
      // Single effect.
      new_effects.push_back(GetOutputRec(vm, method_name, {1}));
    } else {
      for (int64_t i = 0; i < eff_arity; ++i) {
        new_effects.push_back(GetOutputRec(vm, method_name, {1, static_cast<int>(i)}));
      }
    }
    effects = new_effects;
    return output;
  } else {
    // No effect output: return the raw result.
    return GetOutputRec(vm, method_name, {});
  }
}

CppMethodCaller CppModuleNode::GetMethod(const std::string& method_name) const {
  const ModuleSpecNode* ms_node = spec.get();
  for (size_t i = 0; i < ms_node->method_names.size(); ++i) {
    if (std::string(ms_node->method_names[i]) == method_name) {
      auto opt = ms_node->method_specs[i].try_cast<MethodSpec>();
      TVM_FFI_ICHECK(opt.has_value()) << "CppModule: method_spec[" << i << "] is not a MethodSpec";
      return CppMethodCaller(const_cast<CppModuleNode*>(this), method_name, opt.value());
    }
  }
  TVM_FFI_THROW(ValueError) << "CppModule: method '" << method_name << "' not found in spec";
  TVM_FFI_UNREACHABLE();
}

// ===========================================================================
// CppMethodCaller::operator()
// ===========================================================================

ffi::Any CppMethodCaller::operator()(const std::vector<ffi::Any>& args) const {
  const MethodSpecNode* ms = spec_.get();
  TVM_FFI_ICHECK_EQ(args.size(), ms->arg_specs.size())
      << "CppModule['" << method_name_ << "']: expected " << ms->arg_specs.size()
      << " arguments, got " << args.size();

  // Convert each argument to the VM-level representation:
  //   SpecTensor → runtime::Tensor (passed as-is)
  //   SpecInt    → ffi::Shape([value])  (mirrors Python ShapeTuple([v]))
  //   SpecTuple  → ffi::Array<ffi::Any> (recursively converted)
  std::function<ffi::AnyView(const ffi::Any&, const ffi::Any&)> convert_arg;
  convert_arg = [&](const ffi::Any& val, const ffi::Any& spec_any) -> ffi::AnyView {
    auto spec_obj_opt = spec_any.try_cast<runtime::ObjectRef>();
    TVM_FFI_ICHECK(spec_obj_opt.has_value()) << "CppMethodCaller: spec is not an ObjectRef";
    runtime::ObjectRef spec_obj = spec_obj_opt.value();

    if (spec_obj->IsInstance<SpecTensorNode>()) {
      // Expect runtime::Tensor; pass through.
      return ffi::AnyView(val);
    }
    if (spec_obj->IsInstance<SpecIntNode>()) {
      // Expect ffi::Shape([value]) — already in the right form.
      return ffi::AnyView(val);
    }
    if (spec_obj->IsInstance<SpecTupleNode>()) {
      // val should be ffi::Array<ffi::Any>; pass through.
      return ffi::AnyView(val);
    }
    TVM_FFI_THROW(TypeError) << "CppMethodCaller: unsupported spec type: "
                             << spec_obj->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  };

  std::vector<ffi::AnyView> vm_inputs;
  vm_inputs.reserve(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    vm_inputs.push_back(convert_arg(args[i], ms->arg_specs[i]));
  }

  return owner_->RunMethod(method_name_, vm_inputs);
}

// ===========================================================================
// Jit()
// ===========================================================================

CppModule Jit(ModuleSpec spec, tvm::Device device, ffi::String pipeline, bool debug,
              ffi::Array<runtime::ObjectRef> extern_mods) {
  // 1. Export to IRModule.
  IRModule mod = ExportToIRModule(spec, debug);

  // 2. Attach external modules if any.
  if (!extern_mods.empty()) {
    auto attach_pass = relax::transform::AttachExternModules(extern_mods);
    mod = attach_pass(mod);
  }

  // 3. Compile to VM.
  // Build a Target from the device type.
  std::string target_str;
  switch (device.device_type) {
    case kDLCPU:
      target_str = "llvm";
      break;
    case kDLCUDA:
      target_str = "cuda";
      break;
    default:
      target_str = "llvm";
      break;
  }
  tvm::Target target = tvm::Target::WithHost(tvm::Target(target_str), tvm::Target("llvm"));

  ffi::Map<ffi::Any, ffi::ObjectRef> empty_params;
  ffi::Module vm_mod =
      tvm::driver::Compile(mod, target, empty_params, ffi::Optional<ffi::String>(pipeline),
                           ffi::Optional<ffi::String>(ffi::String("generic")));

  // 4. Load and initialise the VM.
  ffi::Module vm = LoadAndInitVM(vm_mod, device);

  // 5. Collect bound parameter tensors in named_params order.
  // Parameters are unbound at export time (no data); they are passed as
  // function arguments.  For now we return an empty params array — callers
  // that need bound params should bind them via load_state_dict before Jit.
  ffi::Array<runtime::Tensor> params;
  // (named_params from the spec have no data bound at export time)

  // 6. Wrap in CppModule.
  return CppModule(vm, spec, params, device);
}

// ===========================================================================
// CppModule explicit constructor
// ===========================================================================

CppModule::CppModule(ffi::Module vm, ModuleSpec spec, ffi::Array<runtime::Tensor> params,
                     tvm::Device device) {
  data_ =
      ffi::make_object<CppModuleNode>(std::move(vm), std::move(spec), std::move(params), device);
}

// ===========================================================================
// FFI registration
// ===========================================================================

TVM_FFI_STATIC_INIT_BLOCK() {
  CppModuleNode::RegisterReflection();

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.frontend.nn.Jit",
                        [](ModuleSpec spec, int device_type, int device_id, ffi::String pipeline,
                           bool debug, ffi::Array<runtime::ObjectRef> extern_mods) -> CppModule {
                          tvm::Device dev{static_cast<DLDeviceType>(device_type), device_id};
                          return Jit(std::move(spec), dev, std::move(pipeline), debug,
                                     std::move(extern_mods));
                        });
}

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
