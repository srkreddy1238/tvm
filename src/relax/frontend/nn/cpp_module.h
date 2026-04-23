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
 * \file src/relax/frontend/nn/cpp_module.h
 * \brief CppModule: C++ port of Python's TorchModule.
 *
 * CppModule wraps a compiled VirtualMachine together with the module's
 * named parameters and effect state, providing a callable interface that
 * mirrors Python's TorchModule.__getitem__(method_name)(*args).
 *
 * Jit() is the C++ equivalent of Python's Module.jit():
 *   1. Export the ModuleSpec to an IRModule via ExportToIRModule.
 *   2. Attach any ExternModules via the AttachExternModules pass.
 *   3. Compile the IRModule to a VM executable via tvm::driver::Compile.
 *   4. Wrap the result in a CppModule.
 *
 * CppModule::operator[](method_name) returns a CppMethodCaller that accepts
 * runtime::Tensor / ffi::Shape arguments and returns runtime::Tensor outputs,
 * automatically threading the effect state through each call.
 *
 * Design mirrors torch.py:TorchModule:
 *   - effects  : ffi::Array<ffi::Any>  — current effect state (updated in-place)
 *   - params   : ffi::Array<runtime::Tensor> — bound parameter tensors
 *   - spec     : ModuleSpec            — for method/arg-spec lookup
 *   - vm       : ffi::Module           — the loaded VirtualMachine
 */

#ifndef TVM_RELAX_FRONTEND_NN_CPP_MODULE_H_
#define TVM_RELAX_FRONTEND_NN_CPP_MODULE_H_

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/tensor.h>

#include <functional>
#include <string>
#include <vector>

#include "exporter.h"
#include "spec.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
class CppModule;
class CppModuleNode;

// ---------------------------------------------------------------------------
// CppMethodCaller
//
// Returned by CppModule::operator[](method_name).
// Accepts a vector of ffi::Any arguments (runtime::Tensor or ffi::Shape for
// SpecInt), calls the VM function, updates the effect state, and returns the
// output tensor(s).
//
// This is a lightweight value type (not an ObjectRef) since it holds a
// back-pointer to the owning CppModuleNode.
// ---------------------------------------------------------------------------
class CppMethodCaller {
 public:
  /*!
   * \brief Call the method with the given user arguments.
   *
   * \param args  One entry per arg_spec in the MethodSpec:
   *              - SpecTensor → runtime::Tensor
   *              - SpecInt    → ffi::Shape([value])  (mirrors Python ShapeTuple)
   *              - SpecTuple  → ffi::Array<ffi::Any> of the above, recursively
   * \return Output tensor(s):
   *              - Single tensor → runtime::Tensor
   *              - Tuple output  → ffi::Array<ffi::Any>
   */
  ffi::Any operator()(const std::vector<ffi::Any>& args) const;

  /*! \brief The method name (for error messages). */
  const std::string& name() const { return method_name_; }

 private:
  friend class CppModuleNode;
  CppMethodCaller(CppModuleNode* owner, std::string method_name, MethodSpec spec)
      : owner_(owner), method_name_(std::move(method_name)), spec_(std::move(spec)) {}

  CppModuleNode* owner_;
  std::string method_name_;
  MethodSpec spec_;
};

// ---------------------------------------------------------------------------
// CppModuleNode
//
// C++ port of Python's TorchModule.  Holds the VM, params, effects, and spec.
// ---------------------------------------------------------------------------
class CppModuleNode : public runtime::Object {
 public:
  /*! \brief The loaded VirtualMachine module. */
  ffi::Module vm;

  /*! \brief The ModuleSpec (for method/arg-spec lookup). */
  ModuleSpec spec;

  /*! \brief Bound parameter tensors in named_params order. */
  ffi::Array<runtime::Tensor> params;

  /*!
   * \brief Current effect state.
   * Initialised by calling _initialize_effect() on construction.
   * Updated in-place after each method call.
   * Null/empty when the module has no effects.
   */
  ffi::Array<ffi::Any> effects;

  /*! \brief The device on which tensors live. */
  tvm::Device device;

  /*!
   * \brief Construct and initialise effects.
   * \param vm      Loaded VM module.
   * \param spec    ModuleSpec.
   * \param params  Bound parameter tensors.
   * \param device  Execution device.
   */
  CppModuleNode(ffi::Module vm, ModuleSpec spec, ffi::Array<runtime::Tensor> params,
                tvm::Device device);

  /*!
   * \brief Return a CppMethodCaller for the named method.
   * Throws if the method is not found in the spec.
   */
  CppMethodCaller GetMethod(const std::string& method_name) const;

  // ---- VM helpers (mirrors test_nn_debug.cc helpers) ---------------------

  /*!
   * \brief Recursively fetch a (possibly nested) output from the VM.
   * Returns runtime::Tensor for leaf outputs, ffi::Array<ffi::Any> for tuples.
   */
  static ffi::Any GetOutputRec(const ffi::Module& vm, const std::string& func_name,
                               const std::vector<int>& indices);

  /*!
   * \brief Run a VM function with the given inputs and return the raw output.
   * Updates effects in-place when the function returns (output, effect_tuple).
   */
  ffi::Any RunMethod(const std::string& method_name, const std::vector<ffi::AnyView>& inputs);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CppModuleNode>()
        .def(refl::init<ffi::Module, ModuleSpec, ffi::Array<runtime::Tensor>, tvm::Device>())
        .def_rw("vm", &CppModuleNode::vm)
        .def_rw("spec", &CppModuleNode::spec)
        .def_rw("params", &CppModuleNode::params)
        .def_rw("effects", &CppModuleNode::effects);
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.CppModule", CppModuleNode, runtime::Object);
};

class CppModule : public runtime::ObjectRef {
 public:
  explicit CppModule(ffi::Module vm, ModuleSpec spec, ffi::Array<runtime::Tensor> params,
                     tvm::Device device);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CppModule, runtime::ObjectRef, CppModuleNode);

  /*!
   * \brief Return a CppMethodCaller for the named method.
   * Equivalent to Python's TorchModule.__getitem__(method_name).
   */
  CppMethodCaller operator[](const std::string& method_name) const {
    return get()->GetMethod(method_name);
  }
};

// ---------------------------------------------------------------------------
// Jit()
//
// C++ equivalent of Python's Module.jit().
//
// Steps:
//   1. ExportToIRModule(spec, debug)
//   2. AttachExternModules(extern_mods)(mod)   [if extern_mods non-empty]
//   3. tvm::driver::Compile(mod, target, params, pipeline)
//   4. Wrap in CppModule
//
// \param spec          The ModuleSpec to compile.
// \param device        Execution device (default: CPU 0).
// \param pipeline      Relax pipeline name (default: "cpu_generic").
// \param debug         If true, add IOEffect to every method.
// \param extern_mods   External modules to attach (from nn.add_extern).
// \return              A CppModule ready for inference.
// ---------------------------------------------------------------------------
CppModule Jit(ModuleSpec spec, tvm::Device device = {kDLCPU, 0},
              ffi::String pipeline = "cpu_generic", bool debug = false,
              ffi::Array<runtime::ObjectRef> extern_mods = {});

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_CPP_MODULE_H_
