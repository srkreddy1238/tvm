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
 * \file src/relax/frontend/nn/exporter.h
 * \brief IRModule builder for the nn module frontend.
 *
 * ExporterNode drives the compilation of a ModuleSpec into a TVM IRModule:
 *
 *   1. When effects are present or debug=true, emits an _initialize_effect
 *      function that allocates the initial effect state objects.
 *
 *   2. For each (method_name, method_spec) in the ModuleSpec:
 *      a. Builds placeholder Vars for each arg_spec.
 *      b. Builds parameter Vars according to param_mode.
 *      c. Adds effect Vars according to effect_mode.
 *      d. Calls method_spec.forward(named_args) inside a dataflow block.
 *      e. Unwraps the return value to a relax::Expr.
 *      f. Appends effect outputs when effect_mode is not "none".
 *      g. Emits the completed function into the IRModule.
 */

#ifndef TVM_RELAX_FRONTEND_NN_EXPORTER_H_
#define TVM_RELAX_FRONTEND_NN_EXPORTER_H_

#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>

#include "spec.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

/*!
 * \brief Get the current debug _io Var from thread-local storage.
 *
 * Set by the exporter before calling forward(); updated by the debug
 * function wrapper after each call to chain the effect token.
 */
ffi::Optional<Var> GetCurrentIOVar();

/*!
 * \brief Set the current debug _io Var in thread-local storage.
 *
 * \param v  The new _io Var, or nullopt to clear.
 */
void SetCurrentIOVar(ffi::Optional<Var> v);

/*!
 * \brief Builds a TVM IRModule from a ModuleSpec.
 *
 * Maintains a BlockBuilder and a list of external modules across
 * multiple Build() calls.  The Python Exporter class delegates to this
 * node for the common case (no spec.Object arguments).
 */
class ExporterNode : public runtime::Object {
 public:
  /*! \brief The BlockBuilder used to accumulate the IRModule. */
  BlockBuilder builder;
  /*! \brief Whether to add an IOEffect token to every method signature. */
  bool debug;
  /*! \brief External modules registered via nn.add_extern. */
  ffi::Array<runtime::ObjectRef> extern_mods;

  explicit ExporterNode(bool debug);

  /*!
   * \brief Register an external module with this exporter.
   *
   * \param extern_mod  The external module object to register.
   */
  void AddExternalModule(runtime::ObjectRef extern_mod);

  /*!
   * \brief Build the ModuleSpec into a TVM IRModule.
   *
   * \param spec  The ModuleSpec to compile.
   * \return      Array of [IRModule, named_params Map, extern_mods Array].
   */
  ffi::Array<ffi::Any> Build(ModuleSpec spec);

  /*!
   * \brief Internal implementation: compile a ModuleSpec into an IRModule.
   *
   * \param spec   The ModuleSpec to compile.
   * \param debug  If true, add an IOEffect token to every method signature.
   * \return       The compiled IRModule.
   */
  IRModule ExportToIRModule(ModuleSpec spec, bool debug);

  static void RegisterReflection();

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.Exporter", ExporterNode, runtime::Object);
};

class Exporter : public runtime::ObjectRef {
 public:
  explicit Exporter(bool debug);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Exporter, runtime::ObjectRef, ExporterNode);
};

/*!
 * \brief Get the current Exporter from thread-local storage.
 *
 * Returns nullopt when no export is in progress.
 */
ffi::Optional<Exporter> Exporter_Current();

/*!
 * \brief Install or clear the thread-local current Exporter.
 *
 * \param exporter  Pointer to the ExporterNode to install, or nullptr to clear.
 */
void Exporter_SetCurrent(ExporterNode* exporter);

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_EXPORTER_H_
