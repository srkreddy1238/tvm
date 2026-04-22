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
 * \brief C++ Exporter: builds a TVM IRModule from a ModuleSpec.
 *
 * Design
 * ------
 * ExportToIRModule(spec, debug) mirrors the Python Exporter.build() logic:
 *
 *   1. If debug=true, emit an _initialize_effect function that creates the
 *      IOEffect null_value object.
 *
 *   2. For each (method_name, method_spec) in the ModuleSpec:
 *      a. Build placeholder Vars for each arg_spec (SpecInt → ShapeVar,
 *         SpecTensor → TensorVar).
 *      b. Build placeholder Vars for each named_param.
 *      c. If debug, add the _io effect Var.
 *      d. Call method_spec.forward(Map<String,Any>{arg_name -> Tensor})
 *         inside a BlockBuilder dataflow scope.
 *      e. Unwrap the return value (Tensor or tuple of Tensors) to relax.Expr.
 *      f. If debug, append the effect output.
 *      g. Emit the function.
 *
 * The forward function receives a Map<String, Any> where each value is a
 * TensorNode (nn.Tensor).  The implementation casts ffi::Any to NNTensor
 * and calls ops.  No Python inspect.signature is needed.
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
 * \brief Build a TVM IRModule from a ModuleSpec.
 *
 * \param spec    The ModuleSpec describing methods and parameters.
 * \param debug   If true, add IOEffect (_io) to every method signature.
 * \return        The compiled IRModule.
 */
IRModule ExportToIRModule(ModuleSpec spec, bool debug);

/*!
 * \brief Get the current debug _io Var from the thread-local storage.
 * Set by EmitMethod before calling forward(), updated by NNDebugFunc.
 */
ffi::Optional<Var> GetCurrentIOVar();

/*!
 * \brief Set the current debug _io Var in the thread-local storage.
 * Called by NNDebugFunc after emitting each debug call to chain effects.
 */
void SetCurrentIOVar(ffi::Optional<Var> v);

// ===========================================================================
// Exporter class - wraps ExportToIRModule with support for spec.Object
// ===========================================================================

/*!
 * \brief Exporter node that wraps ExportToIRModule functionality.
 *
 * This class provides a stateful interface for building IRModules,
 * maintaining a BlockBuilder and tracking external modules.
 * The Python Exporter class wraps this for the fast path (no spec.Object).
 */
class ExporterNode : public runtime::Object {
 public:
  BlockBuilder builder;
  bool debug;
  ffi::Array<runtime::ObjectRef> extern_mods;  // Array of ExternModule objects

  ExporterNode(bool debug);

  void AddExternalModule(runtime::ObjectRef extern_mod);

  /*!
   * \brief Build the ModuleSpec to TVM IRModule.
   * \param spec The ModuleSpec to export.
   * \return Array[IRModule, named_params as Map, extern_mods as Array]
   */
  ffi::Array<ffi::Any> Build(ModuleSpec spec);

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
 * Returns nullopt if no Exporter is active.
 */
ffi::Optional<Exporter> Exporter_Current();

/*!
 * \brief Install/restore the thread-local current Exporter.
 * Pass nullptr to clear.
 */
void Exporter_SetCurrent(ExporterNode* exporter);

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_EXPORTER_H_
