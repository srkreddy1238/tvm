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
 * \file tvm/relax/backend/pipeline.h
 * \brief Pipeline base
 */
#ifndef TVM_RELAX_BACKEND_PIPELINE_H_
#define TVM_RELAX_BACKEND_PIPELINE_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/transform.h>
namespace tvm {
namespace relax {
namespace backend {

using Pass = tvm::transform::Pass;
using PassContext = tvm::transform::PassContext;
using tvm::transform::CreateModulePass;

// Base of all pipelines

class TVM_DLL RelaxPipeline {
 public:
  // Constructor
  explicit RelaxPipeline(const Target& target) : scope_(target) {}

  // Libarary passes
  virtual IRModule Library(IRModule mod) = 0;
  // Legalization
  virtual IRModule Legalize(IRModule mod) = 0;
  // Dalaflow lowering
  virtual IRModule Dataflow(IRModule mod) = 0;
  // Finalize
  virtual IRModule Finalize(IRModule mod) = 0;

 private:
  With<Target> scope_;
};

// Global passes and reflections
#define JOIN(x, y) x##y
#define MAKE_NAME(a, b) JOIN(a, b)

#define TVM_RELAX_BACKEND_PIPELINE(Backend, PipelineClass)                                       \
  Pass MAKE_NAME(PipelineClass, All)(Target target) {                                            \
    auto pass_func = [=](IRModule mod, PassContext pc) {                                         \
      auto pipeline = PipelineClass(target);                                                     \
      mod = pipeline.Library(mod);                                                               \
      mod = pipeline.Legalize(mod);                                                              \
      mod = pipeline.Dataflow(mod);                                                              \
      mod = pipeline.Finalize(mod);                                                              \
      return mod;                                                                                \
    };                                                                                           \
    return CreateModulePass(/*pass_function=*/pass_func, /*opt_level=*/0,                        \
                            /*pass_name=*/#PipelineClass, /*required=*/{});                      \
  }                                                                                              \
  Pass MAKE_NAME(PipelineClass, Library)(Target target) {                                        \
    auto pass_func = [=](IRModule mod, PassContext pc) {                                         \
      auto pipeline = PipelineClass(target);                                                     \
      mod = pipeline.Library(mod);                                                               \
      return mod;                                                                                \
    };                                                                                           \
    return CreateModulePass(/*pass_function=*/pass_func, /*opt_level=*/0,                        \
                            /*pass_name=*/#PipelineClass "Library", /*required=*/{});            \
  }                                                                                              \
  Pass MAKE_NAME(PipelineClass, Legalize)(Target target) {                                       \
    auto pass_func = [=](IRModule mod, PassContext pc) {                                         \
      auto pipeline = PipelineClass(target);                                                     \
      mod = pipeline.Legalize(mod);                                                              \
      return mod;                                                                                \
    };                                                                                           \
    return CreateModulePass(/*pass_function=*/pass_func, /*opt_level=*/0,                        \
                            /*pass_name=*/#PipelineClass "Legalize", /*required=*/{});           \
  }                                                                                              \
  Pass MAKE_NAME(PipelineClass, Dataflow)(Target target) {                                       \
    auto pass_func = [=](IRModule mod, PassContext pc) {                                         \
      auto pipeline = PipelineClass(target);                                                     \
      mod = pipeline.Dataflow(mod);                                                              \
      return mod;                                                                                \
    };                                                                                           \
    return CreateModulePass(/*pass_function=*/pass_func, /*opt_level=*/0,                        \
                            /*pass_name=*/#PipelineClass "Dataflow", /*required=*/{});           \
  }                                                                                              \
  Pass MAKE_NAME(PipelineClass, Finalize)(Target target) {                                       \
    auto pass_func = [=](IRModule mod, PassContext pc) {                                         \
      auto pipeline = PipelineClass(target);                                                     \
      mod = pipeline.Finalize(mod);                                                              \
      return mod;                                                                                \
    };                                                                                           \
    return CreateModulePass(/*pass_function=*/pass_func, /*opt_level=*/0,                        \
                            /*pass_name=*/#PipelineClass "Finalize", /*required=*/{});           \
  }                                                                                              \
  TVM_FFI_STATIC_INIT_BLOCK() {                                                                  \
    namespace refl = tvm::ffi::reflection;                                                       \
    refl::GlobalDef().def("relax.backend." #Backend ".Pipeline", MAKE_NAME(PipelineClass, All)); \
    refl::GlobalDef().def("relax.backend." #Backend ".PipelineLibrary",                          \
                          MAKE_NAME(PipelineClass, Library));                                    \
    refl::GlobalDef().def("relax.backend." #Backend ".PipelineLegalize",                         \
                          MAKE_NAME(PipelineClass, Legalize));                                   \
    refl::GlobalDef().def("relax.backend." #Backend ".PipelineDataflow",                         \
                          MAKE_NAME(PipelineClass, Dataflow));                                   \
    refl::GlobalDef().def("relax.backend." #Backend ".PipelineFinalize",                         \
                          MAKE_NAME(PipelineClass, Finalize));                                   \
  }

// Transform declarations exported

#define TVM_RELAX_BACKEND_PIPELINE_DECL(PipelineClass)                      \
  class PipelineClass : RelaxPipeline {                                     \
   public:                                                                  \
    explicit PipelineClass(const Target& target) : RelaxPipeline(target) {} \
    virtual IRModule Library(IRModule mod);                                 \
    virtual IRModule Legalize(IRModule mod);                                \
    virtual IRModule Dataflow(IRModule mod);                                \
    virtual IRModule Finalize(IRModule mod);                                \
  };                                                                        \
  TVM_DLL Pass MAKE_NAME(PipelineClass, All)(Target target);                \
  TVM_DLL Pass MAKE_NAME(PipelineClass, Library)(Target target);            \
  TVM_DLL Pass MAKE_NAME(PipelineClass, Legalize)(Target target);           \
  TVM_DLL Pass MAKE_NAME(PipelineClass, Dataflow)(Target target);           \
  TVM_DLL Pass MAKE_NAME(PipelineClass, Finalize)(Target target);

namespace cpu_generic {
/*
 * CPU Generic Relax Pipelines resolved as
 *
 * tvm::relax::backend::CPURelaxPipelineAll
 * tvm::relax::backend::CPURelaxPipelineLibrary
 * tvm::relax::backend::CPURelaxPipelineLegalize
 * tvm::relax::backend::CPURelaxPipelineDataflow
 * tvm::relax::backend::CPURelaxPipelineFInalize
 *
 */

TVM_RELAX_BACKEND_PIPELINE_DECL(CPURelaxPipeline);

}  // namespace cpu_generic

namespace gpu_generic {
/*
 * GPU Generic Relax Pipelines resolved as
 *
 * tvm::relax::backend::GPURelaxPipelineAll
 * tvm::relax::backend::GPURelaxPipelineLibrary
 * tvm::relax::backend::GPURelaxPipelineLegalize
 * tvm::relax::backend::GPURelaxPipelineDataflow
 * tvm::relax::backend::GPURelaxPipelineFInalize
 *
 */

TVM_RELAX_BACKEND_PIPELINE_DECL(GPURelaxPipeline);
}  // namespace gpu_generic

}  // namespace backend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BACKEND_PIPELINE_H_
