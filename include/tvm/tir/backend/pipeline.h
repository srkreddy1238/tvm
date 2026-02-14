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
 * \file tvm/tir/backend/pipeline.h
 * \brief TIRPipeline base
 */
#ifndef TVM_TIR_BACKEND_PIPELINE_H_
#define TVM_TIR_BACKEND_PIPELINE_H_

#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {
namespace backend {

using Pass = tvm::transform::Pass;
using PassContext = tvm::transform::PassContext;
using tvm::transform::CreateModulePass;

// Base of all pipelines
class TVM_DLL TIRPipeline {
 public:
  // Constructor
  TIRPipeline(const Target& target) : scope_(target) {
  }

  // TIR Passes
  virtual IRModule Base(IRModule& mod) = 0;
  // Host mod passes
  virtual IRModule Host(IRModule& mod) = 0;
  // Device mod passes
  virtual IRModule Device(IRModule& mod) = 0;

 private:
  With<Target> scope_;
};

// Global passes and reflections
#define JOIN(x, y) x##y
#define MAKE_NAME(a, b) JOIN(a, b)

#define TVM_TIR_BACKEND_PIPELINE(Backend, PipelineClass)                                         \
  Pass MAKE_NAME(PipelineClass, Base)(Target target) {                                           \
    auto pass_func = [=](IRModule mod, PassContext pc) {                                         \
      auto pipeline = PipelineClass(target);                                                     \
      mod = pipeline.Base(mod);                                                                  \
      return mod;                                                                                \
    };                                                                                           \
    return CreateModulePass(/*pass_function=*/pass_func, /*opt_level=*/0,                        \
                            /*pass_name=*/#PipelineClass "Base", /*required=*/{});               \
  }                                                                                              \
  Pass MAKE_NAME(PipelineClass, Host)(Target target) {                                           \
    auto pass_func = [=](IRModule mod, PassContext pc) {                                         \
      auto pipeline = PipelineClass(target);                                                     \
      mod = pipeline.Host(mod);                                                                  \
      return mod;                                                                                \
    };                                                                                           \
    return CreateModulePass(/*pass_function=*/pass_func, /*opt_level=*/0,                        \
                            /*pass_name=*/#PipelineClass "Host", /*required=*/{});               \
  }                                                                                              \
  Pass MAKE_NAME(PipelineClass, Device)(Target target) {                                         \
    auto pass_func = [=](IRModule mod, PassContext pc) {                                         \
      auto pipeline = PipelineClass(target);                                                     \
      mod = pipeline.Device(mod);                                                                \
      return mod;                                                                                \
    };                                                                                           \
    return CreateModulePass(/*pass_function=*/pass_func, /*opt_level=*/0,                        \
                            /*pass_name=*/#PipelineClass "Device", /*required=*/{});             \
  }                                                                                              \
  TVM_FFI_STATIC_INIT_BLOCK() {                                                                  \
    namespace refl = tvm::ffi::reflection;                                                       \
    refl::GlobalDef().def("tir.pipeline." #Backend ".Base", MAKE_NAME(PipelineClass, Base));     \
    refl::GlobalDef().def("tir.pipeline." #Backend ".Host", MAKE_NAME(PipelineClass, Host));     \
    refl::GlobalDef().def("tir.pipeline." #Backend ".Device", MAKE_NAME(PipelineClass, Device)); \
  }

// Transform declarations exported

#define TVM_TIR_BACKEND_PIPELINE_DECL(PipelineClass)                      \
  class PipelineClass : TIRPipeline {                                     \
   public:                                                                \
    explicit PipelineClass(const Target& target) : TIRPipeline(target) {} \
    virtual IRModule Base(IRModule& mod);                                 \
    virtual IRModule Host(IRModule& mod);                                 \
    virtual IRModule Device(IRModule& mod);                               \
  };                                                                      \
  TVM_DLL Pass MAKE_NAME(PipelineClass, Base)(Target target);             \
  TVM_DLL Pass MAKE_NAME(PipelineClass, Host)(Target target);             \
  TVM_DLL Pass MAKE_NAME(PipelineClass, Device)(Target target);

/*
 * Generic TIR Pipelines resolved as
 *
 * tvm::tir::backend::GenericTIRPipelineBase
 * tvm::tir::backend::GenericTIRPipelineHost
 * tvm::tir::backend::GenericTIRPipelineDevice
 *
 */

TVM_TIR_BACKEND_PIPELINE_DECL(GenericTIRPipeline);

}  // namespace backend
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_BACKEND_PIPELINE_H_
