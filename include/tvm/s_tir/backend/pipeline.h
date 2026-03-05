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
 * \file tvm/s_tir/backend/pipeline.h
 * \brief S-TIR Pipeline
 */
#ifndef TVM_S_TIR_BACKEND_PIPELINE_H_
#define TVM_S_TIR_BACKEND_PIPELINE_H_

#include <tvm/s_tir/transform.h>
#include <tvm/tir/pipeline.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace s_tir {
namespace backend {

using Pass = tvm::transform::Pass;
using PassContext = tvm::transform::PassContext;
using tvm::tir::TIRPipeline;
using tvm::transform::CreateModulePass;

// Global passes and reflections
#define JOIN(x, y) x##y
#define MAKE_NAME(a, b) JOIN(a, b)

#define TVM_S_TIR_BACKEND_PIPELINE(Backend, PipelineClass)                                         \
  Pass MAKE_NAME(PipelineClass, Base)(Target target) {                                             \
    auto pass_func = [=](IRModule mod, PassContext pc) {                                           \
      auto pipeline = PipelineClass(target);                                                       \
      mod = pipeline.Base(mod);                                                                    \
      return mod;                                                                                  \
    };                                                                                             \
    return CreateModulePass(/*pass_function=*/pass_func, /*opt_level=*/0,                          \
                            /*pass_name=*/#PipelineClass "Base", /*required=*/{});                 \
  }                                                                                                \
  Pass MAKE_NAME(PipelineClass, Host)(Target target) {                                             \
    auto pass_func = [=](IRModule mod, PassContext pc) {                                           \
      auto pipeline = PipelineClass(target);                                                       \
      mod = pipeline.Host(mod);                                                                    \
      return mod;                                                                                  \
    };                                                                                             \
    return CreateModulePass(/*pass_function=*/pass_func, /*opt_level=*/0,                          \
                            /*pass_name=*/#PipelineClass "Host", /*required=*/{});                 \
  }                                                                                                \
  Pass MAKE_NAME(PipelineClass, Device)(Target target) {                                           \
    auto pass_func = [=](IRModule mod, PassContext pc) {                                           \
      auto pipeline = PipelineClass(target);                                                       \
      mod = pipeline.Device(mod);                                                                  \
      return mod;                                                                                  \
    };                                                                                             \
    return CreateModulePass(/*pass_function=*/pass_func, /*opt_level=*/0,                          \
                            /*pass_name=*/#PipelineClass "Device", /*required=*/{});               \
  }                                                                                                \
  TVM_FFI_STATIC_INIT_BLOCK() {                                                                    \
    namespace refl = tvm::ffi::reflection;                                                         \
    refl::GlobalDef().def("s_tir.pipeline." #Backend ".Base", MAKE_NAME(PipelineClass, Base));     \
    refl::GlobalDef().def("s_tir.pipeline." #Backend ".Host", MAKE_NAME(PipelineClass, Host));     \
    refl::GlobalDef().def("s_tir.pipeline." #Backend ".Device", MAKE_NAME(PipelineClass, Device)); \
  }

// Generic S-TIR Pipeline that can be extended by any backend
class GenericTIRPipeline : public TIRPipeline {
 public:
  explicit GenericTIRPipeline(const Target& target) : TIRPipeline(target) {}
  virtual IRModule Base(IRModule mod);

  // Host mod passes
  IRModule Host(IRModule mod) { return TIRPipeline::Host(mod); }

  // Device mod passes
  IRModule Device(IRModule mod) { return TIRPipeline::Device(mod); }
};

/*! \brief Generic TIR Pipeline base pass for all */
TVM_DLL Pass GenericTIRPipelineBase(Target target);

/*! \brief Generic TIR Pipeline host pass for all */
TVM_DLL Pass GenericTIRPipelineHost(Target target);

/*! \brief Generic TIR Pipeline device pass for all */
TVM_DLL Pass GenericTIRPipelineDevice(Target target);

}  // namespace backend
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_TIR_PIPELINE_H_
