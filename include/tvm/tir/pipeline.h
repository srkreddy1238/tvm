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
 * \file tvm/tir/pipeline.h
 * \brief TIRPipeline base
 */
#ifndef TVM_TIR_PIPELINE_H_
#define TVM_TIR_PIPELINE_H_

#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

using Pass = tvm::transform::Pass;
using PassContext = tvm::transform::PassContext;
using tvm::transform::CreateModulePass;

// Base of all pipelines
class TVM_DLL TIRPipeline {
 public:
  // Constructor
  explicit TIRPipeline(const Target& target) : scope_(target) {}

  // TIR Passes
  virtual IRModule Base(IRModule mod) = 0;

  // Host mod passes
  IRModule Host(IRModule mod) {
    mod = tir::transform::LowerTVMBuiltin()(mod);
    mod = tir::transform::LowerCustomDatatypes()(mod);
    mod = tir::transform::LowerIntrin()(mod);
    return mod;
  }

  // Device mod passes
  IRModule Device(IRModule mod) {
    mod = tir::transform::LowerWarpMemory()(mod);
    mod = tir::transform::Simplify()(mod);
    mod = tir::transform::LowerCustomDatatypes()(mod);
    mod = tir::transform::LowerIntrin()(mod);
    return mod;
  }

 private:
  With<Target> scope_;
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_PIPELINE_H_
