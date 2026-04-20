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
 * \file src/relax/backend/gpu_generic/pipeline.cc
 * Relax pipeline for GPU
 */

#include <tvm/relax/backend/pipeline.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {
namespace backend {
namespace gpu_generic {

using tvm::relax::backend::RelaxPipeline;

IRModule GPURelaxPipeline::Library(IRModule mod) {
  // TODO(Siva): Below passes are in python, to be ported to cpp
  // relax.backend.DispatchSampling
  // relax.backend.DispatchSortScan
  return mod;
}

IRModule GPURelaxPipeline::Legalize(IRModule mod) {
  mod = relax::transform::LegalizeOps()(mod);
  mod = relax::transform::AnnotateTIROpPattern()(mod);
  mod = relax::transform::FoldConstant()(mod);
  mod = relax::transform::FuseOps()(mod);
  mod = relax::transform::FuseTIR()(mod);
  ffi::Array<ffi::String> rules = {"dl.gpu.Matmul", "dl.gpu.GEMV", "dl.gpu.Reduction",
                                   "dl.gpu.GeneralReduction", "dl.gpu.Fallback"};
  mod = relax::transform::ApplyDlightSchedule(rules)(mod);
  return mod;
}

IRModule GPURelaxPipeline::Dataflow(IRModule mod) {
  mod = relax::transform::RewriteDataflowReshape()(mod);
  mod = relax::transform::ToNonDataflow()(mod);
  mod = relax::transform::RemovePurityChecking()(mod);
  mod = relax::transform::CallTIRRewrite()(mod);
  return mod;
}

IRModule GPURelaxPipeline::Finalize(IRModule mod) {
  mod = relax::transform::StaticPlanBlockMemory()(mod);
  mod = relax::transform::LowerAllocTensor()(mod);
  mod = relax::transform::KillAfterLastUse()(mod);
  mod = relax::transform::LowerRuntimeBuiltin()(mod);
  mod = relax::transform::ComputePrimValue()(mod);
  mod = relax::transform::VMShapeLower()(mod);
  mod = relax::transform::AttachGlobalSymbol()(mod);
  return mod;
}

namespace pipeline {

TVM_RELAX_BACKEND_PIPELINE(gpu_generic, GPURelaxPipeline);

}  // namespace pipeline
}  // namespace gpu_generic
}  // namespace backend
}  // namespace relax
}  // namespace tvm
