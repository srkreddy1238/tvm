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
 * \file src/tir/backend/pipeline.cc
 * S_TIR generic pipeline
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/backend/pipeline.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace s_tir {
namespace backend {

using tvm::tir::TIRPipeline;

IRModule GenericTIRPipeline::Base(IRModule mod) {
  auto pass_ctx = tvm::transform::PassContext::Current();
  mod = s_tir::transform::CanonicalizeLoop()(mod);
  mod = s_tir::transform::LowerCrossThreadReduction()(mod);
  mod = s_tir::transform::LowerInitBlock()(mod);
  mod = s_tir::transform::PlanAndUpdateBufferAllocationLocation()(mod);
  mod = s_tir::transform::ConvertBlocksToOpaque()(mod);
  mod = s_tir::transform::LiftThreadBinding()(mod);
  mod = s_tir::transform::ManifestSharedMemoryLocalStage()(mod);
  mod = s_tir::transform::CompactBufferAllocation()(mod);
  mod = s_tir::transform::LowerAutoCopy()(mod);
  mod = s_tir::transform::UnifyThreadBinding()(mod);
  mod = s_tir::transform::LowerMatchBuffer()(mod);
  mod = tir::transform::Simplify()(mod);
  mod = s_tir::transform::InjectPermutedLayout()(mod);
  mod = s_tir::transform::AnnotateIrregularLoop()(mod);
  mod = s_tir::transform::InjectSoftwarePipeline()(mod);
  mod = s_tir::transform::TransformMmaBufferLayout()(mod);
  mod = s_tir::transform::LowerOpaqueBlock()(mod);
  mod = tir::transform::FlattenBuffer()(mod);
  mod = tir::transform::BF16ComputeLegalize()(mod);
  mod = tir::transform::NarrowDataType(32)(mod);
  mod = s_tir::transform::LoopPartition()(mod);
  mod = tir::transform::VectorizeLoop(
      !(pass_ctx->GetConfig<Bool>("tir.disable_vectorize", Bool(false)).value()))(mod);
  mod = s_tir::transform::InjectVirtualThread()(mod);
  mod = s_tir::transform::InjectDoubleBuffer()(mod);

  if (!(pass_ctx->GetConfig<Bool>("tir.disable_storage_rewrite", Bool(false)).value())) {
    mod = tir::transform::StorageRewrite()(mod);
  }
  if (!(pass_ctx->GetConfig<Bool>("tir.use_async_copy", Bool(false)).value())) {
    mod = s_tir::transform::LowerAsyncDMA()(mod);
  }
  mod = s_tir::transform::HoistIfThenElse()(mod);
  mod = tir::transform::UnrollLoop()(mod);
  mod = s_tir::transform::RenormalizeSplitPattern()(mod);
  mod = tir::transform::Simplify()(mod);
  mod = tir::transform::RemoveNoOp()(mod);
  mod = s_tir::transform::RewriteUnsafeSelect()(mod);

  if (pass_ctx->GetConfig<Bool>("tir.instrument_bound_checkers", Bool(false)).value()) {
    mod = s_tir::transform::InstrumentBoundCheckers()(mod);
  }

  if (pass_ctx->GetConfig<Bool>("tir.ptx_ldg32", Bool(false)).value()) {
    mod = s_tir::transform::InjectPTXLDG32(true)(mod);
  }

  mod = tir::transform::CommonSubexprElimTIR(
      !(pass_ctx->GetConfig<Bool>("tir.disable_cse_tir", Bool(false)).value()),
      pass_ctx->GetConfig<Bool>("tir.enable_equiv_terms_in_cse_tir", Bool(false)).value())(mod);

  if (pass_ctx->GetConfig<Bool>("tir.instrument_lwp", Bool(false)).value()) {
    mod = s_tir::transform::InstrumentProfileIntrinsics()(mod);
  }

  mod = tir::transform::FP8ComputeLegalize()(mod);
  mod = s_tir::transform::VerifyVTCMLimit(Target::Current(true))(mod);
  mod = s_tir::transform::LowerVtcmAlloc()(mod);
  mod = tir::transform::VerifyMemory()(mod);
  mod = tir::transform::AnnotateEntryFunc()(mod);

  if (pass_ctx->GetConfig<Bool>("tir.detect_global_barrier", Bool(false)).value()) {
    mod = s_tir::transform::ThreadSync("global")(mod);
  }

  mod = s_tir::transform::ThreadSync("shared")(mod);
  mod = s_tir::transform::ThreadSync("shared.dyn")(mod);
  mod = s_tir::transform::ThreadSync("warp")(mod);
  mod = s_tir::transform::InferFragment()(mod);
  mod = s_tir::transform::LowerThreadAllreduce()(mod);

  if (pass_ctx->GetConfig<Bool>("tir.use_async_copy", Bool(false)).value()) {
    mod = s_tir::transform::InjectPTXAsyncCopy()(mod);
  }

  if (pass_ctx->GetConfig<Bool>("tir.ptx_ldg32", Bool(false)).value()) {
    mod = s_tir::transform::InjectPTXLDG32()(mod);
  }

  mod = tir::transform::AnnotateDeviceRegions()(mod);
  mod = tir::transform::SplitHostDevice()(mod);
  mod = s_tir::transform::MergeSharedMemoryAllocations()(mod);
  mod = tir::transform::MakePackedAPI()(mod);
  mod = tir::transform::FP8StorageLegalize()(mod);
  mod = tir::transform::BF16StorageLegalize()(mod);
  mod = tir::transform::LowerDeviceKernelLaunch()(mod);

  return mod;
}

namespace pipeline {

TVM_S_TIR_BACKEND_PIPELINE(generic, GenericTIRPipeline);

}  // namespace pipeline
}  // namespace backend
}  // namespace s_tir
}  // namespace tvm
