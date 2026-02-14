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
 * TIR generic pipeline
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/backend/pipeline.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {
namespace backend {

using tvm::tir::backend::TIRPipeline;

IRModule GenericTIRPipeline::Base(IRModule& mod) {
  auto pass_ctx = tvm::transform::PassContext::Current();
  mod = tir::transform::CanonicalizeLoop()(mod);
  mod = tir::transform::LowerCrossThreadReduction()(mod);
  mod = tir::transform::LowerInitBlock()(mod);
  mod = tir::transform::PlanAndUpdateBufferAllocationLocation()(mod);
  mod = tir::transform::ConvertBlocksToOpaque()(mod);
  mod = tir::transform::LiftThreadBinding()(mod);
  mod = tir::transform::ManifestSharedMemoryLocalStage()(mod);
  mod = tir::transform::CompactBufferAllocation()(mod);
  mod = tir::transform::LowerAutoCopy()(mod);
  mod = tir::transform::UnifyThreadBinding()(mod);
  mod = tir::transform::LowerMatchBuffer()(mod);
  mod = tir::transform::Simplify()(mod);
  mod = tir::transform::InjectPermutedLayout()(mod);
  mod = tir::transform::AnnotateIrregularLoop()(mod);
  mod = tir::transform::InjectSoftwarePipeline()(mod);
  mod = tir::transform::TransformMmaBufferLayout()(mod);
  mod = tir::transform::LowerOpaqueBlock()(mod);
  mod = tir::transform::FlattenBuffer()(mod);
  mod = tir::transform::BF16ComputeLegalize()(mod);
  mod = tir::transform::NarrowDataType(32)(mod);
  mod = tir::transform::LoopPartition()(mod);
  mod = tir::transform::VectorizeLoop(
      !(pass_ctx->GetConfig<Bool>("tir.disable_vectorize", Bool(false)).value()))(mod);
  mod = tir::transform::InjectVirtualThread()(mod);
  mod = tir::transform::InjectDoubleBuffer()(mod);

  if (!(pass_ctx->GetConfig<Bool>("tir.disable_storage_rewrite", Bool(false)).value())) {
    mod = tir::transform::StorageRewrite()(mod);
  }
  if (!(pass_ctx->GetConfig<Bool>("tir.use_async_copy", Bool(false)).value())) {
    mod = tir::transform::LowerAsyncDMA()(mod);
  }
  mod = tir::transform::HoistIfThenElse()(mod);
  mod = tir::transform::UnrollLoop()(mod);
  mod = tir::transform::RenormalizeSplitPattern()(mod);
  mod = tir::transform::Simplify()(mod);
  mod = tir::transform::RemoveNoOp()(mod);
  mod = tir::transform::RewriteUnsafeSelect()(mod);

  if (pass_ctx->GetConfig<Bool>("tir.instrument_bound_checkers", Bool(false)).value()) {
    mod = tir::transform::InstrumentBoundCheckers()(mod);
  }

  if (pass_ctx->GetConfig<Bool>("tir.ptx_ldg32", Bool(false)).value()) {
    mod = tir::transform::InjectPTXLDG32(true)(mod);
  }

  mod = tir::transform::CommonSubexprElimTIR(
      !(pass_ctx->GetConfig<Bool>("tir.disable_cse_tir", Bool(false)).value()),
      pass_ctx->GetConfig<Bool>("tir.enable_equiv_terms_in_cse_tir", Bool(false)).value())(mod);

  if (pass_ctx->GetConfig<Bool>("tir.instrument_lwp", Bool(false)).value()) {
    mod = tir::transform::InstrumentProfileIntrinsics()(mod);
  }

  mod = tir::transform::FP8ComputeLegalize()(mod);
  mod = tir::transform::VerifyVTCMLimit(Target::Current(true))(mod);
  mod = tir::transform::LowerVtcmAlloc()(mod);
  mod = tir::transform::VerifyMemory()(mod);
  mod = tir::transform::AnnotateEntryFunc()(mod);

  if (pass_ctx->GetConfig<Bool>("tir.detect_global_barrier", Bool(false)).value()) {
    mod = tir::transform::ThreadSync("global")(mod);
  }

  mod = tir::transform::ThreadSync("shared")(mod);
  mod = tir::transform::ThreadSync("shared.dyn")(mod);
  mod = tir::transform::ThreadSync("warp")(mod);
  mod = tir::transform::InferFragment()(mod);
  mod = tir::transform::LowerThreadAllreduce()(mod);

  if (pass_ctx->GetConfig<Bool>("tir.use_async_copy", Bool(false)).value()) {
    mod = tir::transform::InjectPTXAsyncCopy()(mod);
  }

  if (pass_ctx->GetConfig<Bool>("tir.ptx_ldg32", Bool(false)).value()) {
    mod = tir::transform::InjectPTXLDG32()(mod);
  }

  mod = tir::transform::AnnotateDeviceRegions()(mod);
  mod = tir::transform::SplitHostDevice()(mod);
  mod = tir::transform::MergeSharedMemoryAllocations()(mod);
  mod = tir::transform::MakePackedAPI()(mod);
  mod = tir::transform::FP8StorageLegalize()(mod);
  mod = tir::transform::BF16StorageLegalize()(mod);
  mod = tir::transform::LowerDeviceKernelLaunch()(mod);

  return mod;
}

IRModule GenericTIRPipeline::Host(IRModule& mod) {
  mod = tir::transform::LowerTVMBuiltin()(mod);
  mod = tir::transform::LowerCustomDatatypes()(mod);
  mod = tir::transform::LowerIntrin()(mod);
  mod = tir::transform::LowerDeviceStorageAccessInfo()(mod);
  mod = tir::transform::CombineContextCall()(mod);
  return mod;
}

IRModule GenericTIRPipeline::Device(IRModule& mod) {
  mod = tir::transform::LowerWarpMemory()(mod);
  mod = tir::transform::Simplify()(mod);
  mod = tir::transform::LowerCustomDatatypes()(mod);
  mod = tir::transform::LowerDeviceStorageAccessInfo()(mod);
  mod = tir::transform::LowerIntrin()(mod);
  return mod;
}

namespace pipeline {

TVM_TIR_BACKEND_PIPELINE(generic, GenericTIRPipeline);

}  // namespace pipeline
}  // namespace backend
}  // namespace tir
}  // namespace tvm
