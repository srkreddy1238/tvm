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
 * \file src/relax/backend/adreno/pipeline.cc
 * Relax pipeline for Adreno
 */

#include <tvm/relax/backend/adreno/transform.h>
#include <tvm/relax/backend/pipeline.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {
namespace backend {
namespace adreno {

using tvm::relax::backend::gpu_generic::GPURelaxPipeline;

class AdrenoRelaxPipeline : GPURelaxPipeline {
 public:
  explicit AdrenoRelaxPipeline(const Target& target) : GPURelaxPipeline(target) {}

  IRModule Library(IRModule mod) {
    if (Target::Current(true)->HasKey("clml")) {
      mod = relax::backend::adreno::transform::OpenCLMLOffLoad()(mod);
      mod = relax::backend::adreno::transform::OpenCLMLOffLoadForLLM()(mod);
    }
    return mod;
  }

  IRModule Legalize(IRModule mod) {
    bool is_texture = Target::Current(true)->HasKey("texture");
    bool is_coopmat = Target::Current(true)
                          ->GetAttr<Bool>("supports_khr_cooperative_matrix", Bool(false))
                          .value();
    mod = relax::transform::DecomposeOpsForInference(std::nullopt)(mod);
    if (is_texture || is_coopmat) {
      ffi::Array<ffi::String> skip_ops = {
          "relax.nn.conv2d",
          "relax.nn.max_pool2d",
          "relax.nn.adaptive_avg_pool2d",
      };
      auto cb_fn = ffi::Function::GetGlobal("relax.backend.adreno.legalize.conv2d_convert_layout");
      TVM_FFI_ICHECK(cb_fn)
          << "Global function not found : relax.backend.adreno.legalize.conv2d_convert_layout";
      auto layout_cb = (*cb_fn)(is_coopmat).cast<ffi::Function>();
      mod = relax::transform::ConvertLayout(ffi::Map<ffi::String, ffi::Array<ffi::String>>({}),
                                            layout_cb)(mod);
      mod = relax::transform::Normalize()(mod);
      mod = relax::transform::FoldConstant()(mod);
      mod = relax::transform::LegalizeOps(std::nullopt, skip_ops)(mod);
      mod = relax::transform::AnnotateTIROpPattern()(mod);
      mod =
          relax::backend::adreno::transform::AnnotateCustomMemoryScope(Target::Current(true))(mod);
    }
    mod = relax::transform::LegalizeOps()(mod);

    if (is_texture) {
      ffi::Map<ffi::String, ffi::Function> cmap;
      auto gf = ffi::Function::GetGlobal("relax.backend.adreno.legalize.conv2d_NCHWc_OIHWo");
      TVM_FFI_ICHECK(gf)
          << "Global function not found : relax.backend.adreno.legalize.conv2d_NCHWc_OIHWo";
      cmap.Set("relax.nn.conv2d", *gf);
      mod = relax::transform::LegalizeOps(cmap)(mod);
    } else if (is_coopmat) {
      ffi::Map<ffi::String, ffi::Function> cmap;
      auto gf = ffi::Function::GetGlobal("relax.backend.adreno.legalize.conv2d_matmul");
      TVM_FFI_ICHECK(gf)
          << "Global function not found : relax.backend.adreno.legalize.conv2d_matmul";
      cmap.Set("relax.nn.conv2d", *gf);
      mod = relax::transform::LegalizeOps(cmap)(mod);
    }

    mod = relax::transform::AnnotateTIROpPattern()(mod);
    mod = relax::transform::FoldConstant()(mod);
    mod = relax::transform::FuseOps()(mod);
    mod = relax::transform::FuseTIR()(mod);
    mod = relax::transform::DeadCodeElimination()(mod);

    if (is_texture) {
      mod = relax::backend::adreno::transform::FoldVDeviceScopeChange()(mod);
      mod = relax::transform::DeadCodeElimination()(mod);
      mod = relax::transform::SpecializePrimFuncBasedOnCallSite()(mod);
    }

    mod = relax::transform::Normalize()(mod);

    ffi::Array<ffi::String> rules;
    if (is_coopmat) {
      rules.push_back("dl.adreno.DequantMatmulTensorization");
      rules.push_back("dl.adreno.MatmulTensorization");
    }
    rules.push_back("dl.adreno.Conv2D");
    rules.push_back("dl.adreno.LayoutTransform");
    rules.push_back("dl.adreno.Pool2D");
    rules.push_back("dl.adreno.Fallback");
    rules.push_back("dl.gpu.Matmul");
    rules.push_back("dl.gpu.Reduction");
    rules.push_back("dl.gpu.GeneralReduction");
    rules.push_back("dl.gpu.Fallback");
    mod = relax::transform::ApplyDlightSchedule(rules)(mod);

    return mod;
  }
  IRModule Dataflow(IRModule mod) { return GPURelaxPipeline::Dataflow(mod); }

  IRModule Finalize(IRModule mod) { return GPURelaxPipeline::Finalize(mod); }
};

namespace pipeline {

TVM_RELAX_BACKEND_PIPELINE(adreno, AdrenoRelaxPipeline);

}  // namespace pipeline
}  // namespace adreno
}  // namespace backend
}  // namespace relax
}  // namespace tvm
