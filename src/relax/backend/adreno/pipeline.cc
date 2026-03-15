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
    mod = relax::transform::DecomposeOpsForInference(std::nullopt)(mod);
    if (is_texture) {
      ffi::Map<ffi::String, ffi::Array<ffi::String>> desired_layouts = {
          {"relax.nn.conv2d", {"NCHW4c", "OIHW4o", "NCHW4c"}}};
      ffi::Array<ffi::String> skip_ops = {
          "relax.nn.conv2d",
          "relax.nn.max_pool2d",
          "relax.nn.adaptive_avg_pool2d",
      };
      mod = relax::transform::ConvertLayout(desired_layouts, nullptr)(mod);
      mod = relax::transform::Normalize()(mod);
      mod = relax::transform::FoldConstant()(mod);
      mod = relax::transform::LegalizeOps(std::nullopt, skip_ops)(mod);
      mod = relax::transform::AnnotateTIROpPattern()(mod);
      mod =
          relax::backend::adreno::transform::AnnotateCustomMemoryScope(Target::Current(true))(mod);
    }
    mod = relax::transform::LegalizeOps()(mod);

    if (is_texture) {
      // TODO(Siva): {"relax.nn.conv2d": legalize_adreno.conv2d_NCHWc_OIHWo}
      ffi::Map<ffi::String, ffi::Function> cmap;
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

    ffi::Array<ffi::String> rules = {"dl.adreno.Conv2D",        "dl.adreno.LayoutTransform",
                                     "dl.adreno.Pool2D",        "dl.gpu.Reduction",
                                     "dl.gpu.GeneralReduction", "dl.gpu.Fallback"};
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
