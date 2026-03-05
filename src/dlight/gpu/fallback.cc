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
#include <tvm/dlight/dlight_rule.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/schedule/schedule.h>

#include "../analysis/common_analysis.h"
#include "../base/common_schedules.h"
#include "../base/utils.h"

namespace tvm {
namespace dlight {

class ApplyFallbackRuleNode : public DlightRuleNode {
 public:
  // Inherited from DlightRuleNode
  ffi::Optional<s_tir::Schedule> Apply(const tir::PrimFunc& func, const tvm::Target& target) final {
    if (!target->HasKey("gpu")) {
      return std::nullopt;
    }

    auto max_threads_per_block = GetMaxThreadsPerBlock(target);

    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> func_map;
    func_map.Set(GlobalVar("main"), func);
    auto sch = s_tir::Schedule::Traced(IRModule(func_map), -1, 0,
                                       s_tir::ScheduleErrorRenderLevel::kDetail, true);

    // Normalization
    auto blk_infos = DlightNormalizePrimFunc(sch);
    if (!blk_infos.size()) {
      return std::nullopt;
    }

    // Inline
    DlightTryInline(sch, &blk_infos);
    ffi::Map<s_tir::SBlockRV, s_tir::LoopRV> reduction_blocks;

    // Blockwise
    for (auto it = blk_infos.begin(); it != blk_infos.end(); ++it) {
      auto blk_info = *it;
      ffi::Array<s_tir::LoopRV> s_loops, r_loops, o_loops;
      auto dom_kind = blk_info->DomKind();
      auto blk = blk_info->block_rv_;
      auto blk_loops = sch->GetLoops(blk);

      // Has any binds already
      bool is_thread_bind = false;
      for (auto loop_rv : blk_loops) {
        if (sch->Get(loop_rv)->thread_binding.defined()) {
          is_thread_bind = true;
          break;
        }
      }

      // No loops to schedule
      if (is_thread_bind || (0 == blk_loops.size())) {
        continue;
      }

      // Categorise loops
      for (size_t i = 0; i < blk_loops.size(); ++i) {
        if (dom_kind.at(i) == 'S') {
          s_loops.push_back(blk_loops[i]);
        } else if (dom_kind.at(i) == 'R') {
          r_loops.push_back(blk_loops[i]);
        } else {
          o_loops.push_back(blk_loops[i]);
        }
      }

      if (0 == s_loops.size()) {
        s_loops.push_back(sch->AddUnitLoop(blk));
      }

      // Reorder loops
      ffi::Array<s_tir::LoopRV> all_loops;
      all_loops.reserve(s_loops.size() + r_loops.size() + o_loops.size());
      all_loops.insert(all_loops.end(), s_loops.begin(), s_loops.end());
      all_loops.insert(all_loops.end(), r_loops.begin(), r_loops.end());
      all_loops.insert(all_loops.end(), o_loops.begin(), o_loops.end());
      sch->Reorder(all_loops);

      // Bindings
      ffi::Array<ffi::Optional<s_tir::ExprRV>> factors = {ffi::Optional<s_tir::ExprRV>(),
                                                          s_tir::ExprRV(max_threads_per_block)};
      auto splits = sch->Split(sch->Fuse(s_loops), factors, true, false);
      sch->Bind(splits[0], "blockIdx.x");
      sch->Bind(splits[1], "threadIdx.x");

      // Redustion blocks
      if (r_loops.size() > 0) {
        reduction_blocks.Set(blk, r_loops[0]);
      }
    }

    // Reduction
    for (auto reduction : reduction_blocks) {
      sch->DecomposeReduction(reduction.first, reduction.second);
    }

    return sch;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dlight.ApplyFallbackRule", ApplyFallbackRuleNode,
                                    DlightRuleNode);
};

ffi::Optional<s_tir::Schedule> ApplyFallbackRule(tir::PrimFunc func, tvm::Target target) {
  return ApplyFallbackRuleNode().Apply(func, target);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("dl.gpu.Fallback", ApplyFallbackRule);
}

}  // namespace dlight
}  // namespace tvm
