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
 * \file src/dlight/adreno/pool.cc
 * \brief Pool schedule rule for Adreno GPU operators.
 *
 * Exposes one DlightRule:
 *   - ApplyAdrenoPool2DRule  (registered as "dl.adreno.Pool2D")
 */

#include <tvm/dlight/dlight_rule.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/tir/stmt.h>

#include <string>

#include "../analysis/common_analysis.h"

namespace tvm {
namespace dlight {

/*!
 * \brief Schedule a pad_temp block: vectorize the last loop, fuse the rest,
 *        split into [blockIdx.x, threadIdx.x].
 */
static void SchedulePad(const s_tir::Schedule& sch, const s_tir::SBlockRV& blk) {
  ffi::Array<s_tir::LoopRV> loops = sch->GetLoops(blk);
  if (loops.empty()) return;

  // Last loop is the vector loop; all others are block loops.
  s_tir::LoopRV vec_lp = loops[loops.size() - 1];
  sch->Vectorize(vec_lp);

  ffi::Array<s_tir::LoopRV> block_lps(loops.begin(), loops.begin() + loops.size() - 1);
  if (block_lps.empty()) return;

  s_tir::LoopRV b = sch->Fuse(block_lps);

  // tx_extent = min(largest_pow2_divisor(extent), 256)
  // Mirrors: min(int(sch.get(b).extent) & ~int(sch.get(b).extent - 1), 256)
  tir::For b_loop = sch->Get(b);
  int tx_extent = 256;
  if (const auto* imm = b_loop->extent.as<IntImmNode>()) {
    int64_t e = imm->value;
    int64_t pow2 = e & (-e);  // largest power-of-2 that divides e
    tx_extent = static_cast<int>(std::min(pow2, static_cast<int64_t>(256)));
  }

  ffi::Array<ffi::Optional<s_tir::ExprRV>> factors = {
      ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), tx_extent))};
  ffi::Array<s_tir::LoopRV> splits = sch->Split(b, factors, /*preserve_unit_iters=*/true);
  sch->Bind(splits[0], "blockIdx.x");
  sch->Bind(splits[1], "threadIdx.x");
}

/*!
 * \brief Schedule an adaptive_pool_sum or pool_max block.
 *
 * Expects dom-kind "SSSSSRR" (5 spatial + 2 reduction loops).
 *
 * \return true on success, false if the dom-kind does not match.
 */
static bool ScheduleMaxPool(const s_tir::Schedule& sch, const s_tir::SBlockRV& blk,
                            const DlightSBlockInfo& info) {
  // Require exactly "SSSSSRR"
  const std::string dom = info->DomKind();
  if (dom != "SSSSSRR") return false;

  ffi::Array<s_tir::LoopRV> lps = sch->GetLoops(blk);
  // lps: [s0, s1, s2, s3, vec_lp, r0, r1]
  if (lps.size() < 7) return false;

  ffi::Array<s_tir::LoopRV> block_lps(lps.begin(), lps.begin() + 4);
  s_tir::LoopRV vec_lp = lps[4];

  // Cache-write output to local, reverse-compute-at the vec loop
  s_tir::SBlockRV write_blk = sch->CacheWrite(blk, 0, "local");
  sch->ReverseComputeAt(write_blk, vec_lp, /*preserve_unit_loops=*/true);

  // Fuse the 4 block loops, split into [blockIdx.x, threadIdx.x]
  s_tir::LoopRV b = sch->Fuse(block_lps);
  tir::For b_loop = sch->Get(b);
  int tx_extent = 256;
  if (const auto* imm = b_loop->extent.as<IntImmNode>()) {
    int64_t e = imm->value;
    int64_t pow2 = e & (-e);
    tx_extent = static_cast<int>(std::min(pow2, static_cast<int64_t>(256)));
  }

  ffi::Array<ffi::Optional<s_tir::ExprRV>> factors = {
      ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), tx_extent))};
  ffi::Array<s_tir::LoopRV> splits = sch->Split(b, factors, /*preserve_unit_iters=*/true);
  sch->Bind(splits[0], "blockIdx.x");
  sch->Bind(splits[1], "threadIdx.x");

  // Vectorize the spatial vec loop
  sch->Vectorize(vec_lp);
  return true;
}

/*!
 * \brief DlightRule for Adreno Pool2D operators.
 *
 * Accepts functions containing "adaptive_pool_sum" or "pool_max" blocks.
 * For each block:
 *   - "pad_temp"                          → SchedulePad
 *   - "adaptive_pool_sum" / "pool_max"    → ScheduleMaxPool
 *   - others (before reduction)           → ComputeInline
 *   - others (after reduction)            → ReverseComputeInline
 */
class ApplyAdrenoPool2DRuleNode : public DlightRuleNode {
 public:
  ffi::Optional<s_tir::Schedule> Apply(const tir::PrimFunc& func, const tvm::Target& target) final {
    // Guard: Adreno target only
    bool is_adreno = false;
    for (const auto& key : target->keys) {
      if (key == "adreno") {
        is_adreno = true;
        break;
      }
    }
    if (!is_adreno) return std::nullopt;

    // Build schedule
    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> func_map;
    func_map.Set(GlobalVar("main"), func);
    auto sch = s_tir::Schedule::Traced(IRModule(func_map), /*seed=*/-1, /*debug_mask=*/0,
                                       s_tir::ScheduleErrorRenderLevel::kDetail,
                                       /*enable_check=*/true);

    s_tir::SBlockRV root = sch->GetSBlock("root");
    ffi::Array<s_tir::SBlockRV> blocks = sch->GetChildBlocks(root);

    // Check that at least one pool block is present
    bool has_pool = false;
    for (const s_tir::SBlockRV& blk : blocks) {
      const std::string name = std::string(sch->Get(blk)->name_hint);
      if (name == "adaptive_pool_sum" || name == "pool_max") {
        has_pool = true;
        break;
      }
    }
    if (!has_pool) return std::nullopt;

    // Build name → block-info map using GetSBlockInfo.
    std::unordered_map<std::string, DlightSBlockInfo> name_to_info;
    for (const s_tir::SBlockRV& blk : blocks) {
      std::string n = std::string(sch->Get(blk)->name_hint);
      if (!name_to_info.count(n)) {
        try {
          name_to_info.emplace(n, GetSBlockInfo(sch, blk));
        } catch (const tvm::ffi::Error&) {
        }
      }
    }

    bool passed_reduction = false;
    for (const s_tir::SBlockRV& blk : blocks) {
      const std::string name = std::string(sch->Get(blk)->name_hint);

      if (name == "pad_temp") {
        SchedulePad(sch, blk);
      } else if (name == "adaptive_pool_sum" || name == "pool_max") {
        // Look up the block info for dom-kind check
        auto it = name_to_info.find(name);
        if (it == name_to_info.end()) return std::nullopt;
        bool ok = ScheduleMaxPool(sch, blk, it->second);
        if (!ok) return std::nullopt;
        passed_reduction = true;
      } else {
        // Inline: forward before reduction, reverse after
        try {
          if (passed_reduction) {
            sch->ReverseComputeInline(blk);
          } else {
            sch->ComputeInline(blk);
          }
        } catch (const tvm::ffi::Error&) {
          // Ignore inlining failures
        }
      }
    }
    return sch;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dlight.ApplyAdrenoPool2DRule", ApplyAdrenoPool2DRuleNode,
                                    DlightRuleNode);
};

ffi::Optional<s_tir::Schedule> ApplyAdrenoPool2DRule(tir::PrimFunc func, tvm::Target target) {
  return ApplyAdrenoPool2DRuleNode().Apply(func, target);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("dl.adreno.Pool2D", ApplyAdrenoPool2DRule);
}

}  // namespace dlight
}  // namespace tvm
