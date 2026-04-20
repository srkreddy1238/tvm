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
 * \file src/dlight/adreno/fallback.cc
 * \brief Texture-based fallback schedule rule for Adreno GPU operators.
 *
 * Provides three scheduling helpers:
 *   - ScheduleInlineBlocks  – auto-inline injective / element-wise blocks
 *                             while preserving data-pad blocks.
 *   - ScheduleDefault       – vectorised spatial + optional reduction schedule
 *                             for a single block.
 *   - ScheduleFallback      – top-level driver that classifies all blocks and
 *                             dispatches to the two helpers above.
 *
 * The public entry point is ApplyAdrenoFallbackRule / ApplyAdrenoFallbackRuleNode::Apply,
 * which is registered as "dl.adreno.Fallback".
 */

#include "fallback.h"

#include <tvm/dlight/dlight_rule.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../analysis/common_analysis.h"
#include "../base/utils.h"

namespace tvm {
namespace dlight {

/*!
 * \brief Return true when the buffer region's scope contains the substring
 *        "texture", indicating a texture-backed buffer on Adreno.
 */
static bool HasTextureScope(const tir::BufferRegion& buf_region) {
  std::string scope = buf_region->buffer.scope();
  return scope.find("texture") != std::string::npos;
}

/*!
 * \brief Collect the set of block iter-var pointers that appear in a buffer
 *        access region.
 */
static std::unordered_set<const tir::VarNode*> CollectBlockIterVarsInRegion(
    const tir::SBlock& block, const ffi::Array<Range>& region) {
  // Build the universe of block iter-var pointers
  std::unordered_set<const tir::VarNode*> block_iter_set;
  for (const tir::IterVar& iv : block->iter_vars) {
    block_iter_set.insert(iv->var.get());
  }

  // Walk the region and collect those that appear
  std::unordered_set<const tir::VarNode*> result;
  for (const Range& r : region) {
    tir::PostOrderVisit(r->min, [&](const ObjectRef& node) {
      if (const auto* var = node.as<tir::VarNode>()) {
        if (block_iter_set.count(var)) result.insert(var);
      }
    });
  }
  return result;
}

/*!
 * \brief Auto-inline injective / element-wise blocks, skipping data-pad blocks.
 *
 * For each injective (non-data-pad) block:
 *   - If it has exactly one consumer  → try compute_inline.
 *   - If it has exactly one producer  → try reverse_compute_inline repeatedly
 *     while the producer has a single consumer.
 * Blocks that cannot be inlined are kept in the returned list.
 *
 * \param sch        The schedule to transform.
 * \param blk_infos  In/out: block-info list; successfully inlined blocks are
 *                   removed from this list.
 */
static void ScheduleInlineBlocks(const s_tir::Schedule& sch,
                                 ffi::Array<DlightSBlockInfo>* blk_infos) {
  ffi::Array<DlightSBlockInfo> remaining;

  for (const DlightSBlockInfo& info : *blk_infos) {
    s_tir::SBlockRV blk = info->block_rv_;

    if (!info->IsInjective() || info->IsDataPad(sch)) {
      remaining.push_back(info);
      continue;
    }

    ffi::Array<s_tir::SBlockRV> consumers = sch->GetConsumers(blk);
    ffi::Array<s_tir::SBlockRV> producers = sch->GetProducers(blk);

    if (consumers.size() == 1) {
      // Try forward compute_inline
      try {
        sch->ComputeInline(blk);
        // Successfully inlined – do not add to remaining
      } catch (const tvm::ffi::Error&) {
        remaining.push_back(info);
      }
    } else if (producers.size() == 1) {
      // Try reverse_compute_inline while the producer has a single consumer.
      bool inlined_once = false;
      try {
        while (true) {
          ffi::Array<s_tir::SBlockRV> cur_producers = sch->GetProducers(blk);
          if (cur_producers.size() != 1) break;
          ffi::Array<s_tir::SBlockRV> prod_consumers = sch->GetConsumers(cur_producers[0]);
          if (prod_consumers.size() != 1) break;
          sch->ReverseComputeInline(blk);
          inlined_once = true;
        }
      } catch (const tvm::ffi::Error&) {
        // Stop on first failure
      }
      if (!inlined_once) {
        remaining.push_back(info);
      }
      // If inlined at least once, the block is gone – do not add to remaining
    } else {
      remaining.push_back(info);
    }
  }

  *blk_infos = remaining;
}

// Public alias used by convolution.cc and matmul.cc
void AdrenoScheduleInlineBlocks(const s_tir::Schedule& sch,
                                ffi::Array<DlightSBlockInfo>* blk_infos) {
  ScheduleInlineBlocks(sch, blk_infos);
}

/*!
 * \brief Default vectorised schedule for a single Adreno block.
 *
 * 1. Identify the "vector loop" – the loop associated with the last dimension
 *    of the write buffer.
 * 2. Partition the remaining loops into spatial (S), reduction (R), and opaque
 *    (O) groups, excluding the vector loop.
 * 3. Split the vector loop into [outer, vec] where vec ≤ min(vecsize, vsize).
 *    Append the outer part to the spatial group.
 * 4. Separate opaque loops into those whose iter-vars appear in the write
 *    buffer's access region (o_outer) and those that do not (o_inner).
 *    If the original opaque order differs from o_outer ++ o_inner, bail out
 *    (cannot safely reorder opaque loops).
 * 5. Reorder: [s_loops, o_outer, v_inner, r_loops, o_inner].
 * 6. Fuse all spatial loops, split into [blockIdx.x, threadIdx.x].
 * 7. For reduction blocks: insert a local cache-write, decompose the
 *    reduction, reverse-compute-at the cache block, and vectorize the init
 *    and cache loops.
 *    For non-reduction blocks: vectorize the inner vector loop directly.
 *
 * \param sch   The schedule to transform.
 * \param info  Block-info for the block to schedule.
 */
static void ScheduleDefault(const s_tir::Schedule& sch, const DlightSBlockInfo& info) {
  s_tir::SBlockRV blk = info->block_rv_;
  tir::SBlock blk_stmt = sch->Get(blk);

  // vsize: 4 for vulkan, 8 otherwise
  const std::string target_kind =
      tvm::Target::Current(/*allow_none=*/true).defined()
          ? std::string(tvm::Target::Current(/*allow_none=*/true)->kind->name)
          : "";
  const int vsize = (target_kind == "vulkan") ? 4 : 8;

  // Identify the vector loop: the LoopRV associated with the last dimension
  // of the write buffer's access region.
  std::unordered_map<const tir::VarNode*, s_tir::LoopRV> itervar_to_rv;
  for (const DlightIterInfo& di : info->iters_) {
    itervar_to_rv[di->var_.get()] = di->loop_rv_;
  }

  // Determine the vector loop: the LoopRV associated with the last dimension
  // of the write buffer's access region.
  const tir::BufferRegion& write_region = blk_stmt->writes[0];
  const ffi::Array<Range>& write_access = write_region->region;

  // Find the iter-var used in the last dimension of the write access
  ffi::Optional<s_tir::LoopRV> v_loop_opt;
  if (!write_access.empty()) {
    const Range& last_dim = write_access[write_access.size() - 1];
    tir::PostOrderVisit(last_dim->min, [&](const ObjectRef& node) {
      if (v_loop_opt.defined()) return;
      if (const auto* var = node.as<tir::VarNode>()) {
        auto it = itervar_to_rv.find(var);
        if (it != itervar_to_rv.end()) {
          v_loop_opt = it->second;
        }
      }
    });
  }

  if (!v_loop_opt.defined()) {
    // Cannot identify vector loop – skip scheduling this block
    return;
  }
  s_tir::LoopRV v_loop = v_loop_opt.value();

  // Partition remaining loops into S / R / O groups
  ffi::Array<s_tir::LoopRV> s_loops, r_loops, o_loops;
  const tir::For v_loop_stmt = sch->Get(v_loop);

  // Convert DomKind() to std::string once so operator[] works with a plain index.
  const std::string dom_kind = info->DomKind();
  for (size_t idx = 0; idx < info->iters_.size(); ++idx) {
    const DlightIterInfo& di = info->iters_[idx];
    // Skip the vector loop itself
    tir::For loop_stmt = sch->Get(di->loop_rv_);
    if (loop_stmt.same_as(v_loop_stmt)) continue;

    char kind = dom_kind[idx];
    if (kind == 'S') {
      s_loops.push_back(di->loop_rv_);
    } else if (kind == 'R') {
      r_loops.push_back(di->loop_rv_);
    } else {
      o_loops.push_back(di->loop_rv_);
    }
  }

  // Split the vector loop into [outer, vec]
  int vec_width = vsize;
  if (!info->write_bufs_.empty()) {
    vec_width = std::min(info->write_bufs_[0]->GetVecSize(), vsize);
  }
  if (vec_width < 1) vec_width = 1;

  ffi::Array<ffi::Optional<s_tir::ExprRV>> split_factors = {
      ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), vec_width))};
  ffi::Array<s_tir::LoopRV> v_splits = sch->Split(v_loop, split_factors,
                                                  /*preserve_unit_iters=*/true,
                                                  /*disable_predication=*/false);
  s_tir::LoopRV vo = v_splits[0];       // outer part → appended to s_loops
  s_tir::LoopRV v_inner = v_splits[1];  // inner part → vectorized later
  s_loops.push_back(vo);

  // Split opaque loops into o_outer (iter-var appears in write region)
  // and o_inner (does not). Re-fetch the write region after the split.
  tir::SBlock blk_stmt2 = sch->Get(blk);
  const ffi::Array<Range>& write_region2 = blk_stmt2->writes[0]->region;
  auto write_iter_vars = CollectBlockIterVarsInRegion(blk_stmt2, write_region2);

  ffi::Array<s_tir::LoopRV> o_outer, o_inner;
  for (const s_tir::LoopRV& lrv : o_loops) {
    tir::For loop = sch->Get(lrv);
    const tir::VarNode* lvar = loop->loop_var.get();
    // Map loop var → block iter-var via itervar_to_rv (reverse lookup)
    bool in_write = false;
    for (const DlightIterInfo& di : info->iters_) {
      tir::For di_loop = sch->Get(di->loop_rv_);
      if (di_loop->loop_var.get() == lvar) {
        if (write_iter_vars.count(di->var_.get())) {
          in_write = true;
        }
        break;
      }
    }
    if (in_write) {
      o_outer.push_back(lrv);
    } else {
      o_inner.push_back(lrv);
    }
  }

  // Safety check: cannot reorder opaque loops if the original order differs
  // from o_outer ++ o_inner.
  {
    ffi::Array<s_tir::LoopRV> expected_o;
    for (const s_tir::LoopRV& l : o_outer) expected_o.push_back(l);
    for (const s_tir::LoopRV& l : o_inner) expected_o.push_back(l);
    bool order_ok = (o_loops.size() == expected_o.size());
    if (order_ok) {
      for (size_t i = 0; i < o_loops.size(); ++i) {
        if (!sch->Get(o_loops[i]).same_as(sch->Get(expected_o[i]))) {
          order_ok = false;
          break;
        }
      }
    }
    if (!order_ok) return;
  }

  // Reorder: [s_loops, o_outer, v_inner, r_loops, o_inner]
  {
    ffi::Array<s_tir::LoopRV> reorder_order;
    for (const s_tir::LoopRV& l : s_loops) reorder_order.push_back(l);
    for (const s_tir::LoopRV& l : o_outer) reorder_order.push_back(l);
    reorder_order.push_back(v_inner);
    for (const s_tir::LoopRV& l : r_loops) reorder_order.push_back(l);
    for (const s_tir::LoopRV& l : o_inner) reorder_order.push_back(l);
    sch->Reorder(reorder_order);
  }

  // Fuse all spatial loops, split into [blockIdx.x, threadIdx.x]
  if (s_loops.empty()) return;  // nothing to bind

  tvm::Target cur_target = tvm::Target::Current(/*allow_none=*/true);
  int tx_extent = (cur_target.defined()) ? GetMaxThreadsPerBlock(cur_target) : 256;

  s_tir::LoopRV b = sch->Fuse(s_loops);
  ffi::Array<ffi::Optional<s_tir::ExprRV>> bind_factors = {
      ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), tx_extent))};
  ffi::Array<s_tir::LoopRV> bind_splits = sch->Split(b, bind_factors,
                                                     /*preserve_unit_iters=*/true,
                                                     /*disable_predication=*/false);
  sch->Bind(bind_splits[0], "blockIdx.x");
  sch->Bind(bind_splits[1], "threadIdx.x");

  // Reduction path: cache-write to local, decompose, reverse-compute-at,
  // then vectorize the init block's last loop and the cache block's last loop.
  if (r_loops.size() > 1) {
    s_tir::LoopRV tx = bind_splits[1];

    s_tir::SBlockRV wblk = sch->CacheWrite(blk, 0, "local");

    // Decompose at v_inner (the loop just before the reduction loops)
    s_tir::SBlockRV init_block = sch->DecomposeReduction(blk, v_inner);

    sch->ReverseComputeAt(wblk, tx, /*preserve_unit_loops=*/true);

    // Vectorize last loop of init_block and wblk
    {
      ffi::Array<s_tir::LoopRV> init_loops = sch->GetLoops(init_block);
      if (!init_loops.empty()) {
        sch->Vectorize(init_loops[init_loops.size() - 1]);
      }
    }
    {
      ffi::Array<s_tir::LoopRV> wblk_loops = sch->GetLoops(wblk);
      if (!wblk_loops.empty()) {
        sch->Vectorize(wblk_loops[wblk_loops.size() - 1]);
      }
    }
  } else {
    // Non-reduction (or single-reduction) path: vectorize the inner vec loop
    sch->Vectorize(v_inner);
  }
}

// Public alias used by convolution.cc and matmul.cc
void AdrenoScheduleDefault(const s_tir::Schedule& sch, const s_tir::SBlockRV& blk) {
  try {
    DlightSBlockInfo info = GetSBlockInfo(sch, blk);
    ScheduleDefault(sch, info);
  } catch (const tvm::ffi::Error&) {
    // If GetSBlockInfo fails (e.g. after normalization), fall back to
    // DlightNormalizePrimFunc to find the block info.
    ffi::Array<DlightSBlockInfo> infos = DlightNormalizePrimFunc(sch);
    for (const DlightSBlockInfo& info : infos) {
      if (sch->Get(info->block_rv_).same_as(sch->Get(blk))) {
        ScheduleDefault(sch, info);
        return;
      }
    }
  }
}

/*!
 * \brief Top-level fallback scheduler for Adreno texture-based workloads.
 *
 * 1. Collect all leaf blocks under the root.
 * 2. Separate them into:
 *      schedule_blocks  – reduction blocks and data-pad blocks
 *      remaining_blocks – everything else
 * 3. Apply ScheduleDefault to every schedule_block.
 * 4. Try to inline the remaining blocks (ScheduleInlineBlocks).
 * 5. Apply ScheduleDefault to any blocks that could not be inlined.
 *
 * \param sch  The schedule to transform (modified in-place).
 */
static void ScheduleFallback(const s_tir::Schedule& sch) {
  // Get root block and all leaf blocks
  s_tir::SBlockRV root_block = sch->GetSBlock("root");
  ffi::Array<s_tir::SBlockRV> blocks = sch->GetChildBlocks(root_block);

  // Build block-info for every leaf block.
  // DlightNormalizePrimFunc returns info for all non-root blocks in
  // program order, which matches the order of GetChildBlocks(root).
  ffi::Array<DlightSBlockInfo> all_infos = DlightNormalizePrimFunc(sch);

  // Match info entries to blocks by position.
  if (all_infos.size() != blocks.size()) {
    // Mismatch – fall back to a best-effort approach using the info array
    // ordered by DlightNormalizePrimFunc.
    // Apply ScheduleDefault to every block.
    for (const DlightSBlockInfo& info : all_infos) {
      try {
        ScheduleDefault(sch, info);
      } catch (const tvm::ffi::Error&) {
        // Skip blocks that cannot be scheduled
      }
    }
    return;
  }

  // Classify blocks
  ffi::Array<DlightSBlockInfo> schedule_infos;   // reduction + data-pad
  ffi::Array<DlightSBlockInfo> remaining_infos;  // everything else

  for (size_t i = 0; i < all_infos.size(); ++i) {
    const DlightSBlockInfo& info = all_infos[i];
    if (info->IsReduction() || info->IsDataPad(sch)) {
      schedule_infos.push_back(info);
    } else {
      remaining_infos.push_back(info);
    }
  }

  // Step 3: schedule reduction / data-pad blocks
  for (const DlightSBlockInfo& info : schedule_infos) {
    try {
      ScheduleDefault(sch, info);
    } catch (const tvm::ffi::Error&) {
      // Skip blocks that cannot be scheduled
    }
  }

  // Step 4: try to inline the remaining blocks
  ScheduleInlineBlocks(sch, &remaining_infos);

  // Step 5: schedule whatever could not be inlined
  for (const DlightSBlockInfo& info : remaining_infos) {
    try {
      ScheduleDefault(sch, info);
    } catch (const tvm::ffi::Error&) {
      // Skip blocks that cannot be scheduled
    }
  }
}

/*!
 * \brief DlightRule node for the Adreno texture-based fallback schedule.
 *
 * Apply() checks:
 *   1. Rejects non-Adreno targets.
 *   2. Rejects if any block has child blocks (non-leaf scope).
 *   3. Rejects if no block reads or writes a texture-scoped buffer.
 *   4. Runs ScheduleFallback and returns the schedule.
 */
class ApplyAdrenoFallbackRuleNode : public DlightRuleNode {
 public:
  ffi::Optional<s_tir::Schedule> Apply(const tir::PrimFunc& func, const tvm::Target& target) final {
    // ---- Guard: Adreno target only ----
    bool is_adreno = false;
    for (const auto& key : target->keys) {
      if (key == "adreno") {
        is_adreno = true;
        break;
      }
    }
    if (!is_adreno) return std::nullopt;

    // ---- Build schedule ----
    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> func_map;
    func_map.Set(GlobalVar("main"), func);
    auto sch = s_tir::Schedule::Traced(IRModule(func_map), /*seed=*/-1, /*debug_mask=*/0,
                                       s_tir::ScheduleErrorRenderLevel::kDetail,
                                       /*enable_check=*/true);

    // ---- Collect leaf blocks ----
    s_tir::SBlockRV root_block = sch->GetSBlock("root");
    ffi::Array<s_tir::SBlockRV> blocks = sch->GetChildBlocks(root_block);

    // ---- Reject if any block has child blocks (non-leaf scope) ----
    for (const s_tir::SBlockRV& blk : blocks) {
      if (!sch->GetChildBlocks(blk).empty()) return std::nullopt;
    }

    // ---- Reject if no texture-scoped buffer is present ----
    bool has_texture = false;
    for (const s_tir::SBlockRV& blk : blocks) {
      tir::SBlock stmt = sch->Get(blk);
      // Check write buffers
      for (const tir::BufferRegion& wr : stmt->writes) {
        if (HasTextureScope(wr)) {
          has_texture = true;
          break;
        }
      }
      if (has_texture) break;
      // Check read buffers
      for (const tir::BufferRegion& rd : stmt->reads) {
        if (HasTextureScope(rd)) {
          has_texture = true;
          break;
        }
      }
      if (has_texture) break;
    }
    if (!has_texture) return std::nullopt;

    // ---- Run the fallback scheduler ----
    ScheduleFallback(sch);
    return sch;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dlight.ApplyAdrenoFallbackRule", ApplyAdrenoFallbackRuleNode,
                                    DlightRuleNode);
};

ffi::Optional<s_tir::Schedule> ApplyAdrenoFallbackRule(tir::PrimFunc func, tvm::Target target) {
  return ApplyAdrenoFallbackRuleNode().Apply(func, target);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("dl.adreno.Fallback", ApplyAdrenoFallbackRule);
}

}  // namespace dlight
}  // namespace tvm
