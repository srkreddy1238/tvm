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
 * \file src/dlight/adreno/layout_transform.cc
 * \brief Texture-based layout-transform schedule rule for Adreno GPU operators.
 *
 * Exposes one DlightRule:
 *   - ApplyAdrenoLayoutTransformRule  (registered as "dl.adreno.LayoutTransform")
 */

#include <tvm/dlight/dlight_rule.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/tir/stmt.h>

#include <algorithm>
#include <string>
#include <vector>

#include "../analysis/common_analysis.h"

namespace tvm {
namespace dlight {

/*!
 * \brief DlightRule for Adreno texture-based layout-transform operators.
 *
 * The rule accepts a single-block function whose block is either named
 * "te_layout_transform" (use_op_name=true, the default) or satisfies the
 * layout-transform pattern (is_layout_transform).
 *
 * Schedule outline:
 *   1. Identify the "read vector loop" (last assoc_lp of the read buffer)
 *      and the "write vector loop" (last assoc_lp of the write buffer).
 *   2. Collect all other loops as "block loops".
 *   3. Reorder: block_loops, vec_loops.
 *   4. If the two vector loops differ or their vectorisation widths differ
 *      (local_cache = true):
 *        a. If the loops differ: split each, insert a local cache-read,
 *           compute-at the cache at the last block loop, vectorize.
 *        b. If the loops are the same but widths differ: split and cache.
 *   5. Otherwise (same loop, same width): split and vectorize directly.
 *   6. Fuse all block loops, split into [blockIdx.x, threadIdx.x=256].
 */
class ApplyAdrenoLayoutTransformRuleNode : public DlightRuleNode {
 public:
  explicit ApplyAdrenoLayoutTransformRuleNode(bool use_op_name = true)
      : use_op_name_(use_op_name) {}

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

    s_tir::SBlockRV root_block = sch->GetSBlock("root");

    // Require exactly one child block
    ffi::Array<s_tir::SBlockRV> child_blocks = sch->GetChildBlocks(root_block);
    if (child_blocks.size() != 1) return std::nullopt;

    s_tir::SBlockRV blk = child_blocks[0];
    tir::SBlock blk_stmt = sch->Get(blk);

    // Check block name / pattern
    const std::string blk_name = std::string(blk_stmt->name_hint);
    if (use_op_name_) {
      if (blk_name != "te_layout_transform") return std::nullopt;
    } else {
      // is_layout_transform: injective, 1 read, 1 write, not elementwise,
      // same ndim, no IfThenElse. Use GetSBlockInfo to check the pattern.
      DlightSBlockInfo info = GetSBlockInfo(sch, blk);
      if (!info->IsLayoutTransform(sch)) return std::nullopt;
    }

    // Build buffer infos for the single read and write buffer
    ffi::Array<s_tir::LoopRV> all_loops = sch->GetLoops(blk);
    DlightBufferInfo read_buf_info(sch, blk, blk_stmt->reads[0], all_loops);
    DlightBufferInfo write_buf_info(sch, blk, blk_stmt->writes[0], all_loops);

    // Identify the vector loops (assoc_lps[-1])
    if (read_buf_info->assoc_lps_.empty() || write_buf_info->assoc_lps_.empty())
      return std::nullopt;

    const ffi::Optional<s_tir::LoopRV>& lpv_read_opt = read_buf_info->assoc_lps_.back();
    const ffi::Optional<s_tir::LoopRV>& lpv_write_opt = write_buf_info->assoc_lps_.back();
    if (!lpv_read_opt.defined() || !lpv_write_opt.defined()) return std::nullopt;

    s_tir::LoopRV lpv_read = lpv_read_opt.value();
    s_tir::LoopRV lpv_write = lpv_write_opt.value();

    int vlen_read = read_buf_info->GetVecSize();
    int vlen_write = write_buf_info->GetVecSize();
    if (vlen_read < 1) vlen_read = 1;
    if (vlen_write < 1) vlen_write = 1;

    // Determine whether the two vector loops are the same For node
    bool same_vec_loop = sch->Get(lpv_read).same_as(sch->Get(lpv_write));
    bool local_cache = !same_vec_loop || (vlen_read != vlen_write);

    // Collect block loops = all loops except the vector loop(s)
    std::vector<s_tir::LoopRV> block_loops;
    for (const s_tir::LoopRV& lp : all_loops) {
      tir::For lp_stmt = sch->Get(lp);
      bool is_vec_read = lp_stmt.same_as(sch->Get(lpv_read));
      bool is_vec_write = lp_stmt.same_as(sch->Get(lpv_write));
      if (!is_vec_read && !is_vec_write) {
        block_loops.push_back(lp);
      }
    }

    // Reorder: block_loops, then vec_loops
    {
      ffi::Array<s_tir::LoopRV> reorder;
      for (const s_tir::LoopRV& l : block_loops) reorder.push_back(l);
      if (!same_vec_loop) {
        reorder.push_back(lpv_read);
        reorder.push_back(lpv_write);
      } else {
        reorder.push_back(lpv_read);
      }
      sch->Reorder(reorder);
    }

    if (local_cache) {
      if (!same_vec_loop) {
        // Split each vector loop, reorder, insert local cache-read
        ffi::Array<ffi::Optional<s_tir::ExprRV>> rf_read = {
            ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), vlen_read))};
        ffi::Array<ffi::Optional<s_tir::ExprRV>> rf_write = {
            ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), vlen_write))};

        auto read_splits = sch->Split(lpv_read, rf_read, /*preserve_unit_iters=*/true);
        auto write_splits = sch->Split(lpv_write, rf_write, /*preserve_unit_iters=*/true);
        s_tir::LoopRV blp_read = read_splits[0];
        s_tir::LoopRV vlp_read = read_splits[1];
        s_tir::LoopRV blp_write = write_splits[0];
        s_tir::LoopRV vlp_write = write_splits[1];

        // Reorder: block_loops, blp_read, blp_write, vlp_read, vlp_write
        {
          ffi::Array<s_tir::LoopRV> reorder;
          for (const s_tir::LoopRV& l : block_loops) reorder.push_back(l);
          reorder.push_back(blp_read);
          reorder.push_back(blp_write);
          reorder.push_back(vlp_read);
          reorder.push_back(vlp_write);
          sch->Reorder(reorder);
        }
        block_loops.push_back(blp_read);
        block_loops.push_back(blp_write);

        // Insert local cache-read, compute-at the last block loop
        s_tir::SBlockRV rblk = sch->CacheRead(blk, 0, "local");
        sch->ComputeAt(rblk, block_loops.back(), /*preserve_unit_loops=*/true);
        ffi::Array<s_tir::LoopRV> rblk_loops = sch->GetLoops(rblk);
        sch->Vectorize(rblk_loops[rblk_loops.size() - 1]);
        sch->Vectorize(vlp_write);
      } else {
        // Same loop, different widths
        if (vlen_read > vlen_write) {
          // Split at vlen_write, cache-read the outer part
          ffi::Array<ffi::Optional<s_tir::ExprRV>> rf = {
              ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), vlen_write))};
          auto splits = sch->Split(lpv_read, rf, /*preserve_unit_iters=*/true);
          s_tir::LoopRV read_lp = splits[0];
          s_tir::LoopRV vec_lp = splits[1];
          s_tir::SBlockRV rblk = sch->CacheRead(blk, 0, "local");
          sch->ComputeAt(rblk, read_lp, /*preserve_unit_loops=*/true);
          ffi::Array<s_tir::LoopRV> rblk_loops = sch->GetLoops(rblk);
          sch->Vectorize(rblk_loops[rblk_loops.size() - 1]);
          sch->Vectorize(vec_lp);
        } else {
          // vlen_write >= vlen_read: cache-read at last block loop, split inner
          s_tir::SBlockRV rblk = sch->CacheRead(blk, 0, "local");
          if (!block_loops.empty()) {
            sch->ComputeAt(rblk, block_loops.back(), /*preserve_unit_loops=*/true);
          }
          ffi::Array<s_tir::LoopRV> rblk_loops = sch->GetLoops(rblk);
          ffi::Array<ffi::Optional<s_tir::ExprRV>> rf = {
              ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), vlen_read))};
          auto inner_splits =
              sch->Split(rblk_loops[rblk_loops.size() - 1], rf, /*preserve_unit_iters=*/true);
          sch->Vectorize(inner_splits[1]);
          // Vectorize the write vector loop (lpv_write == lpv_read here)
          sch->Vectorize(lpv_write);
        }
      }
    } else {
      // No local cache needed: split and vectorize directly
      ffi::Array<ffi::Optional<s_tir::ExprRV>> rf = {
          ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), vlen_read))};
      auto splits = sch->Split(lpv_read, rf, /*preserve_unit_iters=*/true);
      s_tir::LoopRV blp = splits[0];
      s_tir::LoopRV vlp = splits[1];
      block_loops.push_back(blp);
      sch->Vectorize(vlp);
    }

    // Fuse all block loops, split into [blockIdx.x, threadIdx.x=256]
    if (block_loops.empty()) return std::nullopt;

    ffi::Array<s_tir::LoopRV> block_loops_arr;
    for (const s_tir::LoopRV& l : block_loops) block_loops_arr.push_back(l);
    s_tir::LoopRV b = sch->Fuse(block_loops_arr);

    ffi::Array<ffi::Optional<s_tir::ExprRV>> bind_factors = {
        ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), 256))};
    ffi::Array<s_tir::LoopRV> bind_splits = sch->Split(b, bind_factors,
                                                       /*preserve_unit_iters=*/true);
    sch->Bind(bind_splits[0], "blockIdx.x");
    sch->Bind(bind_splits[1], "threadIdx.x");

    return sch;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dlight.ApplyAdrenoLayoutTransformRule",
                                    ApplyAdrenoLayoutTransformRuleNode, DlightRuleNode);

 private:
  bool use_op_name_;
};

ffi::Optional<s_tir::Schedule> ApplyAdrenoLayoutTransformRule(tir::PrimFunc func,
                                                              tvm::Target target) {
  return ApplyAdrenoLayoutTransformRuleNode().Apply(func, target);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("dl.adreno.LayoutTransform", ApplyAdrenoLayoutTransformRule);
}

}  // namespace dlight
}  // namespace tvm
