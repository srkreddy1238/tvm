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
 * \file src/dlight/adreno/convolution.cc
 * \brief Conv2D schedule rules for Adreno GPU operators.
 *
 * Exposes two DlightRules:
 *   - ApplyAdrenoConv2DTensorizationRule  (registered as "dl.adreno.Conv2DTensorization")
 *   - ApplyAdrenoConv2DRule               (registered as "dl.adreno.Conv2D")
 */

#include <tvm/dlight/dlight_rule.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/tir/stmt.h>

#include <string>
#include <vector>

#include "../analysis/common_analysis.h"
#include "fallback.h"

namespace tvm {
namespace dlight {

// Forward-declare the helpers that live in fallback.cc and are exposed via fallback.h.

/*!
 * \brief Configuration for Conv2DTensorization.
 */
struct Conv2DTensorizationConfig {
  int thread_size_x;
  int thread_size_y;
  int thread_size_z;
  int splits_hw_inner;  // tile_m
  int splits_oc_inner;  // tile_n
  int splits_kc_inner;  // tile_k
  int intrin_tile_m;
  int intrin_tile_n;
  int intrin_tile_k;
  int vector_size;
  bool unroll;
  std::string ldst_scope;  // "local" or "shared"
  std::string in_dtype;
  std::string out_dtype;
};

/*!
 * \brief DlightRule for Conv2D with cooperative-matrix tensorization on Adreno.
 *
 * Applicability conditions:
 *   - Target has "supports_khr_cooperative_matrix" = true.
 *   - The function contains blocks named:
 *       "input_imed", "weight_imed", "conv_matrix", "conv_reinterpret".
 *
 * Schedule outline:
 *   1. Split pad_out_hw → [hwo, hwi], pad_gg_oc → [oco, oci], kc → [kco, kci].
 *   2. Reorder and bind: nn→blockIdx.z, gg→blockIdx.y, hwo→blockIdx.x,
 *      oco→threadIdx.y.
 *   3. compute_at input_blk and weight_blk at kco.
 *   4. Set scopes for input/weight blocks.
 *   5. Insert wmma cache-read/write blocks, decompose reduction.
 *   6. Bind threadIdx.x for load/store blocks.
 *   7. Vectorize last loops of input/weight/output blocks.
 *   8. Tensorize using the adreno wmma intrinsic group.
 *   9. Schedule data_pad (if present) and reinterpret block with ScheduleDefault.
 *  10. Inline remaining blocks.
 */
class ApplyAdrenoConv2DTensorizationRuleNode : public DlightRuleNode {
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

    // Check khr_cooperative_matrix support
    bool has_khr_coopmat = false;
    {
      auto it = target->attrs.find("supports_khr_cooperative_matrix");
      if (it != target->attrs.end()) {
        has_khr_coopmat = (*it).second.cast<bool>();
      }
    }
    if (!has_khr_coopmat) return std::nullopt;

    // Build schedule
    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> func_map;
    func_map.Set(GlobalVar("main"), func);
    auto sch = s_tir::Schedule::Traced(IRModule(func_map), /*seed=*/-1, /*debug_mask=*/0,
                                       s_tir::ScheduleErrorRenderLevel::kDetail,
                                       /*enable_check=*/true);

    // Check required block names
    s_tir::SBlockRV root_block = sch->GetSBlock("root");
    ffi::Array<s_tir::SBlockRV> child_blocks = sch->GetChildBlocks(root_block);
    std::vector<std::string> required = {"input_imed", "weight_imed", "conv_matrix",
                                         "conv_reinterpret"};
    for (const std::string& req : required) {
      bool found = false;
      for (const s_tir::SBlockRV& blk : child_blocks) {
        if (std::string(sch->Get(blk)->name_hint) == req) {
          found = true;
          break;
        }
      }
      if (!found) return std::nullopt;
    }

    // Retrieve the four convolution blocks
    s_tir::SBlockRV input_blk = sch->GetSBlock("input_imed");
    s_tir::SBlockRV weight_blk = sch->GetSBlock("weight_imed");
    s_tir::SBlockRV compute_blk = sch->GetSBlock("conv_matrix");
    s_tir::SBlockRV reinterpret_blk = sch->GetSBlock("conv_reinterpret");

    // Build config from the compute block
    tir::SBlock compute_stmt = sch->Get(compute_blk);
    auto [in_dtypes, out_dtypes] = GetInOutDtypes(compute_stmt);
    if (in_dtypes.empty() || out_dtypes.empty()) return std::nullopt;

    // Get wmma tile sizes via the global function
    auto get_tiles_fn = tvm::ffi::Function::GetGlobal("tir.tensor_intrin.adreno.GetWmmaTileSizes");
    if (!get_tiles_fn.has_value()) return std::nullopt;
    auto tiles_any = (*get_tiles_fn)(in_dtypes[0], out_dtypes[0]);
    ffi::Array<Integer> tiles;
    try {
      tiles = tiles_any.cast<ffi::Array<Integer>>();
    } catch (...) {
      return std::nullopt;
    }
    if (tiles.size() != 3) return std::nullopt;

    int tile_m = static_cast<int>(tiles[0]->value);
    int tile_n = static_cast<int>(tiles[1]->value);
    int tile_k = static_cast<int>(tiles[2]->value);

    bool has_qcom_extn = false;
    {
      auto it = target->attrs.find("supports_qcom_cooperative_matrix_conversion");
      if (it != target->attrs.end()) {
        has_qcom_extn = (*it).second.cast<bool>();
      }
    }
    std::string ldst_scope = has_qcom_extn ? "local" : "shared";

    Conv2DTensorizationConfig config;
    config.thread_size_x = tile_m;
    config.thread_size_y = 1;
    config.thread_size_z = 4;
    config.splits_hw_inner = tile_m;
    config.splits_oc_inner = tile_n;
    config.splits_kc_inner = tile_k;
    config.intrin_tile_m = tile_m;
    config.intrin_tile_n = tile_n;
    config.intrin_tile_k = tile_k;
    config.vector_size = 4;
    config.unroll = true;
    config.ldst_scope = ldst_scope;
    config.in_dtype = in_dtypes[0];
    config.out_dtype = out_dtypes[0];

    // Optional data_pad block
    ffi::Optional<s_tir::SBlockRV> data_pad_opt;
    try {
      data_pad_opt = sch->GetSBlock("data_pad");
    } catch (...) {
      data_pad_opt = std::nullopt;
    }

    // ---- Loop transformations ----
    // compute_blk loops: nn, gg, pad_out_hw, pad_gg_oc, kh, kw, kc
    ffi::Array<s_tir::LoopRV> compute_loops = sch->GetLoops(compute_blk);
    if (compute_loops.size() < 7) return std::nullopt;
    s_tir::LoopRV nn = compute_loops[0];
    s_tir::LoopRV gg = compute_loops[1];
    s_tir::LoopRV pad_out_hw = compute_loops[2];
    s_tir::LoopRV pad_gg_oc = compute_loops[3];
    s_tir::LoopRV kh = compute_loops[4];
    s_tir::LoopRV kw = compute_loops[5];
    s_tir::LoopRV kc = compute_loops[6];

    // Split hw, oc, kc
    auto hw_splits = sch->Split(pad_out_hw,
                                {ffi::Optional<s_tir::ExprRV>(),
                                 s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_hw_inner))},
                                /*preserve_unit_iters=*/true);
    s_tir::LoopRV hwo = hw_splits[0], hwi = hw_splits[1];

    auto oc_splits = sch->Split(pad_gg_oc,
                                {ffi::Optional<s_tir::ExprRV>(),
                                 s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_oc_inner))},
                                /*preserve_unit_iters=*/true);
    s_tir::LoopRV oco = oc_splits[0], oci = oc_splits[1];

    auto kc_splits = sch->Split(kc,
                                {ffi::Optional<s_tir::ExprRV>(),
                                 s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_kc_inner))},
                                /*preserve_unit_iters=*/true);
    s_tir::LoopRV kco = kc_splits[0], kci = kc_splits[1];

    sch->Reorder({nn, gg, hwo, oco, kh, kw, kco, hwi, oci, kci});

    sch->Bind(nn, "blockIdx.z");
    sch->Bind(gg, "blockIdx.y");
    sch->Bind(hwo, "blockIdx.x");
    sch->Bind(oco, "threadIdx.y");

    sch->ComputeAt(input_blk, kco, /*preserve_unit_loops=*/true);
    sch->ComputeAt(weight_blk, kco, /*preserve_unit_loops=*/true);

    sch->SetScope(weight_blk, 0, config.ldst_scope);
    sch->SetScope(input_blk, 0, config.ldst_scope);

    // wmma cache blocks
    s_tir::SBlockRV A_wmma = sch->CacheRead(compute_blk, 0, "wmma.matrix_a");
    s_tir::SBlockRV B_wmma = sch->CacheRead(compute_blk, 1, "wmma.matrix_b");
    s_tir::SBlockRV C_store = sch->CacheWrite(compute_blk, 0, "wmma.accumulator");
    s_tir::SBlockRV C_init = sch->DecomposeReduction(compute_blk, kh);

    s_tir::SBlockRV output_blk = sch->CacheWrite(C_store, 0, config.ldst_scope);

    sch->ReverseComputeAt(C_store, oco, /*preserve_unit_loops=*/true);
    sch->ReverseComputeAt(output_blk, oco, /*preserve_unit_loops=*/true);

    // Bind threadIdx.x for load/store blocks
    {
      auto lps = sch->GetLoops(input_blk);
      sch->Bind(lps[lps.size() - 2], "threadIdx.x");
    }
    {
      auto lps = sch->GetLoops(weight_blk);
      sch->Bind(lps[lps.size() - 2], "threadIdx.x");
    }
    {
      auto lps = sch->GetLoops(output_blk);
      sch->Bind(lps[lps.size() - 2], "threadIdx.x");
    }

    // Scope-specific bindings for wmma blocks
    if (config.ldst_scope == "shared") {
      auto b_lps = sch->GetLoops(B_wmma);
      auto ty_tx = sch->Split(b_lps[b_lps.size() - 2],
                              {ffi::Optional<s_tir::ExprRV>(),
                               s_tir::ExprRV(IntImm(DataType::Int(32), config.intrin_tile_n))},
                              /*preserve_unit_iters=*/true);
      sch->Bind(ty_tx[0], "threadIdx.y");
    } else if (config.ldst_scope == "local") {
      {
        auto lps = sch->GetLoops(A_wmma);
        sch->Bind(lps[lps.size() - 2], "threadIdx.x");
      }
      {
        auto lps = sch->GetLoops(B_wmma);
        sch->Bind(lps[lps.size() - 2], "threadIdx.x");
      }
      {
        auto lps = sch->GetLoops(C_store);
        sch->Bind(lps[lps.size() - 2], "threadIdx.x");
      }
    } else {
      return std::nullopt;  // unsupported ldst_scope
    }

    // Vectorize last loops of input/weight/output blocks
    {
      auto lps = sch->GetLoops(input_blk);
      auto splits = sch->Split(lps.back(),
                               {ffi::Optional<s_tir::ExprRV>(),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.vector_size))},
                               /*preserve_unit_iters=*/true);
      sch->Vectorize(splits[1]);
    }
    {
      auto lps = sch->GetLoops(weight_blk);
      auto splits = sch->Split(lps.back(),
                               {ffi::Optional<s_tir::ExprRV>(),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.vector_size))},
                               /*preserve_unit_iters=*/true);
      sch->Vectorize(splits[1]);
    }
    {
      auto lps = sch->GetLoops(output_blk);
      sch->Split(lps.back(),
                 {ffi::Optional<s_tir::ExprRV>(),
                  s_tir::ExprRV(IntImm(DataType::Int(32), config.vector_size))},
                 /*preserve_unit_iters=*/true);
      // Note: Python leaves vx un-vectorized here (commented out)
    }

    // intrin_lp_idx: -2 for shared, -1 for local
    auto intrin_lp_idx = [&](const std::string& scope) -> int {
      return (scope == "shared") ? -2 : -1;
    };
    int idx = intrin_lp_idx(config.ldst_scope);

    // Tensorize using the adreno wmma intrinsic group
    auto get_intrin_fn =
        tvm::ffi::Function::GetGlobal("tir.tensor_intrin.adreno.GetAdrenoWmmaIntrinGroup");
    if (!get_intrin_fn.has_value()) return std::nullopt;

    auto intrin_map_any =
        (*get_intrin_fn)(config.intrin_tile_m, config.intrin_tile_n, config.intrin_tile_k,
                         config.ldst_scope, config.ldst_scope,
                         /*a_trans=*/false, /*b_trans=*/true, "float32", "float32");
    ffi::Map<ffi::String, ffi::String> intrin_map;
    try {
      intrin_map = intrin_map_any.cast<ffi::Map<ffi::String, ffi::String>>();
    } catch (...) {
      return std::nullopt;
    }

    auto tensorize_at = [&](const s_tir::SBlockRV& blk_rv, int lp_idx,
                            const std::string& intrin_key) {
      auto lps = sch->GetLoops(blk_rv);
      int resolved = (lp_idx < 0) ? static_cast<int>(lps.size()) + lp_idx : lp_idx;
      if (resolved < 0 || resolved >= static_cast<int>(lps.size())) return;
      sch->Tensorize(lps[resolved], intrin_map[intrin_key]);
    };

    tensorize_at(A_wmma, idx, "load_a");
    tensorize_at(B_wmma, idx, "load_b");
    tensorize_at(C_store, idx, "store");
    {
      auto lps = sch->GetLoops(C_init);
      sch->Tensorize(lps[lps.size() - 2], intrin_map["init"]);
    }
    {
      auto lps = sch->GetLoops(compute_blk);
      sch->Tensorize(lps[lps.size() - 3], intrin_map["compute"]);
    }

    // Schedule data_pad if present
    if (data_pad_opt.defined()) {
      AdrenoScheduleDefault(sch, data_pad_opt.value());
    }

    // Schedule reinterpret block
    AdrenoScheduleDefault(sch, reinterpret_blk);

    // Inline remaining blocks
    ffi::Array<DlightSBlockInfo> all_infos = DlightNormalizePrimFunc(sch);
    ffi::Array<DlightSBlockInfo> remaining_infos;
    std::vector<std::string> handled = {"input_imed", "weight_imed", "conv_matrix",
                                        "conv_reinterpret", "data_pad"};
    for (const DlightSBlockInfo& info : all_infos) {
      bool is_handled = false;
      for (const std::string& h : handled) {
        if (std::string(info->name_) == h) {
          is_handled = true;
          break;
        }
      }
      if (!is_handled) remaining_infos.push_back(info);
    }
    AdrenoScheduleInlineBlocks(sch, &remaining_infos);
    for (const DlightSBlockInfo& info : remaining_infos) {
      try {
        AdrenoScheduleDefault(sch, info->block_rv_);
      } catch (const tvm::ffi::Error&) {
      }
    }

    return sch;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dlight.ApplyAdrenoConv2DTensorizationRule",
                                    ApplyAdrenoConv2DTensorizationRuleNode, DlightRuleNode);
};

/*!
 * \brief DlightRule for Conv2D NCHWc on Adreno.
 *
 * Applicability: exactly one reduction block whose name contains "conv2d_NCHWc".
 *
 * Schedule outline (ScheduleConv2D):
 *   Loops: n, oc, oh, ow, ob, ic, kh, kw
 *   1. Split oc→[bz,vz,tz], oh→[by,vy,ty], ow→[bx,vx,tx].
 *   2. Fuse n into bz.
 *   3. Reorder and bind block/vthread/thread axes.
 *   4. Split ic→[icc,icb], icc→[ico,ici], kh→[kho,khi], kw→[kwo,kwi].
 *   5. Reorder inner loops, vectorize ob.
 *   6. Cache-read kernel to local, vectorize.
 *   7. Cache-write output to local, reverse-compute-at tx, vectorize.
 *   8. Decompose reduction at tx, vectorize init.
 */
class ApplyAdrenoConv2DRuleNode : public DlightRuleNode {
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

    s_tir::SBlockRV root_block = sch->GetSBlock("root");
    ffi::Array<s_tir::SBlockRV> blocks = sch->GetChildBlocks(root_block);

    // Find reduction blocks and remaining blocks.
    ffi::Array<s_tir::SBlockRV> reduction_blocks;
    ffi::Array<s_tir::SBlockRV> remaining_blocks;
    for (const s_tir::SBlockRV& blk : blocks) {
      DlightSBlockInfo info = GetSBlockInfo(sch, blk);
      if (info->IsReduction()) {
        reduction_blocks.push_back(blk);
      } else {
        remaining_blocks.push_back(blk);
      }
    }

    // Exactly one reduction block, must be conv2d_NCHWc
    if (reduction_blocks.size() != 1) return std::nullopt;
    s_tir::SBlockRV conv_blk = reduction_blocks[0];
    if (std::string(sch->Get(conv_blk)->name_hint).find("conv2d_NCHWc") == std::string::npos) {
      return std::nullopt;
    }

    // NOTE: Must NOT call DlightNormalizePrimFunc before ScheduleConv2D — it
    // collapses unit-extent loops (e.g. oc=1), reducing GetLoops from 8 to 7
    // and breaking ScheduleConv2D which expects exactly 8 loops.
    ScheduleConv2D(sch, conv_blk);

    // Inline remaining blocks after scheduling the conv block.
    ffi::Array<s_tir::SBlockRV> still_remaining;
    for (const s_tir::SBlockRV& blk : remaining_blocks) {
      DlightSBlockInfo info = GetSBlockInfo(sch, blk);
      if (!info->IsInjective() || info->IsDataPad(sch)) {
        still_remaining.push_back(blk);
        continue;
      }
      ffi::Array<s_tir::SBlockRV> consumers = sch->GetConsumers(blk);
      ffi::Array<s_tir::SBlockRV> producers = sch->GetProducers(blk);
      if (consumers.size() == 1) {
        try {
          sch->ComputeInline(blk);
        } catch (const tvm::ffi::Error&) {
          still_remaining.push_back(blk);
        }
      } else if (producers.size() == 1) {
        bool inlined_once = false;
        try {
          while (true) {
            ffi::Array<s_tir::SBlockRV> cur_prod = sch->GetProducers(blk);
            if (cur_prod.size() != 1) break;
            if (sch->GetConsumers(cur_prod[0]).size() != 1) break;
            sch->ReverseComputeInline(blk);
            inlined_once = true;
          }
        } catch (const tvm::ffi::Error&) {
        }
        if (!inlined_once) still_remaining.push_back(blk);
      } else {
        still_remaining.push_back(blk);
      }
    }

    // schedule_default(sch, remaining_blocks) for anything not inlined
    for (const s_tir::SBlockRV& blk : still_remaining) {
      try {
        AdrenoScheduleDefault(sch, blk);
      } catch (const tvm::ffi::Error&) {
      }
    }

    return sch;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dlight.ApplyAdrenoConv2DRule", ApplyAdrenoConv2DRuleNode,
                                    DlightRuleNode);

 private:
  static void ScheduleConv2D(const s_tir::Schedule& sch, const s_tir::SBlockRV& blk) {
    const int vsize = 4;

    // Loops: n, oc, oh, ow, ob, ic, kh, kw
    ffi::Array<s_tir::LoopRV> loops = sch->GetLoops(blk);
    if (loops.size() < 8) return;
    s_tir::LoopRV n = loops[0];
    s_tir::LoopRV oc = loops[1];
    s_tir::LoopRV oh = loops[2];
    s_tir::LoopRV ow = loops[3];
    s_tir::LoopRV ob = loops[4];
    s_tir::LoopRV ic = loops[5];
    s_tir::LoopRV kh = loops[6];
    s_tir::LoopRV kw = loops[7];

    // ic_size = block.iters[-3].dom  (the ic iter)
    tir::SBlock blk_stmt = sch->Get(blk);
    int64_t ic_size = -1;
    if (blk_stmt->iter_vars.size() >= 3) {
      size_t ic_idx = blk_stmt->iter_vars.size() - 3;
      const PrimExpr& ext = blk_stmt->iter_vars[ic_idx]->dom->extent;
      if (const auto* imm = ext.as<IntImmNode>()) ic_size = imm->value;
    }

    // Split oc → [bz, vz, tz]
    auto oc_splits =
        sch->Split(oc,
                   {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), 8)),
                    s_tir::ExprRV(IntImm(DataType::Int(32), 1))},
                   /*preserve_unit_iters=*/true);
    s_tir::LoopRV bz = oc_splits[0], vz = oc_splits[1], tz = oc_splits[2];

    // Split oh → [by, vy, ty]
    auto oh_splits =
        sch->Split(oh,
                   {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), 1)),
                    s_tir::ExprRV(IntImm(DataType::Int(32), 16))},
                   /*preserve_unit_iters=*/true);
    s_tir::LoopRV by = oh_splits[0], vy = oh_splits[1], ty = oh_splits[2];

    // Split ow → [bx, vx, tx]
    auto ow_splits =
        sch->Split(ow,
                   {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), 1)),
                    s_tir::ExprRV(IntImm(DataType::Int(32), 16))},
                   /*preserve_unit_iters=*/true);
    s_tir::LoopRV bx = ow_splits[0], vx = ow_splits[1], tx = ow_splits[2];

    // Fuse n into bz
    bz = sch->Fuse({n, bz});

    sch->Reorder({bz, by, bx, vz, vy, vx, tz, ty, tx, ob});
    sch->Bind(bz, "blockIdx.z");
    sch->Bind(by, "blockIdx.y");
    sch->Bind(bx, "blockIdx.x");
    sch->Bind(vz, "vthread.z");
    sch->Bind(vy, "vthread.y");
    sch->Bind(vx, "vthread.x");
    sch->Bind(tz, "threadIdx.z");
    sch->Bind(ty, "threadIdx.y");
    sch->Bind(tx, "threadIdx.x");

    // Split ic → [icc, icb]
    int icb_factor = (ic_size > 0 && ic_size % vsize == 0) ? vsize : static_cast<int>(ic_size);
    if (icb_factor <= 0) icb_factor = vsize;
    auto ic_splits = sch->Split(
        ic, {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), icb_factor))},
        /*preserve_unit_iters=*/true);
    s_tir::LoopRV icc = ic_splits[0], icb = ic_splits[1];

    // Split icc → [ico, ici]
    auto icc_splits = sch->Split(
        icc, {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), 1))},
        /*preserve_unit_iters=*/true);
    s_tir::LoopRV ico = icc_splits[0], ici = icc_splits[1];

    // Split kh → [kho, khi], kw → [kwo, kwi]
    auto kh_splits = sch->Split(
        kh, {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), 1))},
        /*preserve_unit_iters=*/true);
    s_tir::LoopRV kho = kh_splits[0], khi = kh_splits[1];

    auto kw_splits = sch->Split(
        kw, {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), 1))},
        /*preserve_unit_iters=*/true);
    s_tir::LoopRV kwo = kw_splits[0], kwi = kw_splits[1];

    sch->Reorder({ico, kho, kwo, ici, khi, kwi, icb, ob});
    sch->Vectorize(ob);

    // Cache-read kernel to local
    s_tir::SBlockRV kernel_rblk = sch->CacheRead(blk, 1, "local");
    sch->ComputeAt(kernel_rblk, icb, /*preserve_unit_loops=*/true);
    {
      auto lps = sch->GetLoops(kernel_rblk);
      sch->Vectorize(lps.back());
    }

    // Cache-write output to local
    s_tir::SBlockRV wblk = sch->CacheWrite(blk, 0, "local");
    sch->ReverseComputeAt(wblk, tx, /*preserve_unit_loops=*/true);
    {
      auto lps = sch->GetLoops(wblk);
      sch->Vectorize(lps.back());
    }

    // Decompose reduction at tx, vectorize init
    s_tir::SBlockRV init_blk = sch->DecomposeReduction(blk, tx);
    {
      auto lps = sch->GetLoops(init_blk);
      sch->Vectorize(lps.back());
    }
  }
};

ffi::Optional<s_tir::Schedule> ApplyAdrenoConv2DTensorizationRule(tir::PrimFunc func,
                                                                  tvm::Target target) {
  return ApplyAdrenoConv2DTensorizationRuleNode().Apply(func, target);
}

ffi::Optional<s_tir::Schedule> ApplyAdrenoConv2DRule(tir::PrimFunc func, tvm::Target target) {
  return ApplyAdrenoConv2DRuleNode().Apply(func, target);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("dl.adreno.Conv2DTensorization", ApplyAdrenoConv2DTensorizationRule);
  refl::GlobalDef().def("dl.adreno.Conv2D", ApplyAdrenoConv2DRule);
}

}  // namespace dlight
}  // namespace tvm
