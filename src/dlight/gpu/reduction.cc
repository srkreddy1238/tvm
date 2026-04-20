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
 * \file src/dlight/gpu/reduction.cc
 * \brief Reduction schedule rule for GPU operators.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/dlight/dlight_rule.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <optional>
#include <unordered_map>
#include <vector>

#include "../analysis/common_analysis.h"
#include "../base/common_schedules.h"
#include "../base/utils.h"

namespace tvm {
namespace dlight {

/*!
 * \brief Find the dominant read index expression of a block.
 *
 * Selects the buffer region in `block->reads` that uses the greatest number
 * of distinct block iter-vars, then returns the flattened offset expression
 * `buffer.OffsetOf(region.min)[0]`.
 *
 * \param block The TIR block to analyse.
 * \return The flattened index expression of the dominant read.
 */
static PrimExpr DetectDominantRead(const tir::SBlock& block) {
  ffi::Optional<tir::BufferRegion> dominant_read = std::nullopt;
  int num_read_iters = -1;

  // Collect the set of block iter-var pointers for fast lookup.
  std::unordered_set<const tir::VarNode*> block_iter_vars;
  for (const tir::IterVar& iv : block->iter_vars) {
    block_iter_vars.insert(iv->var.get());
  }

  for (const tir::BufferRegion& buf_region : block->reads) {
    // Count how many distinct block iter-vars appear in this region.
    std::unordered_set<const tir::VarNode*> tir_vars;
    for (const Range& r : buf_region->region) {
      TVM_FFI_ICHECK(tir::is_one(r->extent))
          << "DetectDominantRead: expected unit-extent ranges in buffer region";
      tir::PostOrderVisit(r->min, [&](const ObjectRef& node) {
        if (const auto* var = node.as<tir::VarNode>()) {
          if (block_iter_vars.count(var)) {
            tir_vars.insert(var);
          }
        }
      });
    }
    int n = static_cast<int>(tir_vars.size());
    if (n > num_read_iters) {
      num_read_iters = n;
      dominant_read = buf_region;
    }
  }

  TVM_FFI_ICHECK(dominant_read.defined()) << "DetectDominantRead: block has no reads";

  // Collect the min indices of the dominant read region.
  ffi::Array<PrimExpr> indices;
  for (const Range& r : dominant_read.value()->region) {
    indices.push_back(r->min);
  }
  ffi::Array<PrimExpr> offsets = dominant_read.value()->buffer.OffsetOf(indices);
  TVM_FFI_ICHECK_EQ(offsets.size(), 1U);
  return offsets[0];
}

/*!
 * \brief Extract the reduction operand from a block body of the form `X[...] = X[...] + Y`.
 *
 * \param block The TIR block to inspect.
 * \return The expression `Y`, or nullopt if the body does not match the pattern.
 */
static ffi::Optional<PrimExpr> GetReductionExpr(const tir::SBlock& block) {
  // block.body must be a BufferStore
  const auto* buf_store = block->body.as<tir::BufferStoreNode>();
  if (buf_store == nullptr) return std::nullopt;

  // The stored value must be an Add
  const auto* add = buf_store->value.as<tir::AddNode>();
  if (add == nullptr) return std::nullopt;

  // The LHS of the Add must be a BufferLoad from the same buffer with the same indices
  tir::BufferLoad expected_load(buf_store->buffer, buf_store->indices);
  if (!ffi::StructuralEqual::Equal(add->a, expected_load, /*map_free_vars=*/true)) {
    return std::nullopt;
  }

  return add->b;
}

/*!
 * \brief Check whether a block has at least one commutative-reduction iter-var.
 *
 * \param block_info Analysis information for the block.
 * \return True if any iter-var in the block has type kCommReduce.
 */
static bool HasReductionLoop(const DlightSBlockInfo& block_info) {
  for (const DlightIterInfo& iter : block_info->iters_) {
    if (iter->kind_ == 'R') return true;
  }
  return false;
}

/*!
 * \brief Check whether an epilogue block broadcasts the output of a reduction block.
 *
 * Returns true when the epilogue reads from the reduction block's output buffer
 * using fewer distinct iter-vars than the epilogue itself has, indicating a
 * broadcast pattern.
 *
 * \param sch The schedule containing both blocks.
 * \param block The upstream reduction block.
 * \param epilogue The downstream epilogue block to test.
 * \return True if the epilogue broadcasts the reduction result.
 */
static bool IsBroadcastEpilogue(const s_tir::Schedule& sch, const s_tir::SBlockRV& block,
                                const s_tir::SBlockRV& epilogue) {
  // Collect the set of buffers written by `block`.
  std::unordered_set<const tir::BufferNode*> write_buffers;
  for (const tir::BufferRegion& bw : sch->Get(block)->writes) {
    write_buffers.insert(bw->buffer.get());
  }

  // Build a map from epilogue block iter-var pointer -> IterVar, excluding
  // trivial (extent == 1) iters.
  std::unordered_map<const tir::VarNode*, tir::IterVar> epilogue_iters;
  for (const tir::IterVar& iv : sch->Get(epilogue)->iter_vars) {
    if (!tir::is_one(iv->dom->extent)) {
      epilogue_iters[iv->var.get()] = iv;
    }
  }

  // For each read of the epilogue that touches a write-buffer of `block`,
  // count how many distinct epilogue iter-vars appear in the access region.
  for (const tir::BufferRegion& buf_region : sch->Get(epilogue)->reads) {
    if (!write_buffers.count(buf_region->buffer.get())) continue;

    std::unordered_set<const tir::VarNode*> tir_vars;
    for (const Range& r : buf_region->region) {
      TVM_FFI_ICHECK(tir::is_one(r->extent));
      tir::PostOrderVisit(r->min, [&](const ObjectRef& node) {
        if (const auto* var = node.as<tir::VarNode>()) {
          if (epilogue_iters.count(var)) tir_vars.insert(var);
        }
      });
    }
    if (tir_vars.size() < epilogue_iters.size()) return true;
  }
  return false;
}

/*!
 * \brief Suggest the number of threads to assign to a single loop.
 *
 * Selects a power-of-two thread count that fits within the target's maximum
 * and evenly divides the loop extent. For dynamic extents a conservative
 * default is returned.
 *
 * \param target The compilation target.
 * \param loop The loop for which to suggest a thread count.
 * \return The suggested number of threads per block for this loop.
 */
static int SuggestThreadsPerBlock(const tvm::Target& target, const tir::For& loop) {
  int threads;
  if (target->kind->name == "cuda") {
    threads = 1024;
  } else if (target->kind->name == "rocm" || target->kind->name == "metal" ||
             target->kind->name == "opencl") {
    threads = 256;
  } else {
    threads = 64;
  }

  const int max_threads_for_dynamic_loop = 32;
  const PrimExpr& loop_extent = loop->extent;

  if (const auto* imm = loop_extent.as<IntImmNode>()) {
    int64_t loop_val = imm->value;
    int64_t extent = 1;
    while (extent <= loop_val && extent <= threads) extent *= 2;
    extent /= 2;
    TVM_FFI_ICHECK_GE(extent, 1);
    TVM_FFI_ICHECK_EQ(threads % extent, 0);
    return static_cast<int>(extent);
  } else {
    // Dynamic loop: use max_threads_for_dynamic_loop, then multiply by remaining threads.
    int extent = 1;
    while (extent <= max_threads_for_dynamic_loop && extent <= threads) extent *= 2;
    extent /= 2;
    TVM_FFI_ICHECK_GE(extent, 1);
    TVM_FFI_ICHECK_EQ(threads % extent, 0);
    threads /= extent;
    // The first dynamic loop absorbs all remaining threads.
    return extent * threads;
  }
}

/*!
 * \brief Result returned by Normalize().
 *
 * If `is_inner_reduction` is std::nullopt, normalization failed and the
 * caller should return nullopt.
 */
struct NormalizeResult {
  // std::nullopt means normalization failed.
  std::optional<bool> is_inner_reduction;
  std::optional<int64_t> c_factor;
  // loop_order[i] = j means the i-th spatial block-var maps to the j-th s_loop.
  std::unordered_map<int, int> loop_order;
  std::optional<int> s_split_index;
};

/*!
 * \brief Normalize a reduction block into the canonical loop order [s, r, c].
 *
 * Reorders and fuses the block's loops so that all spatial loops come first,
 * followed by all reduction loops, followed by a single grouped (c) loop.
 * The spatial and reduction groups are each fused into one loop.
 *
 * \param sch The schedule to transform.
 * \param block_info Analysis information for the reduction block.
 * \param access The normalized iter-sum expression of the dominant read.
 * \return A NormalizeResult describing the outcome; `is_inner_reduction` is
 *         std::nullopt if normalization fails.
 */
static NormalizeResult Normalize(const s_tir::Schedule& sch, const DlightSBlockInfo& block_info,
                                 const arith::IterSumExpr& access) {
  NormalizeResult result;

  // access.base must be 0
  if (!tir::is_zero(access->base)) return result;

  // Build a map from iter-var pointer -> DlightIterInfo (will be consumed).
  std::unordered_map<const tir::VarNode*, DlightIterInfo> iter_to_info;
  for (const DlightIterInfo& info : block_info->iters_) {
    iter_to_info.emplace(info->var_.get(), info);
  }

  ffi::Array<s_tir::LoopRV> s_loops, r_loops, c_loops;
  std::optional<int64_t> c_factor;
  s_tir::LoopRV s_split_loop;
  std::optional<int> s_split_index;
  bool is_inner_reduction = false;  // value set on last iteration

  for (const arith::IterSplitExpr& split_expr : access->args) {
    // The source of the split must be a plain Var.
    const auto* var_node = split_expr->source->source.as<tir::VarNode>();
    if (var_node == nullptr) return result;

    auto it = iter_to_info.find(var_node);
    if (it == iter_to_info.end()) return result;

    DlightIterInfo info = it->second;
    iter_to_info.erase(it);

    s_tir::LoopRV loop = info->loop_rv_;
    is_inner_reduction = (info->kind_ == 'R');

    // Check lower_factor
    const auto* lf_imm = split_expr->lower_factor.as<IntImmNode>();
    int64_t lower_factor = lf_imm ? lf_imm->value : 1;

    if (lower_factor > 1) {
      // Only one c_loop is allowed.
      if (!c_loops.empty()) return result;

      s_split_loop = loop;
      s_split_index = static_cast<int>(s_loops.size());

      // Split: loop -> [outer, c_loop]
      ffi::Array<ffi::Optional<s_tir::ExprRV>> factors = {
          ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), lower_factor))};
      auto splits = sch->Split(loop, factors, /*preserve_unit_iters=*/true,
                               /*disable_predication=*/false);
      loop = splits[0];
      c_loops.push_back(splits[1]);

      if (!is_inner_reduction) {
        c_factor = lower_factor;
      }
    }

    if (is_inner_reduction) {
      r_loops.push_back(loop);
    } else {
      s_loops.push_back(loop);
    }
  }

  // Any remaining iters must be spatial with dom == 1.
  for (auto& [var, info] : iter_to_info) {
    if (info->kind_ == 'S') {
      const auto* ext_imm = info->dom_.as<IntImmNode>();
      if (ext_imm && ext_imm->value == 1) {
        s_loops.push_back(info->loop_rv_);
        continue;
      }
    }
    return result;  // unexpected leftover iter
  }

  // Build loop_order: maps spatial block-var index -> s_loops index.
  // We iterate over block_info->iters_ in order and match against s_loops /
  // s_split_loop (the original loop before the split).
  std::unordered_map<int, int> loop_order;
  std::vector<s_tir::LoopRV> s_block_var_loops;
  for (const DlightIterInfo& info : block_info->iters_) {
    s_tir::LoopRV lrv = info->loop_rv_;
    // Check if this loop is in s_loops or is the s_split_loop.
    bool in_s = false;
    for (const s_tir::LoopRV& sl : s_loops) {
      if (sl.same_as(lrv)) {
        in_s = true;
        break;
      }
    }
    bool is_split =
        s_split_index.has_value() && s_split_loop.defined() && s_split_loop.same_as(lrv);
    if (in_s || is_split) {
      s_block_var_loops.push_back(lrv);
    }
  }

  for (int i = 0; i < static_cast<int>(s_block_var_loops.size()); ++i) {
    for (int j = 0; j < static_cast<int>(s_loops.size()); ++j) {
      if (s_block_var_loops[i].same_as(s_loops[j])) {
        loop_order[i] = j;
        break;
      }
      if (s_split_index.has_value() && s_split_loop.defined() &&
          s_block_var_loops[i].same_as(s_split_loop)) {
        loop_order[i] = *s_split_index;
        break;
      }
    }
  }

  // Sanity checks
  if (s_loops.empty() || r_loops.empty()) return result;

  // Count spatial iters in block_info
  int num_s_iters = 0;
  for (const DlightIterInfo& info : block_info->iters_) {
    if (info->kind_ == 'S') ++num_s_iters;
  }
  if (static_cast<int>(s_loops.size()) != num_s_iters) return result;

  // If no c_loop was created from a split, add a unit loop above the block.
  if (c_loops.empty()) {
    c_loops.push_back(sch->AddUnitLoop(block_info->block_rv_));
  }

  // Reorder: [s..., r..., c...]
  ffi::Array<s_tir::LoopRV> reordered;
  for (auto& l : s_loops) reordered.push_back(l);
  for (auto& l : r_loops) reordered.push_back(l);
  for (auto& l : c_loops) reordered.push_back(l);
  sch->Reorder(reordered);

  // Fuse spatial loops and reduction loops separately.
  sch->Fuse(s_loops);
  sch->Fuse(r_loops);

  result.is_inner_reduction = is_inner_reduction;
  result.c_factor = c_factor;
  result.loop_order = std::move(loop_order);
  result.s_split_index = s_split_index;
  return result;
}

/*!
 * \brief Schedule a reduction block whose innermost axis is the reduction dimension.
 *
 * Splits the reduction loop to expose `threadIdx.x` parallelism, applies
 * rfactor, and schedules the write-back block. Optionally stages the output
 * through shared memory when the epilogue broadcasts.
 *
 * \param sch The schedule to transform.
 * \param target The compilation target (used to select thread count).
 * \param block The reduction block to schedule.
 * \param unroll_spatial_factor If set, the spatial loop is split by this
 *        factor and the tail is unrolled.
 * \param epilogue_info Optional epilogue block to fuse after the reduction.
 * \param loop_order Mapping from spatial block-var index to s_loop index,
 *        used to restore the original spatial loop order after rfactor.
 * \param s_split_index Index of the spatial loop that was split to produce
 *        the c loop, or std::nullopt if no split occurred.
 */
static void SchInnerReduction(const s_tir::Schedule& sch, const tvm::Target& target,
                              const s_tir::SBlockRV& block,
                              const std::optional<int64_t>& unroll_spatial_factor,
                              const std::optional<DlightSBlockInfo>& epilogue_info,
                              const std::unordered_map<int, int>& loop_order,
                              const std::optional<int>& s_split_index) {
  // Get loops: [s_fused, r_fused, c]
  auto loops = sch->GetLoops(block);
  TVM_FFI_ICHECK_EQ(loops.size(), 3U);
  s_tir::LoopRV r = loops[1];

  // Suggest thread count for the reduction loop.
  int len_tx = SuggestThreadsPerBlock(target, sch->Get(r));

  // Split r_fused -> [r_outer, tx]. Loops become [s_fused, r_outer, tx, c].
  ffi::Array<ffi::Optional<s_tir::ExprRV>> r_factors = {
      ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), len_tx))};
  auto r_splits = sch->Split(r, r_factors);
  s_tir::LoopRV tx_rf = r_splits[1];

  // Schedule the RF block
  s_tir::SBlockRV rf = sch->RFactor(tx_rf, 0);
  // After RFactor(tx_rf, factor_axis=0), rf loops are: [bx, r_outer, tx, c]
  auto rf_loops = sch->GetLoops(rf);
  TVM_FFI_ICHECK_GE(rf_loops.size(), 4U);
  s_tir::LoopRV bx = rf_loops[0];
  s_tir::LoopRV r_rf = rf_loops[1];
  s_tir::LoopRV tx = rf_loops[2];
  // rf_loops[3] is the c loop

  sch->Reorder({bx, tx, r_rf});
  sch->Bind(bx, "blockIdx.x");
  sch->Bind(tx, "threadIdx.x");
  sch->Annotate(tx, "pragma_auto_unroll_max_step", IntImm(DataType::Int(32), 256));
  sch->Annotate(tx, "pragma_unroll_explicit", IntImm(DataType::Int(32), 1));
  sch->SetScope(rf, 0, "local");
  sch->DecomposeReduction(rf, r_rf);

  // Schedule the write-back block
  sch->ReverseComputeAt(block, bx, /*preserve_unit_loops=*/true);
  auto wb_loops = sch->GetLoops(block);
  // wb_loops layout: [bx, tx, s...]
  TVM_FFI_ICHECK_GE(wb_loops.size(), 2U);
  s_tir::LoopRV wb_tx = wb_loops[1];

  if (unroll_spatial_factor.has_value()) {
    // Collect the s loops (everything after bx and tx).
    std::vector<s_tir::LoopRV> s(wb_loops.begin() + 2, wb_loops.end());
    TVM_FFI_ICHECK_EQ(static_cast<int>(s.size()), static_cast<int>(loop_order.size()));

    // Reorder s according to loop_order.
    std::vector<s_tir::LoopRV> new_order_s(s.size());
    for (int i = 0; i < static_cast<int>(s.size()); ++i) {
      new_order_s[loop_order.at(i)] = s[i];
    }

    // Split the s_split_index loop.
    TVM_FFI_ICHECK(s_split_index.has_value());
    ffi::Array<ffi::Optional<s_tir::ExprRV>> split_factors = {
        ffi::Optional<s_tir::ExprRV>(),
        s_tir::ExprRV(IntImm(DataType::Int(32), *unroll_spatial_factor))};
    auto split_result = sch->Split(new_order_s[*s_split_index], split_factors);
    new_order_s[*s_split_index] = split_result[0];
    s_tir::LoopRV c = split_result[1];

    // Reorder: [new_order_s..., c]
    ffi::Array<s_tir::LoopRV> reorder_arr;
    for (auto& l : new_order_s) reorder_arr.push_back(l);
    reorder_arr.push_back(c);
    sch->Reorder(reorder_arr);

    // Fuse new_order_s -> s_fused
    ffi::Array<s_tir::LoopRV> s_arr;
    for (auto& l : new_order_s) s_arr.push_back(l);
    s_tir::LoopRV s_fused = sch->Fuse(s_arr);

    // Final reorder: [s_fused, wb_tx, c]
    sch->Reorder({s_fused, wb_tx, c});
  } else {
    // Fuse all s loops.
    std::vector<s_tir::LoopRV> s(wb_loops.begin() + 2, wb_loops.end());
    ffi::Array<s_tir::LoopRV> s_arr;
    for (auto& l : s) s_arr.push_back(l);
    s_tir::LoopRV s_fused = sch->Fuse(s_arr);
    sch->Reorder({s_fused, wb_tx});
  }
  sch->Bind(wb_tx, "threadIdx.x");

  // Schedule epilogue
  if (epilogue_info.has_value()) {
    s_tir::SBlockRV epilogue = epilogue_info->get()->block_rv_;
    sch->ReverseComputeAt(epilogue, bx, /*preserve_unit_loops=*/true);
    if (IsBroadcastEpilogue(sch, block, epilogue)) {
      sch->SetScope(block, 0, "shared");
      auto epi_loops = sch->GetLoops(epilogue);
      // Skip the first loop (bx), fuse the rest.
      ffi::Array<s_tir::LoopRV> epi_s(epi_loops.begin() + 1, epi_loops.end());
      ffi::Array<ffi::Optional<s_tir::ExprRV>> epi_factors = {
          ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), len_tx))};
      auto epi_splits = sch->Split(sch->Fuse(epi_s), epi_factors);
      sch->Bind(epi_splits[1], "threadIdx.x");
    } else {
      sch->SetScope(block, 0, "local");
    }
  }
}

/*!
 * \brief Schedule a reduction block whose innermost axis is spatial.
 *
 * Splits the spatial loop to expose `threadIdx.x` and the reduction loop to
 * expose `threadIdx.y`, applies rfactor, and schedules the write-back block.
 * Optionally stages the output through shared memory when the epilogue
 * broadcasts.
 *
 * \param sch The schedule to transform.
 * \param block The reduction block to schedule.
 * \param block_info Analysis information for the block (used to determine
 *        the spatial tile size).
 * \param unroll_spatial_factor If set, the spatial loop is split by this
 *        factor and the tail is unrolled.
 * \param epilogue_info Optional epilogue block to fuse after the reduction.
 * \param loop_order Mapping from spatial block-var index to s_loop index,
 *        used to restore the original spatial loop order after rfactor.
 * \param s_split_index Index of the spatial loop that was split to produce
 *        the c loop, or std::nullopt if no split occurred.
 */
static void SchInnerSpatial(const s_tir::Schedule& sch, const s_tir::SBlockRV& block,
                            const DlightSBlockInfo& block_info,
                            const std::optional<int64_t>& unroll_spatial_factor,
                            const std::optional<DlightSBlockInfo>& epilogue_info,
                            const std::unordered_map<int, int>& loop_order,
                            const std::optional<int>& s_split_index) {
  // Get loops: [s_fused, r_fused, c]
  auto loops = sch->GetLoops(block);
  TVM_FFI_ICHECK_EQ(loops.size(), 3U);
  s_tir::LoopRV s = loops[0];
  s_tir::LoopRV r = loops[1];

  const int len_tx = 16;
  const int len_ty = 16;

  // Compute the perfect spatial factor: the extent of the last spatial iter,
  // rounded down to the largest divisor of len_tx.
  int64_t s_factor_val = 1;
  for (const DlightIterInfo& info : block_info->iters_) {
    if (info->kind_ == 'S') {
      if (const auto* imm = info->dom_.as<IntImmNode>()) {
        s_factor_val = imm->value;
      }
    }
  }
  int actual_len_tx = len_tx;
  while (actual_len_tx > 1 && s_factor_val % actual_len_tx != 0) {
    --actual_len_tx;
  }

  // Split s_fused -> [s_outer, tx_s]
  ffi::Array<ffi::Optional<s_tir::ExprRV>> s_factors = {
      ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), actual_len_tx))};
  sch->Split(s, s_factors);

  // Split r_fused -> [r_outer, ty]
  ffi::Array<ffi::Optional<s_tir::ExprRV>> r_factors = {
      ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), len_ty))};
  auto r_splits = sch->Split(r, r_factors);
  s_tir::LoopRV ty_rf = r_splits[1];

  // Schedule the RF block
  s_tir::SBlockRV rf = sch->RFactor(ty_rf, 0);
  // After RFactor(ty_rf, factor_axis=0), rf loops are: [bx, tx, r_outer, ty, c]
  auto rf_loops = sch->GetLoops(rf);
  TVM_FFI_ICHECK_GE(rf_loops.size(), 5U);
  s_tir::LoopRV bx = rf_loops[0];
  s_tir::LoopRV tx = rf_loops[1];
  s_tir::LoopRV r_rf = rf_loops[2];
  s_tir::LoopRV ty = rf_loops[3];
  // rf_loops[4] is the c loop

  sch->Reorder({bx, tx, ty, r_rf});
  sch->Bind(tx, "threadIdx.x");
  sch->Bind(ty, "threadIdx.y");
  sch->Bind(bx, "blockIdx.x");
  sch->SetScope(rf, 0, "local");
  sch->DecomposeReduction(rf, r_rf);

  // Schedule the write-back block
  sch->ReverseComputeAt(block, bx, /*preserve_unit_loops=*/true);
  auto wb_loops = sch->GetLoops(block);
  // wb_loops layout: [bx, r, s...]
  TVM_FFI_ICHECK_GE(wb_loops.size(), 2U);
  s_tir::LoopRV wb_r = wb_loops[1];

  if (unroll_spatial_factor.has_value()) {
    std::vector<s_tir::LoopRV> s_vec(wb_loops.begin() + 2, wb_loops.end());
    TVM_FFI_ICHECK_EQ(static_cast<int>(s_vec.size()), static_cast<int>(loop_order.size()));

    // Reorder s according to loop_order.
    std::vector<s_tir::LoopRV> new_order_s(s_vec.size());
    for (int i = 0; i < static_cast<int>(s_vec.size()); ++i) {
      new_order_s[loop_order.at(i)] = s_vec[i];
    }

    // Split the s_split_index loop.
    TVM_FFI_ICHECK(s_split_index.has_value());
    ffi::Array<ffi::Optional<s_tir::ExprRV>> split_factors = {
        ffi::Optional<s_tir::ExprRV>(),
        s_tir::ExprRV(IntImm(DataType::Int(32), *unroll_spatial_factor))};
    auto split_result = sch->Split(new_order_s[*s_split_index], split_factors);
    new_order_s[*s_split_index] = split_result[0];
    s_tir::LoopRV c = split_result[1];

    // Reorder: [new_order_s..., c]
    ffi::Array<s_tir::LoopRV> reorder_arr;
    for (auto& l : new_order_s) reorder_arr.push_back(l);
    reorder_arr.push_back(c);
    sch->Reorder(reorder_arr);

    // Fuse new_order_s -> s_fused
    ffi::Array<s_tir::LoopRV> s_arr;
    for (auto& l : new_order_s) s_arr.push_back(l);
    s_tir::LoopRV s_fused = sch->Fuse(s_arr);

    // Final reorder: [s_fused, c, wb_r]
    sch->Reorder({s_fused, c, wb_r});
  } else {
    std::vector<s_tir::LoopRV> s_vec(wb_loops.begin() + 2, wb_loops.end());
    ffi::Array<s_tir::LoopRV> s_arr;
    for (auto& l : s_vec) s_arr.push_back(l);
    s_tir::LoopRV s_fused = sch->Fuse(s_arr);
    sch->Reorder({s_fused, wb_r});
  }

  // Re-fetch loops after reorder/fuse to bind correctly.
  auto final_wb_loops = sch->GetLoops(block);
  // layout: [bx, s_fused, wb_r]  or  [bx, s_fused, c, wb_r]
  TVM_FFI_ICHECK_GE(final_wb_loops.size(), 3U);
  s_tir::LoopRV bind_tx = final_wb_loops[1];
  s_tir::LoopRV bind_ty = final_wb_loops[final_wb_loops.size() - 1];
  sch->Bind(bind_tx, "threadIdx.x");
  sch->Bind(bind_ty, "threadIdx.y");

  // Schedule epilogue
  if (epilogue_info.has_value()) {
    s_tir::SBlockRV epilogue = epilogue_info->get()->block_rv_;
    sch->ReverseComputeAt(epilogue, bx, /*preserve_unit_loops=*/true);
    if (IsBroadcastEpilogue(sch, block, epilogue)) {
      sch->SetScope(block, 0, "shared");
      auto epi_loops = sch->GetLoops(epilogue);
      // Skip bx, fuse the rest, then split into [_, tx, ty].
      ffi::Array<s_tir::LoopRV> epi_s(epi_loops.begin() + 1, epi_loops.end());
      ffi::Array<ffi::Optional<s_tir::ExprRV>> epi_factors = {
          ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), actual_len_tx)),
          s_tir::ExprRV(IntImm(DataType::Int(32), len_ty))};
      auto epi_splits = sch->Split(sch->Fuse(epi_s), epi_factors);
      sch->Bind(epi_splits[1], "threadIdx.x");
      sch->Bind(epi_splits[2], "threadIdx.y");
    } else {
      // Element-wise epilogue without broadcasting: bind remaining spatial part to tx.
      sch->SetScope(block, 0, "local");
      auto epi_loops = sch->GetLoops(epilogue);
      ffi::Array<s_tir::LoopRV> epi_s(epi_loops.begin() + 1, epi_loops.end());
      ffi::Array<ffi::Optional<s_tir::ExprRV>> epi_factors = {
          s_tir::ExprRV(IntImm(DataType::Int(32), actual_len_tx)), ffi::Optional<s_tir::ExprRV>()};
      auto epi_splits = sch->Split(sch->Fuse(epi_s), epi_factors);
      sch->Bind(epi_splits[0], "threadIdx.x");
    }
  }
}

class ApplyReductionRuleNode : public DlightRuleNode {
 public:
  ffi::Optional<s_tir::Schedule> Apply(const tir::PrimFunc& func, const tvm::Target& target) final {
    if (!target->HasKey("gpu")) return std::nullopt;

    // Build schedule
    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> func_map;
    func_map.Set(GlobalVar("main"), func);
    auto sch = s_tir::Schedule::Traced(IRModule(func_map), -1, 0,
                                       s_tir::ScheduleErrorRenderLevel::kDetail, true);

    // Step 0: Normalize and inline contiguous spatial blocks.
    auto block_infos = DlightNormalizePrimFunc(sch);
    if (block_infos.empty()) return std::nullopt;
    DlightTryInlineContiguousSpatial(sch, &block_infos);

    // Expect exactly 1 or 2 blocks after inlining.
    std::optional<DlightSBlockInfo> epilogue_info;
    if (block_infos.size() == 1) {
      epilogue_info = std::nullopt;
    } else if (block_infos.size() == 2) {
      // The second block must be injective (no 'R' in its dom_kind).
      ffi::String epi_dom = block_infos[1]->DomKind();
      if (epi_dom.find("R") != std::string::npos) return std::nullopt;
      epilogue_info = block_infos[1];
    } else {
      return std::nullopt;
    }

    DlightSBlockInfo block_info = block_infos[0];
    s_tir::SBlockRV block = block_info->block_rv_;
    tir::SBlock block_stmt = sch->Get(block);

    // Step 1: Check reduction block.
    //   - must be a reduction block
    //   - must have at least one reduction loop
    //   - must write exactly one buffer
    //   - body must match X[...] = X[...] + Y
    bool is_reduction = block_info->IsReduction();
    if (!is_reduction || !HasReductionLoop(block_info) || block_stmt->writes.size() != 1 ||
        !GetReductionExpr(block_stmt).defined()) {
      return std::nullopt;
    }

    // Step 2: Normalize the block.
    //   Build input_iters map: iter_var -> Range
    ffi::Map<tir::Var, Range> input_iters;
    for (const tir::IterVar& iv : block_stmt->iter_vars) {
      input_iters.Set(iv->var, iv->dom);
    }

    PrimExpr dominant_read = DetectDominantRead(block_stmt);
    arith::Analyzer analyzer;
    arith::IterSumExpr access = arith::NormalizeToIterSum(dominant_read, input_iters, &analyzer);

    NormalizeResult norm = Normalize(sch, block_info, access);
    if (!norm.is_inner_reduction.has_value()) return std::nullopt;

    // Step 3: Schedule.
    if (*norm.is_inner_reduction) {
      SchInnerReduction(sch, target, block, norm.c_factor, epilogue_info, norm.loop_order,
                        norm.s_split_index);
    } else {
      SchInnerSpatial(sch, block, block_info, norm.c_factor, epilogue_info, norm.loop_order,
                      norm.s_split_index);
    }

    return sch;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dlight.ApplyReductionRule", ApplyReductionRuleNode,
                                    DlightRuleNode);
};

ffi::Optional<s_tir::Schedule> ApplyReductionRule(tir::PrimFunc func, tvm::Target target) {
  return ApplyReductionRuleNode().Apply(func, target);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("dl.gpu.Reduction", ApplyReductionRule);
}

}  // namespace dlight
}  // namespace tvm
