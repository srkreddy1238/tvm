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
 * \file src/dlight/gpu/gemv.cc
 * \brief GEMV / DecodeGEMV schedule rule for GPU operators.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/dlight/dlight_rule.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "../analysis/common_analysis.h"
#include "../base/common_schedules.h"
#include "../base/utils.h"

namespace tvm {
namespace dlight {

/*!
 * \brief Returns the largest value in `factors` that evenly divides `n`.
 *
 * \param n The value to test divisibility against.
 * \param factors Candidate factor values, searched in descending order.
 * \return The largest factor that divides `n`, or 1 if none does.
 */
static int GetMaxFactor(int n, std::vector<int> factors) {
  std::sort(factors.begin(), factors.end(), std::greater<int>());
  for (int f : factors) {
    if (n % f == 0) return f;
  }
  return 1;
}

/*!
 * \brief Returns the static extent of a loop as an int64_t.
 *
 * \param sch The schedule that owns the loop.
 * \param loop_rv The loop whose extent is queried.
 * \return The compile-time constant extent, or -1 if the extent is dynamic.
 */
static int64_t GetLoopExtent(const s_tir::Schedule& sch, const s_tir::LoopRV& loop_rv) {
  tir::For loop = sch->Get(loop_rv);
  if (const auto* imm = loop->extent.as<IntImmNode>()) {
    return imm->value;
  }
  return -1;  // dynamic
}

/*!
 * \brief Vectorize a loop up to a maximum vector width.
 *
 * If the loop extent is a compile-time constant that is at most `max_vec`,
 * the loop is vectorized directly. Otherwise the loop is split and only
 * the inner tail of length `max_vec` is vectorized.
 *
 * \param sch The schedule that owns the loop.
 * \param loop The loop to vectorize.
 * \param max_vec Maximum number of lanes to vectorize.
 */
static void AutoVectorize(const s_tir::Schedule& sch, const s_tir::LoopRV& loop, int max_vec) {
  int64_t extent = GetLoopExtent(sch, loop);
  if (extent < 0) return;  // dynamic — skip
  if (extent <= max_vec) {
    sch->Vectorize(loop);
  } else {
    ffi::Array<ffi::Optional<s_tir::ExprRV>> factors = {
        ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), max_vec))};
    auto splits = sch->Split(loop, factors, /*preserve_unit_iters=*/true,
                             /*disable_predication=*/false);
    sch->Vectorize(splits[1]);
  }
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
  std::unordered_set<const tir::BufferNode*> write_buffers;
  for (const tir::BufferRegion& bw : sch->Get(block)->writes) {
    write_buffers.insert(bw->buffer.get());
  }
  std::unordered_map<const tir::VarNode*, tir::IterVar> epilogue_iters;
  for (const tir::IterVar& iv : sch->Get(epilogue)->iter_vars) {
    if (!tir::is_one(iv->dom->extent)) {
      epilogue_iters[iv->var.get()] = iv;
    }
  }
  for (const tir::BufferRegion& buf_region : sch->Get(epilogue)->reads) {
    if (!write_buffers.count(buf_region->buffer.get())) continue;
    std::unordered_set<const tir::VarNode*> tir_vars;
    for (const Range& r : buf_region->region) {
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
  std::unordered_set<const tir::VarNode*> block_iter_vars;
  for (const tir::IterVar& iv : block->iter_vars) {
    block_iter_vars.insert(iv->var.get());
  }
  ffi::Optional<tir::BufferRegion> dominant_read = std::nullopt;
  int num_read_iters = -1;
  for (const tir::BufferRegion& buf_region : block->reads) {
    std::unordered_set<const tir::VarNode*> tir_vars;
    for (const Range& r : buf_region->region) {
      tir::PostOrderVisit(r->min, [&](const ObjectRef& node) {
        if (const auto* var = node.as<tir::VarNode>()) {
          if (block_iter_vars.count(var)) tir_vars.insert(var);
        }
      });
    }
    int n = static_cast<int>(tir_vars.size());
    if (n > num_read_iters) {
      num_read_iters = n;
      dominant_read = buf_region;
    }
  }
  TVM_FFI_ICHECK(dominant_read.defined());
  ffi::Array<PrimExpr> indices;
  for (const Range& r : dominant_read.value()->region) indices.push_back(r->min);
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
  const auto* buf_store = block->body.as<tir::BufferStoreNode>();
  if (!buf_store) return std::nullopt;
  const auto* add = buf_store->value.as<tir::AddNode>();
  if (!add) return std::nullopt;
  tir::BufferLoad expected_load(buf_store->buffer, buf_store->indices);
  if (!ffi::StructuralEqual::Equal(add->a, expected_load, /*map_free_vars=*/true))
    return std::nullopt;
  return add->b;
}

/*!
 * \brief Collect the block iter-vars that appear in a buffer access region.
 *
 * \param block The TIR block whose iter-vars are the universe of interest.
 * \param region The buffer access region to scan.
 * \return The set of block iter-var pointers referenced in `region`.
 */
static std::unordered_set<const tir::VarNode*> CollectBlockIterVarsInRegion(
    const tir::SBlock& block, const ffi::Array<Range>& region) {
  std::unordered_set<const tir::VarNode*> block_iter_set;
  for (const tir::IterVar& iv : block->iter_vars) {
    block_iter_set.insert(iv->var.get());
  }
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
 * \brief Determine whether a block implements a GEMV (or DecodeGEMV) pattern.
 *
 * A block is considered GEMV if it is a reduction block with at least two reads
 * and exactly one write, its body matches `X[...] = X[...] + Y`, and at least
 * one read buffer is accessed by strictly fewer iter-vars than the total number
 * of block iter-vars (the "vector" input).
 *
 * \param sch The schedule containing the block.
 * \param block_info Analysis information for the block.
 * \return The list of vector input buffers if the block is GEMV, else nullopt.
 */
static ffi::Optional<ffi::Array<tir::Buffer>> IsGemv(const s_tir::Schedule& sch,
                                                     const DlightSBlockInfo& block_info) {
  s_tir::SBlockRV block_rv = block_info->block_rv_;
  tir::SBlock block_stmt = sch->Get(block_rv);

  // Basic structural checks
  bool is_reduction = block_info->IsReduction();
  if (!is_reduction) return std::nullopt;
  if (block_stmt->reads.size() < 2) return std::nullopt;
  if (block_stmt->writes.size() != 1) return std::nullopt;
  if (!GetReductionExpr(block_stmt).defined()) return std::nullopt;

  // The write region must use at least one block iter-var
  auto write_vars = CollectBlockIterVarsInRegion(block_stmt, block_stmt->writes[0]->region);
  if (write_vars.empty()) return std::nullopt;

  int iter_num = static_cast<int>(block_stmt->iter_vars.size());

  // Collect reads that use fewer iter-vars than the total (but at least 1)
  ffi::Array<tir::Buffer> vector_bufs;
  for (const tir::BufferRegion& read : block_stmt->reads) {
    auto vars = CollectBlockIterVarsInRegion(block_stmt, read->region);
    int n = static_cast<int>(vars.size());
    if (n < iter_num && n > 0) {
      vector_bufs.push_back(read->buffer);
    }
  }

  // Must have at least one vector buffer but not all reads are vector buffers
  if (vector_bufs.empty()) return std::nullopt;
  if (static_cast<int>(vector_bufs.size()) >= static_cast<int>(block_stmt->reads.size()))
    return std::nullopt;

  return vector_bufs;
}

/*!
 * \brief Normalize a GEMV block into the canonical loop order [batch, s, r, c].
 *
 * Classifies each iter-var in the dominant-read access expression as batch,
 * spatial, reduction, or grouped (lower_factor > 1), then reorders and fuses
 * the loops into four fused loops: batch_fused, s_fused, r_fused, and c.
 *
 * \param sch The schedule to transform.
 * \param block_info Analysis information for the GEMV block.
 * \return true if the innermost axis is a reduction (inner-reduction path),
 *         false if it is spatial (outer-reduction path), or
 *         std::nullopt if normalization fails.
 */
static std::optional<bool> NormalizeGemv(const s_tir::Schedule& sch,
                                         const DlightSBlockInfo& block_info) {
  tir::SBlock block_stmt = sch->Get(block_info->block_rv_);

  // Build input_iters map for NormalizeToIterSum
  ffi::Map<tir::Var, Range> input_iters;
  for (const tir::IterVar& iv : block_stmt->iter_vars) {
    input_iters.Set(iv->var, iv->dom);
  }

  PrimExpr dominant_read_expr = DetectDominantRead(block_stmt);
  arith::Analyzer analyzer;
  arith::IterSumExpr access = arith::NormalizeToIterSum(dominant_read_expr, input_iters, &analyzer);

  // access.base must not involve any block iter-var
  {
    std::unordered_set<const tir::VarNode*> block_iter_set;
    for (const tir::IterVar& iv : block_stmt->iter_vars) {
      block_iter_set.insert(iv->var.get());
    }
    bool base_uses_iter = false;
    tir::PostOrderVisit(access->base, [&](const ObjectRef& node) {
      if (const auto* var = node.as<tir::VarNode>()) {
        if (block_iter_set.count(var)) base_uses_iter = true;
      }
    });
    if (base_uses_iter) return std::nullopt;
  }

  // Collect the set of block iter-vars used in EVERY buffer region
  // (writes + reads).  A var that appears in all regions is a "batch" var.
  std::vector<std::unordered_set<const tir::VarNode*>> all_buf_vars;
  for (const tir::BufferRegion& bw : block_stmt->writes) {
    all_buf_vars.push_back(CollectBlockIterVarsInRegion(block_stmt, bw->region));
  }
  for (const tir::BufferRegion& br : block_stmt->reads) {
    all_buf_vars.push_back(CollectBlockIterVarsInRegion(block_stmt, br->region));
  }

  // Build iter_var_ptr -> DlightIterInfo lookup.
  std::vector<std::pair<const tir::VarNode*, DlightIterInfo>> iter_to_info_vec;
  iter_to_info_vec.reserve(block_info->iters_.size());
  for (const DlightIterInfo& info : block_info->iters_) {
    iter_to_info_vec.emplace_back(info->var_.get(), info);
  }
  auto find_iter_info = [&](const tir::VarNode* var) -> const DlightIterInfo* {
    for (const auto& kv : iter_to_info_vec) {
      if (kv.first == var) return &kv.second;
    }
    return nullptr;
  };

  // Determine the inner-most axis kind from the last split expr
  const tir::VarNode* inner_axis_var = access->args.back()->source->source.as<tir::VarNode>();
  if (!inner_axis_var) return std::nullopt;
  const DlightIterInfo* inner_info_ptr = find_iter_info(inner_axis_var);
  if (!inner_info_ptr) return std::nullopt;
  bool is_inner_reduction = ((*inner_info_ptr)->kind_ == 'R');

  // Walk the split exprs and classify each loop
  ffi::Array<s_tir::LoopRV> batch_loops, s_loops, r_loops, c_loops;

  for (const arith::IterSplitExpr& split_expr : access->args) {
    const auto* var_node = split_expr->source->source.as<tir::VarNode>();
    if (!var_node) return std::nullopt;

    const DlightIterInfo* info_ptr = find_iter_info(var_node);
    if (!info_ptr) return std::nullopt;

    DlightIterInfo info = *info_ptr;
    s_tir::LoopRV loop = info->loop_rv_;
    bool is_reduction = (info->kind_ == 'R');

    // Handle lower_factor > 1: split the loop into [outer, c_loop]
    const auto* lf_imm = split_expr->lower_factor.as<IntImmNode>();
    int64_t lower_factor = lf_imm ? lf_imm->value : 1;

    if (lower_factor > 1) {
      if (!c_loops.empty()) return std::nullopt;  // only one c_loop allowed
      // Only reduction dims may be grouped (matches Python assert)
      if (!is_reduction) return std::nullopt;

      ffi::Array<ffi::Optional<s_tir::ExprRV>> factors = {
          ffi::Optional<s_tir::ExprRV>(),
          s_tir::ExprRV(IntImm(DataType::Int(32), static_cast<int>(lower_factor)))};
      auto splits = sch->Split(loop, factors, /*preserve_unit_iters=*/true,
                               /*disable_predication=*/false);
      loop = splits[0];
      c_loops.push_back(splits[1]);
    }

    if (is_reduction) {
      r_loops.push_back(loop);
    } else {
      // Check whether this var appears in ALL buffer regions (batch) or not (spatial)
      bool in_all = true;
      for (const auto& buf_vars : all_buf_vars) {
        if (!buf_vars.count(var_node)) {
          in_all = false;
          break;
        }
      }
      if (in_all) {
        batch_loops.push_back(loop);
      } else {
        s_loops.push_back(loop);
      }
    }
  }

  // Ensure we have at least one spatial and one reduction loop
  if (s_loops.empty() || r_loops.empty()) return std::nullopt;

  // Add unit loops for missing batch / c groups
  if (c_loops.empty()) {
    c_loops.push_back(sch->AddUnitLoop(block_info->block_rv_));
  }
  if (batch_loops.empty()) {
    batch_loops.push_back(sch->AddUnitLoop(block_info->block_rv_));
  }

  // Reorder: [batch..., s..., r..., c...]
  ffi::Array<s_tir::LoopRV> reordered;
  for (auto& l : batch_loops) reordered.push_back(l);
  for (auto& l : s_loops) reordered.push_back(l);
  for (auto& l : r_loops) reordered.push_back(l);
  for (auto& l : c_loops) reordered.push_back(l);
  sch->Reorder(reordered);

  // Fuse each group into a single loop
  sch->Fuse(batch_loops);
  sch->Fuse(s_loops);
  sch->Fuse(r_loops);

  return is_inner_reduction;
}

/*!
 * \brief Schedule a GEMV block whose innermost axis is the reduction dimension.
 *
 * Applies two rfactor steps to expose TR-way thread-level parallelism along
 * the reduction axis, then optionally stages the vector input through shared
 * memory. Tuning knobs (TS, TR, TILE_S, TILE_R, VEC_C, etc.) are supplied by
 * the caller based on the target.
 *
 * \param sch The schedule to transform.
 * \param target The compilation target (used for shared-memory capacity checks).
 * \param gemv The main GEMV block.
 * \param vector_input_buffers The vector input buffers identified by IsGemv.
 * \param epilogue_info Optional epilogue block to fuse after the reduction.
 * \param TAG_S Thread-axis tag for the spatial dimension (e.g. "threadIdx.y").
 * \param TAG_R Thread-axis tag for the reduction dimension (e.g. "threadIdx.x").
 * \param TS Number of spatial threads per block.
 * \param TR Number of reduction threads per block.
 * \param TILE_S Spatial tile size per thread.
 * \param TILE_R Reduction tile size per thread.
 * \param VEC_LOAD Vector load width for the quantized weight buffer.
 * \param VEC_C Vectorization factor for the innermost compute loop.
 * \param LOAD_V_SHARED Whether to stage the vector input through shared memory.
 * \param LOAD_V_VEC Vectorization factor for the shared-memory load.
 * \param UNROLL Unroll depth hint for the reduction loop.
 * \param SUPPORT_WARP_SHUFFLE Whether the target supports warp-shuffle allreduce.
 * \return The transformed schedule, or nullopt if scheduling fails.
 */
static ffi::Optional<s_tir::Schedule> SchInnerReduction(
    const s_tir::Schedule& sch, const tvm::Target& target, const s_tir::SBlockRV& gemv,
    const ffi::Array<tir::Buffer>& vector_input_buffers,
    const std::optional<DlightSBlockInfo>& epilogue_info,
    // Tuning knobs (set by the caller from target-specific config)
    const std::string& TAG_S, const std::string& TAG_R, int TS, int TR, int TILE_S, int TILE_R,
    int VEC_LOAD, int VEC_C, bool LOAD_V_SHARED, int LOAD_V_VEC, int UNROLL,
    bool SUPPORT_WARP_SHUFFLE) {
  // ---- rfactor step 1: reduce to TR * VEC_C ----
  // Loops after normalize: [_, s, r, c]
  auto loops0 = sch->GetLoops(gemv);
  TVM_FFI_ICHECK_EQ(loops0.size(), 4U);
  s_tir::LoopRV batch_loop = loops0[0];
  s_tir::LoopRV s_loop = loops0[1];
  s_tir::LoopRV r_loop = loops0[2];
  s_tir::LoopRV c_loop = loops0[3];

  // s = fuse(_, s);  r = fuse(r, c)
  s_tir::LoopRV s_fused = sch->Fuse({batch_loop, s_loop});
  s_tir::LoopRV r_fused = sch->Fuse({r_loop, c_loop});

  // split s_fused -> [bx, ts, tile_s]
  auto s_splits =
      sch->Split(s_fused,
                 {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), TS)),
                  s_tir::ExprRV(IntImm(DataType::Int(32), TILE_S))},
                 /*preserve_unit_iters=*/true);
  s_tir::LoopRV bx = s_splits[0];
  s_tir::LoopRV ts = s_splits[1];
  s_tir::LoopRV tile_s = s_splits[2];

  // split r_fused -> [r, tr, tile_r_vec_n, vec_c]
  int tile_r_vec_n_factor = TILE_R / VEC_C;
  auto r_splits =
      sch->Split(r_fused,
                 {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), TR)),
                  s_tir::ExprRV(IntImm(DataType::Int(32), tile_r_vec_n_factor)),
                  s_tir::ExprRV(IntImm(DataType::Int(32), VEC_C))},
                 /*preserve_unit_iters=*/true);
  s_tir::LoopRV r_outer = r_splits[0];
  s_tir::LoopRV tr = r_splits[1];
  s_tir::LoopRV tile_r_vec_n = r_splits[2];
  s_tir::LoopRV vec_c = r_splits[3];

  // reorder(r, tile_r_vec_n, tr, vec_c)
  sch->Reorder({r_outer, tile_r_vec_n, tr, vec_c});

  // fuse(tr, vec_c) -> tr_vec_c;  rfactor on tr_vec_c
  s_tir::LoopRV tr_vec_c = sch->Fuse({tr, vec_c});
  s_tir::SBlockRV rf = sch->RFactor(tr_vec_c, 0);

  // ---- rfactor step 2: reduce to TR ----
  // After rfactor, gemv loops are: [bx, ts, tile_s, tr_vec_c]
  auto gemv_loops1 = sch->GetLoops(gemv);
  TVM_FFI_ICHECK_EQ(gemv_loops1.size(), 4U);
  s_tir::LoopRV tr_vec_c2 = gemv_loops1[3];
  auto tr_vec_c2_splits = sch->Split(
      tr_vec_c2, {s_tir::ExprRV(IntImm(DataType::Int(32), TR)), ffi::Optional<s_tir::ExprRV>()},
      /*preserve_unit_iters=*/true);
  s_tir::LoopRV tr2 = tr_vec_c2_splits[0];
  s_tir::SBlockRV rf2 = sch->RFactor(tr2, 0);

  // ---- bind / vectorize the rf block ----
  // rf loops: [bx, ts, tile_s, r_outer, tile_r_vec_n, tr_vec_c]
  auto rf_loops = sch->GetLoops(rf);
  TVM_FFI_ICHECK_GE(rf_loops.size(), 6U);
  s_tir::LoopRV rf_bx = rf_loops[0];
  s_tir::LoopRV rf_ts = rf_loops[1];
  s_tir::LoopRV rf_tile_s = rf_loops[2];
  s_tir::LoopRV rf_r = rf_loops[3];
  s_tir::LoopRV rf_tile_r_vec_n = rf_loops[4];
  s_tir::LoopRV rf_tr_vec_c = rf_loops[5];

  // split tr_vec_c -> [tr, vec_c]
  auto rf_tr_splits = sch->Split(
      rf_tr_vec_c, {s_tir::ExprRV(IntImm(DataType::Int(32), TR)), ffi::Optional<s_tir::ExprRV>()},
      /*preserve_unit_iters=*/true);
  s_tir::LoopRV rf_tr = rf_tr_splits[0];
  s_tir::LoopRV rf_vec_c = rf_tr_splits[1];

  // reorder(bx, ts, tr, r, tile_s, tile_r_vec_n, vec_c)
  sch->Reorder({rf_bx, rf_ts, rf_tr, rf_r, rf_tile_s, rf_tile_r_vec_n, rf_vec_c});
  sch->Bind(rf_bx, "blockIdx.x");
  sch->Bind(rf_ts, TAG_S);
  sch->Bind(rf_tr, TAG_R);
  sch->Vectorize(rf_vec_c);

  // ---- decide whether LOAD_V_SHARED is feasible ----
  // Compute total shared memory needed for all vector input buffers.
  // If SUPPORT_WARP_SHUFFLE is false, add TS*TR*dtype_bytes for allreduce shmem.
  if (LOAD_V_SHARED) {
    bool shmem_ok = false;
    auto target_attrs = target->attrs;
    if (target_attrs.find("max_shared_memory_per_block") != target_attrs.end()) {
      int64_t max_shmem = target_attrs["max_shared_memory_per_block"].cast<int64_t>();
      int64_t total = 0;
      bool all_static = true;
      for (const tir::Buffer& buf : vector_input_buffers) {
        int dtype_bytes = buf->dtype.bytes();
        int64_t buf_size = 1;
        for (const PrimExpr& dim : buf->shape) {
          if (const auto* imm = dim.as<IntImmNode>()) {
            buf_size *= imm->value;
          } else {
            all_static = false;
            break;
          }
        }
        if (!all_static) break;
        total += buf_size * dtype_bytes;
        if (!SUPPORT_WARP_SHUFFLE) {
          total += static_cast<int64_t>(TS) * TR * dtype_bytes;
        }
      }
      if (all_static && total <= max_shmem) shmem_ok = true;
    }
    LOAD_V_SHARED = shmem_ok;
  }

  // ---- vectorize load of Aq (quantized weight) into local ----
  // cache_read(rf, index=1, "local")
  s_tir::SBlockRV Aq_local = sch->CacheRead(rf, 1, "local");
  sch->ComputeAt(Aq_local, rf_r, /*preserve_unit_loops=*/true);
  auto aq_loops = sch->GetLoops(Aq_local);
  // last two loops are the s_local and r_local dims
  s_tir::LoopRV s_local = aq_loops[aq_loops.size() - 2];
  s_tir::LoopRV r_local = aq_loops[aq_loops.size() - 1];
  s_tir::LoopRV fused_load = sch->Fuse({s_local, r_local});

  // aq_vec_len = max(1, VEC_LOAD // dtype_bytes)
  auto aq_dtype = sch->Get(Aq_local)->reads[0]->buffer->dtype;
  int aq_dtype_bytes = aq_dtype.bytes();
  int aq_vec_len = std::max(1, VEC_LOAD / aq_dtype_bytes);

  auto fused_load_splits = sch->Split(
      fused_load,
      {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), aq_vec_len))},
      /*preserve_unit_iters=*/true);
  sch->Vectorize(fused_load_splits[1]);

  // ---- optionally load vector V into shared memory ----
  s_tir::SBlockRV V_shared;
  bool has_V_shared = false;
  if (LOAD_V_SHARED) {
    if (vector_input_buffers.size() != 1) {
      // Python returns None here; we signal failure
      return std::nullopt;
    }
    V_shared = sch->CacheRead(rf, 0, "shared");
    has_V_shared = true;
    sch->ComputeAt(V_shared, rf_tr, /*preserve_unit_loops=*/true);

    auto v_loops = sch->GetLoops(V_shared);
    s_tir::LoopRV v_last = v_loops.back();
    int64_t v_extent = GetLoopExtent(sch, v_last);

    int vec_length;
    if (v_extent > 0) {
      // avoid introducing predicates when vector length is too large
      int cand = GetMaxFactor(static_cast<int>(v_extent),
                              {TS * TR * 1, TS * TR * 2, TS * TR * 4, TS * TR * 8});
      vec_length = std::max(1, std::min(cand / TS / TR, LOAD_V_VEC));
    } else {
      vec_length = LOAD_V_VEC;
    }

    // split v_last -> [_, ty, tx, vec]
    // When TAG_R == "threadIdx.x": factors = [None, TS, TR, vec_length]
    // Otherwise (TAG_R == "threadIdx.y"): factors = [None, TR, TS, vec_length]
    ffi::Array<ffi::Optional<s_tir::ExprRV>> v_factors;
    if (TAG_R == "threadIdx.x") {
      v_factors = {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), TS)),
                   s_tir::ExprRV(IntImm(DataType::Int(32), TR)),
                   s_tir::ExprRV(IntImm(DataType::Int(32), vec_length))};
    } else {
      v_factors = {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), TR)),
                   s_tir::ExprRV(IntImm(DataType::Int(32), TS)),
                   s_tir::ExprRV(IntImm(DataType::Int(32), vec_length))};
    }
    auto v_splits = sch->Split(v_last, v_factors, /*preserve_unit_iters=*/true);
    // v_splits: [_, ty, tx, vec]
    sch->Bind(v_splits[1], "threadIdx.y");
    sch->Bind(v_splits[2], "threadIdx.x");
    sch->Vectorize(v_splits[3]);
  }

  // ---- reduce rf2: tile_s * TR * vec -> tile_s * TR ----
  sch->ReverseComputeAt(rf2, bx, /*preserve_unit_loops=*/true);
  // rf2 loops after reverse_compute_at: [bx, tr, vec_c, ts_tile_s...]
  auto rf2_loops = sch->GetLoops(rf2);
  TVM_FFI_ICHECK_GE(rf2_loops.size(), 4U);
  // loops[1] = tr, loops[2] = vec_c, loops[3..] = ts_tile_s
  ffi::Array<s_tir::LoopRV> ts_tile_s_loops(rf2_loops.begin() + 3, rf2_loops.end());
  s_tir::LoopRV ts_tile_s_fused = sch->Fuse(ts_tile_s_loops);

  // split ts_tile_s_fused -> [ts_o, ts_i, tile_s2]
  auto ts_tile_s_splits =
      sch->Split(ts_tile_s_fused,
                 {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), TS)),
                  s_tir::ExprRV(IntImm(DataType::Int(32), TILE_S))},
                 /*preserve_unit_iters=*/true);
  s_tir::LoopRV ts_o2 = ts_tile_s_splits[0];
  s_tir::LoopRV ts_i2 = ts_tile_s_splits[1];
  s_tir::LoopRV tile_s2 = ts_tile_s_splits[2];

  // split tile_s2 -> [tile_s2_outer, vec_s]
  int vec_s_factor = GetMaxFactor(TILE_S, {1, 2, 4, 8});
  auto tile_s2_splits = sch->Split(
      tile_s2,
      {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), vec_s_factor))},
      /*preserve_unit_iters=*/true);
  s_tir::LoopRV tile_s2_outer = tile_s2_splits[0];
  s_tir::LoopRV vec_s = tile_s2_splits[1];

  // fuse(ts_o2, ts_i2) -> ts2
  s_tir::LoopRV ts2 = sch->Fuse({ts_o2, ts_i2});

  // rf2 loops after the above: [bx, tr_rf2, vec_c_rf2, ts2, tile_s2_outer, vec_s]
  // reorder(ts2, tr_rf2, tile_s2_outer, vec_s, vec_c_rf2)
  auto rf2_loops2 = sch->GetLoops(rf2);
  TVM_FFI_ICHECK_GE(rf2_loops2.size(), 2U);
  s_tir::LoopRV tr_rf2 = rf2_loops2[1];
  s_tir::LoopRV vec_c_rf2 = rf2_loops2[2];
  sch->Reorder({ts2, tr_rf2, tile_s2_outer, vec_s, vec_c_rf2});
  sch->Bind(ts2, TAG_S);
  sch->Bind(tr_rf2, TAG_R);
  sch->Vectorize(vec_s);

  // ---- reduce gemv: tile_s * TR -> tile_s ----
  sch->ReverseComputeAt(gemv, bx, /*preserve_unit_loops=*/true);
  // gemv loops after reverse_compute_at: [bx, tr_gemv, ts_tile_s_gemv...]
  auto gemv_loops2 = sch->GetLoops(gemv);
  TVM_FFI_ICHECK_GE(gemv_loops2.size(), 3U);
  s_tir::LoopRV tr_gemv = gemv_loops2[1];
  ffi::Array<s_tir::LoopRV> ts_tile_s_gemv_loops(gemv_loops2.begin() + 2, gemv_loops2.end());
  s_tir::LoopRV ts_tile_s_gemv_fused = sch->Fuse(ts_tile_s_gemv_loops);

  auto ts_tile_s_gemv_splits =
      sch->Split(ts_tile_s_gemv_fused,
                 {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), TS)),
                  s_tir::ExprRV(IntImm(DataType::Int(32), TILE_S))},
                 /*preserve_unit_iters=*/true);
  s_tir::LoopRV ts_o_gemv = ts_tile_s_gemv_splits[0];
  s_tir::LoopRV ts_i_gemv = ts_tile_s_gemv_splits[1];
  s_tir::LoopRV tile_s_gemv = ts_tile_s_gemv_splits[2];

  s_tir::LoopRV ts_gemv = sch->Fuse({ts_o_gemv, ts_i_gemv});
  sch->Reorder({tile_s_gemv, ts_gemv, tr_gemv});
  sch->Bind(ts_gemv, TAG_S);
  sch->Bind(tr_gemv, TAG_R);

  // ---- decompose reductions ----
  // rf:  decompose at loop index 3
  {
    auto rf_l = sch->GetLoops(rf);
    TVM_FFI_ICHECK_GE(rf_l.size(), 4U);
    sch->DecomposeReduction(rf, rf_l[3]);
  }
  // rf2: decompose at last loop
  {
    auto rf2_l = sch->GetLoops(rf2);
    sch->DecomposeReduction(rf2, rf2_l.back());
  }

  sch->SetScope(rf, 0, "local");
  sch->SetScope(rf2, 0, "local");

  // ---- unroll annotations ----
  {
    auto rf_l = sch->GetLoops(rf);
    TVM_FFI_ICHECK_GE(rf_l.size(), 4U);
    sch->Annotate(rf_l[3], "pragma_auto_unroll_max_step", IntImm(DataType::Int(32), UNROLL));
    sch->Annotate(rf_l[3], "pragma_unroll_explicit", IntImm(DataType::Int(32), 1));
  }
  {
    auto rf2_l = sch->GetLoops(rf2);
    TVM_FFI_ICHECK_GE(rf2_l.size(), 4U);
    sch->Annotate(rf2_l[3], "pragma_auto_unroll_max_step", IntImm(DataType::Int(32), UNROLL));
    sch->Annotate(rf2_l[3], "pragma_unroll_explicit", IntImm(DataType::Int(32), 1));
  }

  // ---- V_shared unroll / vectorize annotations ----
  if (has_V_shared) {
    auto v_l = sch->GetLoops(V_shared);
    // annotate at index -4 (4th from the end)
    TVM_FFI_ICHECK_GE(v_l.size(), 4U);
    s_tir::LoopRV ann_loop = v_l[v_l.size() - 4];
    sch->Annotate(ann_loop, "pragma_unroll_explicit", IntImm(DataType::Int(32), UNROLL));
    sch->Annotate(ann_loop, "pragma_vectorize", IntImm(DataType::Int(32), 1));
  }

  // ---- epilogue scheduling ----
  if (epilogue_info.has_value()) {
    s_tir::SBlockRV epilogue = epilogue_info->get()->block_rv_;
    if (IsBroadcastEpilogue(sch, gemv, epilogue)) {
      sch->ReverseComputeAt(epilogue, bx, /*preserve_unit_loops=*/false);
      sch->SetScope(gemv, 0, "shared");
      auto epi_loops = sch->GetLoops(epilogue);
      // skip first two loops (bx, _), fuse the rest, split -> [_, TX]
      // TX = TS * TR  (total threads per block)
      int TX = TS * TR;
      ffi::Array<s_tir::LoopRV> epi_s(epi_loops.begin() + 2, epi_loops.end());
      auto epi_splits = sch->Split(
          sch->Fuse(epi_s),
          {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), TX))});
      sch->Bind(epi_splits[1], "threadIdx.x");
    } else {
      sch->ReverseComputeAt(epilogue, bx, /*preserve_unit_loops=*/true);
      // fuse all loops after bx, then split -> [ts_o, ts_i, tile_s]
      auto epi_loops = sch->GetLoops(epilogue);
      ffi::Array<s_tir::LoopRV> epi_rest(epi_loops.begin() + 1, epi_loops.end());
      // re-fetch after fuse
      s_tir::LoopRV epi_last = sch->GetLoops(epilogue).back();
      auto epi_splits =
          sch->Split(epi_last,
                     {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), TS)),
                      s_tir::ExprRV(IntImm(DataType::Int(32), TILE_S))},
                     /*preserve_unit_iters=*/true);
      s_tir::LoopRV epi_ts = sch->Fuse({epi_splits[0], epi_splits[1]});
      sch->Bind(epi_ts, TAG_S);
      sch->SetScope(gemv, 0, "local");
    }
  }

  return sch;
}

/*!
 * \brief Schedule a dequantized GEMV block whose innermost axis is spatial.
 *
 * Applies two rfactor steps over TR threads with explicit scale and dequant
 * tiling factors (SCALE_PACK, DEC_PACK). Only supported on Adreno
 * (opencl/vulkan) and Metal targets; returns nullopt for all others.
 *
 * \param sch The schedule to transform.
 * \param target The compilation target.
 * \param gemv The main GEMV block.
 * \param vector_input_buffers The vector input buffers identified by IsGemv.
 * \param epilogue_info Optional epilogue block to fuse after the reduction.
 * \param TAG_S Thread-axis tag for the spatial dimension.
 * \param TAG_R Thread-axis tag for the reduction dimension.
 * \param TS Number of spatial threads per block.
 * \param TR Number of reduction threads per block.
 * \param SCALE_PACK Tiling factor for the dequantization scale dimension.
 * \param DEC_PACK Tiling factor for the dequantization data dimension.
 * \param VEC_LOAD Vector load width (unused; reserved for future use).
 * \param VEC_C Vectorization factor for the innermost compute loop.
 * \param LOAD_V_SHARED Whether to stage the vector input through shared memory.
 * \param LOAD_V_VEC Vectorization factor for the shared-memory load.
 * \param UNROLL Unroll depth hint for the reduction loop.
 * \param LOAD_V_TILE Tile size for the vector-input shared-memory load.
 * \return The transformed schedule, or nullopt if the target or shape
 *         constraints are not satisfied.
 */
static ffi::Optional<s_tir::Schedule> SchOuterReduction(
    const s_tir::Schedule& sch, const tvm::Target& target, const s_tir::SBlockRV& gemv,
    const ffi::Array<tir::Buffer>& vector_input_buffers,
    const std::optional<DlightSBlockInfo>& epilogue_info, const std::string& TAG_S,
    const std::string& TAG_R, int TS, int TR, int SCALE_PACK, int DEC_PACK, int /*VEC_LOAD*/,
    int VEC_C, bool LOAD_V_SHARED, int LOAD_V_VEC, int UNROLL, int LOAD_V_TILE) {
  // ---- rfactor step 1: reduce to TR * DEC_PACK ----
  // Loops after normalize: [batch, s, r, c]
  auto loops0 = sch->GetLoops(gemv);
  TVM_FFI_ICHECK_EQ(loops0.size(), 4U);
  s_tir::LoopRV batch_loop = loops0[0];
  s_tir::LoopRV s_loop = loops0[1];
  s_tir::LoopRV r_loop = loops0[2];
  s_tir::LoopRV c_loop = loops0[3];

  // s = fuse(batch, s);  r = fuse(r, c)
  s_tir::LoopRV s_fused = sch->Fuse({batch_loop, s_loop});
  s_tir::LoopRV r_fused = sch->Fuse({r_loop, c_loop});

  // split s_fused -> [bx, ts]
  auto s_splits = sch->Split(
      s_fused, {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), TS))},
      /*preserve_unit_iters=*/true);
  s_tir::LoopRV bx = s_splits[0];
  s_tir::LoopRV ts = s_splits[1];

  // split r_fused -> [r, v_tile, tr, tile_r, vec_c]
  auto r_splits = sch->Split(
      r_fused,
      {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), LOAD_V_TILE)),
       s_tir::ExprRV(IntImm(DataType::Int(32), TR)),
       s_tir::ExprRV(IntImm(DataType::Int(32), SCALE_PACK)),
       s_tir::ExprRV(IntImm(DataType::Int(32), DEC_PACK))},
      /*preserve_unit_iters=*/true);
  s_tir::LoopRV r_outer = r_splits[0];
  s_tir::LoopRV v_tile = r_splits[1];
  s_tir::LoopRV tr = r_splits[2];
  s_tir::LoopRV tile_r = r_splits[3];
  s_tir::LoopRV vec_c = r_splits[4];

  // reorder(bx, ts, r, v_tile, tile_r, tr, vec_c)
  sch->Reorder({bx, ts, r_outer, v_tile, tile_r, tr, vec_c});

  // fuse(tr, vec_c) -> tr_vec_c;  rfactor on tr_vec_c
  s_tir::LoopRV tr_vec_c = sch->Fuse({tr, vec_c});
  s_tir::SBlockRV rf = sch->RFactor(tr_vec_c, 0);

  // ---- rfactor step 2: reduce to TR ----
  // gemv loops after rfactor: [bx, ts, tr_vec_c]
  auto gemv_loops1 = sch->GetLoops(gemv);
  TVM_FFI_ICHECK_EQ(gemv_loops1.size(), 3U);
  s_tir::LoopRV tr_vec_c2 = gemv_loops1[2];
  auto tr_vec_c2_splits = sch->Split(
      tr_vec_c2, {s_tir::ExprRV(IntImm(DataType::Int(32), TR)), ffi::Optional<s_tir::ExprRV>()},
      /*preserve_unit_iters=*/true);
  s_tir::LoopRV tr2 = tr_vec_c2_splits[0];
  s_tir::SBlockRV rf2 = sch->RFactor(tr2, 0);

  // ---- bind / vectorize the rf block ----
  // rf loops: [bx, ts, r_outer, v_tile, tile_r, tr_vec_c]
  auto rf_loops = sch->GetLoops(rf);
  TVM_FFI_ICHECK_GE(rf_loops.size(), 6U);
  s_tir::LoopRV rf_bx = rf_loops[0];
  s_tir::LoopRV rf_ts = rf_loops[1];
  s_tir::LoopRV rf_r = rf_loops[2];
  s_tir::LoopRV rf_v_tile = rf_loops[3];
  s_tir::LoopRV rf_tile_r = rf_loops[4];
  s_tir::LoopRV rf_tr_vec_c = rf_loops[5];

  // split tr_vec_c -> [tr, vec_c]  with exact factor DEC_PACK
  auto rf_tr_splits = sch->Split(rf_tr_vec_c, {s_tir::ExprRV(IntImm(DataType::Int(32), TR)),
                                               s_tir::ExprRV(IntImm(DataType::Int(32), DEC_PACK))});
  s_tir::LoopRV rf_tr = rf_tr_splits[0];
  s_tir::LoopRV rf_vec_c = rf_tr_splits[1];

  // reorder(bx, ts, tr, r, v_tile, tile_r, vec_c)
  sch->Reorder({rf_bx, rf_ts, rf_tr, rf_r, rf_v_tile, rf_tile_r, rf_vec_c});
  sch->Bind(rf_bx, "blockIdx.x");
  sch->Bind(rf_ts, TAG_S);
  sch->Bind(rf_tr, TAG_R);
  AutoVectorize(sch, rf_vec_c, VEC_C);

  // ---- cache reads ----
  // If rf has >= 3 reads, cache read index 2 (scale) at v_tile
  tir::SBlock rf_stmt = sch->Get(rf);
  if (rf_stmt->reads.size() >= 3) {
    s_tir::SBlockRV As_local = sch->CacheRead(rf, 2, "local");
    sch->ComputeAt(As_local, rf_v_tile, /*preserve_unit_loops=*/true);
  }

  // cache read index 1 (quantized weight Aq) at tile_r
  s_tir::SBlockRV Aq_local = sch->CacheRead(rf, 1, "local");
  sch->ComputeAt(Aq_local, rf_tile_r, /*preserve_unit_loops=*/true);

  // optionally load V into shared memory
  if (LOAD_V_SHARED) {
    s_tir::SBlockRV V_shared = sch->CacheRead(rf, 0, "shared");
    sch->ComputeAt(V_shared, rf_r, /*preserve_unit_loops=*/true);
    auto v_loops = sch->GetLoops(V_shared);
    s_tir::LoopRV v_last = v_loops.back();
    // split -> [_, v_tile, ts, tr, vec]
    auto v_splits = sch->Split(
        v_last,
        {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), LOAD_V_TILE)),
         s_tir::ExprRV(IntImm(DataType::Int(32), TS)), s_tir::ExprRV(IntImm(DataType::Int(32), TR)),
         s_tir::ExprRV(IntImm(DataType::Int(32), LOAD_V_VEC))},
        /*preserve_unit_iters=*/true);
    sch->Bind(v_splits[3], TAG_R);
    sch->Bind(v_splits[2], TAG_S);
    AutoVectorize(sch, v_splits[4], LOAD_V_VEC);
  }

  // ---- reduce rf2: tile_s * TR * vec -> tile_s * TR ----
  sch->ReverseComputeAt(rf2, bx, /*preserve_unit_loops=*/true);
  // rf2 loops: [bx, tr_rf2, vec_c_rf2, ts_rf2]
  auto rf2_loops = sch->GetLoops(rf2);
  TVM_FFI_ICHECK_GE(rf2_loops.size(), 4U);
  s_tir::LoopRV tr_rf2 = rf2_loops[1];
  s_tir::LoopRV vec_c_rf2 = rf2_loops[2];
  s_tir::LoopRV ts_rf2 = rf2_loops[3];
  sch->Reorder({ts_rf2, tr_rf2, vec_c_rf2});
  sch->Bind(ts_rf2, TAG_S);
  sch->Bind(tr_rf2, TAG_R);

  // ---- reduce gemv: tile_s * TR -> tile_s ----
  sch->ReverseComputeAt(gemv, bx, /*preserve_unit_loops=*/true);
  // gemv loops: [bx, tr_gemv, ts_gemv]
  auto gemv_loops2 = sch->GetLoops(gemv);
  TVM_FFI_ICHECK_GE(gemv_loops2.size(), 3U);
  s_tir::LoopRV tr_gemv = gemv_loops2[1];
  s_tir::LoopRV ts_gemv = gemv_loops2[2];
  sch->Reorder({ts_gemv, tr_gemv});
  sch->Bind(ts_gemv, TAG_S);
  sch->Bind(tr_gemv, TAG_R);

  // ---- decompose reductions ----
  // rf:  decompose at loop index 2
  {
    auto rf_l = sch->GetLoops(rf);
    TVM_FFI_ICHECK_GE(rf_l.size(), 3U);
    sch->DecomposeReduction(rf, rf_l[2]);
  }
  // rf2: decompose at last loop
  {
    auto rf2_l = sch->GetLoops(rf2);
    sch->DecomposeReduction(rf2, rf2_l.back());
  }

  sch->SetScope(rf, 0, "local");
  sch->SetScope(rf2, 0, "local");

  // ---- unroll annotations on rf2 loop index 3 ----
  {
    auto rf2_l = sch->GetLoops(rf2);
    TVM_FFI_ICHECK_GE(rf2_l.size(), 4U);
    sch->Annotate(rf2_l[3], "pragma_auto_unroll_max_step", IntImm(DataType::Int(32), UNROLL));
    sch->Annotate(rf2_l[3], "pragma_unroll_explicit", IntImm(DataType::Int(32), 1));
  }

  // ---- epilogue scheduling ----
  if (epilogue_info.has_value()) {
    s_tir::SBlockRV epilogue = epilogue_info->get()->block_rv_;
    if (IsBroadcastEpilogue(sch, gemv, epilogue)) {
      sch->ReverseComputeAt(epilogue, bx, /*preserve_unit_loops=*/false);
      sch->SetScope(gemv, 0, "shared");
      auto epi_loops = sch->GetLoops(epilogue);
      // skip first two loops, fuse the rest, split -> [_, ts]
      ffi::Array<s_tir::LoopRV> epi_s(epi_loops.begin() + 2, epi_loops.end());
      auto epi_splits = sch->Split(
          sch->Fuse(epi_s),
          {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), TS))});
      sch->Bind(epi_splits[1], TAG_S);
    } else {
      sch->ReverseComputeAt(epilogue, bx, /*preserve_unit_loops=*/true);
      s_tir::LoopRV epi_last = sch->GetLoops(epilogue).back();
      auto epi_splits = sch->Split(
          epi_last, {s_tir::ExprRV(IntImm(DataType::Int(32), TS)), ffi::Optional<s_tir::ExprRV>()},
          /*preserve_unit_iters=*/true);
      sch->Bind(epi_splits[0], TAG_S);
      sch->SetScope(gemv, 0, "local");
    }
  }

  return sch;
}

/*!
 * \brief Fallback schedule for outer-reduction GEMV on Adreno (opencl/vulkan).
 *
 * Splits the spatial axis into [bx, tx, vec], loads the vector input into
 * shared memory cooperatively, and decomposes the reduction at the outer
 * reduction loop. Only supported on Adreno opencl/vulkan targets.
 *
 * \param sch The schedule to transform.
 * \param target The compilation target.
 * \param block The main GEMV block.
 * \param vector_input_buffers The vector input buffers identified by IsGemv.
 * \param epilogue_info Optional epilogue block to fuse after the reduction.
 * \return The transformed schedule, or nullopt if the target is not supported.
 */
static ffi::Optional<s_tir::Schedule> SchOuterReductionFallback(
    const s_tir::Schedule& sch, const tvm::Target& target, const s_tir::SBlockRV& block,
    const ffi::Array<tir::Buffer>& vector_input_buffers,
    const std::optional<DlightSBlockInfo>& epilogue_info) {
  // NOTE: Only Android / Adreno is supported so far.
  bool is_adreno = false;
  for (const auto& key : target->keys) {
    if (key == "adreno") {
      is_adreno = true;
      break;
    }
  }
  bool is_opencl = (target->kind->name == "opencl");
  bool is_vulkan = (target->kind->name == "vulkan");
  if (!((is_opencl || is_vulkan) && is_adreno)) return std::nullopt;

  // Loops after NormalizeGemv: [batch, s, r, c]
  auto loops0 = sch->GetLoops(block);
  TVM_FFI_ICHECK_EQ(loops0.size(), 4U);
  s_tir::LoopRV batch_loop = loops0[0];
  s_tir::LoopRV s_loop = loops0[1];
  s_tir::LoopRV r_loop = loops0[2];
  s_tir::LoopRV c_loop = loops0[3];

  int64_t len_s = GetLoopExtent(sch, s_loop);

  // Adreno-tuned config
  // LOAD_V_SHARED is always 1 (true) for this fallback path
  const bool LOAD_V_SHARED = true;
  const int tx_len = 128;
  const int inner_r = 4;

  // vec_len = (4 if len_s > 4096 else 2) if isinstance(len_s, int) else 1
  int vec_len;
  if (len_s > 0) {  // static extent
    vec_len = (len_s > 4096) ? 4 : 2;
  } else {  // dynamic extent
    vec_len = 1;
  }

  // ---- split s -> [bx, tx, vec] ----
  auto s_splits = sch->Split(
      s_loop, {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), tx_len)),
               s_tir::ExprRV(IntImm(DataType::Int(32), vec_len))});
  s_tir::LoopRV bx = s_splits[0];
  s_tir::LoopRV tx = s_splits[1];
  s_tir::LoopRV vec = s_splits[2];

  // ---- split r -> [r0, r1] ----
  auto r_splits = sch->Split(
      r_loop, {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), inner_r))});
  s_tir::LoopRV r0 = r_splits[0];
  s_tir::LoopRV r1 = r_splits[1];

  // ---- bind ----
  sch->Bind(batch_loop, "blockIdx.z");
  sch->Bind(bx, "blockIdx.x");
  sch->Bind(tx, "threadIdx.x");

  // ---- reorder: [bx, tx, r0, r1, c, vec] ----
  sch->Reorder({bx, tx, r0, r1, c_loop, vec});

  // ---- unroll annotations on tx ----
  sch->Annotate(tx, "pragma_auto_unroll_max_step", IntImm(DataType::Int(32), 8));
  sch->Annotate(tx, "pragma_unroll_explicit", IntImm(DataType::Int(32), 1));

  // ---- load V into shared memory ----
  // Python: sch.cache_read(block, vector_input_buffers[0], storage_scope="shared")
  // Note: C++ CacheRead takes a buffer index (int), not a Buffer object.
  // vector_input_buffers[0] is the first vector buffer; it corresponds to
  // read index 0 of the block (the vector / activation input).
  if (LOAD_V_SHARED) {
    s_tir::SBlockRV V_shared = sch->CacheRead(block, 0, "shared");
    sch->ComputeAt(V_shared, bx, /*preserve_unit_loops=*/true);

    // split the last loop of V_shared -> [_, tx_v, vec_r]  with factors [None, tx_len, 8]
    auto v_loops = sch->GetLoops(V_shared);
    s_tir::LoopRV v_last = v_loops.back();
    auto v_splits = sch->Split(
        v_last,
        {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), tx_len)),
         s_tir::ExprRV(IntImm(DataType::Int(32), 8))},
        /*preserve_unit_iters=*/true);
    s_tir::LoopRV tx_v = v_splits[1];
    s_tir::LoopRV vec_r = v_splits[2];

    sch->Bind(tx_v, "threadIdx.x");

    // opencl: vectorize vec_r;  vulkan: unroll vec_r
    if (is_opencl) {
      sch->Vectorize(vec_r);
    } else {
      sch->Unroll(vec_r);
    }
  }

  // ---- vectorize the spatial vec loop ----
  sch->Vectorize(vec);

  // ---- epilogue scheduling ----
  if (epilogue_info.has_value()) {
    s_tir::SBlockRV epilogue = epilogue_info->get()->block_rv_;

    sch->ReverseComputeAt(epilogue, bx, /*preserve_unit_loops=*/true);

    // split the last epilogue loop -> [ts, vec_epi]  with factors [tx_len, vec_len]
    s_tir::LoopRV epi_last = sch->GetLoops(epilogue).back();
    auto epi_splits = sch->Split(epi_last,
                                 {s_tir::ExprRV(IntImm(DataType::Int(32), tx_len)),
                                  s_tir::ExprRV(IntImm(DataType::Int(32), vec_len))},
                                 /*preserve_unit_iters=*/true);
    s_tir::LoopRV ts_epi = epi_splits[0];
    s_tir::LoopRV vec_epi = epi_splits[1];

    sch->Bind(ts_epi, "threadIdx.x");
    sch->Vectorize(vec_epi);
    sch->SetScope(block, 0, "local");
  }

  // ---- decompose reduction at r0 ----
  sch->DecomposeReduction(block, r0);

  return sch;
}

class ApplyGemvRuleNode : public DlightRuleNode {
 public:
  ffi::Optional<s_tir::Schedule> Apply(const tir::PrimFunc& func, const tvm::Target& target) final {
    if (!target->HasKey("gpu")) return std::nullopt;

    // Build schedule
    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> func_map;
    func_map.Set(GlobalVar("main"), func);
    auto sch = s_tir::Schedule::Traced(IRModule(func_map), -1, 0,
                                       s_tir::ScheduleErrorRenderLevel::kDetail, true);

    // Step 0. Normalize and inline contiguous spatial blocks.
    auto block_infos = DlightNormalizePrimFunc(sch);
    if (block_infos.empty()) return std::nullopt;
    DlightTryInlineContiguousSpatial(sch, &block_infos);
    if (block_infos.empty()) return std::nullopt;

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

    // The main block must have exactly 2 or 3 iters:
    //   [S, R]   — plain GEMV
    //   [B, S, R] — batched GEMV
    DlightSBlockInfo block_info = block_infos[0];
    if (block_info->iters_.size() != 2 && block_info->iters_.size() != 3) {
      return std::nullopt;
    }

    s_tir::SBlockRV block = block_info->block_rv_;

    // Step 1. Check GEMV pattern and collect vector input buffers.
    ffi::Optional<ffi::Array<tir::Buffer>> maybe_vec_bufs = IsGemv(sch, block_info);
    if (!maybe_vec_bufs.defined()) return std::nullopt;
    ffi::Array<tir::Buffer> vector_input_buffers = maybe_vec_bufs.value();

    // Step 2. Normalize: reorder + fuse loops into [batch, s, r, c].
    std::optional<bool> is_inner_reduction = NormalizeGemv(sch, block_info);
    if (!is_inner_reduction.has_value()) return std::nullopt;

    // Step 3. Read the 4 normalized loops to derive tuning knobs.
    auto norm_loops = sch->GetLoops(block);
    TVM_FFI_ICHECK_EQ(norm_loops.size(), 4U);
    // loops: [batch_fused, s_fused, r_fused, c]
    int64_t len_batch = GetLoopExtent(sch, norm_loops[0]);
    int64_t len_s = GetLoopExtent(sch, norm_loops[1]);
    int64_t len_r = GetLoopExtent(sch, norm_loops[2]);
    int64_t len_c = GetLoopExtent(sch, norm_loops[3]);

    // len_S = len_batch * len_s  (treat dynamic as dynamic)
    // A value of -1 means "dynamic / unknown".
    bool len_S_static = (len_batch > 0 && len_s > 0);
    int64_t len_S = len_S_static ? len_batch * len_s : -1;
    bool len_R_static = (len_r > 0 && len_c > 0);
    int64_t len_R = len_R_static ? len_r * len_c : -1;

    // ---- target-specific tuning knobs (mirrors the big if/elif chain) ----
    std::string TAG_S = "threadIdx.y";
    std::string TAG_R = "threadIdx.x";
    bool SUPPORT_WARP_SHUFFLE = false;
    int VEC_LOAD = 1;
    int VEC_C = 1;
    bool LOAD_V_SHARED = false;
    int LOAD_V_VEC = -1;
    int UNROLL = 64;
    int TS = 1, TR = 64;

    const std::string& kind = target->kind->name;

    auto has_adreno = [&]() -> bool {
      for (const auto& k : target->keys) {
        if (k == "adreno") return true;
      }
      return false;
    };
    auto has_mali = [&]() -> bool {
      std::ostringstream oss;
      for (const auto& kv : target->attrs) {
        oss << kv.first;
      }
      return oss.str().find("mali") != std::string::npos;
    };

    if (kind == "cuda") {
      VEC_C = 4;
      LOAD_V_SHARED = true;
      LOAD_V_VEC = 8;
      VEC_LOAD = 4;
      UNROLL = 256;
      SUPPORT_WARP_SHUFFLE = true;
      if (len_S_static) {
        TS = 16;
        TR = 32;
      } else {
        TS = 1;
        TR = 64;
      }
    } else if (kind == "metal") {
      TAG_S = "threadIdx.x";
      TAG_R = "threadIdx.y";
      VEC_C = 1;
      LOAD_V_SHARED = false;
      LOAD_V_VEC = -1;
      UNROLL = 256;
      SUPPORT_WARP_SHUFFLE = true;
      if (len_S_static) {
        if (len_S > len_R) {
          TS = 4;
          TR = 16;
        } else {
          TS = 2;
          TR = 64;
        }
      } else {
        TS = 1;
        TR = 64;
      }
    } else if (kind == "rocm") {
      VEC_C = 4;
      LOAD_V_SHARED = false;
      LOAD_V_VEC = 8;
      UNROLL = 256;
      if (len_S_static) {
        if (len_S > len_R) {
          TS = 1;
          TR = 128;
        } else {
          TS = 8;
          TR = 64;
        }
      } else {
        TS = 1;
        TR = 64;
      }
    } else if (kind == "opencl" && has_adreno()) {
      TAG_S = "threadIdx.x";
      TAG_R = "threadIdx.y";
      VEC_C = 8;
      LOAD_V_SHARED = false;
      LOAD_V_VEC = -1;
      UNROLL = 8;
      TS = 2;
      TR = 32;
    } else if (kind == "vulkan" && has_adreno()) {
      TAG_S = "threadIdx.x";
      TAG_R = "threadIdx.y";
      VEC_C = 4;
      LOAD_V_SHARED = false;
      LOAD_V_VEC = -1;
      UNROLL = 8;
      TS = 2;
      TR = 32;
    } else if (kind == "vulkan") {
      VEC_C = 4;
      LOAD_V_SHARED = true;
      LOAD_V_VEC = 4;
      UNROLL = 256;
      if (len_S_static) {
        if (len_S > len_R) {
          TS = 4;
          TR = 32;
        } else {
          TS = 16;
          TR = 32;
        }
      } else {
        TS = 1;
        TR = 64;
      }
    } else if (kind == "opencl" && has_mali()) {
      VEC_C = 8;
      LOAD_V_SHARED = false;
      LOAD_V_VEC = -1;
      UNROLL = 64;
      TS = 1;
      TR = 64;
    } else {
      // default / fallback knobs (already set above)
    }

    // Clamp TS * TR to max_num_threads
    int max_num_threads = GetMaxThreadsPerBlock(target);
    while (TS * TR > max_num_threads) {
      if (TS > 1)
        TS /= 2;
      else
        TR /= 2;
    }

    // TILE_S = 1
    // TILE_R = len_c  (if len_c > 1)
    //        else max(GetMaxFactor(len_r, [TR*1..TR*8]) / TR, 1)
    int TILE_S = 1;
    int TILE_R;
    if (len_c > 1) {
      TILE_R = static_cast<int>(len_c);
    } else if (len_r > 0) {
      TILE_R =
          std::max(GetMaxFactor(static_cast<int>(len_r), {TR * 1, TR * 2, TR * 4, TR * 8}) / TR, 1);
    } else {
      TILE_R = 1;  // dynamic r — conservative
    }

    VEC_C = std::min(GetMaxFactor(TILE_R, {1, 2, 4, 8}), VEC_C);

    // Step 4. Dispatch to the appropriate scheduling path.
    if (*is_inner_reduction) {
      return SchInnerReduction(sch, target, block, vector_input_buffers, epilogue_info, TAG_S,
                               TAG_R, TS, TR, TILE_S, TILE_R, VEC_LOAD, VEC_C, LOAD_V_SHARED,
                               LOAD_V_VEC, UNROLL, SUPPORT_WARP_SHUFFLE);
    } else {
      // Try the target-specific outer-reduction schedule first.
      // It returns nullopt when the target / shape constraints are not met.
      // ---- outer-reduction tuning knobs ----
      const int DEC_PACK = 8;
      const int SCALE_PACK = 4;
      int or_VEC_C = VEC_C;
      bool or_LOAD_V_SHARED = false;
      int or_LOAD_V_VEC = 4;
      int or_LOAD_V_TILE = 1;
      int or_UNROLL = UNROLL;
      int or_TS = TS, or_TR = TR;
      std::string or_TAG_S = TAG_S, or_TAG_R = TAG_R;

      bool outer_target_ok = false;
      if (kind == "opencl" && has_adreno()) {
        or_TAG_S = "threadIdx.x";
        or_TAG_R = "threadIdx.y";
        or_VEC_C = 8;
        or_UNROLL = 8;
        or_TS = 64;
        or_TR = 4;
        or_LOAD_V_SHARED = false;
        or_LOAD_V_VEC = 4;
        or_LOAD_V_TILE = 8;
        outer_target_ok = true;
      } else if (kind == "vulkan" && has_adreno()) {
        or_TAG_S = "threadIdx.x";
        or_TAG_R = "threadIdx.y";
        or_VEC_C = 4;
        or_UNROLL = 8;
        or_TS = 64;
        or_TR = 4;
        or_LOAD_V_SHARED = false;
        or_LOAD_V_VEC = 4;
        or_LOAD_V_TILE = 8;
        outer_target_ok = true;
      } else if (kind == "metal") {
        or_TAG_S = "threadIdx.x";
        or_TAG_R = "threadIdx.y";
        or_VEC_C = 4;
        or_UNROLL = 8;
        or_TS = 128;
        or_TR = 4;
        or_LOAD_V_SHARED = false;
        or_LOAD_V_VEC = 4;
        or_LOAD_V_TILE = 4;
        outer_target_ok = true;
      }

      if (!outer_target_ok) {
        // Unsupported target for outer reduction — go straight to fallback
        return SchOuterReductionFallback(sch, target, block, vector_input_buffers, epilogue_info);
      }

      // LOAD_V_SHARED=false => LOAD_V_TILE = 1
      if (!or_LOAD_V_SHARED) or_LOAD_V_TILE = 1;

      // Shape guard: len_r must be static and >= LOAD_V_TILE * TR * SCALE_PACK
      if (len_r < 0 || len_r < static_cast<int64_t>(or_LOAD_V_TILE) * or_TR * SCALE_PACK) {
        return SchOuterReductionFallback(sch, target, block, vector_input_buffers, epilogue_info);
      }

      // Dynamic len_s: override TS/TR and enable shared
      if (!len_S_static) {
        or_TS = 256;
        or_TR = 1;
        or_LOAD_V_SHARED = true;
      }

      // Static len_s too large
      if (len_S_static && len_S > 96000) {
        return SchOuterReductionFallback(sch, target, block, vector_input_buffers, epilogue_info);
      }

      // TS = min(GetMaxFactor(len_s, [8,16,32,64]), TS)
      if (len_s > 0) {
        or_TS = std::min(GetMaxFactor(static_cast<int>(len_s), {8, 16, 32, 64}), or_TS);
      }

      // TILE_R for outer path
      int or_TILE_R;
      if (len_c > 1) {
        or_TILE_R = static_cast<int>(len_c);
      } else if (len_r > 0) {
        or_TILE_R = std::max(
            GetMaxFactor(static_cast<int>(len_r), {or_TR * 1, or_TR * 2, or_TR * 4, or_TR * 8}) /
                or_TR,
            1);
      } else {
        or_TILE_R = 1;
      }
      or_LOAD_V_VEC = std::min(GetMaxFactor(or_TILE_R, {1, 2, 4, 8}), or_LOAD_V_VEC);

      auto ret =
          SchOuterReduction(sch, target, block, vector_input_buffers, epilogue_info, or_TAG_S,
                            or_TAG_R, or_TS, or_TR, SCALE_PACK, DEC_PACK, /*VEC_LOAD=*/1, or_VEC_C,
                            or_LOAD_V_SHARED, or_LOAD_V_VEC, or_UNROLL, or_LOAD_V_TILE);
      if (!ret.defined()) {
        return SchOuterReductionFallback(sch, target, block, vector_input_buffers, epilogue_info);
      }
      return sch;
    }
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dlight.ApplyGemvRule", ApplyGemvRuleNode, DlightRuleNode);
};

ffi::Optional<s_tir::Schedule> ApplyGemvRule(tir::PrimFunc func, tvm::Target target) {
  return ApplyGemvRuleNode().Apply(func, target);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("dl.gpu.GEMV", ApplyGemvRule);
}

}  // namespace dlight
}  // namespace tvm
