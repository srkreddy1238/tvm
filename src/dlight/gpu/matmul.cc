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
 * \file src/dlight/gpu/matmul.cc
 * \brief Matmul schedule rule for GPU operators.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/dlight/dlight_rule.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "../analysis/common_analysis.h"
#include "../base/common_schedules.h"
#include "../base/utils.h"

namespace tvm {
namespace dlight {

/*!
 * \brief Iter-axis classification for the GEMM pattern
 *        C[S, I, J] += A[S, I, K] * B[S, J, K].
 */
enum class IterKind {
  kIter_S = 0,  // batch / other spatial axes
  kIter_I = 1,  // I axis  (A rows / C rows)
  kIter_J = 2,  // J axis  (B cols / C cols)
  kIter_K = 3,  // K axis  (reduction / inner product)
  kIter_T = 4,  // trivial (extent == 1)
};

/*! \brief Pair of IterKind and the corresponding loop extent. */
struct IterTrait {
  IterKind kind;
  PrimExpr extent;
};

/*!
 * \brief Build an IndexMap that fuses all iter-axes of each kind in
 *        `kind_order` into a single output dimension.
 *
 * \param traits Per-axis (kind, extent) descriptors in input order.
 * \param kind_order The output dimension order; one output dim per kind.
 * \return The constructed IndexMap.
 */
static tir::IndexMap MakeIterFusionIndexMap(const std::vector<IterTrait>& traits,
                                            const std::vector<IterKind>& kind_order) {
  // Build input vars
  std::vector<tir::Var> input_vars;
  input_vars.reserve(traits.size());
  for (int i = 0; i < static_cast<int>(traits.size()); ++i) {
    DataType dt = traits[i].extent->dtype;
    input_vars.push_back(tir::Var("i" + std::to_string(i), dt));
  }

  // Fuse iters of the same kind
  std::unordered_map<int, PrimExpr> fused;  // key = (int)IterKind
  for (int i = 0; i < static_cast<int>(traits.size()); ++i) {
    if (traits[i].kind == IterKind::kIter_T) continue;
    int k = static_cast<int>(traits[i].kind);
    auto it = fused.find(k);
    if (it == fused.end()) {
      fused[k] = input_vars[i];
    } else {
      it->second = it->second * traits[i].extent + input_vars[i];
    }
  }

  // Build output indices in kind_order
  ffi::Array<PrimExpr> final_indices;
  DataType zero_dt = traits.empty() ? DataType::Int(32) : traits[0].extent->dtype;
  for (IterKind kind : kind_order) {
    int k = static_cast<int>(kind);
    auto it = fused.find(k);
    if (it != fused.end()) {
      final_indices.push_back(it->second);
    } else {
      final_indices.push_back(tir::make_const(zero_dt, 0));
    }
  }

  ffi::Array<tir::Var> iv_arr;
  for (auto& v : input_vars) iv_arr.push_back(v);
  return tir::IndexMap(iv_arr, final_indices, /*inverse_index_map=*/std::nullopt);
}

/*!
 * \brief Classify each block iter-var according to the GEMM pattern
 *        C[S, I, J] += A[S, I, K] * B[S, J, K].
 *
 * \param block The TIR block to analyse.
 * \param A_traits_out Receives the per-axis traits for the A operand.
 * \param B_traits_out Receives the per-axis traits for the B operand.
 * \param C_traits_out Receives the per-axis traits for the C operand.
 * \param block_traits_out Receives the traits for all block iter-vars.
 * \return True if the block matches the GEMM pattern, false otherwise.
 */
static bool DetectIterTraits(const tir::SBlock& block, std::vector<IterTrait>* A_traits_out,
                             std::vector<IterTrait>* B_traits_out,
                             std::vector<IterTrait>* C_traits_out,
                             std::vector<IterTrait>* block_traits_out) {
  if (block->reads.size() != 2 || block->writes.size() != 1) return false;

  // Collect the set of block iter-vars that appear in each buffer region.
  auto get_access_vars = [&](const tir::BufferRegion& buf_region,
                             std::unordered_set<const tir::VarNode*>* out) -> bool {
    for (const Range& r : buf_region->region) {
      if (!tir::is_one(r->extent)) return false;  // not element-wise
      tir::PostOrderVisit(r->min, [&](const ObjectRef& node) {
        if (const auto* v = node.as<tir::VarNode>()) out->insert(v);
      });
    }
    return true;
  };

  std::unordered_set<const tir::VarNode*> A_vars, B_vars, C_vars;
  if (!get_access_vars(block->reads[0], &A_vars)) return false;
  if (!get_access_vars(block->reads[1], &B_vars)) return false;
  if (!get_access_vars(block->writes[0], &C_vars)) return false;

  // Classify each block iter-var
  std::unordered_map<const tir::VarNode*, IterTrait> traits;
  for (const tir::IterVar& iv : block->iter_vars) {
    const tir::VarNode* var = iv->var.get();
    IterKind kind;
    if (tir::is_one(iv->dom->extent)) {
      kind = IterKind::kIter_T;
    } else if (iv->iter_type == tir::kDataPar) {
      bool in_A = A_vars.count(var), in_B = B_vars.count(var), in_C = C_vars.count(var);
      if (in_A && in_B && in_C) {
        kind = IterKind::kIter_S;
      } else if (in_A && in_C && !in_B) {
        kind = IterKind::kIter_I;
      } else if (in_B && in_C && !in_A) {
        kind = IterKind::kIter_J;
      } else {
        return false;
      }
    } else if (iv->iter_type == tir::kCommReduce) {
      bool in_A = A_vars.count(var), in_B = B_vars.count(var), in_C = C_vars.count(var);
      if (in_A && in_B && !in_C) {
        kind = IterKind::kIter_K;
      } else {
        return false;
      }
    } else {
      return false;
    }
    traits[var] = IterTrait{kind, iv->dom->extent};
  }

  // Must have I, J, K axes
  bool has_I = false, has_J = false, has_K = false;
  for (auto& [v, t] : traits) {
    if (t.kind == IterKind::kIter_I) has_I = true;
    if (t.kind == IterKind::kIter_J) has_J = true;
    if (t.kind == IterKind::kIter_K) has_K = true;
  }
  if (!has_I || !has_J || !has_K) return false;

  // Build per-buffer trait lists in block iter-var order
  auto build_traits = [&](const std::unordered_set<const tir::VarNode*>& vars,
                          std::vector<IterTrait>* out) {
    for (const tir::IterVar& iv : block->iter_vars) {
      if (vars.count(iv->var.get())) out->push_back(traits.at(iv->var.get()));
    }
  };

  build_traits(A_vars, A_traits_out);
  build_traits(B_vars, B_traits_out);
  build_traits(C_vars, C_traits_out);
  for (const tir::IterVar& iv : block->iter_vars) {
    block_traits_out->push_back(traits.at(iv->var.get()));
  }
  return true;
}

/*!
 * \brief Derive the four index maps needed to normalize a GEMM-like block.
 *
 * \param block The TIR block to analyse.
 * \param matmul_map_out Receives the full block-iter fusion map [S,I,J,K].
 * \param a_map_out Receives the A-buffer fusion map [S,I,K].
 * \param b_map_out Receives the B-buffer fusion map [S,J,K].
 * \param c_map_out Receives the C-buffer fusion map [S,I,J].
 * \return True if the block is a GEMM-like kernel, false otherwise.
 */
static bool GetIndexMap(const tir::SBlock& block, tir::IndexMap* matmul_map_out,
                        tir::IndexMap* a_map_out, tir::IndexMap* b_map_out,
                        tir::IndexMap* c_map_out) {
  std::vector<IterTrait> A_traits, B_traits, C_traits, block_traits;
  if (!DetectIterTraits(block, &A_traits, &B_traits, &C_traits, &block_traits)) return false;

  *a_map_out =
      MakeIterFusionIndexMap(A_traits, {IterKind::kIter_S, IterKind::kIter_I, IterKind::kIter_K});
  *b_map_out =
      MakeIterFusionIndexMap(B_traits, {IterKind::kIter_S, IterKind::kIter_J, IterKind::kIter_K});
  *c_map_out =
      MakeIterFusionIndexMap(C_traits, {IterKind::kIter_S, IterKind::kIter_I, IterKind::kIter_J});
  *matmul_map_out = MakeIterFusionIndexMap(
      block_traits, {IterKind::kIter_S, IterKind::kIter_I, IterKind::kIter_J, IterKind::kIter_K});
  return true;
}

/*!
 * \brief Return the root block of the schedule (always named "root").
 *
 * \param sch The schedule to query.
 * \return The root SBlockRV.
 */
static s_tir::SBlockRV GetRootBlock(const s_tir::Schedule& sch) {
  return sch->GetSBlock("root", ffi::Optional<ffi::String>("main"));
}

/*!
 * \brief Return the unique reduction block from a list of blocks.
 *
 * All blocks must be either purely spatial (DataPar only) or reduction
 * (DataPar + CommReduce). Returns an empty array if the pattern does not
 * match or if there is not exactly one reduction block.
 *
 * \param sch The schedule that owns the blocks.
 * \param blocks The candidate block list (typically all children of root).
 * \return A single-element array containing the reduction block, or empty.
 */
static ffi::Array<s_tir::SBlockRV> GetReductionBlocks(const s_tir::Schedule& sch,
                                                      const ffi::Array<s_tir::SBlockRV>& blocks) {
  auto is_reduction = [&](const s_tir::SBlockRV& blk) -> bool {
    bool has_s = false, has_r = false;
    for (const tir::IterVar& iv : sch->Get(blk)->iter_vars) {
      if (iv->iter_type == tir::kDataPar)
        has_s = true;
      else if (iv->iter_type == tir::kCommReduce)
        has_r = true;
      else
        return false;
    }
    return has_s && has_r;
  };
  auto is_spatial = [&](const s_tir::SBlockRV& blk) -> bool {
    for (const tir::IterVar& iv : sch->Get(blk)->iter_vars) {
      if (iv->iter_type != tir::kDataPar) return false;
    }
    return true;
  };

  // All blocks must be spatial or reduction
  for (const s_tir::SBlockRV& blk : blocks) {
    if (!is_reduction(blk) && !is_spatial(blk)) return {};
  }

  ffi::Array<s_tir::SBlockRV> reduction_blocks;
  for (const s_tir::SBlockRV& blk : blocks) {
    if (is_reduction(blk)) reduction_blocks.push_back(blk);
  }
  // Exactly one reduction block required
  if (reduction_blocks.size() != 1) return {};
  return reduction_blocks;
}

/*!
 * \brief Recursively collect all transitive producer blocks of `block`.
 *
 * \param sch The schedule that owns the blocks.
 * \param block The block whose producers are collected.
 * \param result Accumulated producer list (appended to, not cleared).
 */
static void CollectProducers(const s_tir::Schedule& sch, const s_tir::SBlockRV& block,
                             std::vector<s_tir::SBlockRV>* result) {
  for (const s_tir::SBlockRV& p : sch->GetProducers(block)) {
    result->push_back(p);
    CollectProducers(sch, p, result);
  }
}

/*!
 * \brief Recursively collect all transitive consumer blocks of `block`.
 *
 * \param sch The schedule that owns the blocks.
 * \param block The block whose consumers are collected.
 * \param result Accumulated consumer list (appended to, not cleared).
 */
static void CollectConsumers(const s_tir::Schedule& sch, const s_tir::SBlockRV& block,
                             std::vector<s_tir::SBlockRV>* result) {
  for (const s_tir::SBlockRV& c : sch->GetConsumers(block)) {
    result->push_back(c);
    CollectConsumers(sch, c, result);
  }
}

/*!
 * \brief Repeatedly inline all transitive producers of `block` until
 *        no further inlining is possible.
 *
 * \param sch The schedule to transform.
 * \param block The block whose producers are inlined.
 */
static void AutoInlineProducers(const s_tir::Schedule& sch, const s_tir::SBlockRV& block) {
  while (true) {
    std::vector<s_tir::SBlockRV> producers;
    CollectProducers(sch, block, &producers);
    int inlined = 0;
    for (const s_tir::SBlockRV& p : producers) {
      try {
        sch->ComputeInline(p);
        ++inlined;
      } catch (const tvm::ffi::Error&) {
        continue;
      }
    }
    if (inlined == 0) break;
  }
}

/*!
 * \brief Repeatedly inline all transitive consumers of `block` until
 *        no further inlining is possible.
 *
 * \param sch The schedule to transform.
 * \param block The block whose consumers are inlined.
 */
static void AutoInlineConsumers(const s_tir::Schedule& sch, const s_tir::SBlockRV& block) {
  while (true) {
    std::vector<s_tir::SBlockRV> consumers;
    CollectConsumers(sch, block, &consumers);
    int inlined = 0;
    for (const s_tir::SBlockRV& c : consumers) {
      try {
        sch->ComputeInline(c);
        ++inlined;
      } catch (const tvm::ffi::Error&) {
        continue;
      }
    }
    for (const s_tir::SBlockRV& c : consumers) {
      try {
        sch->ReverseComputeInline(c);
        ++inlined;
      } catch (const tvm::ffi::Error&) {
        continue;
      }
    }
    if (inlined == 0) break;
  }
}

/*!
 * \brief Inline all consumers of `block`, retrying after inlining any
 *        blocking producers of consumers that could not be inlined on the
 *        first pass.
 *
 * \param sch The schedule to transform.
 * \param block The block whose consumer chain is inlined.
 */
static void AutoInlineConsumerChain(const s_tir::Schedule& sch, const s_tir::SBlockRV& block) {
  AutoInlineConsumers(sch, block);
  ffi::Array<s_tir::SBlockRV> remaining = sch->GetConsumers(block);
  if (!remaining.empty()) {
    // Some consumers could not be inlined; inline their other producers first.
    for (const s_tir::SBlockRV& c : remaining) {
      for (const s_tir::SBlockRV& p : sch->GetProducers(c)) {
        if (!sch->Get(p).same_as(sch->Get(block))) {
          AutoInlineProducers(sch, p);
          try {
            sch->ComputeInline(p);
          } catch (const tvm::ffi::Error&) {
          }
        }
      }
    }
    AutoInlineConsumers(sch, block);
  }
}

/*! \brief Tiling and threading configuration for the generic matmul schedule. */
struct MatmulConfig {
  int block_size_x = 8;
  int block_size_y = 8;
  int vthread_x = 1;
  int vthread_y = 1;
  int micro_size_x = 4;
  int micro_size_y = 4;
  int micro_size_k = 8;
  int vector_size = 1;
  int unroll = 256;  // 0 means no unroll
  bool use_shared = true;
  bool storage_align = false;
  bool inner_x = false;
};

/*!
 * \brief Return the MatmulConfig tuned for `target`.
 *
 * \param target The compilation target.
 * \return A MatmulConfig with target-appropriate tiling parameters.
 */
static MatmulConfig GetMatmulConfig(const tvm::Target& target) {
  if (target->kind->name == "cuda" || target->kind->name == "rocm") {
    return MatmulConfig{
        /*block_size_x=*/8,     /*block_size_y=*/16,
        /*vthread_x=*/1,        /*vthread_y=*/1,
        /*micro_size_x=*/4,     /*micro_size_y=*/4,
        /*micro_size_k=*/16,    /*vector_size=*/2,
        /*unroll=*/256,         /*use_shared=*/true,
        /*storage_align=*/true, /*inner_x=*/false,
    };
  }
  return MatmulConfig{};
}

/*!
 * \brief Dlight schedule rule for generic tiled matmul.
 *
 * Normalizes the block to C[S,I,J] += A[S,I,K]*B[S,J,K] via reindex and
 * block-layout transforms, then tiles and binds the loops for GPU execution.
 * Hardware-specific tensorization (WMMA, simdgroup) is not included.
 */
class ApplyMatmulRuleNode : public DlightRuleNode {
 public:
  ffi::Optional<s_tir::Schedule> Apply(const tir::PrimFunc& func, const tvm::Target& target) final {
    if (!target->HasKey("gpu")) return std::nullopt;

    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> func_map;
    func_map.Set(GlobalVar("main"), func);
    auto sch = s_tir::Schedule::Traced(IRModule(func_map), -1, 0,
                                       s_tir::ScheduleErrorRenderLevel::kDetail, true);

    MatmulConfig config = GetMatmulConfig(target);

    s_tir::SBlockRV root_block = GetRootBlock(sch);
    ffi::Array<s_tir::SBlockRV> blocks = sch->GetChildBlocks(root_block);

    ffi::Array<s_tir::SBlockRV> reduction_blocks = GetReductionBlocks(sch, blocks);
    if (reduction_blocks.empty()) return std::nullopt;

    s_tir::SBlockRV main_block = reduction_blocks[0];

    // Step 0. Normalize to C[S,I,J] += A[S,I,K]*B[S,J,K].
    //
    // Phase 0a: reject non-matmul blocks before calling ReIndex.
    // GetIndexMap on the pre-reindex block is a cheap structural guard: it
    // checks that the block has exactly 2 reads and 1 write and that every
    // iter-var can be classified as S/I/J/K/T.  Blocks like pool_max have
    // only 1 read buffer, so they are rejected here before ReIndex is called
    // with an out-of-range buffer index.
    {
      tir::IndexMap dummy_matmul, dummy_a, dummy_b, dummy_c;
      if (!GetIndexMap(sch->Get(main_block), &dummy_matmul, &dummy_a, &dummy_b, &dummy_c)) {
        return std::nullopt;
      }
    }

    // Phase 0b: reindex so that each buffer is accessed with direct
    // one-to-one iter-var indices, then call GetIndexMap on the post-reindex
    // block to build the index maps used for TransformLayout.
    // Calling GetIndexMap on the pre-reindex block for the actual maps would
    // build them from compound access expressions (e.g. oh*stride+rh for
    // conv2d), whose initial_indices count can diverge from the reindex
    // buffer shape and cause a shape mismatch in TransformLayout.
    s_tir::SBlockRV reindex_a = sch->ReIndex(main_block, 0, s_tir::BufferIndexType::kRead);
    s_tir::SBlockRV reindex_b = sch->ReIndex(main_block, 1, s_tir::BufferIndexType::kRead);
    s_tir::SBlockRV reindex_c = sch->ReIndex(main_block, 0, s_tir::BufferIndexType::kWrite);

    tir::IndexMap matmul_index_map, a_index_map, b_index_map, c_index_map;
    if (!GetIndexMap(sch->Get(main_block), &matmul_index_map, &a_index_map, &b_index_map,
                     &c_index_map)) {
      return std::nullopt;
    }

    sch->TransformLayout(reindex_a, 0, s_tir::BufferIndexType::kWrite, a_index_map,
                         /*pad_value=*/std::nullopt, /*assume_injective_transform=*/false);
    sch->TransformLayout(reindex_b, 0, s_tir::BufferIndexType::kWrite, b_index_map,
                         /*pad_value=*/std::nullopt, /*assume_injective_transform=*/false);
    sch->TransformLayout(reindex_c, 0, s_tir::BufferIndexType::kRead, c_index_map,
                         /*pad_value=*/std::nullopt, /*assume_injective_transform=*/false);
    sch->TransformBlockLayout(main_block, matmul_index_map);

    // Step 1. Pad einsum for tiling.
    int y_kernel_size = config.vthread_y * config.block_size_y * config.micro_size_y;
    int x_kernel_size = config.vthread_x * config.block_size_x * config.micro_size_x;

    ffi::Array<Integer> pad_factors;
    if (config.inner_x) {
      pad_factors = {Integer(1), Integer(y_kernel_size), Integer(x_kernel_size),
                     Integer(config.micro_size_k)};
    } else {
      pad_factors = {Integer(1), Integer(x_kernel_size), Integer(y_kernel_size),
                     Integer(config.micro_size_k)};
    }
    sch->PadEinsum(main_block, pad_factors);

    // Step 2. Tile and bind loops.
    // After PadEinsum the block has exactly 4 loops: [batch, x, y, k] when
    // inner_x=false, or [batch, y, x, k] when inner_x=true.
    auto loops = sch->GetLoops(main_block);
    TVM_FFI_ICHECK_EQ(loops.size(), 4U) << "Expected 4 loops after PadEinsum";

    s_tir::LoopRV batch = loops[0];
    s_tir::LoopRV loop_y = config.inner_x ? loops[1] : loops[2];
    s_tir::LoopRV loop_x = config.inner_x ? loops[2] : loops[1];
    s_tir::LoopRV loop_k = loops[3];

    // Split y -> [by, vy, ty, yi]
    auto y_splits =
        sch->Split(loop_y, {ffi::Optional<s_tir::ExprRV>(),
                            s_tir::ExprRV(IntImm(DataType::Int(32), config.vthread_y)),
                            s_tir::ExprRV(IntImm(DataType::Int(32), config.block_size_y)),
                            s_tir::ExprRV(IntImm(DataType::Int(32), config.micro_size_y))});
    s_tir::LoopRV by = y_splits[0], vy = y_splits[1], ty = y_splits[2], yi = y_splits[3];

    // Split x -> [bx, vx, tx, xi]
    auto x_splits =
        sch->Split(loop_x, {ffi::Optional<s_tir::ExprRV>(),
                            s_tir::ExprRV(IntImm(DataType::Int(32), config.vthread_x)),
                            s_tir::ExprRV(IntImm(DataType::Int(32), config.block_size_x)),
                            s_tir::ExprRV(IntImm(DataType::Int(32), config.micro_size_x))});
    s_tir::LoopRV bx = x_splits[0], vx = x_splits[1], tx = x_splits[2], xi = x_splits[3];

    // Split k -> [ko, ki]
    auto k_splits =
        sch->Split(loop_k, {ffi::Optional<s_tir::ExprRV>(),
                            s_tir::ExprRV(IntImm(DataType::Int(32), config.micro_size_k))});
    s_tir::LoopRV ko = k_splits[0], ki = k_splits[1];

    // Reorder: [by, bx, vy, vx, ty, tx, ko, ki, yi, xi]  (or xi, yi for inner_x)
    ffi::Array<s_tir::LoopRV> reorder_arr = {by, bx, vy, vx, ty, tx, ko, ki};
    if (config.inner_x) {
      reorder_arr.push_back(yi);
      reorder_arr.push_back(xi);
    } else {
      reorder_arr.push_back(xi);
      reorder_arr.push_back(yi);
    }
    sch->Reorder(reorder_arr);

    s_tir::LoopRV by_fused = sch->Fuse({batch, by});

    sch->Bind(bx, "blockIdx.x");
    sch->Bind(by_fused, "blockIdx.y");
    sch->Bind(vy, "vthread.y");
    sch->Bind(vx, "vthread.x");
    sch->Bind(ty, "threadIdx.y");
    sch->Bind(tx, "threadIdx.x");

    int inner_loop_size = config.inner_x ? config.micro_size_x : config.micro_size_y;
    if (inner_loop_size % config.vector_size == 0) {
      s_tir::LoopRV inner = config.inner_x ? xi : yi;
      auto vec_splits =
          sch->Split(inner, {ffi::Optional<s_tir::ExprRV>(),
                             s_tir::ExprRV(IntImm(DataType::Int(32), config.vector_size))});
      sch->Vectorize(vec_splits[1]);
    }

    if (config.unroll > 0) {
      sch->Annotate(tx, "pragma_auto_unroll_max_step", IntImm(DataType::Int(32), config.unroll));
      sch->Annotate(tx, "pragma_unroll_explicit", IntImm(DataType::Int(32), 1));
    }

    // Step 3. Cache-write to local (l2g).
    s_tir::SBlockRV l2g = sch->CacheWrite(main_block, 0, "local");
    sch->ReverseComputeAt(l2g, tx, /*preserve_unit_loops=*/true);
    if (config.micro_size_x % config.vector_size == 0) {
      auto l2g_loops = sch->GetLoops(l2g);
      auto l2g_vec_splits = sch->Split(
          l2g_loops.back(), {ffi::Optional<s_tir::ExprRV>(),
                             s_tir::ExprRV(IntImm(DataType::Int(32), config.vector_size))});
      sch->Vectorize(l2g_vec_splits[1]);
    }

    // Step 4. Cooperative fetch into shared memory (optional).
    if (config.use_shared) {
      auto cooperative_fetch = [&](int index, int vec_len) -> s_tir::SBlockRV {
        s_tir::SBlockRV blk = sch->CacheRead(main_block, index, "shared");
        int num_loops = static_cast<int>(sch->GetLoops(blk).size());
        sch->ComputeAt(blk, ko, /*preserve_unit_loops=*/true);
        auto blk_loops = sch->GetLoops(blk);
        int n = static_cast<int>(blk_loops.size());
        ffi::Array<s_tir::LoopRV> to_fuse;
        for (int i = n - num_loops; i < n; ++i) to_fuse.push_back(blk_loops[i]);
        s_tir::LoopRV fused = sch->Fuse(to_fuse);
        // split fused -> [ty_coop, tx_coop, _, vec]
        auto coop_splits =
            sch->Split(fused, {s_tir::ExprRV(IntImm(DataType::Int(32), config.block_size_y)),
                               s_tir::ExprRV(IntImm(DataType::Int(32), config.block_size_x)),
                               ffi::Optional<s_tir::ExprRV>(),
                               s_tir::ExprRV(IntImm(DataType::Int(32), vec_len))});
        sch->Vectorize(coop_splits[3]);
        sch->Bind(coop_splits[0], "threadIdx.y");
        sch->Bind(coop_splits[1], "threadIdx.x");
        if (config.storage_align) {
          sch->StorageAlign(blk, 0, /*axis=*/1, /*factor=*/8, /*offset=*/vec_len);
        }
        return blk;
      };

      s_tir::SBlockRV a_g2s = cooperative_fetch(0, config.vector_size);
      s_tir::SBlockRV b_g2s = cooperative_fetch(1, config.vector_size);

      AutoInlineProducers(sch, a_g2s);
      AutoInlineProducers(sch, b_g2s);
    } else {
      AutoInlineProducers(sch, main_block);
    }

    // Step 5. Inline consumers and decompose reduction.
    AutoInlineConsumerChain(sch, l2g);
    sch->DecomposeReduction(main_block, ko);

    return sch;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dlight.ApplyMatmulRule", ApplyMatmulRuleNode, DlightRuleNode);
};

ffi::Optional<s_tir::Schedule> ApplyMatmulRule(tir::PrimFunc func, tvm::Target target) {
  return ApplyMatmulRuleNode().Apply(func, target);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("dl.gpu.Matmul", ApplyMatmulRule);
}

}  // namespace dlight
}  // namespace tvm
