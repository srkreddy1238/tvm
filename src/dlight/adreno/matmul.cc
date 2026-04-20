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
 * \file src/dlight/adreno/matmul.cc
 * \brief Cooperative-matrix matmul tensorization schedule rule for Adreno GPU.
 *
 * Exposes one DlightRule:
 *   - ApplyAdrenoMatmulTensorizationRule  (registered as "dl.adreno.MatmulTensorization")
 *   - ApplyAdrenoDequantMatmulTensorizationRule
 *     (registered as "dl.adreno.DequantMatmulTensorization")
 *
 * The TIR tensor intrinsics (wmma_fill, wmma_load, wmma_store, wmma_sync,
 * wmma_qcom) are registered by tensor_intrin_adreno.cc via the global
 * function "tir.tensor_intrin.adreno.GetAdrenoWmmaIntrinGroup".
 */

#include <stdint.h>
#include <tvm/dlight/dlight_rule.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/target/target.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../analysis/common_analysis.h"
#include "fallback.h"

namespace tvm {
namespace dlight {

/*! \brief Iter-axis classification for the GEMM pattern
 *        C[S, I, J] += A[S, I, K] * B[S, J, K].
 */
enum class MatmulIterKind {
  kIter_S = 0,  // batch / other spatial axes
  kIter_I = 1,  // I axis  (A rows / C rows)
  kIter_J = 2,  // J axis  (B cols / C cols)
  kIter_K = 3,  // K axis  (reduction / inner product)
  kIter_T = 4,  // trivial (extent == 1)
};

/*! \brief Pair of MatmulIterKind and the corresponding loop extent. */
struct MatmulIterTrait {
  MatmulIterKind kind;
  PrimExpr extent;
};

/*! \brief Tiling and threading configuration for DequantMatmulTensorization. */
struct DequantMatmulTensorizationConfig {
  int thread_size_x;  // = tile_m
  int thread_size_y;  // = tile_m_factor  (2 with qcom_extn, else 1)
  int thread_size_z;  // = get_max_factor(tile_n_factor, [1, 2])

  int splits_m_1;  // threadIdx.y factor when splitting M
  int splits_m_2;  // vthread factor (always 1)
  int splits_m_3;  // wmma tile size along M

  int splits_n_1;  // threadIdx.z factor when splitting N
  int splits_n_2;  // vthread factor (always 1)
  int splits_n_3;  // wmma tile size along N

  int splits_k_1;  // dequant unroll factor (always 2)
  int splits_k_2;  // quant unroll factor (always 1)
  int splits_k_3;  // wmma tile size along K

  int intrin_tile_m;
  int intrin_tile_n;
  int intrin_tile_k;
  int vector_size;
  bool unroll;
  bool use_textures;
  bool b_trans;
  std::string load_scope_a;   // "global"
  std::string load_scope_b;   // "local" or "shared"
  std::string store_scope_c;  // "local" or "shared"
  std::string in_dtype;
  std::string out_dtype;
  bool use_storage_align;
};

int manual_clzll(uint64_t x) {
  if (x == 0) return 64;  // Handle zero separately to avoid undefined behavior
  int n = 0;
  if ((x & 0xFFFFFFFF00000000ULL) == 0) {
    n += 32;
    x <<= 32;
  }
  if ((x & 0xFFFF000000000000ULL) == 0) {
    n += 16;
    x <<= 16;
  }
  if ((x & 0xFF00000000000000ULL) == 0) {
    n += 8;
    x <<= 8;
  }
  if ((x & 0xF000000000000000ULL) == 0) {
    n += 4;
    x <<= 4;
  }
  if ((x & 0xC000000000000000ULL) == 0) {
    n += 2;
    x <<= 2;
  }
  if ((x & 0x8000000000000000ULL) == 0) {
    n += 1;
  }
  return n;
}

/*!
 * \brief Build an IndexMap that fuses all iter-axes of each kind in
 *        `kind_order` into a single output dimension.
 *
 * \param traits Per-axis (kind, extent) descriptors in input order.
 * \param kind_order The output dimension order; one output dim per kind.
 * \return The constructed IndexMap.
 */
static tir::IndexMap MakeMatmulIterFusionIndexMap(const std::vector<MatmulIterTrait>& traits,
                                                  const std::vector<MatmulIterKind>& kind_order) {
  std::vector<tir::Var> input_vars;
  input_vars.reserve(traits.size());
  for (int i = 0; i < static_cast<int>(traits.size()); ++i) {
    DataType dt = traits[i].extent->dtype;
    input_vars.push_back(tir::Var("i" + std::to_string(i), dt));
  }

  // Fuse all axes of the same kind into a single expression
  std::unordered_map<int, PrimExpr> fused;
  for (int i = 0; i < static_cast<int>(traits.size()); ++i) {
    if (traits[i].kind == MatmulIterKind::kIter_T) continue;
    int k = static_cast<int>(traits[i].kind);
    auto it = fused.find(k);
    if (it == fused.end()) {
      fused[k] = input_vars[i];
    } else {
      it->second = it->second * traits[i].extent + input_vars[i];
    }
  }

  DataType zero_dt = traits.empty() ? DataType::Int(32) : traits[0].extent->dtype;
  ffi::Array<PrimExpr> final_indices;
  for (MatmulIterKind kind : kind_order) {
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
 * \param A_traits_out Receives the per-axis traits for the A operand, in
 *        the order the iter-vars appear in A's buffer access region.
 * \param B_traits_out Receives the per-axis traits for the B operand.
 * \param C_traits_out Receives the per-axis traits for the C operand.
 * \param block_traits_out Receives the traits for all block iter-vars.
 * \return True if the block matches the GEMM pattern, false otherwise.
 */
static bool DetectMatmulIterTraits(const tir::SBlock& block,
                                   std::vector<MatmulIterTrait>* A_traits_out,
                                   std::vector<MatmulIterTrait>* B_traits_out,
                                   std::vector<MatmulIterTrait>* C_traits_out,
                                   std::vector<MatmulIterTrait>* block_traits_out) {
  if (block->reads.size() != 2 || block->writes.size() != 1) return false;

  // Collect access vars in the order they appear in the buffer region.
  // Preserving access order is critical: it determines whether B is accessed
  // as [K,J] (transposed) or [J,K] (non-transposed).
  auto get_access_vars_ordered = [&](const tir::BufferRegion& buf_region,
                                     std::vector<const tir::VarNode*>* ordered_out,
                                     std::unordered_set<const tir::VarNode*>* set_out) -> bool {
    for (const Range& r : buf_region->region) {
      if (!tir::is_one(r->extent)) return false;
      tir::PostOrderVisit(r->min, [&](const ObjectRef& node) {
        if (const auto* v = node.as<tir::VarNode>()) {
          if (!set_out->count(v)) {
            set_out->insert(v);
            ordered_out->push_back(v);
          }
        }
      });
    }
    return true;
  };

  std::vector<const tir::VarNode*> A_vars_ord, B_vars_ord, C_vars_ord;
  std::unordered_set<const tir::VarNode*> A_vars, B_vars, C_vars;
  if (!get_access_vars_ordered(block->reads[0], &A_vars_ord, &A_vars)) return false;
  if (!get_access_vars_ordered(block->reads[1], &B_vars_ord, &B_vars)) return false;
  if (!get_access_vars_ordered(block->writes[0], &C_vars_ord, &C_vars)) return false;

  // Classify each iter-var
  std::unordered_map<const tir::VarNode*, MatmulIterTrait> traits;
  for (const tir::IterVar& iv : block->iter_vars) {
    const tir::VarNode* var = iv->var.get();
    MatmulIterKind kind;
    if (tir::is_one(iv->dom->extent)) {
      kind = MatmulIterKind::kIter_T;
    } else if (iv->iter_type == tir::kDataPar) {
      bool in_A = A_vars.count(var), in_B = B_vars.count(var), in_C = C_vars.count(var);
      if (in_A && in_B && in_C) {
        kind = MatmulIterKind::kIter_S;
      } else if (in_A && in_C && !in_B) {
        kind = MatmulIterKind::kIter_I;
      } else if (in_B && in_C && !in_A) {
        kind = MatmulIterKind::kIter_J;
      } else {
        return false;
      }
    } else if (iv->iter_type == tir::kCommReduce) {
      bool in_A = A_vars.count(var), in_B = B_vars.count(var), in_C = C_vars.count(var);
      if (in_A && in_B && !in_C) {
        kind = MatmulIterKind::kIter_K;
      } else {
        return false;
      }
    } else {
      return false;
    }
    traits[var] = MatmulIterTrait{kind, iv->dom->extent};
  }

  // A GEMM kernel requires I, J, and K axes
  bool has_I = false, has_J = false, has_K = false;
  for (auto& [v, t] : traits) {
    if (t.kind == MatmulIterKind::kIter_I) has_I = true;
    if (t.kind == MatmulIterKind::kIter_J) has_J = true;
    if (t.kind == MatmulIterKind::kIter_K) has_K = true;
  }
  if (!has_I || !has_J || !has_K) return false;

  // Build per-buffer trait lists in access order.
  auto build_traits_ordered = [&](const std::vector<const tir::VarNode*>& vars_ord,
                                  std::vector<MatmulIterTrait>* out) {
    for (const tir::VarNode* v : vars_ord) {
      out->push_back(traits.at(v));
    }
  };

  build_traits_ordered(A_vars_ord, A_traits_out);
  build_traits_ordered(B_vars_ord, B_traits_out);
  build_traits_ordered(C_vars_ord, C_traits_out);
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
static bool GetMatmulIndexMap(const tir::SBlock& block, tir::IndexMap* matmul_map_out,
                              tir::IndexMap* a_map_out, tir::IndexMap* b_map_out,
                              tir::IndexMap* c_map_out) {
  std::vector<MatmulIterTrait> A_traits, B_traits, C_traits, block_traits;
  if (!DetectMatmulIterTraits(block, &A_traits, &B_traits, &C_traits, &block_traits)) return false;

  *a_map_out = MakeMatmulIterFusionIndexMap(
      A_traits, {MatmulIterKind::kIter_S, MatmulIterKind::kIter_I, MatmulIterKind::kIter_K});
  *b_map_out = MakeMatmulIterFusionIndexMap(
      B_traits, {MatmulIterKind::kIter_S, MatmulIterKind::kIter_J, MatmulIterKind::kIter_K});
  *c_map_out = MakeMatmulIterFusionIndexMap(
      C_traits, {MatmulIterKind::kIter_S, MatmulIterKind::kIter_I, MatmulIterKind::kIter_J});
  *matmul_map_out = MakeMatmulIterFusionIndexMap(
      block_traits, {MatmulIterKind::kIter_S, MatmulIterKind::kIter_I, MatmulIterKind::kIter_J,
                     MatmulIterKind::kIter_K});
  return true;
}

/*!
 * \brief Determine whether an index map represents a transposed buffer access.
 *
 * The index map is expected to have 2 or 3 initial indices and 3 final
 * indices [S, dim0, dim1]. The last two initial indices are compared against
 * the last two final indices to detect swapping.
 *
 * \return false  if [.., i, j] maps to [.., i, j]  (not transposed)
 * \return true   if [.., i, j] maps to [.., j, i]  (transposed)
 * \return nullopt if the pattern is not recognised
 */
static std::optional<bool> IsTransposed(const tir::IndexMap& index_map) {
  const ffi::Array<tir::Var>& initial = index_map->initial_indices;
  const ffi::Array<PrimExpr>& final_idx = index_map->final_indices;

  if ((initial.size() != 2 && initial.size() != 3) || final_idx.size() != 3) {
    return std::nullopt;
  }

  // The last two initial indices
  tir::Var ax1 = initial[initial.size() - 2];
  tir::Var ax2 = initial[initial.size() - 1];

  // The last two final indices (zx1, zx2)
  const PrimExpr& zx1 = final_idx[1];
  const PrimExpr& zx2 = final_idx[2];

  // Compare the last two final indices against the last two initial indices.
  if (const auto* v1 = zx1.as<tir::VarNode>()) {
    if (const auto* v2 = zx2.as<tir::VarNode>()) {
      if (v1 == ax1.get() && v2 == ax2.get()) return false;
      if (v1 == ax2.get() && v2 == ax1.get()) return true;
    }
  }
  return std::nullopt;
}

/*!
 * \brief Return the (tile_m, tile_n, tile_k) wmma tile sizes for the block.
 *
 * Queries the registered global function
 * "tir.tensor_intrin.adreno.GetWmmaTileSizes" (defined in
 * tensor_intrin_adreno.cc) to look up the hardware-supported tile dimensions
 * for the block's input/output dtype combination.
 *
 * \param block       The TIR block to inspect.
 * \param tile_m_out  Output tile M.
 * \param tile_n_out  Output tile N.
 * \param tile_k_out  Output tile K.
 * \return True on success, false if the dtype combination is unsupported.
 */
static bool GetMatmulSupportedProfile(const tir::SBlock& block, int* tile_m_out, int* tile_n_out,
                                      int* tile_k_out) {
  auto [in_dtypes, out_dtypes] = GetInOutDtypes(block);
  if (in_dtypes.size() < 2 || out_dtypes.size() < 1) return false;
  if (in_dtypes[0] != in_dtypes[1]) return false;

  const std::string& in_dtype = in_dtypes[0];
  const std::string& out_dtype = out_dtypes[0];

  auto get_tiles_fn = tvm::ffi::Function::GetGlobal("tir.tensor_intrin.adreno.GetWmmaTileSizes");
  if (!get_tiles_fn.has_value()) return false;

  auto tiles_any = (*get_tiles_fn)(in_dtype, out_dtype);
  ffi::Array<Integer> tiles;
  try {
    tiles = tiles_any.cast<ffi::Array<Integer>>();
  } catch (...) {
    return false;
  }
  if (tiles.size() != 3) return false;

  *tile_m_out = static_cast<int>(tiles[0]->value);
  *tile_n_out = static_cast<int>(tiles[1]->value);
  *tile_k_out = static_cast<int>(tiles[2]->value);
  return true;
}

/*!
 * \brief Return the unique reduction block from a list of blocks.
 *
 * All blocks must be either purely spatial (DataPar only) or reduction
 * (DataPar + CommReduce). Returns an empty array if the pattern does not
 * match or if there is not exactly one reduction block.
 *
 * \param sch    The schedule that owns the blocks.
 * \param blocks The candidate block list (typically all children of root).
 * \return A single-element array containing the reduction block, or empty.
 */
static ffi::Array<s_tir::SBlockRV> GetAdrenoReductionBlocks(
    const s_tir::Schedule& sch, const ffi::Array<s_tir::SBlockRV>& blocks) {
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

  // All blocks must be either reduction or spatial
  for (const s_tir::SBlockRV& blk : blocks) {
    if (!is_reduction(blk) && !is_spatial(blk)) return {};
  }

  ffi::Array<s_tir::SBlockRV> reduction_blocks;
  for (const s_tir::SBlockRV& blk : blocks) {
    if (is_reduction(blk)) reduction_blocks.push_back(blk);
  }
  if (reduction_blocks.size() != 1) return {};
  return reduction_blocks;
}

/*!
 * \brief Return true when the block matches the GEMM pattern.
 *
 * Checks that the block has exactly 2 read buffers and 1 write buffer,
 * all accessed element-wise (extents == 1), and that every iter-var can
 * be classified as one of:
 *   S (appears in A, B, C), K (in A and B, not C),
 *   I (in A and C, not B),  J (in B and C, not A).
 */
static bool IsGemm(const tir::SBlock& block) {
  if (block->reads.size() != 2 || block->writes.size() != 1) return false;

  auto get_ewise_vars = [&](const tir::BufferRegion& buf_region,
                            std::unordered_set<const tir::VarNode*>* out) -> bool {
    for (const Range& r : buf_region->region) {
      if (!tir::is_one(r->extent)) return false;
      tir::PostOrderVisit(r->min, [&](const ObjectRef& node) {
        if (const auto* v = node.as<tir::VarNode>()) out->insert(v);
      });
    }
    return true;
  };

  std::unordered_set<const tir::VarNode*> A_vars, B_vars, C_vars;
  if (!get_ewise_vars(block->reads[0], &A_vars)) return false;
  if (!get_ewise_vars(block->reads[1], &B_vars)) return false;
  if (!get_ewise_vars(block->writes[0], &C_vars)) return false;

  bool has_K = false, has_I = false, has_J = false;
  for (const tir::IterVar& iv : block->iter_vars) {
    const tir::VarNode* var = iv->var.get();
    bool in_A = A_vars.count(var), in_B = B_vars.count(var), in_C = C_vars.count(var);
    if (in_A && in_B && in_C) {
      // S (batch) axis
    } else if (in_A && in_B && !in_C) {
      has_K = true;
    } else if (in_A && !in_B && in_C) {
      has_I = true;
    } else if (!in_A && in_B && in_C) {
      has_J = true;
    } else {
      return false;
    }
  }
  return has_K && has_I && has_J;
}

/*!
 * \brief Return the loop index offset used when tensorizing a wmma block.
 *
 * For global/shared scopes the tensorize anchor is the second-to-last loop
 * (index -2); for local scope it is the last loop (index -1).
 */
static int IntrinLpIdx(const std::string& scope) {
  return (scope == "global" || scope == "shared") ? -2 : -1;
}

/*!
 * \brief Compute the tiling configuration for the given block and target.
 *
 * Determines wmma tile sizes from the block's dtype, detects B transposition,
 * and selects thread counts based on the output shape and available target
 * extensions (QCOM cooperative matrix conversion).
 *
 * \param block_stmt  The main (reduction) block statement.
 * \param target      The compilation target.
 * \param config_out  Output configuration.
 * \return True on success, false if the block is not a supported GEMM.
 */
static bool GetDequantMatmulConfig(const tir::SBlock& block_stmt, const tvm::Target& target,
                                   DequantMatmulTensorizationConfig* config_out) {
  // get_matmul_supported_profile
  int tile_m, tile_n, tile_k;
  if (!GetMatmulSupportedProfile(block_stmt, &tile_m, &tile_n, &tile_k)) return false;

  // dtypes
  auto [in_dtypes, out_dtypes] = GetInOutDtypes(block_stmt);
  if (in_dtypes.size() < 2 || out_dtypes.size() < 1) return false;
  const std::string& in_dtype = in_dtypes[0];
  const std::string& out_dtype = out_dtypes[0];

  // Detect B transposition from the index maps.
  tir::IndexMap matmul_index_map, a_index_map, b_index_map, c_index_map;
  if (!GetMatmulIndexMap(block_stmt, &matmul_index_map, &a_index_map, &b_index_map, &c_index_map)) {
    return false;
  }
  auto a_trans_opt = IsTransposed(a_index_map);
  auto b_trans_opt = IsTransposed(b_index_map);
  // If b_trans is not detectable, reject the block.
  if (!a_trans_opt.has_value() || !b_trans_opt.has_value()) return false;
  bool b_trans = b_trans_opt.value();

  // target capability flags
  bool has_qcom_extn = false;
  {
    auto it = target->attrs.find("supports_qcom_cooperative_matrix_conversion");
    if (it != target->attrs.end()) {
      has_qcom_extn = (*it).second.cast<bool>();
    }
  }

  // tile_m_factor and intermediate_storage_scope
  // When the QCOM cooperative matrix conversion extension is available,
  // we can use local memory for intermediate storage and process 2 M-tiles
  // per thread. Otherwise we fall back to shared memory with 1 M-tile.
  int tile_m_factor;
  std::string intermediate_storage_scope;
  if (has_qcom_extn) {
    tile_m_factor = 2;
    intermediate_storage_scope = "local";
  } else {
    tile_m_factor = 1;
    intermediate_storage_scope = "shared";
  }

  // Compute thread_size_z from the N dimension of the output buffer.
  // tile_n_factor = N / tile_n; thread_size_z = largest factor in {1, 2}
  // that divides tile_n_factor.
  int64_t n_dim = -1;
  {
    const ffi::Array<PrimExpr>& shape = block_stmt->writes[0]->buffer->shape;
    if (!shape.empty()) {
      if (const auto* imm = shape[shape.size() - 1].as<IntImmNode>()) {
        n_dim = imm->value;
      }
    }
  }

  int thread_size_x = tile_m;
  int thread_size_y = tile_m_factor;
  int thread_size_z = 1;  // default: get_max_factor returns 1 when nothing divides

  if (n_dim > 0) {
    int64_t tile_n_factor = n_dim / tile_n;
    // get_max_factor(tile_n_factor, [1, 2]) – sorted descending: [2, 1]
    for (int factor : {2, 1}) {
      if (tile_n_factor % factor == 0) {
        thread_size_z = factor;
        break;
      }
    }
  }

  // For int32 output with shared memory, shared memory is not reused across
  // N-tiles, so restrict thread_size_z to 1.
  if (intermediate_storage_scope == "shared" && out_dtype == "int32") {
    thread_size_z = 1;
  }

  // populate config
  config_out->thread_size_x = thread_size_x;
  config_out->thread_size_y = thread_size_y;
  config_out->thread_size_z = thread_size_z;

  // splits_m = (None, thread_size_y, 1, tile_m)
  config_out->splits_m_1 = thread_size_y;
  config_out->splits_m_2 = 1;
  config_out->splits_m_3 = tile_m;

  // splits_n = (None, thread_size_z, 1, tile_n)
  config_out->splits_n_1 = thread_size_z;
  config_out->splits_n_2 = 1;
  config_out->splits_n_3 = tile_n;

  // splits_k = (None, 2, 1, tile_k)
  config_out->splits_k_1 = 2;
  config_out->splits_k_2 = 1;
  config_out->splits_k_3 = tile_k;

  config_out->intrin_tile_m = tile_m;
  config_out->intrin_tile_n = tile_n;
  config_out->intrin_tile_k = tile_k;
  config_out->vector_size = 4;
  config_out->unroll = true;
  config_out->use_textures = false;
  config_out->b_trans = b_trans;
  config_out->load_scope_a = "global";
  config_out->load_scope_b = intermediate_storage_scope;
  config_out->store_scope_c = intermediate_storage_scope;
  config_out->in_dtype = in_dtype;
  config_out->out_dtype = out_dtype;
  config_out->use_storage_align = true;

  return true;
}

/*!
 * \brief Return true when the function contains blocks named
 *        "compute", "dequantize", and "matmul".
 */
static bool CheckDequantApplicability(const tir::PrimFunc& func) {
  // Required block names: "compute", "dequantize", "matmul"
  static const std::vector<std::string> kRequiredBlocks = {"compute", "dequantize", "matmul"};

  tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> func_map;
  func_map.Set(GlobalVar("main"), func);
  auto sch = s_tir::Schedule::Traced(IRModule(func_map), /*seed=*/-1, /*debug_mask=*/0,
                                     s_tir::ScheduleErrorRenderLevel::kDetail,
                                     /*enable_check=*/false);
  s_tir::SBlockRV root = sch->GetSBlock("root");
  ffi::Array<s_tir::SBlockRV> child_blocks = sch->GetChildBlocks(root);

  // Collect child block names
  std::unordered_set<std::string> child_names;
  for (const s_tir::SBlockRV& blk : child_blocks) {
    child_names.insert(std::string(sch->Get(blk)->name_hint));
  }

  // All required names must be present
  for (const std::string& name : kRequiredBlocks) {
    if (!child_names.count(name)) return false;
  }
  return true;
}

/*!
 * \brief Return true when the rule can be applied to (func, target).
 *
 * Requires a Vulkan target with KHR cooperative matrix support and a
 * function that contains blocks named "compute", "dequantize", and "matmul".
 */
static bool IsDequantMatmulSupported(const tir::PrimFunc& func, const tvm::Target& target) {
  // Condition 1: target kind must be "vulkan"
  if (target->kind->name != "vulkan") return false;

  // Condition 2: target must have supports_khr_cooperative_matrix = true
  {
    auto it = target->attrs.find("supports_khr_cooperative_matrix");
    if (it == target->attrs.end()) return false;
    if (!(*it).second.cast<bool>()) return false;
  }

  // Condition 3: function must contain "compute", "dequantize", "matmul" blocks
  return CheckDequantApplicability(func);
}

/*!
 * \brief DlightRule for Adreno dequant cooperative-matrix matmul tensorization.
 *
 * Applicability conditions (IsDequantMatmulSupported):
 *   - Target kind is "vulkan".
 *   - Target has "supports_khr_cooperative_matrix" = true.
 *   - The function contains blocks named "compute", "dequantize", "matmul".
 *
 * Schedule outline (Apply):
 *
 *  1.  Guard: Adreno target, IsSupported check.
 *  2.  Build schedule; find root and child blocks.
 *  3.  Find the single reduction block; verify it is a GEMM.
 *  4.  Compute GetConfig to obtain tiling parameters.
 *  5.  Verify index maps exist.
 *  6.  Verify loop count == 4 (bz, mb, ms, n, k after transform).
 *  7.  Verify mb and n are static IntImm extents.
 *  8.  Retrieve compute_blk and dequant_blk by name.
 *  9.  cache_read(compute_blk, 0, "local")  → quant_block.
 * 10.  Collect epilogue block chain (reverse_compute_inline all but last).
 * 11.  transform_block_layout(main_block, matmul_index_map).
 * 12.  pad_einsum(main_block, [1, ty*tile_m, tz*tile_n, tile_k]).
 * 13.  compute_inline(compute_blk).
 * 14.  Detect pada_block (if ms is not a multiple of intrin_tile_m):
 *        - storage_align on pada_block.
 *        - padc_block = consumer of main_block.
 *      else:
 *        - padc_block = reindex(main_block, ("write", 0)).
 * 15.  cache_read(main_block, 0, load_scope_a)  → loada_block.
 * 16.  loadb_block = dequant_blk.
 * 17.  If b_trans: transform_block_layout and transform_layout for
 *        quant_block and dequant_blk.
 * 18.  Split loops:
 *        bz, m, n, k = get_loops(main_block)
 *        m → [mb, ty, vx, tile_m]   (splits_m)
 *        n → [nb, tz, vy, tile_n]   (splits_n)
 *        k → [ko, ks, kq, tile_k]   (splits_k)
 * 19.  Reorder: [bz, mb, nb, vx, vy, ty, tz, ko, ks, kq, tile_m, tile_n, tile_k].
 * 20.  Bind blockIdx / vthread / threadIdx.
 * 21.  cache_read(main_block, 0, "wmma.matrix_a")  → A_wmma.
 * 22.  compute_at / reverse_compute_at the cache blocks.
 * 23.  C_store = cache_read(padc_block, 0, store_scope_c).
 * 24.  C_init  = decompose_reduction(main_block, ko).
 * 25.  set_scope(C_init, 0, "wmma.accumulator").
 * 26.  reverse_compute_inline(epilogue_block) if present.
 * 27.  Bind threadIdx.x for quant_block and padc_block; unroll quant_block.
 * 28.  set_scope(dequant_blk, 0, load_scope_b).
 * 29.  B_wmma = cache_write(dequant_blk, 0, load_scope_b); set_scope wmma.matrix_b.
 * 30.  Schedule loada_block (if load_scope_a in local/shared).
 * 31.  Schedule loadb_block (dequant_blk) based on load_scope_b.
 * 32.  Bind threadIdx.x for WMMA blocks (local scope only).
 * 33.  Tensorize using the adreno wmma intrinsic group.
 * 34.  Annotate unroll on tz.
 * 35.  Schedule pada_block (if present).
 * 36.  Schedule padc_block (split + unroll last loop).
 */

class ApplyAdrenoDequantMatmulTensorizationRuleNode : public DlightRuleNode {
 public:
  ffi::Optional<s_tir::Schedule> Apply(const tir::PrimFunc& func, const tvm::Target& target) final {
    // Step 1: Guard
    bool is_adreno = false;
    for (const auto& key : target->keys) {
      if (key == "adreno") {
        is_adreno = true;
        break;
      }
    }
    if (!is_adreno) return std::nullopt;

    if (!IsDequantMatmulSupported(func, target)) return std::nullopt;

    // Step 2: Build schedule
    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> func_map;
    func_map.Set(GlobalVar("main"), func);
    auto sch = s_tir::Schedule::Traced(IRModule(func_map), /*seed=*/-1, /*debug_mask=*/0,
                                       s_tir::ScheduleErrorRenderLevel::kDetail,
                                       /*enable_check=*/true);

    s_tir::SBlockRV root = sch->GetSBlock("root");
    ffi::Array<s_tir::SBlockRV> blocks = sch->GetChildBlocks(root);

    // Step 3: Find the single reduction block
    ffi::Array<s_tir::SBlockRV> reduction_blocks = GetAdrenoReductionBlocks(sch, blocks);
    if (reduction_blocks.empty()) return std::nullopt;

    s_tir::SBlockRV main_block = reduction_blocks[0];
    tir::SBlock main_block_stmt = sch->Get(main_block);

    if (!IsGemm(main_block_stmt)) return std::nullopt;

    // Step 4: Compute config
    DequantMatmulTensorizationConfig config;
    if (!GetDequantMatmulConfig(main_block_stmt, target, &config)) return std::nullopt;

    // Step 5: Verify index maps
    tir::IndexMap matmul_index_map, a_index_map, b_index_map, c_index_map;
    if (!GetMatmulIndexMap(sch->Get(main_block), &matmul_index_map, &a_index_map, &b_index_map,
                           &c_index_map)) {
      return std::nullopt;
    }

    // Step 6: Verify loop count == 4
    {
      ffi::Array<s_tir::LoopRV> loops = sch->GetLoops(main_block);
      if (loops.size() != 4) return std::nullopt;
    }

    // Step 7: Verify mb and n are static IntImm extents
    {
      ffi::Array<s_tir::LoopRV> loops = sch->GetLoops(main_block);
      PrimExpr mb_ext = sch->Get(loops[0])->extent;
      PrimExpr n_ext = sch->Get(loops[2])->extent;
      if (!mb_ext.as<IntImmNode>() || !n_ext.as<IntImmNode>()) return std::nullopt;
    }

    // Step 8: Retrieve compute_blk and dequant_blk by name
    s_tir::SBlockRV compute_blk = sch->GetSBlock("compute");
    s_tir::SBlockRV dequant_blk = sch->GetSBlock("dequantize");

    // Step 9: cache_read(compute_blk, 0, "local") → quant_block
    s_tir::SBlockRV quant_block = sch->CacheRead(compute_blk, 0, "local");

    // Step 10: Collect epilogue block chain
    // Walk the consumer chain from main_block, inlining all intermediate
    // consumers until only the final epilogue block remains.
    ffi::Optional<s_tir::SBlockRV> epilogue_block_opt;
    {
      ffi::Array<s_tir::SBlockRV> consumers = sch->GetConsumers(main_block);
      if (!consumers.empty()) {
        s_tir::SBlockRV epilogue = consumers[0];
        while (!sch->GetConsumers(epilogue).empty()) {
          sch->ComputeInline(epilogue);
          epilogue = sch->GetConsumers(main_block)[0];
        }
        epilogue_block_opt = epilogue;
      }
    }

    // Step 11: transform_block_layout
    sch->TransformBlockLayout(main_block, matmul_index_map);

    // Step 12: pad_einsum
    // Pad to [1, ty*tile_m, tz*tile_n, tile_k] so that tensorization is valid.
    sch->PadEinsum(main_block,
                   {Integer(1), Integer(config.splits_m_1 * config.splits_m_3),
                    Integer(config.splits_n_1 * config.splits_n_3), Integer(config.intrin_tile_k)});

    // Step 13: compute_inline(compute_blk)
    sch->ComputeInline(compute_blk);

    // Step 14: Detect pada_block / padc_block
    // After TransformBlockLayout + PadEinsum the loops are [bz, m, n, k].
    // loops[1] is the M-spatial dimension. If it is not a multiple of
    // intrin_tile_m, PadEinsum inserts a pad block (pada_block) as a producer
    // of main_block, and the consumer of main_block becomes padc_block.
    // Otherwise we reindex the write buffer directly to get padc_block.
    ffi::Optional<s_tir::SBlockRV> pada_block_opt;
    s_tir::SBlockRV padc_block;
    {
      ffi::Array<s_tir::LoopRV> loops = sch->GetLoops(main_block);
      // loops[1] corresponds to the `ms` (M-spatial) dimension
      PrimExpr ms_ext = sch->Get(loops[1])->extent;
      bool ms_is_multiple = false;
      if (const auto* imm = ms_ext.as<IntImmNode>()) {
        ms_is_multiple = (imm->value % config.intrin_tile_m == 0);
      }

      if (!ms_is_multiple) {
        // pada_block = sch.get_producers(main_block)[0]
        ffi::Array<s_tir::SBlockRV> producers = sch->GetProducers(main_block);
        if (producers.empty()) return std::nullopt;
        s_tir::SBlockRV pada = producers[0];
        pada_block_opt = pada;

        // align_fc = sch.get(sch.get_loops(pada_block)[-1]).extent.value
        ffi::Array<s_tir::LoopRV> pada_loops = sch->GetLoops(pada);
        if (pada_loops.empty()) return std::nullopt;
        PrimExpr align_fc_expr = sch->Get(pada_loops[pada_loops.size() - 1])->extent;
        int align_fc = 1;
        if (const auto* imm = align_fc_expr.as<IntImmNode>()) {
          align_fc = static_cast<int>(imm->value);
        }

        // sch.storage_align(pada_block, buffer_index=0, axis=-2, factor=align_fc, offset=64)
        sch->StorageAlign(pada, /*buffer_index=*/0, /*axis=*/-2, /*factor=*/align_fc,
                          /*offset=*/64);

        // padc_block = sch.get_consumers(main_block)[0]
        ffi::Array<s_tir::SBlockRV> consumers = sch->GetConsumers(main_block);
        if (consumers.empty()) return std::nullopt;
        padc_block = consumers[0];
      } else {
        // padc_block = sch.reindex(main_block, ("write", 0))
        padc_block = sch->ReIndex(main_block, 0, s_tir::BufferIndexType::kWrite);
      }
    }

    // Step 15: cache_read(main_block, 0, load_scope_a) → loada_block
    s_tir::SBlockRV loada_block = sch->CacheRead(main_block, 0, config.load_scope_a);

    // Step 16: loadb_block = dequant_blk
    s_tir::SBlockRV loadb_block = dequant_blk;

    // Step 17: If b_trans, swap K and J axes for quant_block and dequant_blk
    // Applying a (k, j) → (j, k) layout transform ensures the wmma load
    // intrinsic receives the matrix in the expected transposed layout.
    if (config.b_trans) {
      // Build the (k, j) → (j, k) IndexMap
      tir::Var v_k("k", DataType::Int(32)), v_j("j", DataType::Int(32));
      ffi::Array<tir::Var> swap_inputs = {v_k, v_j};
      ffi::Array<PrimExpr> swap_outputs = {PrimExpr(v_j), PrimExpr(v_k)};
      tir::IndexMap swap_map(swap_inputs, swap_outputs, /*inverse_index_map=*/std::nullopt);

      sch->TransformBlockLayout(quant_block, swap_map);
      sch->TransformBlockLayout(dequant_blk, swap_map);
      sch->TransformLayout(quant_block, 0, s_tir::BufferIndexType::kWrite, swap_map,
                           /*pad_value=*/std::nullopt, /*assume_injective_transform=*/false);
      sch->TransformLayout(dequant_blk, 0, s_tir::BufferIndexType::kWrite, swap_map,
                           /*pad_value=*/std::nullopt, /*assume_injective_transform=*/false);
    }

    // Step 18: Split loops
    ffi::Array<s_tir::LoopRV> main_loops = sch->GetLoops(main_block);
    if (main_loops.size() < 4) return std::nullopt;
    s_tir::LoopRV bz = main_loops[0];
    s_tir::LoopRV m = main_loops[1];
    s_tir::LoopRV n = main_loops[2];
    s_tir::LoopRV k = main_loops[3];

    // m → [mb, ty, vx, tile_m]
    auto m_splits = sch->Split(m,
                               {ffi::Optional<s_tir::ExprRV>(),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_m_1)),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_m_2)),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_m_3))},
                               /*preserve_unit_iters=*/true);
    s_tir::LoopRV mb = m_splits[0], ty = m_splits[1], vx = m_splits[2], tile_m_lp = m_splits[3];

    // n → [nb, tz, vy, tile_n]
    auto n_splits = sch->Split(n,
                               {ffi::Optional<s_tir::ExprRV>(),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_n_1)),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_n_2)),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_n_3))},
                               /*preserve_unit_iters=*/true);
    s_tir::LoopRV nb = n_splits[0], tz = n_splits[1], vy = n_splits[2], tile_n_lp = n_splits[3];

    // k → [ko, ks, kq, tile_k]
    auto k_splits = sch->Split(k,
                               {ffi::Optional<s_tir::ExprRV>(),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_k_1)),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_k_2)),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_k_3))},
                               /*preserve_unit_iters=*/true);
    s_tir::LoopRV ko = k_splits[0], ks = k_splits[1], kq = k_splits[2], tile_k_lp = k_splits[3];

    // Step 19: Reorder
    sch->Reorder({bz, mb, nb, vx, vy, ty, tz, ko, ks, kq, tile_m_lp, tile_n_lp, tile_k_lp});

    // Step 20: Bind blockIdx / vthread / threadIdx
    sch->Bind(mb, "blockIdx.x");
    sch->Bind(nb, "blockIdx.y");
    sch->Bind(bz, "blockIdx.z");
    sch->Bind(vx, "vthread.x");
    sch->Bind(vy, "vthread.y");
    sch->Bind(ty, "threadIdx.y");
    sch->Bind(tz, "threadIdx.z");

    // Step 21: cache_read(main_block, 0, "wmma.matrix_a") → A_wmma
    s_tir::SBlockRV A_wmma = sch->CacheRead(main_block, 0, "wmma.matrix_a");

    // Step 22: compute_at / reverse_compute_at
    sch->ComputeAt(A_wmma, kq, /*preserve_unit_loops=*/true);
    if (config.load_scope_a == "local" || config.load_scope_a == "shared") {
      sch->ComputeAt(loada_block, kq, /*preserve_unit_loops=*/true);
    } else {
      sch->ComputeInline(loada_block);
    }
    sch->ComputeAt(loadb_block, kq, /*preserve_unit_loops=*/true);
    sch->ComputeAt(quant_block, ks, /*preserve_unit_loops=*/true);
    sch->ReverseComputeAt(padc_block, tz, /*preserve_unit_loops=*/true);

    // Step 23: C_store = cache_read(padc_block, 0, store_scope_c)
    s_tir::SBlockRV C_store = sch->CacheRead(padc_block, 0, config.store_scope_c);

    // Step 24: C_init = decompose_reduction(main_block, ko)
    s_tir::SBlockRV C_init = sch->DecomposeReduction(main_block, ko);

    // Step 25: set_scope(C_init, 0, "wmma.accumulator")
    sch->SetScope(C_init, 0, "wmma.accumulator");

    // Step 26: reverse_compute_inline(epilogue_block) if present
    if (epilogue_block_opt.defined()) {
      sch->ReverseComputeInline(epilogue_block_opt.value());
    }

    // Step 27: Bind threadIdx.x for quant_block and padc_block; unroll quant_block
    {
      ffi::Array<s_tir::LoopRV> qb_loops = sch->GetLoops(quant_block);
      if (qb_loops.size() >= 2) {
        sch->Bind(qb_loops[qb_loops.size() - 2], "threadIdx.x");
        sch->Unroll(qb_loops[qb_loops.size() - 1]);
      }
    }
    {
      ffi::Array<s_tir::LoopRV> padc_loops = sch->GetLoops(padc_block);
      if (padc_loops.size() >= 2) {
        sch->Bind(padc_loops[padc_loops.size() - 2], "threadIdx.x");
      }
    }

    // Step 28: set_scope(dequant_blk, 0, load_scope_b)
    sch->SetScope(dequant_blk, 0, config.load_scope_b);

    // Step 29: B_wmma = cache_write(dequant_blk, 0, load_scope_b)
    s_tir::SBlockRV B_wmma = sch->CacheWrite(dequant_blk, 0, config.load_scope_b);
    sch->SetScope(B_wmma, 0, "wmma.matrix_b");

    // Step 30: Schedule loada_block (if load_scope_a in local/shared)
    if (config.load_scope_a == "local" || config.load_scope_a == "shared") {
      ffi::Array<s_tir::LoopRV> la_loops = sch->GetLoops(loada_block);
      if (la_loops.size() >= 2) {
        sch->Bind(la_loops[la_loops.size() - 2], "threadIdx.x");
        auto la_vec_splits =
            sch->Split(la_loops[la_loops.size() - 1],
                       {ffi::Optional<s_tir::ExprRV>(),
                        s_tir::ExprRV(IntImm(DataType::Int(32), config.vector_size))},
                       /*preserve_unit_iters=*/true);
        sch->Vectorize(la_vec_splits[1]);
        sch->Unroll(la_vec_splits[0]);
      }
    }

    // Step 31: Schedule loadb_block (dequant_blk) based on load_scope_b
    if (config.load_scope_b == "local") {
      ffi::Array<s_tir::LoopRV> lb_loops = sch->GetLoops(loadb_block);
      if (lb_loops.size() >= 2) {
        sch->Bind(lb_loops[lb_loops.size() - 2], "threadIdx.x");
        sch->Unroll(lb_loops[lb_loops.size() - 1]);
      }
    } else if (config.load_scope_b == "shared") {
      // Schedule loadb_block (dequant_blk)
      {
        ffi::Array<s_tir::LoopRV> lb_loops = sch->GetLoops(loadb_block);
        if (lb_loops.size() >= 2) {
          s_tir::LoopRV dtx = lb_loops[lb_loops.size() - 2];
          s_tir::LoopRV dvec = lb_loops[lb_loops.size() - 1];

          // _, tx = sch.split(dtx, [None, config.intrin_tile_n])
          auto dtx_splits =
              sch->Split(dtx,
                         {ffi::Optional<s_tir::ExprRV>(),
                          s_tir::ExprRV(IntImm(DataType::Int(32), config.intrin_tile_n))},
                         /*preserve_unit_iters=*/true);
          sch->Bind(dtx_splits[1], "threadIdx.x");

          // ux, vx = sch.split(dvec, [None, config.vector_size])
          auto dvec_splits =
              sch->Split(dvec,
                         {ffi::Optional<s_tir::ExprRV>(),
                          s_tir::ExprRV(IntImm(DataType::Int(32), config.vector_size))},
                         /*preserve_unit_iters=*/true);
          sch->Vectorize(dvec_splits[1]);
          sch->Unroll(dvec_splits[0]);
        }
      }
      // Schedule B_wmma annotation for CoopMat B block
      {
        ffi::Array<s_tir::LoopRV> bw_loops = sch->GetLoops(B_wmma);
        if (bw_loops.size() >= 2) {
          s_tir::LoopRV dtx = bw_loops[bw_loops.size() - 2];
          // dtz, dtx = sch.split(dtx, [None, config.intrin_tile_n])
          auto bw_splits =
              sch->Split(dtx,
                         {ffi::Optional<s_tir::ExprRV>(),
                          s_tir::ExprRV(IntImm(DataType::Int(32), config.intrin_tile_n))},
                         /*preserve_unit_iters=*/true);
          sch->Bind(bw_splits[0], "threadIdx.z");
        }
      }
    }

    // Step 32: Bind threadIdx.x for WMMA blocks (local scope only)
    if (config.load_scope_a == "local") {
      ffi::Array<s_tir::LoopRV> aw_loops = sch->GetLoops(A_wmma);
      if (aw_loops.size() >= 2) {
        sch->Bind(aw_loops[aw_loops.size() - 2], "threadIdx.x");
      }
    }
    if (config.load_scope_b == "local") {
      ffi::Array<s_tir::LoopRV> bw_loops = sch->GetLoops(B_wmma);
      if (bw_loops.size() >= 2) {
        sch->Bind(bw_loops[bw_loops.size() - 2], "threadIdx.x");
      }
    }
    if (config.store_scope_c == "local") {
      ffi::Array<s_tir::LoopRV> cs_loops = sch->GetLoops(C_store);
      if (cs_loops.size() >= 2) {
        sch->Bind(cs_loops[cs_loops.size() - 2], "threadIdx.x");
      }
    }

    // Step 33: Tensorize
    auto get_intrin_fn =
        tvm::ffi::Function::GetGlobal("tir.tensor_intrin.adreno.GetAdrenoWmmaIntrinGroup");
    if (!get_intrin_fn.has_value()) return std::nullopt;

    auto intrin_map_any = (*get_intrin_fn)(
        config.intrin_tile_m, config.intrin_tile_n, config.intrin_tile_k,
        ffi::Array<ffi::String>{config.load_scope_a, config.load_scope_b}, config.store_scope_c,
        /*trans_a=*/false,
        /*trans_b=*/true, config.in_dtype, config.out_dtype);
    ffi::Map<ffi::String, ffi::String> intrin_map;
    try {
      intrin_map = intrin_map_any.cast<ffi::Map<ffi::String, ffi::String>>();
    } catch (...) {
      return std::nullopt;
    }

    // Helper: tensorize a block at the loop resolved from a (possibly negative) index.
    auto tensorize_at = [&](const s_tir::SBlockRV& blk_rv, int lp_idx,
                            const std::string& intrin_key) {
      ffi::Array<s_tir::LoopRV> lps = sch->GetLoops(blk_rv);
      int resolved = (lp_idx < 0) ? static_cast<int>(lps.size()) + lp_idx : lp_idx;
      if (resolved < 0 || resolved >= static_cast<int>(lps.size())) return;
      sch->Tensorize(lps[resolved], intrin_map[intrin_key]);
    };

    // C_init: tensorize at [-2]
    {
      ffi::Array<s_tir::LoopRV> lps = sch->GetLoops(C_init);
      if (lps.size() >= 2) {
        sch->Tensorize(lps[lps.size() - 2], intrin_map["init"]);
      }
    }
    // A_wmma: tensorize at [a_idx]
    tensorize_at(A_wmma, IntrinLpIdx(config.load_scope_a), "load_a");
    // B_wmma: tensorize at [b_idx]
    tensorize_at(B_wmma, IntrinLpIdx(config.load_scope_b), "load_b");
    // main_block: tensorize at [-3]
    {
      ffi::Array<s_tir::LoopRV> lps = sch->GetLoops(main_block);
      if (lps.size() >= 3) {
        sch->Tensorize(lps[lps.size() - 3], intrin_map["compute"]);
      }
    }
    // C_store: tensorize at [c_idx]
    tensorize_at(C_store, IntrinLpIdx(config.store_scope_c), "store");

    // Step 34: Annotate unroll on tz
    if (config.unroll) {
      sch->Annotate(tz, "pragma_auto_unroll_max_step", IntImm(DataType::Int(32), 8));
      sch->Annotate(tz, "pragma_unroll_explicit", IntImm(DataType::Int(32), 1));
    }

    // Step 35: Schedule pada_block (if present)
    if (pada_block_opt.defined()) {
      s_tir::SBlockRV pada = pada_block_opt.value();
      ffi::Array<s_tir::LoopRV> pada_loops = sch->GetLoops(pada);
      if (pada_loops.size() >= 3) {
        s_tir::LoopRV by_lp = pada_loops[0];
        s_tir::LoopRV m_lp = pada_loops[1];
        s_tir::LoopRV n_lp = pada_loops[2];

        // ux, tx, vx = sch.split(n, [None, 32, config.vector_size])
        auto n_splits = sch->Split(
            n_lp,
            {ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), 32)),
             s_tir::ExprRV(IntImm(DataType::Int(32), config.vector_size))},
            /*preserve_unit_iters=*/true);
        s_tir::LoopRV ux_lp = n_splits[0], tx_lp = n_splits[1], vx_lp = n_splits[2];

        // b = sch.fuse(m, ux)
        s_tir::LoopRV b_lp = sch->Fuse({m_lp, ux_lp});

        // bx, ty = sch.split(b, [None, config.vector_size])
        auto b_splits = sch->Split(b_lp,
                                   {ffi::Optional<s_tir::ExprRV>(),
                                    s_tir::ExprRV(IntImm(DataType::Int(32), config.vector_size))},
                                   /*preserve_unit_iters=*/true);
        s_tir::LoopRV bx_lp = b_splits[0], ty_lp = b_splits[1];

        sch->Bind(bx_lp, "blockIdx.x");
        sch->Bind(by_lp, "blockIdx.y");
        sch->Bind(tx_lp, "threadIdx.x");
        sch->Bind(ty_lp, "threadIdx.y");
        sch->Vectorize(vx_lp);
        sch->SetScope(pada, 0, "global");
      }
    }

    // Step 36: Schedule padc_block
    // Split the last loop by vector_size and unroll the outer part.
    {
      ffi::Array<s_tir::LoopRV> padc_loops = sch->GetLoops(padc_block);
      if (!padc_loops.empty()) {
        auto padc_vec_splits =
            sch->Split(padc_loops[padc_loops.size() - 1],
                       {ffi::Optional<s_tir::ExprRV>(),
                        s_tir::ExprRV(IntImm(DataType::Int(32), config.vector_size))},
                       /*preserve_unit_iters=*/true);
        sch->Unroll(padc_vec_splits[0]);
        // sch->Vectorize(padc_vec_splits[1]);  // TODO: Uncomment once Codegen Bug is fixed
      }
    }

    return sch;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dlight.ApplyAdrenoDequantMatmulTensorizationRule",
                                    ApplyAdrenoDequantMatmulTensorizationRuleNode, DlightRuleNode);
};

/*! \brief Tiling and threading configuration for the wmma matmul schedule. */
struct MatmulTensorizationConfig {
  int thread_size_x;
  int thread_size_y;
  int thread_size_z;
  int splits_m_1;  // threadIdx.y factor when splitting M
  int splits_m_2;  // wmma tile size along M
  int splits_n_1;  // threadIdx.z factor when splitting N
  int splits_n_2;  // wmma tile size along N
  int splits_k_1;  // wmma tile size along K
  int intrin_tile_m;
  int intrin_tile_n;
  int intrin_tile_k;
  int vector_size;
  bool unroll;
  std::string load_scope_a;
  std::string load_scope_b;
  std::string store_scope_c;
  bool a_trans;
  bool b_trans;
  std::string in_dtype;
  std::string out_dtype;
};

/*!
 * \brief DlightRule for Adreno cooperative-matrix matmul tensorization.
 *
 * Applicability conditions (IsSupported):
 *   - Target kind is "vulkan".
 *   - Target has "supports_khr_cooperative_matrix" = true.
 *   - The function contains a block named "matmul".
 *
 * Schedule outline (Apply):
 *   1. Find the single reduction block; verify it is a GEMM.
 *   2. Compute GetConfig to obtain tiling parameters.
 *   3. Cache-read A and B to "global" (pad blocks if needed).
 *   4. TransformBlockLayout + PadEinsum.
 *   5. Split m → [mb, ty, tile_m], n → [nb, tz, tile_n], k → [ko, tile_k].
 *   6. Reorder and bind blockIdx / threadIdx.
 *   7. Insert wmma cache-read/write blocks (A_wmma, B_wmma, C_store).
 *   8. compute_at / reverse_compute_at the cache blocks.
 *   9. Decompose reduction at ko.
 *  10. Bind threadIdx.x for C_store (local scope only).
 *  11. Handle C_block (pad epilogue) if present.
 *  12. Tensorize using the adreno wmma intrinsic group.
 *  13. Optionally annotate unroll.
 *  14. schedule_default for A_block / B_block if they were not inlined.
 */
class ApplyAdrenoMatmulTensorizationRuleNode : public DlightRuleNode {
 public:
  /*!
   * \brief Return true when the function contains a block named "matmul".
   */
  static bool CheckApplicability(const tir::PrimFunc& func) {
    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> func_map;
    func_map.Set(GlobalVar("main"), func);
    auto sch = s_tir::Schedule::Traced(IRModule(func_map), /*seed=*/-1, /*debug_mask=*/0,
                                       s_tir::ScheduleErrorRenderLevel::kDetail,
                                       /*enable_check=*/false);
    s_tir::SBlockRV root = sch->GetSBlock("root");
    ffi::Array<s_tir::SBlockRV> child_blocks = sch->GetChildBlocks(root);
    for (const s_tir::SBlockRV& blk : child_blocks) {
      if (std::string(sch->Get(blk)->name_hint) == "matmul") return true;
    }
    return false;
  }

  /*!
   * \brief Return true when the rule can be applied to (func, target).
   *
   * Requires a Vulkan target with KHR cooperative matrix support and a
   * function that contains a block named "matmul".
   */
  static bool IsSupported(const tir::PrimFunc& func, const tvm::Target& target) {
    if (target->kind->name != "vulkan") return false;
    {
      auto it = target->attrs.find("supports_khr_cooperative_matrix");
      if (it == target->attrs.end()) return false;
      if (!(*it).second.cast<bool>()) return false;
    }
    return CheckApplicability(func);
  }

  /*!
   * \brief Compute the tiling configuration for the given block and target.
   *
   * Determines wmma tile sizes from the block's dtype, detects A/B
   * transposition, and selects thread counts based on the problem shape
   * and available target extensions (QCOM cooperative matrix conversion).
   *
   * \param block_stmt  The main (reduction) block statement.
   * \param block_rv    The SBlockRV for the main block.
   * \param sch         The schedule (used to query loop extents).
   * \param target      The compilation target.
   * \param config_out  Output configuration.
   * \return True on success, false if the block is not a supported GEMM.
   */
  static bool GetConfig(const tir::SBlock& block_stmt, const s_tir::SBlockRV& block_rv,
                        const s_tir::Schedule& sch, const tvm::Target& target,
                        MatmulTensorizationConfig* config_out) {
    int tile_m = 0, tile_n = 0, tile_k = 0;
    if (!GetMatmulSupportedProfile(block_stmt, &tile_m, &tile_n, &tile_k)) return false;

    auto [in_dtypes, out_dtypes] = GetInOutDtypes(block_stmt);
    if (in_dtypes.size() < 2 || out_dtypes.size() < 1) return false;
    const std::string& A_dtype = in_dtypes[0];
    const std::string& C_dtype = out_dtypes[0];

    // Get transpose information
    tir::IndexMap matmul_index_map, a_index_map, b_index_map, c_index_map;
    if (!GetMatmulIndexMap(block_stmt, &matmul_index_map, &a_index_map, &b_index_map,
                           &c_index_map)) {
      return false;
    }
    auto a_trans_opt = IsTransposed(a_index_map);
    auto b_trans_opt = IsTransposed(b_index_map);
    if (!a_trans_opt.has_value() || !b_trans_opt.has_value()) return false;

    bool a_trans = a_trans_opt.value();
    // B's index map is built with [S, J, K] as the canonical order, so a
    // non-transposed B (accessed as [S, J, K]) maps to false from IsTransposed.
    // The wmma intrinsic convention treats B as transposed when stored in
    // row-major [J, K] order, so we invert the flag here.
    bool b_trans = !b_trans_opt.value();

    bool has_qcom_extn = false;
    {
      auto it = target->attrs.find("supports_qcom_cooperative_matrix_conversion");
      if (it != target->attrs.end()) {
        has_qcom_extn = (*it).second.cast<bool>();
      }
    }

    // Compute thread_size_y and thread_size_z from the problem shape.
    // The number of M-tiles (BM) and N-tiles (BN) are rounded down to the
    // previous power of two (pM, pN) to select a stable thread count.
    // When both pM >= 2 and pN >= 2 the full 2x2 thread grid is used;
    // otherwise the thread counts are clamped to the available tile count.
    int thread_size_x = tile_m;
    int thread_size_y = 1, thread_size_z = 1;

    // For float16 output, two wmma tiles fit in the same shared memory
    // footprint, so we allow a factor-of-2 increase in thread count.
    int shared_factor = (C_dtype == "float16") ? 2 : 1;

    // Derive M and N extents directly from the block iter-vars.
    int64_t M_val = -1, N_val = -1;
    {
      std::vector<MatmulIterTrait> A_tr, B_tr, C_tr, blk_tr;
      if (DetectMatmulIterTraits(block_stmt, &A_tr, &B_tr, &C_tr, &blk_tr)) {
        int64_t I_ext = 1, J_ext = 1;
        for (const auto& t : blk_tr) {
          if (t.kind == MatmulIterKind::kIter_I) {
            if (const auto* imm = t.extent.as<IntImmNode>()) I_ext *= imm->value;
          } else if (t.kind == MatmulIterKind::kIter_J) {
            if (const auto* imm = t.extent.as<IntImmNode>()) J_ext *= imm->value;
          }
        }
        M_val = I_ext;
        N_val = J_ext;
      }
    }

    if (M_val > 0 && N_val > 0) {
      int64_t BM = (M_val + tile_m - 1) / tile_m;
      int64_t BN = (N_val + tile_n - 1) / tile_n;

      // Round down to the previous power of two.
      auto p_two = [](int64_t x) -> int64_t {
        if (x <= 0) return 1;
        return static_cast<int64_t>(1) << (63 - manual_clzll(static_cast<unsigned long long>(x)));
      };
      int64_t pM = p_two(BM), pN = p_two(BN);

      if (pM >= 2 && pN >= 2) {
        if (has_qcom_extn) {
          thread_size_y = 2;
          thread_size_z = 2;
        } else {
          thread_size_y = 1;
          thread_size_z = shared_factor;
        }
      } else {
        if (has_qcom_extn) {
          thread_size_y = static_cast<int>(std::min(pM, static_cast<int64_t>(4)));
          thread_size_z = static_cast<int>(std::min(pN, static_cast<int64_t>(4)));
        } else {
          thread_size_y = static_cast<int>(std::min(pM, static_cast<int64_t>(shared_factor)));
          thread_size_z = static_cast<int>(std::min(pN, static_cast<int64_t>(shared_factor)));
        }
      }
    }

    config_out->thread_size_x = thread_size_x;
    config_out->thread_size_y = thread_size_y;
    config_out->thread_size_z = thread_size_z;
    config_out->splits_m_1 = thread_size_y;
    config_out->splits_m_2 = tile_m;
    config_out->splits_n_1 = thread_size_z;
    config_out->splits_n_2 = tile_n;
    config_out->splits_k_1 = tile_k;
    config_out->intrin_tile_m = tile_m;
    config_out->intrin_tile_n = tile_n;
    config_out->intrin_tile_k = tile_k;
    config_out->vector_size = 4;
    config_out->unroll = false;  // TODO(sanjs): unrolling causes vk_pipeline failure
    config_out->a_trans = a_trans;
    config_out->b_trans = b_trans;
    config_out->load_scope_a = "global";
    config_out->load_scope_b = "global";
    config_out->store_scope_c = has_qcom_extn ? "local" : "shared";
    config_out->in_dtype = A_dtype;
    config_out->out_dtype = C_dtype;
    return true;
  }

  // Apply
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

    if (!IsSupported(func, target)) return std::nullopt;

    // Build schedule
    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> func_map;
    func_map.Set(GlobalVar("main"), func);
    auto sch = s_tir::Schedule::Traced(IRModule(func_map), /*seed=*/-1, /*debug_mask=*/0,
                                       s_tir::ScheduleErrorRenderLevel::kDetail,
                                       /*enable_check=*/true);

    // Find root and all child blocks
    s_tir::SBlockRV root_block = sch->GetSBlock("root");
    ffi::Array<s_tir::SBlockRV> blocks = sch->GetChildBlocks(root_block);

    // Find the single reduction block
    ffi::Array<s_tir::SBlockRV> reduction_blocks = GetAdrenoReductionBlocks(sch, blocks);
    if (reduction_blocks.empty()) return std::nullopt;

    s_tir::SBlockRV main_block = reduction_blocks[0];
    tir::SBlock main_block_stmt = sch->Get(main_block);

    // Verify GEMM pattern
    if (!IsGemm(main_block_stmt)) return std::nullopt;

    // Compute config (needs the pre-transform block stmt for shape info)
    MatmulTensorizationConfig config;
    if (!GetConfig(main_block_stmt, main_block, sch, target, &config)) return std::nullopt;

    // Verify index maps exist
    tir::IndexMap matmul_index_map, a_index_map, b_index_map, c_index_map;
    if (!GetMatmulIndexMap(sch->Get(main_block), &matmul_index_map, &a_index_map, &b_index_map,
                           &c_index_map)) {
      return std::nullopt;
    }

    // Step 1. Cache-read A and B into global memory staging buffers so that
    // PadEinsum can insert padding blocks between the cache and the main block.
    s_tir::SBlockRV A_block = sch->CacheRead(main_block, 0, "global");
    s_tir::SBlockRV B_block = sch->CacheRead(main_block, 1, "global");

    // Step 2. Normalize the block layout to [S, I, J, K] and pad to multiples
    // of the wmma tile sizes so that tensorization is always valid.
    sch->TransformBlockLayout(main_block, matmul_index_map);
    sch->PadEinsum(main_block, {Integer(1), Integer(config.thread_size_y * config.intrin_tile_m),
                                Integer(config.thread_size_z * config.intrin_tile_n),
                                Integer(config.intrin_tile_k)});

    // Step 3. Detect whether PadEinsum inserted a padding block between the
    // cache-read block and the main block. If a pad block was inserted, the
    // first consumer of the cache block is no longer the main block itself.
    auto is_padded = [&](const s_tir::SBlockRV& blk) -> bool {
      ffi::Array<s_tir::SBlockRV> consumers = sch->GetConsumers(blk);
      if (consumers.empty()) return false;
      return !sch->Get(consumers[0]).same_as(sch->Get(main_block));
    };

    bool is_Apad = is_padded(A_block);
    bool is_Bpad = is_padded(B_block);
    bool is_Cpad = !sch->GetConsumers(main_block).empty();

    // When a pad block was inserted, inline the original cache-read block into
    // the pad block and redirect A_block/B_block to the pad block.
    // When no padding was needed, inline the cache-read block entirely.
    if (is_Apad) {
      s_tir::SBlockRV new_A = sch->GetConsumers(A_block)[0];
      sch->ComputeInline(A_block);
      A_block = new_A;
    } else {
      sch->ComputeInline(A_block);
      is_Apad = false;
    }

    if (is_Bpad) {
      s_tir::SBlockRV new_B = sch->GetConsumers(B_block)[0];
      sch->ComputeInline(B_block);
      B_block = new_B;
    } else {
      sch->ComputeInline(B_block);
      is_Bpad = false;
    }

    // Apply a layout transform to the pad block's write buffer so that the
    // data is stored in the canonical [S, I, K] / [S, J, K] order expected
    // by the wmma load intrinsics. The original index maps (built before
    // TransformBlockLayout) are used here intentionally.
    if (is_Apad) {
      sch->TransformLayout(A_block, 0, s_tir::BufferIndexType::kWrite, a_index_map,
                           /*pad_value=*/std::nullopt, /*assume_injective_transform=*/false);
      config.a_trans = false;
    }
    if (is_Bpad) {
      sch->TransformLayout(B_block, 0, s_tir::BufferIndexType::kWrite, b_index_map,
                           /*pad_value=*/std::nullopt, /*assume_injective_transform=*/false);
      config.b_trans = true;
    }

    // Capture the output pad block (if any) before splitting loops.
    ffi::Optional<s_tir::SBlockRV> C_block_opt;
    if (is_Cpad) {
      C_block_opt = sch->GetConsumers(main_block)[0];
    }

    // Step 4. Split and bind loops.
    // After TransformBlockLayout the block has loops [bz, m, n, k].
    // m → [mb, ty, tile_m]  (blockIdx.x, threadIdx.y, wmma tile)
    // n → [nb, tz, tile_n]  (blockIdx.y, threadIdx.z, wmma tile)
    // k → [ko, tile_k]      (outer reduction, wmma tile)
    ffi::Array<s_tir::LoopRV> main_loops = sch->GetLoops(main_block);
    if (main_loops.size() < 4) return std::nullopt;
    s_tir::LoopRV bz = main_loops[0];
    s_tir::LoopRV m = main_loops[1];
    s_tir::LoopRV n = main_loops[2];
    s_tir::LoopRV k = main_loops[3];

    auto m_splits = sch->Split(m,
                               {ffi::Optional<s_tir::ExprRV>(),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_m_1)),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_m_2))},
                               /*preserve_unit_iters=*/true);
    s_tir::LoopRV mb = m_splits[0], ty = m_splits[1], tile_m_lp = m_splits[2];

    auto n_splits = sch->Split(n,
                               {ffi::Optional<s_tir::ExprRV>(),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_n_1)),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_n_2))},
                               /*preserve_unit_iters=*/true);
    s_tir::LoopRV nb = n_splits[0], tz = n_splits[1], tile_n_lp = n_splits[2];

    auto k_splits = sch->Split(k,
                               {ffi::Optional<s_tir::ExprRV>(),
                                s_tir::ExprRV(IntImm(DataType::Int(32), config.splits_k_1))},
                               /*preserve_unit_iters=*/true);
    s_tir::LoopRV ko = k_splits[0], tile_k_lp = k_splits[1];

    sch->Reorder({bz, mb, nb, ty, tz, ko, tile_m_lp, tile_n_lp, tile_k_lp});

    sch->Bind(mb, "blockIdx.x");
    sch->Bind(nb, "blockIdx.y");
    sch->Bind(bz, "blockIdx.z");
    sch->Bind(ty, "threadIdx.y");
    sch->Bind(tz, "threadIdx.z");

    // Step 5. Insert wmma register-file cache blocks.
    // A_wmma and B_wmma are loaded once per k-tile (compute_at ko).
    // C_store accumulates the output and is written back after all k-tiles
    // (reverse_compute_at tz).
    s_tir::SBlockRV A_wmma = sch->CacheRead(main_block, 0, "wmma.matrix_a");
    s_tir::SBlockRV B_wmma = sch->CacheRead(main_block, 1, "wmma.matrix_b");
    s_tir::SBlockRV C_store = sch->CacheWrite(main_block, 0, "wmma.accumulator");

    sch->ComputeAt(A_wmma, ko, /*preserve_unit_loops=*/true);
    sch->ComputeAt(B_wmma, ko, /*preserve_unit_loops=*/true);
    sch->ReverseComputeAt(C_store, tz, /*preserve_unit_loops=*/true);

    // Step 6. Decompose the reduction so that the wmma fill (init) block is
    // emitted before the k-loop and the wmma sync (compute) block is inside.
    s_tir::SBlockRV C_init = sch->DecomposeReduction(main_block, ko);

    // When there is no output pad block the accumulator is written directly
    // to global memory; update the scope accordingly.
    if (!is_Cpad) {
      config.store_scope_c = "global";
    }
    sch->SetScope(C_store, 0, config.store_scope_c);

    // For local-scope stores each thread owns its own fragment, so we bind
    // the second-to-last loop of C_store to threadIdx.x.
    if (config.store_scope_c == "local") {
      auto c_store_loops = sch->GetLoops(C_store);
      sch->Bind(c_store_loops[c_store_loops.size() - 2], "threadIdx.x");
    }

    // Step 7. Schedule the output pad block (epilogue) if present.
    // Anchor it under tz, split the second-to-last loop by intrin_tile_m,
    // and bind the outer part to threadIdx.x.
    if (C_block_opt.defined()) {
      s_tir::SBlockRV C_block = C_block_opt.value();
      sch->ReverseComputeAt(C_block, tz, /*preserve_unit_loops=*/true);
      auto c_blk_loops = sch->GetLoops(C_block);
      auto pad_splits = sch->Split(c_blk_loops[c_blk_loops.size() - 2],
                                   {s_tir::ExprRV(IntImm(DataType::Int(32), config.intrin_tile_m)),
                                    ffi::Optional<s_tir::ExprRV>()},
                                   /*preserve_unit_iters=*/true);
      sch->Bind(pad_splits[0], "threadIdx.x");
    }

    // Tensorize
    auto get_intrin_fn =
        tvm::ffi::Function::GetGlobal("tir.tensor_intrin.adreno.GetAdrenoWmmaIntrinGroup");
    if (!get_intrin_fn.has_value()) return std::nullopt;

    auto intrin_map_any = (*get_intrin_fn)(
        config.intrin_tile_m, config.intrin_tile_n, config.intrin_tile_k,
        ffi::Array<ffi::String>{config.load_scope_a, config.load_scope_b}, config.store_scope_c,
        config.a_trans, config.b_trans, config.in_dtype, config.out_dtype);
    ffi::Map<ffi::String, ffi::String> intrin_map;
    try {
      intrin_map = intrin_map_any.cast<ffi::Map<ffi::String, ffi::String>>();
    } catch (...) {
      return std::nullopt;
    }

    // Helper: tensorize a block at the loop resolved from a (possibly negative) index.
    auto tensorize_at = [&](const s_tir::SBlockRV& blk_rv, int lp_idx,
                            const std::string& intrin_key) {
      auto lps = sch->GetLoops(blk_rv);
      int resolved = (lp_idx < 0) ? static_cast<int>(lps.size()) + lp_idx : lp_idx;
      if (resolved < 0 || resolved >= static_cast<int>(lps.size())) return;
      sch->Tensorize(lps[resolved], intrin_map[intrin_key]);
    };

    // Step 8. Tensorize each wmma block.
    // C_init, A_wmma, B_wmma: anchor at the second-to-last loop (index -2).
    // main_block (compute): anchor at the third-to-last loop (index -3).
    // C_store: anchor depends on scope (-2 for global/shared, -1 for local).
    {
      auto lps = sch->GetLoops(C_init);
      sch->Tensorize(lps[lps.size() - 2], intrin_map["init"]);
    }
    tensorize_at(A_wmma, -2, "load_a");
    tensorize_at(B_wmma, -2, "load_b");
    {
      auto lps = sch->GetLoops(main_block);
      sch->Tensorize(lps[lps.size() - 3], intrin_map["compute"]);
    }
    tensorize_at(C_store, IntrinLpIdx(config.store_scope_c), "store");

    // Step 9. Optionally annotate the threadIdx.z loop for auto-unrolling.
    if (config.unroll) {
      sch->Annotate(tz, "pragma_auto_unroll_max_step", IntImm(DataType::Int(32), 8));
      sch->Annotate(tz, "pragma_unroll_explicit", IntImm(DataType::Int(32), 1));
    }

    // Step 10. Apply the default Adreno schedule to any pad blocks that
    // could not be inlined (i.e. when is_Apad / is_Bpad is true).
    if (is_Apad) {
      AdrenoScheduleDefault(sch, A_block);
    }
    if (is_Bpad) {
      AdrenoScheduleDefault(sch, B_block);
    }

    return sch;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dlight.ApplyAdrenoMatmulTensorizationRule",
                                    ApplyAdrenoMatmulTensorizationRuleNode, DlightRuleNode);
};

ffi::Optional<s_tir::Schedule> ApplyAdrenoMatmulTensorizationRule(tir::PrimFunc func,
                                                                  tvm::Target target) {
  return ApplyAdrenoMatmulTensorizationRuleNode().Apply(func, target);
}

ffi::Optional<s_tir::Schedule> ApplyAdrenoDequantMatmulTensorizationRule(tir::PrimFunc func,
                                                                         tvm::Target target) {
  return ApplyAdrenoDequantMatmulTensorizationRuleNode().Apply(func, target);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("dl.adreno.DequantMatmulTensorization",
                        ApplyAdrenoDequantMatmulTensorizationRule);
  refl::GlobalDef().def("dl.adreno.MatmulTensorization", ApplyAdrenoMatmulTensorizationRule);
}

}  // namespace dlight
}  // namespace tvm
