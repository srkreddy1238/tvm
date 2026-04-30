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
 * \file src/relax/frontend/nn/llm/kv_cache_attention_prefill_gpu_schedule.cc
 * \brief Schedule pass for GPU prefill attention kernels.
 *        Mirrors Python _schedule_prefill_kernel().
 */

#include "kv_cache_attention_prefill_gpu_schedule.h"

#include <tvm/ir/module.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/tir/index_map.h>
#include <tvm/ffi/optional.h>
#include <tvm/tir/index_map.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

using namespace tvm::tir;

// ---------------------------------------------------------------------------
// Internal helpers  (mirror the nested helpers in Python _schedule_prefill_kernel)
// ---------------------------------------------------------------------------

// get_vecsize: min(load_vec, lowest_set_bit(extent))
static int64_t GetVecSize(int64_t load_vec, int64_t extent) {
  int64_t lsb = extent & (-extent);
  return std::min(load_vec, lsb);
}

// getxy_vecsize(x, y, t) = min(get_vecsize(y), get_vecsize(x*y/t))
static int64_t GetXYVecSize(int64_t load_vec, int64_t x, int64_t y, int64_t t) {
  return std::min(GetVecSize(load_vec, y), GetVecSize(load_vec, x * y / t));
}

// get_tile_size(x, y, t) -> (tile_x, tile_y)
static std::pair<int64_t, int64_t> GetTileSize(int64_t x, int64_t y, int64_t t) {
  int64_t cnt = (x * y) / t;
  int64_t ty = static_cast<int64_t>(std::ceil(std::sqrt(static_cast<double>(cnt))));
  while ((cnt % ty != 0 || y % ty != 0 || x % (cnt / ty) != 0) && ty <= cnt) ty++;
  return {cnt / ty, ty};
}

// ---------------------------------------------------------------------------
// SchedulePrefillKernel
// ---------------------------------------------------------------------------

tir::PrimFunc SchedulePrefillKernel(tir::PrimFunc func, const PrefillKernelConfig& cfg,
                                    bool transform_k_load, bool merged_qk_load) {
  int64_t load_vec = cfg.LOAD_VEC;
  int64_t bdx = cfg.bdx;
  int64_t num_warps = cfg.num_warps;
  int64_t tile_x = cfg.tile_x;
  int64_t tile_y = cfg.tile_y;
  int64_t tile_z = cfg.tile_z;

  // Wrap in IRModule so Schedule can work on it
  ffi::Map<GlobalVar, BaseFunc> func_map;
  func_map.Set(GlobalVar("main"), func);
  IRModule mod(func_map);

  s_tir::Schedule sch = s_tir::Schedule::Traced(
      mod, /*seed=*/-1, /*debug_mask=*/0,
      s_tir::ScheduleErrorRenderLevel::kDetail, /*enable_check=*/false);

  // Helper: get static extent of a loop
  auto get_extent = [&](s_tir::LoopRV lp) -> int64_t {
    return sch->Get(lp)->extent.as<IntImmNode>()->value;
  };

  // apply_to_qkv_load: vectorize + tile + bind threads
  auto apply_to_qkv_load = [&](s_tir::SBlockRV block) {
    auto loops = sch->GetLoops(block);
    auto loop_x = loops[loops.size() - 2];
    auto loop_y = loops[loops.size() - 1];
    int64_t x_ext = get_extent(loop_x);
    int64_t y_ext = get_extent(loop_y);
    int64_t vec_size = GetXYVecSize(load_vec, x_ext, y_ext, bdx * num_warps);
    auto split_y = sch->Split(loop_y, {std::nullopt, Integer(vec_size)});
    auto yo = split_y[0], yv = split_y[1];
    int64_t yo_ext = y_ext / vec_size;
    auto [ts_x, ts_y] = GetTileSize(x_ext, yo_ext, bdx * num_warps);
    auto split_x = sch->Split(loop_x, {Integer(ts_x), std::nullopt});
    auto xo = split_x[0], xi = split_x[1];
    auto split_yo = sch->Split(yo, {Integer(ts_y), std::nullopt});
    auto yo2 = split_yo[0], yi = split_yo[1];
    sch->Reorder({xi, yi, xo, yo2});
    auto t = sch->Fuse({xi, yi});
    auto split_t = sch->Split(t, {Integer(num_warps), Integer(bdx)});
    sch->Bind(split_t[0], "threadIdx.y");
    sch->Bind(split_t[1], "threadIdx.x");
    sch->Vectorize(yv);
  };

  // apply_to_so_ewise: tile + vectorize + bind threads
  auto apply_to_so_ewise = [&](s_tir::SBlockRV block, std::pair<int64_t, int64_t> tile) {
    auto loops = sch->GetLoops(block);
    auto loop_x = loops[loops.size() - 2];
    auto loop_y = loops[loops.size() - 1];
    auto split_x = sch->Split(loop_x, {std::nullopt, Integer(tile.first)});
    auto xo = split_x[0], xi = split_x[1];
    auto split_y = sch->Split(loop_y, {std::nullopt, Integer(tile.second)});
    auto yo = split_y[0], yi = split_y[1];
    sch->Reorder({xo, yo, xi, yi});
    int64_t yiv_ext = GetVecSize(load_vec, tile.second);
    auto split_yi = sch->Split(yi, {std::nullopt, Integer(yiv_ext)});
    sch->Unroll(split_yi[0]);
    sch->Vectorize(split_yi[1]);
    auto t = sch->Fuse({xo, yo});
    auto split_t = sch->Split(t, {std::nullopt, Integer(bdx)});
    sch->Bind(split_t[0], "threadIdx.y");
    sch->Bind(split_t[1], "threadIdx.x");
  };

  // apply_to_gemm: tile + bind + split reduction + vectorize + decompose
  auto apply_to_gemm = [&](s_tir::SBlockRV block, std::pair<int64_t, int64_t> tile,
                            int64_t r_len, bool k_major) {
    auto loops = sch->GetLoops(block);
    auto loop_x = loops[loops.size() - 3];
    auto loop_y = loops[loops.size() - 2];
    auto loop_z = loops[loops.size() - 1];
    auto split_x = sch->Split(loop_x, {std::nullopt, Integer(tile.first)});
    auto xo = split_x[0], xi = split_x[1];
    auto split_y = sch->Split(loop_y, {std::nullopt, Integer(tile.second)});
    auto yo = split_y[0], yi = split_y[1];
    sch->Reorder({xo, yo, xi, yi});
    auto t = sch->Fuse({xo, yo});
    auto split_t = sch->Split(t, {std::nullopt, Integer(bdx)});
    auto ty = split_t[0], tx = split_t[1];
    sch->Bind(ty, "threadIdx.y");
    sch->Bind(tx, "threadIdx.x");
    auto split_z = sch->Split(loop_z, {std::nullopt, Integer(r_len)});
    auto ko = split_z[0], ki = split_z[1];
    if (k_major) {
      sch->Reorder({ko, xi, yi, ki});
    } else {
      sch->Reorder({ko, ki, xi, yi});
    }
    int64_t yiv_ext = GetVecSize(load_vec, tile.second);
    auto split_yi = sch->Split(yi, {std::nullopt, Integer(yiv_ext)});
    sch->Unroll(split_yi[0]);
    sch->Vectorize(split_yi[1]);
    sch->Unroll(xi);
    sch->DecomposeReduction(block, ty);
  };

  // apply_to_md: split last loop into (_, ty, tx)
  auto apply_to_md = [&](s_tir::SBlockRV block) {
    auto loops = sch->GetLoops(block);
    auto loop = loops[loops.size() - 1];
    auto split = sch->Split(loop, {std::nullopt, Integer(num_warps), Integer(bdx)});
    sch->Bind(split[1], "threadIdx.y");
    sch->Bind(split[2], "threadIdx.x");
  };

  // Transform K_load write layout (i,j) -> (j,i) for ragged variant
  if (transform_k_load && !merged_qk_load) {
    auto k_load_block = sch->GetSBlock("K_load");
    auto index_map = IndexMap::FromFunc(
        2, [](ffi::Array<Var> axes) -> ffi::Array<PrimExpr> {
          return {axes[1], axes[0]};
        });
    sch->TransformLayout(k_load_block, 0, s_tir::BufferIndexType::kWrite,
                         index_map, std::nullopt, /*assume_injective_transform=*/false);
  }

  // Compute tile sizes for S and O gemms
  auto tile_s = GetTileSize(tile_x, tile_z, bdx * num_warps);
  auto tile_o = GetTileSize(tile_x, tile_y, bdx * num_warps);

  // Apply all schedule primitives
  apply_to_gemm(sch->GetSBlock("S_gemm"), tile_s, 16, /*k_major=*/true);
  apply_to_gemm(sch->GetSBlock("O_gemm"), tile_o, 16, /*k_major=*/false);
  apply_to_so_ewise(sch->GetSBlock("S_store"), tile_s);
  apply_to_so_ewise(sch->GetSBlock("O_init"), tile_o);
  apply_to_so_ewise(sch->GetSBlock("O_store"), tile_o);
  apply_to_qkv_load(sch->GetSBlock("Q_load"));
  if (!merged_qk_load) {
    apply_to_qkv_load(sch->GetSBlock("K_load"));
    apply_to_qkv_load(sch->GetSBlock("V_load"));
  } else {
    apply_to_qkv_load(sch->GetSBlock("KV_load"));
  }
  apply_to_md(sch->GetSBlock("lse_store"));

  // Extract the scheduled function and mark it as scheduled
  tir::PrimFunc scheduled = Downcast<tir::PrimFunc>(sch->mod()->Lookup("main"));
  return WithAttr(scheduled, "tir.is_scheduled", ffi::Any(true));
}

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
