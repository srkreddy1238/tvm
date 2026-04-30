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
 * \file src/relax/frontend/nn/llm/kv_cache_attention_decode_gpu_helpers.cc
 * \brief Implementations of helper functions for the GPU batched-decode kernel.
 */

#include "kv_cache_attention_decode_gpu_helpers.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

using namespace tvm::tir;

// ============================================================================
// ComputeDecodeGpuConfig
// ============================================================================

DecodeGpuConfig ComputeDecodeGpuConfig(int64_t H_kv, int64_t H_qo, int64_t D,
                                       const std::string& dtype, bool sliding_window,
                                       Target target) {
  // Python hardcodes qkv_dtype_bytes = 2 regardless of actual dtype.
  // This matches the Python _attention_decode() behaviour exactly.
  int64_t qkv_dtype_bytes = 2;
  int64_t THREAD_LIMIT = 512;
  int64_t TILE_SIZE_PER_BDX = 2;

  // Adreno / Android mobile GPU: lower thread limit to avoid register spill
  std::string target_str = target->str();
  bool is_adreno = (target_str.find("adreno") != std::string::npos) ||
                   (target_str.find("android") != std::string::npos);
  if (is_adreno) {
    THREAD_LIMIT = 256;
    TILE_SIZE_PER_BDX = 1;
  }

  int64_t VEC_SIZE = std::min(std::max(8LL / qkv_dtype_bytes, D / 32), 4LL);

  // Query max threads per block from target attrs
  int64_t max_threads = 1024;
  auto attr = target->GetAttr<Integer>("max_num_threads");
  if (attr.defined()) max_threads = attr.value()->value;
  int64_t thread_limit = std::min(max_threads, THREAD_LIMIT);

  int64_t GROUP_SIZE = H_qo / H_kv;
  int64_t bdx = D / VEC_SIZE;
  int64_t bdy = GROUP_SIZE;
  while (bdx * bdy > thread_limit && bdy > 1) bdy /= 2;
  int64_t gdz = GROUP_SIZE / bdy;
  int64_t threads_per_CTA = std::max(thread_limit, bdx * bdy);
  int64_t bdz = threads_per_CTA / (bdx * bdy);
  int64_t tile_size_per_bdx = (GROUP_SIZE == 1) ? TILE_SIZE_PER_BDX : 1;

  std::string global_symbol = "batch_decode_paged_kv";
  if (sliding_window) global_symbol += "_sliding_window";

  return {VEC_SIZE, bdx, bdy, bdz, gdz, tile_size_per_bdx, GROUP_SIZE, global_symbol};
}

// ============================================================================
// BuildDecodeRopeExpr
// ============================================================================

PrimExpr BuildDecodeRopeExpr(tir::Buffer buf, ffi::Array<PrimExpr> base_indices, PrimExpr d_expr,
                              int64_t D, PrimExpr pos_expr, tir::Var rope_scale,
                              tir::Var rope_theta, tir::Var rotary_mode, const std::string& dtype,
                              const ffi::Map<ffi::String, ffi::Any>& rope_scaling) {
  bool is_f16 = (dtype == "float16");
  DataType dt = DataType(runtime::StringToDLDataType(dtype));
  int64_t half_d = D / 2;

  // Current element: buf[base_indices..., d_expr]
  ffi::Array<PrimExpr> full_idx = base_indices;
  full_idx.push_back(d_expr);
  PrimExpr elem = tir::BufferLoad(buf, full_idx);

  // Rotated partner: if d < D/2 → buf[..., d+D/2]*(-1)  else → buf[..., d-D/2]
  ffi::Array<PrimExpr> idx_plus = base_indices;
  idx_plus.push_back(d_expr + I32(half_d));
  ffi::Array<PrimExpr> idx_minus = base_indices;
  idx_minus.push_back(d_expr - I32(half_d));
  PrimExpr neg_one = tir::make_const(dt, -1.0);
  PrimExpr partner = tvm::if_then_else(d_expr < I32(half_d),
                                        tir::BufferLoad(buf, idx_plus) * neg_one,
                                        tir::BufferLoad(buf, idx_minus));

  // Rope frequency: use d_expr directly (no extra d_var binding, matches Python TVMScript)
  RopeFreqFunc rope_freq_func = SwitchRopeFreqFunc(rope_scaling);
  PrimExpr pos_f32 = CastTo(pos_expr, "float32") * rope_scale;
  // Always compute rope freq in float32 (matches Python which always passes "float32")
  auto freq = rope_freq_func(pos_f32, d_expr, D, rope_theta, "float32", rope_scaling);

  // Build rope value: cast<dtype>(cos*cast<f32>(elem) + sin*cast<f32>(partner))
  // For f16: cos_freq/sin_freq are already float32 (from tvm::cos/sin), so no extra cast needed
  PrimExpr rope_val;
  if (is_f16) {
    rope_val = CastTo(freq.cos_freq * CastTo(elem, "float32") +
                      freq.sin_freq * CastTo(partner, "float32"),
                      dtype);
  } else {
    rope_val = freq.cos_freq * elem + freq.sin_freq * partner;
  }

  // Wrap in Let bindings for intermediate freq vars (outermost first = forward order)
  // var_map is ordered: [outer_var, inner_var, ...] where outer_var is defined first
  // We wrap from innermost to outermost: iterate in reverse so outer var wraps last
  // But Python TVMScript wraps outer var LAST (outermost), so we iterate forward:
  // Let(outer, ..., Let(inner, ..., expr)) = forward iteration
  for (auto it = freq.var_map.begin(); it != freq.var_map.end(); ++it) {
    rope_val = tir::Let(it->first, it->second, rope_val);
  }

  return tvm::if_then_else(rotary_mode == I32(1), rope_val, elem);
}

// ============================================================================
// BuildDecodeKVLoadBody
// ============================================================================

Stmt BuildDecodeKVLoadBody(const DecodeGpuConfig& cfg, tir::Buffer pages_buf,
                           tir::Buffer page_values_buf, tir::Buffer length_info_buf,
                           tir::Buffer k_rope_pos_offset_buf, tir::Buffer K_smem,
                           tir::Buffer V_smem, tir::Buffer kv_chunk_len_buf, tir::Var cur_begin,
                           tir::Var batch_idx, tir::Var by_var, tir::Var tx, PrimExpr tile_start_s,
                           PrimExpr tile_start_g, tir::Var j_var, tir::Var rope_scale,
                           tir::Var rope_theta, tir::Var rotary_mode,
                           const ffi::Map<ffi::String, ffi::Any>& rope_scaling) {
  bool sliding_window = (length_info_buf->shape.size() == 2);
  int64_t page_size = cfg.VEC_SIZE > 0 ? pages_buf->shape[3].as<IntImmNode>()->value : 16;
  // Recover page_size from pages_buf shape: (max_num_pages, 2, H_kv, page_size, D)
  page_size = pages_buf->shape[3].as<IntImmNode>()->value;

  PrimExpr kv_len = tir::BufferLoad(kv_chunk_len_buf, {I32(0)});

  // row_g is a SizeVar LetStmt wrapping the entire if/else (matches Python TVMScript)
  tir::SizeVar row_g("row_g", DataType::Int(32));
  PrimExpr row_g_expr = tile_start_g + j_var;

  // seq_offset: raw pos or sliding-window adjusted
  tir::SizeVar seq_offset("seq_offset", DataType::Int(32));
  PrimExpr seq_offset_expr;
  if (sliding_window) {
    seq_offset_expr = tvm::if_then_else(
        row_g < tir::BufferLoad(length_info_buf, {I32(2), batch_idx}),
        row_g,
        row_g - tir::BufferLoad(length_info_buf, {I32(2), batch_idx}) +
            tir::BufferLoad(length_info_buf, {I32(1), batch_idx}));
  } else {
    seq_offset_expr = row_g;
  }

  tir::SizeVar page_no("page_no", DataType::Int(32));
  tir::SizeVar page_offset_var("page_offset", DataType::Int(32));

  // Single vectorized loop for both K and V (matches Python TVMScript)
  tir::Var vec("vec", DataType::Int(32));
  PrimExpr d_kv = tx * I32(cfg.VEC_SIZE) + vec;
  PrimExpr k_pos = tir::BufferLoad(k_rope_pos_offset_buf, {batch_idx}) + row_g;
  DataType pages_dt = pages_buf->dtype;
  std::string pages_dtype_str = ffi::DLDataTypeToString(pages_dt);
  PrimExpr k_rope = BuildDecodeRopeExpr(
      pages_buf, ffi::Array<PrimExpr>{page_no, I32(0), by_var, page_offset_var},
      d_kv, pages_buf->shape[4].as<IntImmNode>()->value,
      k_pos, rope_scale, rope_theta, rotary_mode, pages_dtype_str, rope_scaling);
  PrimExpr v_elem = tir::BufferLoad(pages_buf, {page_no, I32(1), by_var, page_offset_var, d_kv});

  PrimExpr zero_kv = tir::make_const(K_smem->dtype, 0.0);

  // Build the "row_g < kv_len" branch: seq_offset -> page_no -> page_offset -> single vec loop
  Stmt load_branch = tir::LetStmt(
      seq_offset, seq_offset_expr,
      tir::LetStmt(
          page_no,
          tir::BufferLoad(page_values_buf,
                          {cur_begin + floordiv(seq_offset, I32(page_size))}),
          tir::LetStmt(
              page_offset_var, floormod(seq_offset, I32(page_size)),
              tir::For(vec, I32(0), I32(cfg.VEC_SIZE), tir::ForKind::kVectorized,
                       tir::SeqStmt({
                           tir::BufferStore(K_smem, k_rope, {tile_start_s + j_var, d_kv}),
                           tir::BufferStore(V_smem, v_elem, {tile_start_s + j_var, d_kv})})))));

  // Build the "row_g >= kv_len" zero-fill branch: single vec loop for both K and V
  tir::Var vec0("vec", DataType::Int(32));
  PrimExpr d_zero = tx * I32(cfg.VEC_SIZE) + vec0;
  Stmt zero_branch = tir::For(
      vec0, I32(0), I32(cfg.VEC_SIZE), tir::ForKind::kVectorized,
      tir::SeqStmt({
          tir::BufferStore(K_smem, zero_kv, {tile_start_s + j_var, d_zero}),
          tir::BufferStore(V_smem, zero_kv, {tile_start_s + j_var, d_zero})}));

  // Wrap in row_g LetStmt
  return tir::LetStmt(row_g, row_g_expr,
                      tir::IfThenElse(row_g < kv_len, load_branch, zero_branch));
}

// ============================================================================
// BuildDecodeAllreduceBlock
// ============================================================================

Stmt BuildDecodeAllreduceBlock(const DecodeGpuConfig& cfg, tir::Buffer O_allreduce,
                               tir::Buffer md_allreduce, tir::Buffer O_local, tir::Buffer st_m,
                               tir::Buffer st_d, tir::Buffer m_prev, tir::Buffer d_prev,
                               tir::Buffer other_m, tir::Buffer other_d, tir::Buffer other_o,
                               tir::Buffer exp_mprev, tir::Buffer exp_otherm, tir::Var tz,
                               tir::Var ty, tir::Var tx) {
  auto ld0 = [](tir::Buffer b) { return tir::BufferLoad(b, {I32(0)}); };
  auto st0 = [](tir::Buffer b, PrimExpr v) { return tir::BufferStore(b, v, {I32(0)}); };
  auto ldi = [](tir::Buffer b, PrimExpr i) { return tir::BufferLoad(b, {i}); };
  auto sti = [](tir::Buffer b, PrimExpr i, PrimExpr v) { return tir::BufferStore(b, v, {i}); };

  // Store O_local → O_allreduce[tz, ty, tx*VEC_SIZE+vec]
  tir::Var vec_store("vec", DataType::Int(32));
  Stmt store_O = tir::For(vec_store, I32(0), I32(cfg.VEC_SIZE), tir::ForKind::kVectorized,
                           tir::BufferStore(O_allreduce, ldi(O_local, vec_store),
                                            {tz, ty, tx * I32(cfg.VEC_SIZE) + vec_store}));

  // Store st_m, st_d → md_allreduce[tz, ty, 0/1]
  Stmt store_md = tir::SeqStmt({
      tir::BufferStore(md_allreduce, ld0(st_m), {tz, ty, I32(0)}),
      tir::BufferStore(md_allreduce, ld0(st_d), {tz, ty, I32(1)})});

  Stmt sync = tir::Evaluate(tir::Call(DataType::Int(32), tir::builtin::tvm_storage_sync(),
                                       ffi::Array<PrimExpr>{tir::StringImm("shared")}));

  // Reset st_m, st_d, O_local
  tir::Var vec_reset("vec", DataType::Int(32));
  Stmt reset = tir::SeqStmt({
      st0(st_m, F32(-50000.0)),
      st0(st_d, F32(1.0)),
      tir::For(vec_reset, I32(0), I32(cfg.VEC_SIZE), tir::ForKind::kVectorized,
               sti(O_local, vec_reset, F32(0.0)))});

  // Merge loop: for j in range(bdz)
  tir::Var j_merge("j", DataType::Int(32));
  tir::Var vec_oo("vec", DataType::Int(32));
  tir::Var vec_om("vec", DataType::Int(32));

  Stmt load_other_o = tir::For(
      vec_oo, I32(0), I32(cfg.VEC_SIZE), tir::ForKind::kVectorized,
      sti(other_o, vec_oo,
          tir::BufferLoad(O_allreduce, {j_merge, ty, tx * I32(cfg.VEC_SIZE) + vec_oo})));

  Stmt merge_body = tir::SeqStmt({
      st0(m_prev, ld0(st_m)),
      st0(d_prev, ld0(st_d)),
      st0(other_m, tir::BufferLoad(md_allreduce, {j_merge, ty, I32(0)})),
      st0(other_d, tir::BufferLoad(md_allreduce, {j_merge, ty, I32(1)})),
      load_other_o,
      st0(st_m, tvm::max(ld0(st_m), ld0(other_m))),
      st0(st_d, ld0(d_prev) * tvm::exp2(ld0(m_prev) - ld0(st_m)) +
                ld0(other_d) * tvm::exp2(ld0(other_m) - ld0(st_m))),
      st0(exp_mprev, tvm::exp2(ld0(m_prev) - ld0(st_m))),
      st0(exp_otherm, tvm::exp2(ld0(other_m) - ld0(st_m))),
      tir::For(vec_om, I32(0), I32(cfg.VEC_SIZE), tir::ForKind::kVectorized,
               sti(O_local, vec_om,
                   ldi(O_local, vec_om) * ld0(exp_mprev) +
                   ldi(other_o, vec_om) * ld0(exp_otherm)))});

  Stmt merge_loop = tir::For(j_merge, I32(0), I32(cfg.bdz), tir::ForKind::kSerial, merge_body);

  return tir::SeqStmt({store_O, store_md, sync, reset, merge_loop});
}

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
