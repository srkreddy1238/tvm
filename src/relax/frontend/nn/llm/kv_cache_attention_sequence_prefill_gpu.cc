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
 * \file src/relax/frontend/nn/llm/kv_cache_attention_sequence_prefill_gpu.cc
 * \brief GPU TIR kernel for sequence (non-ragged, non-paged) prefill.
 *        C++ port of _attention_sequence_prefill() in kv_cache.py.
 *
 * Differences from paged prefill:
 *   - K/V are dense batched tensors: (batch, kv_len, h_kv, d).
 *   - Q is a dense batched tensor: (batch, qo_len, h_q, d).
 *   - No page table, no sliding window, no RoPE.
 *   - Grid: blockIdx.x = batch * batch_tiles, blockIdx.y = h_kv.
 *   - No tile dispatch while loop — each block handles exactly one tile.
 */

#include <tvm/s_tir/stmt.h>
#include <tvm/tir/op.h>

#include <cmath>
#include <string>

#include "../../../../tir/ir/script/script_complete.h"
#include "kv_cache.h"
#include "kv_cache_attention_prefill_gpu_helpers.h"
#include "kv_cache_attention_prefill_gpu_schedule.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

using namespace tvm::tir;

static tir::Buffer AllocLocal(const std::string& name, ffi::Array<PrimExpr> shape,
                              const std::string& dtype = "float32") {
  return tir::decl_buffer(shape, DataType(runtime::StringToDLDataType(dtype)), name, "local");
}
static tir::Buffer AllocShared(const std::string& name, ffi::Array<PrimExpr> shape,
                               const std::string& dtype = "float32") {
  return tir::decl_buffer(shape, DataType(runtime::StringToDLDataType(dtype)), name, "shared");
}
static Stmt Sync() {
  return tir::Evaluate(tir::Call(DataType::Int(32), tir::builtin::tvm_storage_sync(),
                                 ffi::Array<PrimExpr>{tir::StringImm("shared")}));
}

tir::PrimFunc AttentionSequencePrefill(int64_t h_kv, int64_t h_q, int64_t d_qk, int64_t d_v,
                                       const std::string& dtype,
                                       const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
                                       Target target, int64_t causal = 0, double sm_scale = 1.0) {
  const PrefillKernelConfig cfg = ComputePrefillKernelConfig(h_kv, h_q, d_qk, dtype, target);
  const int64_t group_size = cfg.group_size;
  const int64_t bdx = cfg.bdx, num_warps = cfg.num_warps;
  const int64_t tile_x = cfg.tile_x, tile_z = cfg.tile_z;
  const bool is_f16 = (dtype == "float16");
  DataType dt = DataType(runtime::StringToDLDataType(dtype));
  const double log2e = 1.4426950408889634;

  // ---- symbolic size vars ----
  tir::SizeVar batch_size("batch_size", DataType::Int(32));
  tir::SizeVar qo_len("qo_len", DataType::Int(32));
  tir::SizeVar kv_len("kv_len", DataType::Int(32));

  // ---- parameter handles ----
  tir::Var h_q_h("var_q", DataType::Handle());
  tir::Var h_k_h("var_k", DataType::Handle());
  tir::Var h_v_h("var_v", DataType::Handle());
  tir::Var h_output_h("var_output", DataType::Handle());
  tir::Var h_lse_h("var_lse", DataType::Handle());

  // ---- buffers ----
  tir::Buffer q_buf = tir::decl_buffer({batch_size, qo_len, I32(h_q), I32(d_qk)}, dt, "q");
  tir::Buffer k_buf = tir::decl_buffer({batch_size, kv_len, I32(h_kv), I32(d_qk)}, dt, "k");
  tir::Buffer v_buf = tir::decl_buffer({batch_size, kv_len, I32(h_kv), I32(d_v)}, dt, "v");
  tir::Buffer output_buf = tir::decl_buffer({batch_size, qo_len, I32(h_q), I32(d_v)}, dt, "output");
  // lse dtype matches the compute dtype (float16 or float32), matching Python reference
  tir::Buffer lse_buf = tir::decl_buffer({batch_size, qo_len, I32(h_q)}, dt, "lse");

  // ---- thread index vars ----
  // blockIdx.x = b_idx * batch_tiles + tile_id
  // blockIdx.y = by (head index)
  tir::Var lbx("lbx", DataType::Int(32)), lby("lby", DataType::Int(32));
  tir::Var lty("lty", DataType::Int(32)), ltx("ltx", DataType::Int(32));
  tir::Var vbx("vbx", DataType::Int(32)), by("by", DataType::Int(32));
  tir::Var ty("ty", DataType::Int(32)), tx("tx", DataType::Int(32));

  // ---- sblock-local alloc buffers ----
  tir::Buffer Q_smem = AllocShared("Q_smem", {I32(tile_x), I32(d_qk)}, dtype);
  tir::Buffer K_smem = AllocShared("K_smem", {I32(tile_z), I32(d_qk)}, dtype);
  tir::Buffer V_smem = AllocShared("V_smem", {I32(tile_z), I32(d_v)}, dtype);
  tir::Buffer S_smem = AllocShared("S_smem", {I32(tile_x), I32(tile_z)}, "float32");
  tir::Buffer S_local = AllocLocal("S_local", {I32(tile_x), I32(tile_z)}, "float32");
  tir::Buffer O_local = AllocLocal("O_local", {I32(tile_x), I32(d_v)}, "float32");
  tir::Buffer m_smem = AllocShared("m_smem", {I32(tile_x)}, "float32");
  tir::Buffer m_prev_smem = AllocShared("m_prev_smem", {I32(tile_x)}, "float32");
  tir::Buffer d_smem = AllocShared("d_smem", {I32(tile_x)}, "float32");
  int64_t md_sz = static_cast<int64_t>(std::ceil((double)tile_x / (bdx * num_warps)));
  tir::Buffer m_new_buf = AllocLocal("m_new", {I32(md_sz)}, "float32");
  tir::Buffer m_prev_buf = AllocLocal("m_prev", {I32(md_sz)}, "float32");
  tir::Buffer d_new_buf = AllocLocal("d_new", {I32(md_sz)}, "float32");

  auto ld0 = [](tir::Buffer b) { return tir::BufferLoad(b, {I32(0)}); };
  (void)ld0;

  // Annotation for auto-detecting reads/writes in sblocks
  ffi::Map<ffi::String, ffi::Any> detect_annots;
  detect_annots.Set("tir.script_parsing_detect_access", IntImm(DataType::Int(32), 3));

  // ---- derived vars ----
  // batch_tiles = ceil(qo_len * group_size / tile_x)
  // When group_size == 1 (MHA/MQA), this simplifies to ceil(qo_len / tile_x).
  // batch_tiles is a LetStmt variable in the root sblock body (matching Python reference IR).
  tir::Var batch_tiles_var("batch_tiles", DataType::Int(32));
  PrimExpr batch_tiles_expr =
      floordiv(qo_len * I32(group_size) + I32(tile_x) - I32(1), I32(tile_x));
  // Use batch_tiles_var everywhere batch_tiles is referenced
  PrimExpr batch_tiles = batch_tiles_var;
  tir::Var b_idx("b_idx", DataType::Int(32));
  tir::Var tile_id("tile_id", DataType::Int(32));
  tir::Var LH_start("LH_start", DataType::Int(32));
  // When group_size > 1, the Python reference renames the KV loop variable to "iterator_1"
  // to avoid collision with the outer "iterator" variable from _var("int32").
  tir::Var iterator(group_size > 1 ? "iterator_1" : "iterator", DataType::Int(32));
  tir::Var L_kv_start("L_kv_start", DataType::Int(32));
  tir::Var L_kv_base("L_kv_base", DataType::Int(32));

  // init states
  tir::Var i_init("i", DataType::Int(32)), row_init("row", DataType::Int(32));
  Stmt init_md = tir::For(
      i_init, I32(0), I32(md_sz), tir::ForKind::kSerial,
      tir::LetStmt(
          row_init, i_init * I32(bdx) * I32(num_warps) + ty * I32(bdx) + tx,
          tir::IfThenElse(row_init < I32(tile_x),
                          tir::SeqStmt({tir::BufferStore(m_smem, F32(-50000.0), {row_init}),
                                        tir::BufferStore(d_smem, F32(1.0), {row_init})}))));

  // O_init
  tir::Var li_oi("li", DataType::Int(32)), lj_oi("lj", DataType::Int(32));
  tir::Var vi_oi("i", DataType::Int(32)), vj_oi("j", DataType::Int(32));
  ffi::Array<tir::BufferRegion> o_init_writes = {
      tir::BufferRegion(O_local, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(vi_oi, I32(1)),
                                                        tvm::Range::FromMinExtent(vj_oi, I32(1))})};
  auto o_init_block = tir::SBlock(
      ffi::Array<IterVar>{
          IterVar(Range::FromMinExtent(I32(0), I32(tile_x)), vi_oi, tir::kDataPar, ""),
          IterVar(Range::FromMinExtent(I32(0), I32(d_v)), vj_oi, tir::kDataPar, "")},
      {}, o_init_writes, "O_init",
      tir::BufferStore(O_local, F32(0.0), ffi::Array<PrimExpr>{vi_oi, vj_oi}));
  Stmt o_init = tir::For(li_oi, I32(0), I32(tile_x), tir::ForKind::kSerial,
                         tir::For(lj_oi, I32(0), I32(d_v), tir::ForKind::kSerial,
                                  tir::SBlockRealize({PrimExpr(li_oi), PrimExpr(lj_oi)},
                                                     tir::const_true(), o_init_block)));

  // Q_load: q[b_idx, LH_start + i, cur_H_qo, j]
  tir::Var li_ql("li", DataType::Int(32)), lj_ql("lj", DataType::Int(32));
  tir::Var vi_ql("i", DataType::Int(32)), vj_ql("j", DataType::Int(32));
  tir::Var cur_L_ql("cur_L", DataType::Int(32)), cur_H_qo_ql("cur_H_qo", DataType::Int(32));
  PrimExpr zero_val = is_f16 ? PrimExpr(tvm::FloatImm(dt, 0.0)) : F32(0.0);
  Stmt q_load = tir::For(
      li_ql, I32(0), I32(tile_x), tir::ForKind::kSerial,
      tir::For(
          lj_ql, I32(0), I32(d_qk), tir::ForKind::kSerial,
          tir::SBlockRealize(
              {PrimExpr(li_ql), PrimExpr(lj_ql)}, tir::const_true(),
              tir::SBlock(
                  ffi::Array<IterVar>{
                      IterVar(Range::FromMinExtent(I32(0), I32(tile_x)), vi_ql, tir::kDataPar, ""),
                      IterVar(Range::FromMinExtent(I32(0), I32(d_qk)), vj_ql, tir::kDataPar, "")},
                  {}, {}, "Q_load",
                  tir::LetStmt(
                      cur_L_ql, floordiv(LH_start + vi_ql, I32(group_size)),
                      tir::LetStmt(
                          cur_H_qo_ql,
                          by * I32(group_size) + floormod(LH_start + vi_ql, I32(group_size)),
                          tir::IfThenElse(
                              cur_L_ql < qo_len,
                              tir::BufferStore(
                                  Q_smem,
                                  tir::BufferLoad(q_buf, ffi::Array<PrimExpr>{b_idx, cur_L_ql,
                                                                              cur_H_qo_ql, vj_ql}),
                                  ffi::Array<PrimExpr>{vi_ql, vj_ql}),
                              tir::BufferStore(Q_smem, zero_val,
                                               ffi::Array<PrimExpr>{vi_ql, vj_ql}))))))));

  // K_load: k[b_idx, L_kv_base + cur_L, by, j]
  tir::Var lz_kl("lz", DataType::Int(32)), ly_kl("ly", DataType::Int(32));
  tir::Var vi_kl("i", DataType::Int(32)), vj_kl("j", DataType::Int(32));
  tir::Var cur_L_kl("cur_L", DataType::Int(32));
  Stmt k_load = tir::For(
      lz_kl, I32(0), I32(tile_z), tir::ForKind::kSerial,
      tir::For(
          ly_kl, I32(0), I32(d_qk), tir::ForKind::kSerial,
          tir::SBlockRealize(
              {PrimExpr(lz_kl), PrimExpr(ly_kl)}, tir::const_true(),
              tir::SBlock(
                  ffi::Array<IterVar>{
                      IterVar(Range::FromMinExtent(I32(0), I32(tile_z)), vi_kl, tir::kDataPar, ""),
                      IterVar(Range::FromMinExtent(I32(0), I32(d_qk)), vj_kl, tir::kDataPar, "")},
                  {}, {}, "K_load",
                  tir::LetStmt(cur_L_kl, L_kv_start + vi_kl,
                               tir::IfThenElse(
                                   cur_L_kl < kv_len,
                                   tir::BufferStore(
                                       K_smem,
                                       tir::BufferLoad(
                                           k_buf, ffi::Array<PrimExpr>{b_idx, L_kv_base + cur_L_kl,
                                                                       by, vj_kl}),
                                       ffi::Array<PrimExpr>{vi_kl, vj_kl}),
                                   tir::BufferStore(K_smem, zero_val,
                                                    ffi::Array<PrimExpr>{vi_kl, vj_kl})))))));

  // V_load
  tir::Var lz_vl("lz", DataType::Int(32)), ly_vl("ly", DataType::Int(32));
  tir::Var vi_vl("i", DataType::Int(32)), vj_vl("j", DataType::Int(32));
  tir::Var cur_L_vl("cur_L", DataType::Int(32));
  Stmt v_load = tir::For(
      lz_vl, I32(0), I32(tile_z), tir::ForKind::kSerial,
      tir::For(
          ly_vl, I32(0), I32(d_v), tir::ForKind::kSerial,
          tir::SBlockRealize(
              {PrimExpr(lz_vl), PrimExpr(ly_vl)}, tir::const_true(),
              tir::SBlock(
                  ffi::Array<IterVar>{
                      IterVar(Range::FromMinExtent(I32(0), I32(tile_z)), vi_vl, tir::kDataPar, ""),
                      IterVar(Range::FromMinExtent(I32(0), I32(d_v)), vj_vl, tir::kDataPar, "")},
                  {}, {}, "V_load",
                  tir::LetStmt(cur_L_vl, L_kv_start + vi_vl,
                               tir::IfThenElse(
                                   cur_L_vl < kv_len,
                                   tir::BufferStore(
                                       V_smem,
                                       tir::BufferLoad(
                                           v_buf, ffi::Array<PrimExpr>{b_idx, L_kv_base + cur_L_vl,
                                                                       by, vj_vl}),
                                       ffi::Array<PrimExpr>{vi_vl, vj_vl}),
                                   tir::BufferStore(V_smem, zero_val,
                                                    ffi::Array<PrimExpr>{vi_vl, vj_vl})))))));

  // S_gemm
  tir::Var li_sg("li", DataType::Int(32)), lj_sg("lj", DataType::Int(32)),
      lk_sg("lk", DataType::Int(32));
  PrimExpr q_elem =
      is_f16 ? CastTo(tir::BufferLoad(Q_smem, ffi::Array<PrimExpr>{li_sg, lk_sg}), "float32")
             : tir::BufferLoad(Q_smem, ffi::Array<PrimExpr>{li_sg, lk_sg});
  PrimExpr k_elem =
      is_f16 ? CastTo(tir::BufferLoad(K_smem, ffi::Array<PrimExpr>{lj_sg, lk_sg}), "float32")
             : tir::BufferLoad(K_smem, ffi::Array<PrimExpr>{lj_sg, lk_sg});

  tir::IterVar iv_sg_i =
      tir::IterVar(Range(I32(0), I32(tile_x)), tir::Var("i", DataType::Int(32)), tir::kDataPar);
  tir::IterVar iv_sg_j =
      tir::IterVar(Range(I32(0), I32(tile_z)), tir::Var("j", DataType::Int(32)), tir::kDataPar);
  tir::IterVar iv_sg_k =
      tir::IterVar(Range(I32(0), I32(d_qk)), tir::Var("k_ax", DataType::Int(32)), tir::kCommReduce);
  ffi::Array<tir::IterVar> iter_vars_sg = {iv_sg_i, iv_sg_j, iv_sg_k};
  ffi::Array<tir::BufferRegion> s_gemm_reads = {
      tir::BufferRegion(S_local,
                        ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_sg_i->var, I32(1)),
                                               tvm::Range::FromMinExtent(iv_sg_j->var, I32(1))}),
      tir::BufferRegion(Q_smem,
                        ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_sg_i->var, I32(1)),
                                               tvm::Range::FromMinExtent(iv_sg_k->var, I32(1))}),
      tir::BufferRegion(K_smem,
                        ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_sg_j->var, I32(1)),
                                               tvm::Range::FromMinExtent(iv_sg_k->var, I32(1))})};
  ffi::Array<tir::BufferRegion> s_gemm_writes = {tir::BufferRegion(
      S_local, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_sg_i->var, I32(1)),
                                      tvm::Range::FromMinExtent(iv_sg_j->var, I32(1))})};
  Stmt s_gemm_inner_sblock = tir::SBlockRealize(
      ffi::Array<PrimExpr>{li_sg, lj_sg, lk_sg}, tir::const_true(),
      tir::SBlock(
          iter_vars_sg, s_gemm_reads, s_gemm_writes, "S_gemm",
          tir::BufferStore(
              S_local,
              tir::BufferLoad(S_local, ffi::Array<PrimExpr>{iv_sg_i->var, iv_sg_j->var}) +
                  (is_f16 ? CastTo(tir::BufferLoad(
                                       Q_smem, ffi::Array<PrimExpr>{iv_sg_i->var, iv_sg_k->var}),
                                   "float32")
                          : tir::BufferLoad(Q_smem,
                                            ffi::Array<PrimExpr>{iv_sg_i->var, iv_sg_k->var})) *
                      (is_f16 ? CastTo(tir::BufferLoad(K_smem, ffi::Array<PrimExpr>{iv_sg_j->var,
                                                                                    iv_sg_k->var}),
                                       "float32")
                              : tir::BufferLoad(K_smem,
                                                ffi::Array<PrimExpr>{iv_sg_j->var, iv_sg_k->var})) *
                      F32(log2e),
              ffi::Array<PrimExpr>{iv_sg_i->var, iv_sg_j->var}),
          /*init=*/
          tir::BufferStore(S_local, F32(0.0), ffi::Array<PrimExpr>{iv_sg_i->var, iv_sg_j->var})));
  ffi::Array<tir::BufferRegion> s_gemm_outer_reads = {
      tir::BufferRegion(Q_smem,
                        ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(I32(0), I32(tile_x)),
                                               tvm::Range::FromMinExtent(I32(0), I32(d_qk))}),
      tir::BufferRegion(K_smem,
                        ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(I32(0), I32(tile_z)),
                                               tvm::Range::FromMinExtent(I32(0), I32(d_qk))})};
  ffi::Array<tir::BufferRegion> s_gemm_outer_writes = {tir::BufferRegion(
      S_local, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(I32(0), I32(tile_x)),
                                      tvm::Range::FromMinExtent(I32(0), I32(tile_z))})};
  Stmt s_gemm = tir::SBlockRealize(
      {}, tir::const_true(),
      tir::SBlock({}, s_gemm_outer_reads, s_gemm_outer_writes, "",
                  tir::For(li_sg, I32(0), I32(tile_x), tir::ForKind::kSerial,
                           tir::For(lj_sg, I32(0), I32(tile_z), tir::ForKind::kSerial,
                                    tir::For(lk_sg, I32(0), I32(d_qk), tir::ForKind::kSerial,
                                             s_gemm_inner_sblock)))));

  // S_store
  tir::Var li_ss("li", DataType::Int(32)), lj_ss("lj", DataType::Int(32));
  tir::Var vi_ss("i", DataType::Int(32)), vj_ss("j", DataType::Int(32));
  ffi::Array<tir::BufferRegion> s_store_reads = {
      tir::BufferRegion(S_local, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(vi_ss, I32(1)),
                                                        tvm::Range::FromMinExtent(vj_ss, I32(1))})};
  ffi::Array<tir::BufferRegion> s_store_writes = {
      tir::BufferRegion(S_smem, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(vi_ss, I32(1)),
                                                       tvm::Range::FromMinExtent(vj_ss, I32(1))})};
  auto s_store_block = tir::SBlock(
      ffi::Array<IterVar>{
          IterVar(Range::FromMinExtent(I32(0), I32(tile_x)), vi_ss, tir::kDataPar, ""),
          IterVar(Range::FromMinExtent(I32(0), I32(tile_z)), vj_ss, tir::kDataPar, "")},
      s_store_reads, s_store_writes, "S_store",
      tir::BufferStore(S_smem, tir::BufferLoad(S_local, ffi::Array<PrimExpr>{vi_ss, vj_ss}),
                       ffi::Array<PrimExpr>{vi_ss, vj_ss}));
  Stmt s_store = tir::For(li_ss, I32(0), I32(tile_x), tir::ForKind::kSerial,
                          tir::For(lj_ss, I32(0), I32(tile_z), tir::ForKind::kSerial,
                                   tir::SBlockRealize({PrimExpr(li_ss), PrimExpr(lj_ss)},
                                                      tir::const_true(), s_store_block)));

  // update1: supports both causal and non-causal via causal_mask condition
  // row_ variable name: "row_" for causal, "_" for non-causal (matches Python reference IR)
  tir::Var i_u1("i", DataType::Int(32)), row_u1("row", DataType::Int(32)),
      j_u1("j", DataType::Int(32));
  tir::Var row_prime_u1(causal ? "row_" : "_", DataType::Int(32));
  // causal_mask: if causal>0: col < kv_len - qo_len + row_ + 1, else: col < kv_len
  auto causal_mask_u1 = [&](PrimExpr col) -> PrimExpr {
    return tvm::if_then_else(IntImm(DataType::Int(64), causal) > I32(0),
                             col < kv_len - qo_len + row_prime_u1 + I32(1), col < kv_len);
  };
  Stmt update1 = tir::For(
      i_u1, I32(0), I32(md_sz), tir::ForKind::kSerial,
      tir::LetStmt(
          row_u1, i_u1 * I32(bdx) * I32(num_warps) + ty * I32(bdx) + tx,
          tir::IfThenElse(
              row_u1 < I32(tile_x),
              tir::SBlockRealize(
                  {}, tir::const_true(),
                  tir::SBlock(
                      {}, {}, {}, "update1",
                      // Sequence: store_m_prev, store_m_new, LetStmt(row_, SeqStmt(for_j,
                      // store_d_new)) This matches the Python reference IR order
                      [&]() -> Stmt {
                        Stmt inner = tir::LetStmt(
                            row_prime_u1, floordiv(LH_start + row_u1, I32(group_size)),
                            tir::SeqStmt(
                                {tir::For(j_u1, I32(0), I32(tile_z), tir::ForKind::kSerial,
                                          tir::IfThenElse(
                                              causal_mask_u1(L_kv_start + j_u1),
                                              tir::BufferStore(
                                                  m_new_buf,
                                                  tvm::max(tir::BufferLoad(m_new_buf, {i_u1}),
                                                           tir::BufferLoad(
                                                               S_smem,
                                                               ffi::Array<PrimExpr>{row_u1, j_u1})),
                                                  {i_u1}))),
                                 tir::BufferStore(
                                     d_new_buf,
                                     tir::BufferLoad(d_smem, {row_u1}) *
                                         tvm::exp2(tir::BufferLoad(m_prev_buf, {i_u1}) -
                                                   tir::BufferLoad(m_new_buf, {i_u1})),
                                     {i_u1})}));
                        return tir::SeqStmt(
                            {tir::BufferStore(m_prev_buf, tir::BufferLoad(m_smem, {row_u1}),
                                              {i_u1}),
                             tir::BufferStore(m_new_buf, tir::BufferLoad(m_smem, {row_u1}), {i_u1}),
                             inner});
                      }(),
                      std::nullopt, {}, {}, detect_annots)))));

  tir::Var i_u2("i", DataType::Int(32)), row_u2("row", DataType::Int(32)),
      j_u2("j", DataType::Int(32));
  tir::Var row_prime_u2(causal ? "row_" : "_", DataType::Int(32));
  auto causal_mask_u2 = [&](PrimExpr col) -> PrimExpr {
    return tvm::if_then_else(IntImm(DataType::Int(64), causal) > I32(0),
                             col < kv_len - qo_len + row_prime_u2 + I32(1), col < kv_len);
  };
  Stmt update2_inner_ite = tir::IfThenElse(
      causal_mask_u2(L_kv_start + j_u2),
      tir::BufferStore(S_smem,
                       tvm::exp2(tir::BufferLoad(S_smem, ffi::Array<PrimExpr>{row_u2, j_u2}) -
                                 tir::BufferLoad(m_new_buf, {i_u2})),
                       ffi::Array<PrimExpr>{row_u2, j_u2}),
      tir::BufferStore(S_smem, tvm::exp2(F32(-50000.0) - tir::BufferLoad(m_new_buf, {i_u2})),
                       ffi::Array<PrimExpr>{row_u2, j_u2}));
  // update2: row_ is inside the if(row < tile_x) body, matching Python reference IR
  Stmt update2_sblock = tir::SBlockRealize(
      {}, tir::const_true(),
      tir::SBlock(
          {}, {}, {}, "update",
          tir::For(j_u2, I32(0), I32(tile_z), tir::ForKind::kSerial,
                   tir::IfThenElse(
                       row_u2 < I32(tile_x),
                       tir::LetStmt(row_prime_u2, floordiv(LH_start + row_u2, I32(group_size)),
                                    Stmt(update2_inner_ite)))),
          std::nullopt, {}, {}, detect_annots));
  Stmt update2 = tir::For(
      i_u2, I32(0), I32(md_sz), tir::ForKind::kSerial,
      tir::LetStmt(row_u2, i_u2 * I32(bdx) * I32(num_warps) + ty * I32(bdx) + tx, update2_sblock));

  // update3
  tir::Var i_u3("i", DataType::Int(32)), row_u3("row", DataType::Int(32)),
      j_u3("j", DataType::Int(32));
  Stmt update3 = tir::For(
      i_u3, I32(0), I32(md_sz), tir::ForKind::kSerial,
      tir::LetStmt(
          row_u3, i_u3 * I32(bdx) * I32(num_warps) + ty * I32(bdx) + tx,
          tir::IfThenElse(
              row_u3 < I32(tile_x),
              tir::SBlockRealize(
                  {}, tir::const_true(),
                  tir::SBlock(
                      {}, {}, {}, "update",
                      tir::SeqStmt(
                          {tir::For(j_u3, I32(0), I32(tile_z), tir::ForKind::kSerial,
                                    tir::BufferStore(
                                        d_new_buf,
                                        tir::BufferLoad(d_new_buf, {i_u3}) +
                                            tir::BufferLoad(S_smem,
                                                            ffi::Array<PrimExpr>{row_u3, j_u3}),
                                        {i_u3})),
                           tir::BufferStore(m_smem, tir::BufferLoad(m_new_buf, {i_u3}), {row_u3}),
                           tir::BufferStore(d_smem, tir::BufferLoad(d_new_buf, {i_u3}), {row_u3}),
                           tir::BufferStore(m_prev_smem, tir::BufferLoad(m_prev_buf, {i_u3}),
                                            {row_u3})}),
                      std::nullopt, {}, {}, detect_annots)))));

  // O_gemm
  tir::Var li_og("li", DataType::Int(32)), lj_og("lj", DataType::Int(32)),
      lk_og("lk", DataType::Int(32));
  PrimExpr v_elem =
      is_f16 ? CastTo(tir::BufferLoad(V_smem, ffi::Array<PrimExpr>{lk_og, lj_og}), "float32")
             : tir::BufferLoad(V_smem, ffi::Array<PrimExpr>{lk_og, lj_og});

  tir::IterVar iv_og_i =
      tir::IterVar(Range(I32(0), I32(tile_x)), tir::Var("i", DataType::Int(32)), tir::kDataPar);
  tir::IterVar iv_og_j =
      tir::IterVar(Range(I32(0), I32(d_v)), tir::Var("j", DataType::Int(32)), tir::kDataPar);
  tir::IterVar iv_og_k = tir::IterVar(Range(I32(0), I32(tile_z)),
                                      tir::Var("k_ax", DataType::Int(32)), tir::kCommReduce);
  ffi::Array<tir::IterVar> iter_vars_og = {iv_og_i, iv_og_j, iv_og_k};
  ffi::Array<tir::BufferRegion> o_gemm_reads = {
      tir::BufferRegion(O_local,
                        ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_og_i->var, I32(1)),
                                               tvm::Range::FromMinExtent(iv_og_j->var, I32(1))}),
      tir::BufferRegion(m_prev_smem,
                        ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_og_i->var, I32(1))}),
      tir::BufferRegion(m_smem,
                        ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_og_i->var, I32(1))}),
      tir::BufferRegion(S_smem,
                        ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_og_i->var, I32(1)),
                                               tvm::Range::FromMinExtent(iv_og_k->var, I32(1))}),
      tir::BufferRegion(V_smem,
                        ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_og_k->var, I32(1)),
                                               tvm::Range::FromMinExtent(iv_og_j->var, I32(1))})};
  ffi::Array<tir::BufferRegion> o_gemm_writes = {tir::BufferRegion(
      O_local, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_og_i->var, I32(1)),
                                      tvm::Range::FromMinExtent(iv_og_j->var, I32(1))})};
  Stmt o_gemm_inner_sblock = tir::SBlockRealize(
      ffi::Array<PrimExpr>{li_og, lj_og, lk_og}, tir::const_true(),
      tir::SBlock(
          iter_vars_og, o_gemm_reads, o_gemm_writes, "O_gemm",
          tir::BufferStore(
              O_local,
              tir::BufferLoad(O_local, ffi::Array<PrimExpr>{iv_og_i->var, iv_og_j->var}) +
                  tir::BufferLoad(S_smem, ffi::Array<PrimExpr>{iv_og_i->var, iv_og_k->var}) *
                      (is_f16 ? CastTo(tir::BufferLoad(V_smem, ffi::Array<PrimExpr>{iv_og_k->var,
                                                                                    iv_og_j->var}),
                                       "float32")
                              : tir::BufferLoad(V_smem,
                                                ffi::Array<PrimExpr>{iv_og_k->var, iv_og_j->var})),
              ffi::Array<PrimExpr>{iv_og_i->var, iv_og_j->var}),
          /*init=*/
          tir::BufferStore(
              O_local,
              tir::BufferLoad(O_local, ffi::Array<PrimExpr>{iv_og_i->var, iv_og_j->var}) *
                  tvm::exp2(tir::BufferLoad(m_prev_smem, {iv_og_i->var}) -
                            tir::BufferLoad(m_smem, {iv_og_i->var})),
              ffi::Array<PrimExpr>{iv_og_i->var, iv_og_j->var})));
  ffi::Array<tir::BufferRegion> o_gemm_outer_reads = {
      tir::BufferRegion(m_prev_smem,
                        ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(I32(0), I32(tile_x))}),
      tir::BufferRegion(m_smem,
                        ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(I32(0), I32(tile_x))}),
      tir::BufferRegion(S_smem,
                        ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(I32(0), I32(tile_x)),
                                               tvm::Range::FromMinExtent(I32(0), I32(tile_z))}),
      tir::BufferRegion(V_smem,
                        ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(I32(0), I32(tile_z)),
                                               tvm::Range::FromMinExtent(I32(0), I32(d_v))})};
  ffi::Array<tir::BufferRegion> o_gemm_outer_writes = {tir::BufferRegion(
      O_local, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(I32(0), I32(tile_x)),
                                      tvm::Range::FromMinExtent(I32(0), I32(d_v))})};
  Stmt o_gemm = tir::SBlockRealize(
      {}, tir::const_true(),
      tir::SBlock({}, o_gemm_outer_reads, o_gemm_outer_writes, "",
                  tir::For(li_og, I32(0), I32(tile_x), tir::ForKind::kSerial,
                           tir::For(lj_og, I32(0), I32(d_v), tir::ForKind::kSerial,
                                    tir::For(lk_og, I32(0), I32(tile_z), tir::ForKind::kSerial,
                                             o_gemm_inner_sblock)))));

  // KV iterator
  Stmt kv_iter = tir::LetStmt(
      L_kv_start, iterator * I32(tile_z),
      tir::LetStmt(L_kv_base, I32(0),
                   tir::SeqStmt({k_load, Sync(), v_load, Sync(), s_gemm, Sync(), s_store, Sync(),
                                 update1, update2, update3, Sync(), o_gemm})));
  // Use (kv_len + tile_z - 1) // tile_z = ceildiv(kv_len, tile_z)
  // Written as kv_len + (tile_z-1) to match Python reference IR form
  Stmt kv_loop = tir::For(iterator, I32(0), floordiv(kv_len + I32(tile_z - 1), I32(tile_z)),
                          tir::ForKind::kSerial, kv_iter);

  // O_store
  tir::Var li_os("li", DataType::Int(32)), lj_os("lj", DataType::Int(32));
  tir::Var vi_os("i", DataType::Int(32)), vj_os("j", DataType::Int(32));
  tir::Var cur_L_os("cur_L", DataType::Int(32)), cur_H_qo_os("cur_H_qo", DataType::Int(32));
  auto o_store_block = tir::SBlock(
      ffi::Array<IterVar>{
          IterVar(Range::FromMinExtent(I32(0), I32(tile_x)), vi_os, tir::kDataPar, ""),
          IterVar(Range::FromMinExtent(I32(0), I32(d_v)), vj_os, tir::kDataPar, "")},
      {}, {}, "O_store",
      tir::LetStmt(
          cur_L_os, floordiv(LH_start + vi_os, I32(group_size)),
          tir::LetStmt(
              cur_H_qo_os, by * I32(group_size) + floormod(LH_start + vi_os, I32(group_size)),
              tir::IfThenElse(
                  cur_L_os < qo_len,
                  tir::BufferStore(
                      output_buf,
                      is_f16
                          ? CastTo(tvm::div(
                                       tir::BufferLoad(O_local, ffi::Array<PrimExpr>{vi_os, vj_os}),
                                       tir::BufferLoad(d_smem, {vi_os})),
                                   dtype)
                          : tvm::div(tir::BufferLoad(O_local, ffi::Array<PrimExpr>{vi_os, vj_os}),
                                     tir::BufferLoad(d_smem, {vi_os})),
                      ffi::Array<PrimExpr>{b_idx, cur_L_os, cur_H_qo_os, vj_os})))),
      std::nullopt, {}, {}, detect_annots);
  Stmt o_store = tir::For(li_os, I32(0), I32(tile_x), tir::ForKind::kSerial,
                          tir::For(lj_os, I32(0), I32(d_v), tir::ForKind::kSerial,
                                   tir::SBlockRealize({PrimExpr(li_os), PrimExpr(lj_os)},
                                                      tir::const_true(), o_store_block)));

  // lse_store
  tir::Var li_lse("li", DataType::Int(32));
  tir::Var vi_lse("i", DataType::Int(32));
  tir::Var cur_L_lse("cur_L", DataType::Int(32)), cur_H_qo_lse("cur_H_qo", DataType::Int(32));
  auto lse_store_block = tir::SBlock(
      ffi::Array<IterVar>{
          IterVar(Range::FromMinExtent(I32(0), I32(tile_x)), vi_lse, tir::kDataPar, "")},
      {}, {}, "lse_store",
      tir::LetStmt(
          cur_L_lse, floordiv(LH_start + vi_lse, I32(group_size)),
          tir::LetStmt(
              cur_H_qo_lse, by * I32(group_size) + floormod(LH_start + vi_lse, I32(group_size)),
              tir::IfThenElse(
                  cur_L_lse < qo_len,
                  tir::BufferStore(lse_buf,
                                   is_f16 ? CastTo(tir::BufferLoad(m_smem, {vi_lse}) +
                                                       tvm::log2(tir::BufferLoad(d_smem, {vi_lse})),
                                                   dtype)
                                          : tir::BufferLoad(m_smem, {vi_lse}) +
                                                tvm::log2(tir::BufferLoad(d_smem, {vi_lse})),
                                   ffi::Array<PrimExpr>{b_idx, cur_L_lse, cur_H_qo_lse})))),
      std::nullopt, {}, {}, detect_annots);
  Stmt lse_store =
      tir::For(li_lse, I32(0), I32(tile_x), tir::ForKind::kSerial,
               tir::SBlockRealize({PrimExpr(li_lse)}, tir::const_true(), lse_store_block));

  // sblock body: b_idx = vbx // batch_tiles, tile_id = vbx % batch_tiles
  ffi::Array<tir::Buffer> alloc_bufs = {Q_smem,  K_smem,    V_smem,     S_smem,
                                        S_local, O_local,   m_smem,     m_prev_smem,
                                        d_smem,  m_new_buf, m_prev_buf, d_new_buf};
  Stmt sblock_body =
      tir::LetStmt(b_idx, floordiv(vbx, batch_tiles),
                   tir::LetStmt(tile_id, floormod(vbx, batch_tiles),
                                tir::LetStmt(LH_start, tile_id * I32(tile_x),
                                             tir::SeqStmt({Sync(), init_md, o_init, Sync(), q_load,
                                                           Sync(), kv_loop, o_store, lse_store}))));

  // Thread-binding IterVars for the For loops
  IterVar lbx_iv(Range(nullptr), tir::Var("iter", DataType::Int(32)), tir::kThreadIndex,
                 "blockIdx.x");
  IterVar lby_iv(Range(nullptr), tir::Var("iter", DataType::Int(32)), tir::kThreadIndex,
                 "blockIdx.y");
  IterVar lty_iv(Range(nullptr), tir::Var("iter", DataType::Int(32)), tir::kThreadIndex,
                 "threadIdx.y");
  IterVar ltx_iv(Range(nullptr), tir::Var("iter", DataType::Int(32)), tir::kThreadIndex,
                 "threadIdx.x");
  // Spatial IterVars for the "attn" sblock (T.axis.remap("SSSS", [lbx, lby, lty, ltx]))
  PrimExpr bx_extent = batch_size * batch_tiles;
  IterVar vbx_iv(Range(I32(0), bx_extent), vbx, tir::kDataPar, "");
  IterVar by_iv(Range(I32(0), I32(h_kv)), by, tir::kDataPar, "");
  IterVar ty_iv(Range(I32(0), I32(num_warps)), ty, tir::kDataPar, "");
  IterVar tx_iv(Range(I32(0), I32(bdx)), tx, tir::kDataPar, "");

  Stmt body = tir::SBlockRealize(
      {}, tir::const_true(),
      tir::SBlock(
          {}, {}, {}, "root",
          // batch_tiles is a LetStmt in the root sblock body, matching Python reference IR
          tir::LetStmt(
              batch_tiles_var, batch_tiles_expr,
              tir::For(
                  lbx, I32(0), bx_extent, tir::ForKind::kThreadBinding,
                  tir::For(lby, I32(0), I32(h_kv), tir::ForKind::kThreadBinding,
                           tir::For(lty, I32(0), I32(num_warps), tir::ForKind::kThreadBinding,
                                    tir::For(ltx, I32(0), I32(bdx), tir::ForKind::kThreadBinding,
                                             tir::SBlockRealize(
                                                 ffi::Array<PrimExpr>{PrimExpr(lbx), PrimExpr(lby),
                                                                      PrimExpr(lty), PrimExpr(ltx)},
                                                 tir::const_true(),
                                                 tir::SBlock(ffi::Array<IterVar>{vbx_iv, by_iv,
                                                                                 ty_iv, tx_iv},
                                                             {}, {}, "attn", sblock_body,
                                                             std::nullopt, alloc_bufs, {}, {})),
                                             ltx_iv),
                                    lty_iv),
                           lby_iv),
                  lbx_iv))));

  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_q_h, q_buf);
  buf_map.Set(h_k_h, k_buf);
  buf_map.Set(h_v_h, v_buf);
  buf_map.Set(h_output_h, output_buf);
  buf_map.Set(h_lse_h, lse_buf);

  ffi::Array<tir::Var> params = {h_q_h, h_k_h, h_v_h, h_output_h, h_lse_h};

  tir::PrimFunc fn(params, body, VoidType(), buf_map);
  fn = WithAttr(fn, "global_symbol", ffi::Any(ffi::String("batch_sequence_prefill_kv")));
  fn = tir::ScriptComplete(fn, {});
  return SchedulePrefillKernel(fn, cfg, /*transform_k_load=*/false, /*merged_qk_load=*/false);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.frontend.nn.llm.kv_cache.attention_sequence_prefill",
                        [](int64_t h_kv, int64_t h_q, int64_t d_qk, int64_t d_v, ffi::String dtype,
                           ffi::Map<ffi::String, ffi::Any> rope_scaling, Target target,
                           int64_t causal = 0, double sm_scale = 1.0) -> tir::PrimFunc {
                          return AttentionSequencePrefill(h_kv, h_q, d_qk, d_v, std::string(dtype),
                                                          rope_scaling, target, causal, sm_scale);
                        });
}

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
