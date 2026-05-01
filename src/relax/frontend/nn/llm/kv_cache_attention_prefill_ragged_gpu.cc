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
 * \file src/relax/frontend/nn/llm/kv_cache_attention_prefill_ragged_gpu.cc
 * \brief GPU TIR kernel for ragged prefill (non-paged KV).
 *        C++ port of _attention_prefill_ragged() in kv_cache.py.
 *
 * Key difference from paged prefill:
 *   - K/V are dense ragged tensors (kv_len, h_kv, d) not paged.
 *   - K_smem is stored transposed: shape (d_qk, tile_z) instead of (tile_z, d_qk).
 *     This requires transform_k_load=true in SchedulePrefillKernel.
 *   - No sliding window.
 */

#include "kv_cache.h"
#include "kv_cache_attention_prefill_gpu_helpers.h"
#include "kv_cache_attention_prefill_gpu_schedule.h"
#include "../../../../tir/ir/script/script_complete.h"

#include <tvm/s_tir/stmt.h>
#include <tvm/tir/op.h>

#include <cmath>
#include <string>

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

tir::PrimFunc AttentionPrefillRagged(int64_t h_kv, int64_t h_q, int64_t d_qk, int64_t d_v,
                                     const std::string& dtype,
                                     const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
                                     Target target) {
  const PrefillKernelConfig cfg = ComputePrefillKernelConfig(h_kv, h_q, d_qk, dtype, target);
  const int64_t NUM_BLKS = cfg.NUM_BLKS, group_size = cfg.group_size;
  const int64_t bdx = cfg.bdx, num_warps = cfg.num_warps;
  const int64_t tile_x = cfg.tile_x, tile_z = cfg.tile_z;
  const bool is_f16 = (dtype == "float16");
  DataType dt = DataType(runtime::StringToDLDataType(dtype));
  const double log2e = 1.4426950408889634;

  // ---- symbolic size vars ----
  tir::SizeVar batch_size("batch_size", DataType::Int(32));
  tir::SizeVar qo_len("qo_len", DataType::Int(32));
  tir::SizeVar kv_len("kv_len", DataType::Int(32));
  tir::SizeVar q_indptr_eo("q_indptr_elem_offset", DataType::Int(32));
  tir::SizeVar kv_indptr_eo("kv_indptr_elem_offset", DataType::Int(32));
  tir::SizeVar q_rope_pos_eo("q_rope_position_elem_offset", DataType::Int(32));
  tir::SizeVar k_rope_pos_eo("k_rope_pos_offset_elem_offset", DataType::Int(32));

  // ---- parameter handles ----
  tir::Var h_q_h("var_q", DataType::Handle());
  tir::Var h_q_indptr_h("var_q_indptr", DataType::Handle());
  tir::Var h_k_h("var_k", DataType::Handle());
  tir::Var h_v_h("var_v", DataType::Handle());
  tir::Var h_kv_indptr_h("var_kv_indptr", DataType::Handle());
  tir::Var h_q_rope_pos_h("var_q_rope_position", DataType::Handle());
  tir::Var h_k_rope_pos_h("var_k_rope_pos_offset", DataType::Handle());
  tir::Var h_output_h("var_output", DataType::Handle());
  tir::Var h_lse_h("var_lse", DataType::Handle());
  tir::Var causal("causal", DataType::Int(32));
  tir::Var rotary_mode("rotary_mode", DataType::Int(32));
  tir::Var rope_scale("rope_scale", DataType::Float(32));
  tir::Var rope_theta("rope_theta", DataType::Float(32));
  tir::Var sm_scale("sm_scale", DataType::Float(32));

  // ---- buffers ----
  tir::Buffer q_buf = tir::decl_buffer({qo_len, I32(h_q), I32(d_qk)}, dt, "q");
  tir::Buffer q_indptr_buf = tir::Buffer(
      tir::decl_buffer({batch_size + I32(1)}, DataType::Int(32), "q_indptr")->data,
      DataType::Int(32), {batch_size + I32(1)}, {}, q_indptr_eo, "q_indptr", 0, 0, tir::kDefault);
  tir::Buffer k_buf = tir::decl_buffer({kv_len, I32(h_kv), I32(d_qk)}, dt, "k");
  tir::Buffer v_buf = tir::decl_buffer({kv_len, I32(h_kv), I32(d_v)}, dt, "v");
  tir::Buffer kv_indptr_buf = tir::Buffer(
      tir::decl_buffer({batch_size + I32(1)}, DataType::Int(32), "kv_indptr")->data,
      DataType::Int(32), {batch_size + I32(1)}, {}, kv_indptr_eo, "kv_indptr", 0, 0, tir::kDefault);
  tir::Buffer q_rope_pos_buf = tir::Buffer(
      tir::decl_buffer({qo_len}, DataType::Int(32), "q_rope_position")->data,
      DataType::Int(32), {qo_len}, {}, q_rope_pos_eo, "q_rope_position", 0, 0, tir::kDefault);
  tir::Buffer k_rope_pos_buf = tir::Buffer(
      tir::decl_buffer({batch_size}, DataType::Int(32), "k_rope_pos_offset")->data,
      DataType::Int(32), {batch_size}, {}, k_rope_pos_eo, "k_rope_pos_offset", 0, 0, tir::kDefault);
  tir::Buffer output_buf = tir::decl_buffer({qo_len, I32(h_q), I32(d_v)}, dt, "output");
  tir::Buffer lse_buf = tir::decl_buffer({qo_len, I32(h_q)}, DataType::Float(32), "lse");

  // ---- thread index vars ----
  tir::Var lbx("lbx", DataType::Int(32)), lby("lby", DataType::Int(32));
  tir::Var lty("lty", DataType::Int(32)), ltx("ltx", DataType::Int(32));
  tir::Var bx("bx", DataType::Int(32)), by("by", DataType::Int(32));
  tir::Var ty("ty", DataType::Int(32)), tx("tx", DataType::Int(32));

  // ---- sblock-local alloc buffers ----
  tir::Buffer tile_id_buf    = AllocLocal("tile_id",    {I32(1)}, "int32");
  tir::Buffer batch_idx_buf  = AllocLocal("batch_idx",  {I32(1)}, "int32");
  tir::Buffer batch_tiles_buf= AllocLocal("batch_tiles",{I32(1)}, "int32");
  tir::Buffer batch_rows_buf = AllocLocal("batch_rows", {I32(1)}, "int32");
  tir::Buffer dummy_buf      = AllocLocal("_",          {I32(1)}, "int32");
  tir::Buffer kv_chunk_buf   = AllocLocal("kv_chunk_len",{I32(1)},"int32");
  tir::Buffer Q_smem  = AllocShared("Q_smem", {I32(tile_x), I32(d_qk)}, dtype);
  // K_smem is transposed: (d_qk, tile_z) for ragged variant
  tir::Buffer K_smem  = AllocShared("K_smem", {I32(d_qk), I32(tile_z)}, dtype);
  tir::Buffer V_smem  = AllocShared("V_smem", {I32(tile_z), I32(d_v)}, dtype);
  tir::Buffer S_smem  = AllocShared("S_smem", {I32(tile_x), I32(tile_z)}, "float32");
  tir::Buffer S_local = AllocLocal("S_local", {I32(tile_x), I32(tile_z)}, "float32");
  tir::Buffer O_local = AllocLocal("O_local", {I32(tile_x), I32(d_v)}, "float32");
  tir::Buffer m_smem      = AllocShared("m_smem",      {I32(tile_x)}, "float32");
  tir::Buffer m_prev_smem = AllocShared("m_prev_smem", {I32(tile_x)}, "float32");
  tir::Buffer d_smem      = AllocShared("d_smem",      {I32(tile_x)}, "float32");
  int64_t md_sz = static_cast<int64_t>(std::ceil((double)tile_x / (bdx * num_warps)));
  tir::Buffer m_new_buf  = AllocLocal("m_new",  {I32(md_sz)}, "float32");
  tir::Buffer m_prev_buf = AllocLocal("m_prev", {I32(md_sz)}, "float32");
  tir::Buffer d_new_buf  = AllocLocal("d_new",  {I32(md_sz)}, "float32");

  auto ld0 = [](tir::Buffer b) { return tir::BufferLoad(b, {I32(0)}); };
  auto st0 = [](tir::Buffer b, PrimExpr v) -> Stmt { return tir::BufferStore(b, v, {I32(0)}); };
  auto causal_mask = [&](PrimExpr row, PrimExpr col, PrimExpr kv_len_e, PrimExpr qo_len_e) {
    return tvm::if_then_else(causal > I32(0),
                              col < kv_len_e - qo_len_e + row + I32(1),
                              col < kv_len_e);
  };

  tir::Var b_idx("b_idx", DataType::Int(32));
  tir::Var LH_start("LH_start", DataType::Int(32));
  tir::Var q_indptr_val("q_indptr_val", DataType::Int(32));
  tir::Var iterator("iterator", DataType::Int(32));
  tir::Var L_kv_start("L_kv_start", DataType::Int(32));
  tir::Var L_kv_base("L_kv_base", DataType::Int(32));

  PrimExpr kv_chunk_expr = tir::BufferLoad(kv_indptr_buf, {b_idx + I32(1)}) -
                           tir::BufferLoad(kv_indptr_buf, {b_idx});
  PrimExpr qo_len_expr = tir::BufferLoad(q_indptr_buf, {b_idx + I32(1)}) -
                         tir::BufferLoad(q_indptr_buf, {b_idx});

  // init states
  tir::Var i_init("i", DataType::Int(32)), row_init("row", DataType::Int(32));
  Stmt init_md = tir::For(i_init, I32(0), I32(md_sz), tir::ForKind::kSerial,
      tir::LetStmt(row_init, i_init * I32(bdx * num_warps) + ty * I32(bdx) + tx,
          tir::IfThenElse(row_init < I32(tile_x),
              tir::SeqStmt({tir::BufferStore(m_smem, F32(-50000.0), {row_init}),
                            tir::BufferStore(d_smem, F32(1.0), {row_init})}))));

  // O_init
  tir::Var li_oi("li", DataType::Int(32)), lj_oi("lj", DataType::Int(32));
  Stmt o_init = tir::For(li_oi, I32(0), I32(tile_x), tir::ForKind::kSerial,
      tir::For(lj_oi, I32(0), I32(d_v), tir::ForKind::kSerial,
          tir::SBlockRealize({}, tir::const_true(),
              tir::SBlock({}, {}, {}, "O_init",
                  tir::BufferStore(O_local, F32(0.0), ffi::Array<PrimExpr>{li_oi, lj_oi})))));

  // Q_load
  tir::Var li_ql("li", DataType::Int(32)), lj_ql("lj", DataType::Int(32));
  tir::Var cur_L_ql("cur_L", DataType::Int(32)), cur_H_qo_ql("cur_H_qo", DataType::Int(32));
  PrimExpr zero_val = is_f16 ? CastTo(F32(0.0), dtype) : F32(0.0);
  Stmt q_load = tir::For(li_ql, I32(0), I32(tile_x), tir::ForKind::kSerial,
      tir::For(lj_ql, I32(0), I32(d_qk), tir::ForKind::kSerial,
          tir::SBlockRealize({}, tir::const_true(),
              tir::SBlock({}, {}, {}, "Q_load",
                  tir::LetStmt(cur_L_ql,
                      q_indptr_val + floordiv(LH_start + li_ql, I32(group_size)),
                      tir::LetStmt(cur_H_qo_ql,
                          by * I32(group_size) + floormod(LH_start + li_ql, I32(group_size)),
                          tir::IfThenElse(
                              cur_L_ql < tir::BufferLoad(q_indptr_buf, {b_idx + I32(1)}),
                              tir::BufferStore(Q_smem,
                                  BuildPrefillRopeExpr(q_buf, {cur_L_ql, cur_H_qo_ql}, lj_ql, d_qk,
                                      tir::BufferLoad(q_rope_pos_buf, {cur_L_ql}),
                                      rope_scale, rope_theta, rotary_mode, dtype, rope_scaling), ffi::Array<PrimExpr>{li_ql, lj_ql}),
                              tir::BufferStore(Q_smem, zero_val, ffi::Array<PrimExpr>{li_ql, lj_ql}))))))));

  // K_load — stored transposed: K_smem[j, i] = k[L_kv_base+cur_L, by, j]
  tir::Var lz_kl("lz", DataType::Int(32)), ly_kl("ly", DataType::Int(32));
  tir::Var cur_L_kl("cur_L", DataType::Int(32));
  Stmt k_load = tir::For(lz_kl, I32(0), I32(tile_z), tir::ForKind::kSerial,
      tir::For(ly_kl, I32(0), I32(d_qk), tir::ForKind::kSerial,
          tir::SBlockRealize({}, tir::const_true(),
              tir::SBlock({}, {},
    ffi::Array<tir::BufferRegion>{
        tir::BufferRegion(K_smem, ffi::Array<tvm::Range>{
            tvm::Range::FromMinExtent(ly_kl, I32(1)),
            tvm::Range::FromMinExtent(lz_kl, I32(1))})},
    "K_load",
                  tir::LetStmt(cur_L_kl, L_kv_start + lz_kl,
                      tir::IfThenElse(cur_L_kl < ld0(kv_chunk_buf),
                          tir::BufferStore(K_smem,
                              BuildPrefillRopeExpr(k_buf,
                                  {L_kv_base + cur_L_kl, by}, ly_kl, d_qk,
                                  tir::BufferLoad(k_rope_pos_buf, {b_idx}) + cur_L_kl,
                                  rope_scale, rope_theta, rotary_mode, dtype, rope_scaling), ffi::Array<PrimExpr>{ly_kl, lz_kl}),  // transposed!
                          tir::BufferStore(K_smem, zero_val, ffi::Array<PrimExpr>{ly_kl, lz_kl})))))));

  // V_load
  tir::Var lz_vl("lz", DataType::Int(32)), ly_vl("ly", DataType::Int(32));
  tir::Var cur_L_vl("cur_L", DataType::Int(32));
  Stmt v_load = tir::For(lz_vl, I32(0), I32(tile_z), tir::ForKind::kSerial,
      tir::For(ly_vl, I32(0), I32(d_v), tir::ForKind::kSerial,
          tir::SBlockRealize({}, tir::const_true(),
              tir::SBlock({}, {}, {}, "V_load",
                  tir::LetStmt(cur_L_vl, L_kv_start + lz_vl,
                      tir::IfThenElse(cur_L_vl < ld0(kv_chunk_buf),
                          tir::BufferStore(V_smem,
                              tir::BufferLoad(v_buf, ffi::Array<PrimExpr>{L_kv_base + cur_L_vl, by, ly_vl}), ffi::Array<PrimExpr>{lz_vl, ly_vl}),
                          tir::BufferStore(V_smem, zero_val, ffi::Array<PrimExpr>{lz_vl, ly_vl})))))));

  // S_gemm
  tir::Var li_sg("li", DataType::Int(32)), lj_sg("lj", DataType::Int(32)), lk_sg("lk", DataType::Int(32));
  PrimExpr q_elem = is_f16 ? CastTo(tir::BufferLoad(Q_smem, ffi::Array<PrimExpr>{li_sg, lk_sg}), "float32")
                           : tir::BufferLoad(Q_smem, ffi::Array<PrimExpr>{li_sg, lk_sg});
  PrimExpr k_elem = is_f16 ? CastTo(tir::BufferLoad(K_smem, ffi::Array<PrimExpr>{lj_sg, lk_sg}), "float32")
                           : tir::BufferLoad(K_smem, ffi::Array<PrimExpr>{lj_sg, lk_sg});


    tir::IterVar iv_sg_i = tir::IterVar(Range(I32(0), I32(tile_x)), tir::Var("i", DataType::Int(32)), tir::kDataPar);
  tir::IterVar iv_sg_j = tir::IterVar(Range(I32(0), I32(tile_z)), tir::Var("j", DataType::Int(32)), tir::kDataPar);
  tir::IterVar iv_sg_k = tir::IterVar(Range(I32(0), I32(d_qk)), tir::Var("k", DataType::Int(32)), tir::kCommReduce);
  ffi::Array<tir::IterVar> iter_vars_sg = {iv_sg_i, iv_sg_j, iv_sg_k};
    ffi::Array<tir::BufferRegion> s_gemm_reads = {
      tir::BufferRegion(Q_smem, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_sg_i->var, I32(1)), tvm::Range::FromMinExtent(iv_sg_k->var, I32(1))}),
      tir::BufferRegion(K_smem, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_sg_j->var, I32(1)), tvm::Range::FromMinExtent(iv_sg_k->var, I32(1))}),
      tir::BufferRegion(S_local, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_sg_i->var, I32(1)), tvm::Range::FromMinExtent(iv_sg_j->var, I32(1))})};
    ffi::Array<tir::BufferRegion> s_gemm_writes = {
      tir::BufferRegion(S_local, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_sg_i->var, I32(1)), tvm::Range::FromMinExtent(iv_sg_j->var, I32(1))})};
  Stmt s_gemm_inner_sblock =
      tir::SBlockRealize(
          ffi::Array<PrimExpr>{li_sg, lj_sg, lk_sg}, tir::const_true(),
          tir::SBlock(iter_vars_sg, s_gemm_reads, s_gemm_writes, "S_gemm",
              tir::BufferStore(S_local,
                  tir::BufferLoad(S_local, ffi::Array<PrimExpr>{iv_sg_i->var, iv_sg_j->var}) +
                  (is_f16 ? CastTo(tir::BufferLoad(Q_smem, ffi::Array<PrimExpr>{iv_sg_i->var, iv_sg_k->var}), "float32") : tir::BufferLoad(Q_smem, ffi::Array<PrimExpr>{iv_sg_i->var, iv_sg_k->var})) * (is_f16 ? CastTo(tir::BufferLoad(K_smem, ffi::Array<PrimExpr>{iv_sg_j->var, iv_sg_k->var}), "float32") : tir::BufferLoad(K_smem, ffi::Array<PrimExpr>{iv_sg_j->var, iv_sg_k->var})) * sm_scale * F32(log2e),
                  ffi::Array<PrimExpr>{iv_sg_i->var, iv_sg_j->var}),
              /*init=*/tir::BufferStore(S_local, F32(0.0),
                  ffi::Array<PrimExpr>{iv_sg_i->var, iv_sg_j->var})));
  Stmt s_gemm =
      tir::SBlockRealize({}, tir::const_true(),
          tir::SBlock({}, {}, {}, "",
              tir::For(li_sg, I32(0), I32(tile_x), tir::ForKind::kSerial,
                  tir::For(lj_sg, I32(0), I32(tile_z), tir::ForKind::kSerial,
                      tir::For(lk_sg, I32(0), I32(d_qk), tir::ForKind::kSerial,
                          s_gemm_inner_sblock)))));

  // S_store
  tir::Var li_ss("li", DataType::Int(32)), lj_ss("lj", DataType::Int(32));
  Stmt s_store = tir::For(li_ss, I32(0), I32(tile_x), tir::ForKind::kSerial,
      tir::For(lj_ss, I32(0), I32(tile_z), tir::ForKind::kSerial,
          tir::SBlockRealize({}, tir::const_true(),
              tir::SBlock({}, {}, {}, "S_store",
                  tir::BufferStore(S_smem,
                      tir::BufferLoad(S_local, ffi::Array<PrimExpr>{li_ss, lj_ss}), ffi::Array<PrimExpr>{li_ss, lj_ss})))));

  // update1
  tir::Var i_u1("i", DataType::Int(32)), row_u1("row", DataType::Int(32));
  tir::Var row__u1("row_", DataType::Int(32)), j_u1("j", DataType::Int(32));
  Stmt update1 = tir::For(i_u1, I32(0), I32(md_sz), tir::ForKind::kSerial,
      tir::LetStmt(row_u1, i_u1 * I32(bdx * num_warps) + ty * I32(bdx) + tx,
          tir::IfThenElse(row_u1 < I32(tile_x),
              tir::SBlockRealize({}, tir::const_true(),
                  tir::SBlock({}, {}, {}, "update1",
                      tir::SeqStmt({
                          tir::BufferStore(m_prev_buf, tir::BufferLoad(m_smem, {row_u1}), {i_u1}),
                          tir::BufferStore(m_new_buf,  tir::BufferLoad(m_smem, {row_u1}), {i_u1}),
                          tir::LetStmt(row__u1, floordiv(LH_start + row_u1, I32(group_size)),
                              tir::For(j_u1, I32(0), I32(tile_z), tir::ForKind::kSerial,
                                  tir::IfThenElse(
                                      causal_mask(row__u1, L_kv_start + j_u1,
                                                  ld0(kv_chunk_buf), qo_len_expr),
                                      tir::BufferStore(m_new_buf,
                                          tvm::max(tir::BufferLoad(m_new_buf, {i_u1}),
                                                   tir::BufferLoad(S_smem, ffi::Array<PrimExpr>{row_u1, j_u1})),
                                          {i_u1})))),
                          tir::BufferStore(d_new_buf,
                              tir::BufferLoad(d_smem, {row_u1}) *
                              tvm::exp2(tir::BufferLoad(m_prev_buf, {i_u1}) -
                                        tir::BufferLoad(m_new_buf, {i_u1})),
                              {i_u1})}))))));

  // update2
  tir::Var i_u2("i", DataType::Int(32)), row_u2("row", DataType::Int(32));
  tir::Var row__u2("row_", DataType::Int(32)), j_u2("j", DataType::Int(32));
  Stmt update2_sblock =
      tir::SBlockRealize({}, tir::const_true(),
              tir::SBlock({}, {}, {}, "update",
                  tir::For(j_u2, I32(0), I32(tile_z), tir::ForKind::kSerial,
                      tir::IfThenElse(row_u2 < I32(tile_x),
                          tir::LetStmt(row__u2, floordiv(LH_start + row_u2, I32(group_size)),
                              tir::IfThenElse(
                                  causal_mask(row__u2, L_kv_start + j_u2,
                                              ld0(kv_chunk_buf), qo_len_expr),
                                  tir::BufferStore(S_smem,
                                      tvm::exp2(tir::BufferLoad(S_smem, ffi::Array<PrimExpr>{row_u2, j_u2}) -
                                                tir::BufferLoad(m_new_buf, {i_u2})), ffi::Array<PrimExpr>{row_u2, j_u2}),
                                  tir::BufferStore(S_smem,
                                      tvm::exp2(F32(-50000.0) -
                                                tir::BufferLoad(m_new_buf, {i_u2})), ffi::Array<PrimExpr>{row_u2, j_u2})))))));
  Stmt update2 = tir::For(i_u2, I32(0), I32(md_sz), tir::ForKind::kSerial,
      tir::LetStmt(row_u2, i_u2 * I32(bdx * num_warps) + ty * I32(bdx) + tx,
          update2_sblock));

  // update3
  tir::Var i_u3("i", DataType::Int(32)), row_u3("row", DataType::Int(32)), j_u3("j", DataType::Int(32));
  Stmt update3 = tir::For(i_u3, I32(0), I32(md_sz), tir::ForKind::kSerial,
      tir::LetStmt(row_u3, i_u3 * I32(bdx * num_warps) + ty * I32(bdx) + tx,
          tir::IfThenElse(row_u3 < I32(tile_x),
              tir::SBlockRealize({}, tir::const_true(),
                  tir::SBlock({}, {}, {}, "update",
                      tir::SeqStmt({
                          tir::For(j_u3, I32(0), I32(tile_z), tir::ForKind::kSerial,
                              tir::BufferStore(d_new_buf,
                                  tir::BufferLoad(d_new_buf, {i_u3}) +
                                  tir::BufferLoad(S_smem, ffi::Array<PrimExpr>{row_u3, j_u3}), {i_u3})),
                          tir::BufferStore(m_smem,      tir::BufferLoad(m_new_buf,  {i_u3}), {row_u3}),
                          tir::BufferStore(d_smem,      tir::BufferLoad(d_new_buf,  {i_u3}), {row_u3}),
                          tir::BufferStore(m_prev_smem, tir::BufferLoad(m_prev_buf, {i_u3}), {row_u3})}))))));

  // O_gemm
  tir::Var li_og("li", DataType::Int(32)), lj_og("lj", DataType::Int(32)), lk_og("lk", DataType::Int(32));
  PrimExpr v_elem = is_f16 ? CastTo(tir::BufferLoad(V_smem, ffi::Array<PrimExpr>{lk_og, lj_og}), "float32")
                           : tir::BufferLoad(V_smem, ffi::Array<PrimExpr>{lk_og, lj_og});


    tir::IterVar iv_og_i = tir::IterVar(Range(I32(0), I32(tile_x)), tir::Var("i", DataType::Int(32)), tir::kDataPar);
  tir::IterVar iv_og_j = tir::IterVar(Range(I32(0), I32(d_v)), tir::Var("j", DataType::Int(32)), tir::kDataPar);
  tir::IterVar iv_og_k = tir::IterVar(Range(I32(0), I32(tile_z)), tir::Var("k", DataType::Int(32)), tir::kCommReduce);
  ffi::Array<tir::IterVar> iter_vars_og = {iv_og_i, iv_og_j, iv_og_k};
    ffi::Array<tir::BufferRegion> o_gemm_reads = {
      tir::BufferRegion(S_smem, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_og_i->var, I32(1)), tvm::Range::FromMinExtent(iv_og_k->var, I32(1))}),
      tir::BufferRegion(V_smem, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_og_k->var, I32(1)), tvm::Range::FromMinExtent(iv_og_j->var, I32(1))}),
      tir::BufferRegion(O_local, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_og_i->var, I32(1)), tvm::Range::FromMinExtent(iv_og_j->var, I32(1))}),
      tir::BufferRegion(m_prev_smem, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_og_i->var, I32(1))}),
      tir::BufferRegion(m_smem, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_og_i->var, I32(1))})};
    ffi::Array<tir::BufferRegion> o_gemm_writes = {
      tir::BufferRegion(O_local, ffi::Array<tvm::Range>{tvm::Range::FromMinExtent(iv_og_i->var, I32(1)), tvm::Range::FromMinExtent(iv_og_j->var, I32(1))})};
  Stmt o_gemm_inner_sblock =
      tir::SBlockRealize(
          ffi::Array<PrimExpr>{li_og, lj_og, lk_og}, tir::const_true(),
          tir::SBlock(iter_vars_og, o_gemm_reads, o_gemm_writes, "O_gemm",
              tir::BufferStore(O_local,
                  tir::BufferLoad(O_local, ffi::Array<PrimExpr>{iv_og_i->var, iv_og_j->var}) +
                  tir::BufferLoad(S_smem, ffi::Array<PrimExpr>{iv_og_i->var, iv_og_k->var}) * (is_f16 ? CastTo(tir::BufferLoad(V_smem, ffi::Array<PrimExpr>{iv_og_k->var, iv_og_j->var}), "float32") : tir::BufferLoad(V_smem, ffi::Array<PrimExpr>{iv_og_k->var, iv_og_j->var})),
                  ffi::Array<PrimExpr>{iv_og_i->var, iv_og_j->var}),
              /*init=*/tir::BufferStore(O_local,
                  tir::BufferLoad(O_local, ffi::Array<PrimExpr>{iv_og_i->var, iv_og_j->var}) *
                  tvm::exp2(tir::BufferLoad(m_prev_smem, {iv_og_i->var}) -
                            tir::BufferLoad(m_smem, {iv_og_i->var})),
                  ffi::Array<PrimExpr>{iv_og_i->var, iv_og_j->var})));
  Stmt o_gemm =
      tir::SBlockRealize({}, tir::const_true(),
          tir::SBlock({}, {}, {}, "",
              tir::For(li_og, I32(0), I32(tile_x), tir::ForKind::kSerial,
                  tir::For(lj_og, I32(0), I32(d_v), tir::ForKind::kSerial,
                      tir::For(lk_og, I32(0), I32(tile_z), tir::ForKind::kSerial,
                          o_gemm_inner_sblock)))));

  // KV iterator
  Stmt kv_iter = tir::LetStmt(L_kv_start, iterator * I32(tile_z),
      tir::LetStmt(L_kv_base, tir::BufferLoad(kv_indptr_buf, {b_idx}),
          tir::SeqStmt({k_load, Sync(), v_load, Sync(), s_gemm, Sync(), s_store, Sync(),
                        update1, update2, update3, Sync(), o_gemm})));
  Stmt kv_loop = tir::For(iterator, I32(0),
      floordiv(ld0(kv_chunk_buf) + I32(tile_z - 1), I32(tile_z)),
      tir::ForKind::kSerial, kv_iter);

  // O_store
  tir::Var li_os("li", DataType::Int(32)), lj_os("lj", DataType::Int(32));
  tir::Var cur_L_os("cur_L", DataType::Int(32)), cur_H_qo_os("cur_H_qo", DataType::Int(32));
  Stmt o_store = tir::For(li_os, I32(0), I32(tile_x), tir::ForKind::kSerial,
      tir::For(lj_os, I32(0), I32(d_v), tir::ForKind::kSerial,
          tir::SBlockRealize({}, tir::const_true(),
              tir::SBlock({}, {}, {}, "O_store",
                  tir::LetStmt(cur_L_os,
                      tir::BufferLoad(q_indptr_buf, {b_idx}) +
                      floordiv(LH_start + li_os, I32(group_size)),
                      tir::LetStmt(cur_H_qo_os,
                          by * I32(group_size) + floormod(LH_start + li_os, I32(group_size)),
                          tir::IfThenElse(
                              cur_L_os < tir::BufferLoad(q_indptr_buf, {b_idx + I32(1)}),
                              tir::BufferStore(output_buf,
                                  is_f16
                                  ? CastTo(tvm::div(tir::BufferLoad(O_local, ffi::Array<PrimExpr>{li_os, lj_os}),
                                    tir::BufferLoad(d_smem, {li_os})), dtype)
                                  : tvm::div(tir::BufferLoad(O_local, ffi::Array<PrimExpr>{li_os, lj_os}),
                                    tir::BufferLoad(d_smem, {li_os})), ffi::Array<PrimExpr>{cur_L_os, cur_H_qo_os, lj_os}))))))));

  // lse_store
  tir::Var li_lse("li", DataType::Int(32));
  tir::Var cur_L_lse("cur_L", DataType::Int(32)), cur_H_qo_lse("cur_H_qo", DataType::Int(32));
  Stmt lse_store = tir::For(li_lse, I32(0), I32(tile_x), tir::ForKind::kSerial,
      tir::SBlockRealize({}, tir::const_true(),
          tir::SBlock({}, {}, {}, "lse_store",
              tir::LetStmt(cur_L_lse,
                  tir::BufferLoad(q_indptr_buf, {b_idx}) +
                  floordiv(LH_start + li_lse, I32(group_size)),
                  tir::LetStmt(cur_H_qo_lse,
                      by * I32(group_size) + floormod(LH_start + li_lse, I32(group_size)),
                      tir::IfThenElse(
                          cur_L_lse < tir::BufferLoad(q_indptr_buf, {b_idx + I32(1)}),
                          tir::BufferStore(lse_buf,
                              tir::BufferLoad(m_smem, {li_lse}) +
                              tvm::log2(tir::BufferLoad(d_smem, {li_lse})), ffi::Array<PrimExpr>{cur_L_lse, cur_H_qo_lse})))))));

  // tile dispatch while loop
  tir::Var b_idx2("b_idx", DataType::Int(32));
  Stmt advance = tir::SeqStmt({
      st0(tile_id_buf, tir::BufferLoad(tile_id_buf, {I32(0)}) - tir::BufferLoad(batch_tiles_buf, {I32(0)})),
      st0(batch_idx_buf, tir::BufferLoad(batch_idx_buf, {I32(0)}) + I32(1)),
      tir::IfThenElse(tir::BufferLoad(batch_idx_buf, {I32(0)}) < batch_size,
          tir::LetStmt(b_idx2, tir::BufferLoad(batch_idx_buf, {I32(0)}),
              tir::SeqStmt({
                  st0(batch_rows_buf,
                      (tir::BufferLoad(q_indptr_buf, {b_idx2 + I32(1)}) -
                       tir::BufferLoad(q_indptr_buf, {b_idx2})) * I32(group_size)),
                  st0(batch_tiles_buf,
                      floordiv(tir::BufferLoad(batch_rows_buf, {I32(0)}) + I32(tile_x - 1),
                               I32(tile_x)))})))});
  Stmt inner_while = tir::While(
      logical_and(tir::BufferLoad(tile_id_buf, {I32(0)}) >= tir::BufferLoad(batch_tiles_buf, {I32(0)}),
               tir::BufferLoad(batch_idx_buf, {I32(0)}) < batch_size),
      advance);

  Stmt tile_body = tir::LetStmt(b_idx, tir::BufferLoad(batch_idx_buf, {I32(0)}),
      tir::LetStmt(q_indptr_val, tir::BufferLoad(q_indptr_buf, {b_idx}),
          tir::LetStmt(LH_start, tir::BufferLoad(tile_id_buf, {I32(0)}) * I32(tile_x),
              tir::SeqStmt({
                  st0(kv_chunk_buf, kv_chunk_expr),
                  Sync(),
                  init_md, o_init, Sync(),
                  q_load, Sync(),
                  kv_loop,
                  o_store, lse_store,
                  st0(tile_id_buf,
                      tir::BufferLoad(tile_id_buf, {I32(0)}) + I32(NUM_BLKS))}))));

  Stmt outer_while = tir::While(
      tir::Call(DataType::Bool(), tir::builtin::tvm_thread_invariant(),
                ffi::Array<PrimExpr>{tir::BufferLoad(batch_idx_buf, {I32(0)}) < batch_size}),
      tir::SeqStmt({inner_while,
                    tir::IfThenElse(
                        tir::Call(DataType::Bool(), tir::builtin::tvm_thread_invariant(),
                                  ffi::Array<PrimExpr>{
                                      tir::BufferLoad(batch_idx_buf, {I32(0)}) < batch_size}),
                        tile_body)}));

  Stmt sblock_init = tir::SeqStmt({
      st0(tile_id_buf, bx),
      st0(batch_idx_buf, I32(0)),
      st0(batch_rows_buf,
          (tir::BufferLoad(q_indptr_buf, {I32(1)}) -
           tir::BufferLoad(q_indptr_buf, {I32(0)})) * I32(group_size)),
      st0(batch_tiles_buf,
          floordiv(tir::BufferLoad(batch_rows_buf, {I32(0)}) + I32(tile_x - 1), I32(tile_x)))});

  ffi::Array<tir::Buffer> alloc_bufs = {
      tile_id_buf, batch_idx_buf, batch_tiles_buf, batch_rows_buf, dummy_buf, kv_chunk_buf,
      Q_smem, K_smem, V_smem, S_smem, S_local, O_local,
      m_smem, m_prev_smem, d_smem, m_new_buf, m_prev_buf, d_new_buf};
  IterVar bx_iv(Range(nullptr), tir::Var("iter", DataType::Int(32)), tir::kThreadIndex, "blockIdx.x");
  IterVar by_iv(Range(nullptr), tir::Var("iter", DataType::Int(32)), tir::kThreadIndex, "blockIdx.y");
  IterVar ty_iv(Range(nullptr), tir::Var("iter", DataType::Int(32)), tir::kThreadIndex, "threadIdx.y");
  IterVar tx_iv(Range(nullptr), tir::Var("iter", DataType::Int(32)), tir::kThreadIndex, "threadIdx.x");

  Stmt body = tir::SBlockRealize({}, tir::const_true(),
      tir::SBlock({}, {}, {}, "root",
          tir::For(lbx, I32(0), I32(NUM_BLKS), tir::ForKind::kThreadBinding,
              tir::For(lby, I32(0), I32(h_kv), tir::ForKind::kThreadBinding,
                  tir::For(lty, I32(0), I32(num_warps), tir::ForKind::kThreadBinding,
                      tir::For(ltx, I32(0), I32(bdx), tir::ForKind::kThreadBinding,
                          tir::SBlockRealize({}, tir::const_true(),
                              tir::SBlock({}, {}, {}, "attn",
                                  tir::LetStmt(bx, lbx,
                                      tir::LetStmt(by, lby,
                                          tir::LetStmt(ty, lty,
                                              tir::LetStmt(tx, ltx,
                                                  tir::SeqStmt({sblock_init, outer_while}))))),
                                  std::nullopt, alloc_bufs, {}, {})),
                          tx_iv), ty_iv), by_iv), bx_iv)));

  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_q_h, q_buf);
  buf_map.Set(h_q_indptr_h, q_indptr_buf);
  buf_map.Set(h_k_h, k_buf);
  buf_map.Set(h_v_h, v_buf);
  buf_map.Set(h_kv_indptr_h, kv_indptr_buf);
  buf_map.Set(h_q_rope_pos_h, q_rope_pos_buf);
  buf_map.Set(h_k_rope_pos_h, k_rope_pos_buf);
  buf_map.Set(h_output_h, output_buf);
  buf_map.Set(h_lse_h, lse_buf);

  ffi::Array<tir::Var> params = {
      h_q_h, h_q_indptr_h, h_k_h, h_v_h, h_kv_indptr_h,
      h_q_rope_pos_h, h_k_rope_pos_h,
      h_output_h, h_lse_h, causal, rotary_mode, rope_scale, rope_theta, sm_scale};

  tir::PrimFunc fn(params, body, VoidType(), buf_map);
  fn = tir::ScriptComplete(fn, {});
  // transform_k_load=true: K_smem write layout is transposed (j,i) -> (i,j)
  return SchedulePrefillKernel(fn, cfg, /*transform_k_load=*/true, /*merged_qk_load=*/false);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.frontend.nn.llm.kv_cache.attention_prefill_ragged",
      [](int64_t h_kv, int64_t h_q, int64_t d_qk, int64_t d_v, ffi::String dtype,
         ffi::Map<ffi::String, ffi::Any> rope_scaling, Target target) -> tir::PrimFunc {
        return AttentionPrefillRagged(h_kv, h_q, d_qk, d_v, std::string(dtype),
                                      rope_scaling, target);
      });
}

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
