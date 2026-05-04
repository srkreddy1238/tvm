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
 * \file src/relax/frontend/nn/llm/kv_cache_attention_decode_gpu.cc
 * \brief GPU TIR kernel for batched decode with paged KV cache.
 *        C++ port of _attention_decode() in kv_cache.py.
 */

#include <tvm/s_tir/stmt.h>

#include "../../../../tir/ir/script/script_complete.h"
#include "kv_cache.h"
#include "kv_cache_attention_decode_gpu_helpers.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

using namespace tvm::tir;

static tir::Buffer AllocLocal(const std::string& name, ffi::Array<PrimExpr> shape,
                              const std::string& dtype) {
  DataType dt = DataType(runtime::StringToDLDataType(dtype));
  return tir::decl_buffer(shape, dt, name, "local");
}

static tir::Buffer AllocShared(const std::string& name, ffi::Array<PrimExpr> shape,
                               const std::string& dtype) {
  DataType dt = DataType(runtime::StringToDLDataType(dtype));
  return tir::decl_buffer(shape, dt, name, "shared");
}

tir::PrimFunc AttentionDecode(int64_t num_kv_heads, int64_t num_qo_heads, int64_t head_dim,
                              const std::string& qkv_dtype, bool sliding_window,
                              const ffi::Map<ffi::String, ffi::Any>& rope_scaling, Target target,
                              int64_t page_size) {
  const DecodeGpuConfig cfg = ComputeDecodeGpuConfig(num_kv_heads, num_qo_heads, head_dim,
                                                     qkv_dtype, sliding_window, target);

  const int64_t H_kv = num_kv_heads, H_qo = num_qo_heads, D = head_dim;
  const int64_t GROUP_SIZE = cfg.GROUP_SIZE, VEC_SIZE = cfg.VEC_SIZE;
  const int64_t bdx = cfg.bdx, bdy = cfg.bdy, bdz = cfg.bdz, gdz = cfg.gdz;
  const int64_t tile = cfg.tile_size_per_bdx;
  const bool is_f16 = (qkv_dtype == "float16");
  DataType dt = DataType(runtime::StringToDLDataType(qkv_dtype));

  tir::SizeVar B_var("B", DataType::Int(32));
  tir::SizeVar nnz_pages("nnz_pages", DataType::Int(32));
  tir::SizeVar max_num_pages("max_num_pages", DataType::Int(32));
  tir::SizeVar pages_elem_offset("pages_elem_offset", DataType::Int(64));
  tir::SizeVar page_indptr_eo("page_indptr_elem_offset", DataType::Int(32));
  tir::SizeVar page_values_eo("page_values_elem_offset", DataType::Int(32));
  tir::SizeVar k_rope_pos_offset_eo("k_rope_pos_offset_elem_offset", DataType::Int(32));
  tir::SizeVar q_rope_position_eo("q_rope_position_elem_offset", DataType::Int(32));
  tir::SizeVar length_info_eo("length_info_elem_offset", DataType::Int(32));

  tir::Var h_Q("Q_handle", DataType::Handle());
  tir::Var h_pages("pages_handle", DataType::Handle());
  tir::Var h_page_indptr("page_table_indptr_handle", DataType::Handle());
  tir::Var h_page_values("page_table_values_handle", DataType::Handle());
  tir::Var h_length_info("var_length_info", DataType::Handle());
  tir::Var h_k_rope_pos_offset("k_rope_pos_offset_handle", DataType::Handle());
  tir::Var h_q_rope_position("q_rope_position_handle", DataType::Handle());
  tir::Var h_output("output_handle", DataType::Handle());
  tir::Var h_lse("lse_handle", DataType::Handle());
  tir::Var rotary_mode("rotary_mode", DataType::Int(32));
  tir::Var rope_scale("rope_scale", DataType::Float(32));
  tir::Var rope_theta("rope_theta", DataType::Float(32));
  tir::Var sm_scale("sm_scale", DataType::Float(32));

  tir::Buffer Q_buf = tir::decl_buffer({B_var, I32(H_qo), I32(D)}, dt, "Q");
  tir::Buffer pages_buf = tir::Buffer(
      tir::decl_buffer({max_num_pages, I32(2), I32(H_kv), I32(page_size), I32(D)}, dt, "pages")
          ->data,
      dt, {max_num_pages, I32(2), I32(H_kv), I32(page_size), I32(D)}, {}, pages_elem_offset,
      "pages", 0, 0, tir::kDefault);
  tir::Buffer page_indptr_buf =
      tir::Buffer(tir::decl_buffer({B_var + I32(1)}, DataType::Int(32), "page_table_indptr")->data,
                  DataType::Int(32), {B_var + I32(1)}, {}, page_indptr_eo, "page_table_indptr", 0,
                  0, tir::kDefault);
  tir::Buffer page_values_buf = tir::Buffer(
      tir::decl_buffer({nnz_pages}, DataType::Int(32), "page_table_values")->data,
      DataType::Int(32), {nnz_pages}, {}, page_values_eo, "page_table_values", 0, 0, tir::kDefault);
  tir::Buffer length_info_buf;
  if (sliding_window) {
    length_info_buf = tir::Buffer(
        tir::decl_buffer({I32(3), B_var}, DataType::Int(32), "length_info")->data,
        DataType::Int(32), {I32(3), B_var}, {}, length_info_eo, "length_info", 0, 0, tir::kDefault);
  } else {
    length_info_buf = tir::Buffer(tir::decl_buffer({B_var}, DataType::Int(32), "length_info")->data,
                                  DataType::Int(32), {B_var}, {}, length_info_eo, "length_info", 0,
                                  0, tir::kDefault);
  }
  tir::Buffer k_rope_pos_offset_buf = tir::Buffer(
      tir::decl_buffer({B_var}, DataType::Int(32), "k_rope_pos_offset")->data, DataType::Int(32),
      {B_var}, {}, k_rope_pos_offset_eo, "k_rope_pos_offset", 0, 0, tir::kDefault);
  tir::Buffer q_rope_position_buf = tir::Buffer(
      tir::decl_buffer({B_var}, DataType::Int(32), "q_rope_position")->data, DataType::Int(32),
      {B_var}, {}, q_rope_position_eo, "q_rope_position", 0, 0, tir::kDefault);
  tir::Buffer output_buf = tir::decl_buffer({B_var, I32(H_qo), I32(D)}, dt, "output");
  tir::Buffer lse_buf = tir::decl_buffer({B_var, I32(H_qo)}, DataType::Float(32), "lse");

  tir::Var bx("bx", DataType::Int(32));
  tir::Var fused_by_bz("fused_by_bz", DataType::Int(32));
  tir::Var ty("ty", DataType::Int(32));
  tir::Var tx("tx", DataType::Int(32));
  tir::Var tz("tz", DataType::Int(32));

  tir::Buffer Q_local = AllocLocal("Q_local", {I32(VEC_SIZE)}, qkv_dtype);
  tir::Buffer kv_chunk_len = AllocLocal("kv_chunk_len", {I32(1)}, "int32");
  tir::Buffer K_smem = AllocShared("K_smem", {I32(bdz * bdy * tile), I32(D)}, qkv_dtype);
  tir::Buffer V_smem = AllocShared("V_smem", {I32(bdz * bdy * tile), I32(D)}, qkv_dtype);
  tir::Buffer O_allreduce = AllocShared("O_allreduce", {I32(bdz), I32(bdy), I32(D)}, "float32");
  tir::Buffer md_allreduce = AllocShared("md_allreduce", {I32(bdz), I32(bdy), I32(2)}, "float32");
  tir::Buffer S_reduce_local = AllocLocal("S_reduce_local", {I32(1)}, "float32");
  tir::Buffer t0 = AllocLocal("t0", {I32(1)}, "float32");
  tir::Buffer S_local = AllocLocal("S_local", {I32(bdy * tile)}, "float32");
  tir::Buffer QK_local = AllocLocal("QK_local", {I32(VEC_SIZE)}, "float32");
  tir::Buffer V_local = AllocLocal("V_local", {I32(VEC_SIZE)}, qkv_dtype);
  tir::Buffer m_prev_buf = AllocLocal("m_prev", {I32(1)}, "float32");
  tir::Buffer d_prev_buf = AllocLocal("d_prev", {I32(1)}, "float32");
  tir::Buffer other_m_buf = AllocLocal("other_m", {I32(1)}, "float32");
  tir::Buffer other_d_buf = AllocLocal("other_d", {I32(1)}, "float32");
  tir::Buffer exp_mprev_buf = AllocLocal("exp_mprev", {I32(1)}, "float32");
  tir::Buffer exp_otherm_buf = AllocLocal("exp_otherm", {I32(1)}, "float32");
  tir::Buffer other_o_buf = AllocLocal("other_o", {I32(VEC_SIZE)}, "float32");
  tir::Buffer st_m_buf = AllocLocal("st_m", {I32(1)}, "float32");
  tir::Buffer st_d_buf = AllocLocal("st_d", {I32(1)}, "float32");
  tir::Buffer O_local_buf = AllocLocal("O_local", {I32(VEC_SIZE)}, "float32");

  auto ld0 = [](tir::Buffer b) { return tir::BufferLoad(b, {I32(0)}); };
  auto st0 = [](tir::Buffer b, PrimExpr v) { return tir::BufferStore(b, v, {I32(0)}); };
  auto ldi = [](tir::Buffer b, PrimExpr i) { return tir::BufferLoad(b, {i}); };
  auto sti = [](tir::Buffer b, PrimExpr i, PrimExpr v) { return tir::BufferStore(b, v, {i}); };

  tir::Var by_var("by", DataType::Int(32));
  tir::Var bz_var("bz", DataType::Int(32));
  tir::Var batch_idx("batch_idx", DataType::Int(32));
  tir::Var cur_begin("cur_page_indptr_begin", DataType::Int(32));
  tir::Var cur_end("cur_page_indptr_end", DataType::Int(32));

  PrimExpr num_pages = cur_end - cur_begin;
  PrimExpr last_page_len_expr;
  if (sliding_window) {
    last_page_len_expr = (num_pages - I32(1)) * I32(page_size) +
                         tir::BufferLoad(length_info_buf, {I32(0), batch_idx}) -
                         tir::BufferLoad(length_info_buf, {I32(1), batch_idx}) +
                         tir::BufferLoad(length_info_buf, {I32(2), batch_idx});
  } else {
    last_page_len_expr =
        (num_pages - I32(1)) * I32(page_size) + tir::BufferLoad(length_info_buf, {batch_idx});
  }
  PrimExpr kv_len_expr = tvm::if_then_else(cur_begin != cur_end, last_page_len_expr, I32(0));
  PrimExpr h_qo_idx = by_var * I32(GROUP_SIZE) + bz_var * I32(bdy) + ty;

  tir::Var vec_init("vec", DataType::Int(32));
  Stmt init_states =
      tir::SeqStmt({st0(st_m_buf, F32(-50000.0)), st0(st_d_buf, F32(1.0)),
                    tir::For(vec_init, I32(0), I32(VEC_SIZE), tir::ForKind::kVectorized,
                             sti(O_local_buf, vec_init, F32(0.0)))});

  tir::Var vec_q("vec", DataType::Int(32));
  PrimExpr d_q = tx * I32(VEC_SIZE) + vec_q;
  PrimExpr q_pos = tir::BufferLoad(q_rope_position_buf, {batch_idx});
  PrimExpr q_rope_expr = BuildDecodeRopeExpr(Q_buf, {bx, h_qo_idx}, d_q, D, q_pos, rope_scale,
                                             rope_theta, rotary_mode, qkv_dtype, rope_scaling);
  Stmt q_load = tir::For(vec_q, I32(0), I32(VEC_SIZE), tir::ForKind::kVectorized,
                         sti(Q_local, vec_q, q_rope_expr));

  tir::Var iterator("iterator", DataType::Int(32));
  tir::SizeVar tile_start_s_var("tile_start_s", DataType::Int(32));
  tir::SizeVar tile_start_g_var("tile_start_g", DataType::Int(32));
  PrimExpr tile_start_s_expr = (tz * I32(bdy) + ty) * I32(tile);
  PrimExpr tile_start_g_expr = ((iterator * I32(bdz) + tz) * I32(bdy) + ty) * I32(tile);

  tir::Var j_kv("j", DataType::Int(32));
  Stmt kv_load_body = BuildDecodeKVLoadBody(
      cfg, pages_buf, page_values_buf, length_info_buf, k_rope_pos_offset_buf, K_smem, V_smem,
      kv_chunk_len, cur_begin, batch_idx, by_var, tx, tile_start_s_var, tile_start_g_var, j_kv,
      rope_scale, rope_theta, rotary_mode, rope_scaling);
  Stmt kv_load_sblock =
      tir::SBlockRealize({}, tir::const_true(), tir::SBlock({}, {}, {}, "KV_load", kv_load_body));
  Stmt j_kv_loop = tir::For(j_kv, I32(0), I32(tile), tir::ForKind::kSerial, kv_load_sblock);

  Stmt sync_shared = tir::Evaluate(tir::Call(DataType::Int(32), tir::builtin::tvm_storage_sync(),
                                             ffi::Array<PrimExpr>{tir::StringImm("shared")}));

  Stmt save_m_prev = st0(m_prev_buf, ld0(st_m_buf));
  tir::Var j_qk("j", DataType::Int(32));
  const double log2e = 1.4426950408889634;
  tir::Var vec_qk("vec", DataType::Int(32));
  PrimExpr d_qk_idx = tx * I32(VEC_SIZE) + vec_qk;
  PrimExpr k_smem_elem = tir::BufferLoad(K_smem, {tz * I32(bdy * tile) + j_qk, d_qk_idx});
  PrimExpr qk_elem = is_f16 ? CastTo(ldi(Q_local, vec_qk), "float32") *
                                  CastTo(k_smem_elem, "float32") * sm_scale * F32(log2e)
                            : ldi(Q_local, vec_qk) * k_smem_elem * sm_scale * F32(log2e);
  Stmt qk_vec_loop = tir::For(vec_qk, I32(0), I32(VEC_SIZE), tir::ForKind::kVectorized,
                              sti(QK_local, vec_qk, qk_elem));
  tir::Var vec_red("vec", DataType::Int(32));
  Stmt s_reduce_init = st0(S_reduce_local, F32(0.0));
  Stmt s_reduce_loop = tir::For(vec_red, I32(0), I32(VEC_SIZE), tir::ForKind::kUnrolled,
                                st0(S_reduce_local, ld0(S_reduce_local) + ldi(QK_local, vec_red)));

  tir::Var x0("x0", DataType::Float(32)), y0("y0", DataType::Float(32));
  tir::CommReducer add_reducer(ffi::Array<Var>{x0}, ffi::Array<Var>{y0},
                               ffi::Array<PrimExpr>{x0 + y0}, ffi::Array<PrimExpr>{F32(0.0)});
  Stmt allreduce_sblock = tir::SBlockRealize(
      {}, tir::const_true(),
      tir::SBlock(
          {}, {tir::BufferRegion(S_reduce_local, {Range::FromMinExtent(I32(0), I32(1))})},
          {tir::BufferRegion(t0, {Range::FromMinExtent(I32(0), I32(1))})}, "block_cross_thread",
          tir::AttrStmt(add_reducer, "reduce_scope",
                        tvm::reinterpret(DataType::Handle(), IntImm(DataType::UInt(64), 0)),
                        tir::Evaluate(tir::Call(
                            DataType::Handle(), tir::builtin::tvm_thread_allreduce(),
                            ffi::Array<PrimExpr>{(PrimExpr)IntImm(DataType::UInt(32), 1),
                                                 ld0(S_reduce_local), (PrimExpr)tir::const_true(),
                                                 tir::BufferLoad(t0, {I32(0)}), (PrimExpr)tx})))));

  PrimExpr s_cond = (iterator * I32(bdz) + tz) * I32(bdy * tile) + j_qk < ld0(kv_chunk_len);
  Stmt s_update = tir::SeqStmt({sti(S_local, j_qk, F32(-50000.0)),
                                tir::IfThenElse(s_cond, sti(S_local, j_qk, ld0(t0))),
                                st0(st_m_buf, tvm::max(ld0(st_m_buf), ldi(S_local, j_qk)))});
  Stmt j_qk_body =
      tir::SeqStmt({qk_vec_loop, s_reduce_init, s_reduce_loop, allreduce_sblock, s_update});
  Stmt j_qk_loop = tir::For(j_qk, I32(0), I32(bdy * tile), tir::ForKind::kSerial, j_qk_body);

  tir::Var o_scale_var("o_scale", DataType::Float(32));
  PrimExpr o_scale_expr = tvm::exp2(ld0(m_prev_buf) - ld0(st_m_buf));
  tir::Var j_s("j", DataType::Int(32));
  Stmt s_exp_loop =
      tir::For(j_s, I32(0), I32(bdy * tile), tir::ForKind::kSerial,
               tir::SeqStmt({sti(S_local, j_s, tvm::exp2(ldi(S_local, j_s) - ld0(st_m_buf))),
                             st0(st_d_buf, ld0(st_d_buf) + ldi(S_local, j_s))}));
  tir::Var vec_oscale("j", DataType::Int(32));
  Stmt o_scale_loop =
      tir::For(vec_oscale, I32(0), I32(VEC_SIZE), tir::ForKind::kVectorized,
               sti(O_local_buf, vec_oscale, ldi(O_local_buf, vec_oscale) * o_scale_var));
  tir::Var j_v("j", DataType::Int(32));
  tir::Var vec_vl("vec", DataType::Int(32));
  tir::Var vec_oa("vec", DataType::Int(32));
  PrimExpr v_smem_elem =
      tir::BufferLoad(V_smem, {tz * I32(bdy * tile) + j_v, tx * I32(VEC_SIZE) + vec_vl});
  PrimExpr o_acc = is_f16 ? ldi(O_local_buf, vec_oa) +
                                CastTo(ldi(V_local, vec_oa), "float32") * ldi(S_local, j_v)
                          : ldi(O_local_buf, vec_oa) + ldi(V_local, vec_oa) * ldi(S_local, j_v);
  Stmt v_load_loop = tir::For(vec_vl, I32(0), I32(VEC_SIZE), tir::ForKind::kVectorized,
                              sti(V_local, vec_vl, v_smem_elem));
  Stmt o_acc_loop = tir::For(vec_oa, I32(0), I32(VEC_SIZE), tir::ForKind::kVectorized,
                             sti(O_local_buf, vec_oa, o_acc));
  Stmt j_v_loop = tir::For(j_v, I32(0), I32(bdy * tile), tir::ForKind::kSerial,
                           tir::SeqStmt({v_load_loop, o_acc_loop}));

  // j_v_loop is inside softmax_update LetStmt body (matches Python TVMScript)
  Stmt softmax_update = tir::LetStmt(o_scale_var, o_scale_expr,
                                     tir::SeqStmt({st0(st_d_buf, ld0(st_d_buf) * o_scale_var),
                                                   s_exp_loop, o_scale_loop, j_v_loop}));

  Stmt iter_body = tir::LetStmt(
      tile_start_s_var, tile_start_s_expr,
      tir::LetStmt(tile_start_g_var, tile_start_g_expr,
                   tir::SeqStmt({j_kv_loop, sync_shared, save_m_prev, j_qk_loop, softmax_update})));
  int64_t tile_total = tile * bdy * bdz;
  PrimExpr iter_extent = floordiv(ld0(kv_chunk_len) + I32(tile_total - 1), I32(tile_total));
  Stmt iter_loop = tir::For(iterator, I32(0), iter_extent, tir::ForKind::kSerial, iter_body);

  Stmt allreduce_block = BuildDecodeAllreduceBlock(
      cfg, O_allreduce, md_allreduce, O_local_buf, st_m_buf, st_d_buf, m_prev_buf, d_prev_buf,
      other_m_buf, other_d_buf, other_o_buf, exp_mprev_buf, exp_otherm_buf, tz, ty, tx);

  tir::Var vec_norm("vec", DataType::Int(32));
  Stmt normalize =
      tir::For(vec_norm, I32(0), I32(VEC_SIZE), tir::ForKind::kVectorized,
               sti(O_local_buf, vec_norm, tvm::div(ldi(O_local_buf, vec_norm), ld0(st_d_buf))));
  tir::Var vec_out("vec", DataType::Int(32));
  PrimExpr out_val =
      is_f16 ? CastTo(ldi(O_local_buf, vec_out), qkv_dtype) : ldi(O_local_buf, vec_out);
  Stmt store_output = tir::For(
      vec_out, I32(0), I32(VEC_SIZE), tir::ForKind::kVectorized,
      tir::BufferStore(output_buf, out_val, {batch_idx, h_qo_idx, tx * I32(VEC_SIZE) + vec_out}));
  Stmt store_lse =
      tir::BufferStore(lse_buf, ld0(st_m_buf) + tvm::log2(ld0(st_d_buf)), {batch_idx, h_qo_idx});

  ffi::Array<tir::Buffer> alloc_bufs = {
      Q_local,        kv_chunk_len, K_smem,      V_smem,        O_allreduce,    md_allreduce,
      S_reduce_local, t0,           S_local,     QK_local,      V_local,        m_prev_buf,
      d_prev_buf,     other_m_buf,  other_d_buf, exp_mprev_buf, exp_otherm_buf, other_o_buf,
      st_m_buf,       st_d_buf,     O_local_buf};

  Stmt main_body;
  if (bdz > 1) {
    main_body = tir::SeqStmt(
        {init_states, q_load, iter_loop, allreduce_block, normalize, store_output, store_lse});
  } else {
    main_body = tir::SeqStmt({init_states, q_load, iter_loop, normalize, store_output, store_lse});
  }

  Stmt sblock_body = tir::LetStmt(
      by_var, floormod(fused_by_bz, I32(H_kv)),
      tir::LetStmt(
          bz_var, floordiv(fused_by_bz, I32(H_kv)),
          tir::LetStmt(
              batch_idx, bx,
              tir::LetStmt(
                  cur_begin, tir::BufferLoad(page_indptr_buf, {batch_idx}),
                  tir::LetStmt(cur_end, tir::BufferLoad(page_indptr_buf, {batch_idx + I32(1)}),
                               tir::SeqStmt({tir::BufferStore(kv_chunk_len, kv_len_expr, {I32(0)}),
                                             main_body}))))));

  // Add script_parsing_detect_access=3 so ScriptComplete fills in reads/writes
  ffi::Map<ffi::String, ffi::Any> attn_annotations;
  attn_annotations.Set(s_tir::attr::script_parsing_detect_access,
                       ffi::Any(IntImm(DataType::Int(32), 3)));
  Stmt sblock = tir::SBlockRealize(
      {}, tir::const_true(),
      tir::SBlock({}, {}, {}, "attn", sblock_body, std::nullopt, alloc_bufs, {}, attn_annotations));

  // Use Range(nullptr) and separate "iter" vars for thread binding IterVars
  // (matches Python TVMScript T.thread_binding() output)
  tir::Var iter_bx("iter", DataType::Int(32));
  tir::Var iter_fused("iter", DataType::Int(32));
  tir::Var iter_ty("iter", DataType::Int(32));
  tir::Var iter_tx("iter", DataType::Int(32));
  tir::Var iter_tz("iter", DataType::Int(32));
  IterVar bx_iv(Range(nullptr), iter_bx, tir::kThreadIndex, "blockIdx.x");
  IterVar fused_iv(Range(nullptr), iter_fused, tir::kThreadIndex, "blockIdx.y");
  IterVar ty_iv(Range(nullptr), iter_ty, tir::kThreadIndex, "threadIdx.y");
  IterVar tx_iv(Range(nullptr), iter_tx, tir::kThreadIndex, "threadIdx.x");
  IterVar tz_iv(Range(nullptr), iter_tz, tir::kThreadIndex, "threadIdx.z");

  Stmt body = sblock;
  body = tir::For(tz, I32(0), I32(bdz), tir::ForKind::kThreadBinding, body, tz_iv);
  body = tir::For(tx, I32(0), I32(bdx), tir::ForKind::kThreadBinding, body, tx_iv);
  body = tir::For(ty, I32(0), I32(bdy), tir::ForKind::kThreadBinding, body, ty_iv);
  body =
      tir::For(fused_by_bz, I32(0), I32(H_kv * gdz), tir::ForKind::kThreadBinding, body, fused_iv);
  body = tir::For(bx, I32(0), B_var, tir::ForKind::kThreadBinding, body, bx_iv);
  // Wrap in root SBlockRealize (matches Python TVMScript ScriptComplete output)
  body = tir::SBlockRealize({}, tir::const_true(), tir::SBlock({}, {}, {}, "root", body));

  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_Q, Q_buf);
  buf_map.Set(h_pages, pages_buf);
  buf_map.Set(h_page_indptr, page_indptr_buf);
  buf_map.Set(h_page_values, page_values_buf);
  buf_map.Set(h_length_info, length_info_buf);
  buf_map.Set(h_k_rope_pos_offset, k_rope_pos_offset_buf);
  buf_map.Set(h_q_rope_position, q_rope_position_buf);
  buf_map.Set(h_output, output_buf);
  buf_map.Set(h_lse, lse_buf);

  ffi::Array<tir::Var> params = {h_Q,
                                 h_pages,
                                 h_page_indptr,
                                 h_page_values,
                                 h_length_info,
                                 h_k_rope_pos_offset,
                                 h_q_rope_position,
                                 h_output,
                                 h_lse,
                                 rotary_mode,
                                 rope_scale,
                                 rope_theta,
                                 sm_scale};

  tir::PrimFunc fn(params, body, VoidType(), buf_map);
  fn = WithAttr(fn, "tir.is_scheduled", ffi::Any(true));
  fn = WithAttr(fn, "global_symbol", ffi::Any(ffi::String(cfg.global_symbol)));
  // Run ScriptComplete to fill in reads/writes annotations (matches Python TVMScript output)
  fn = tir::ScriptComplete(fn, {});
  return fn;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.frontend.nn.llm.kv_cache.attention_decode",
      [](int64_t num_kv_heads, int64_t num_qo_heads, int64_t head_dim, ffi::String qkv_dtype,
         bool sliding_window, ffi::Map<ffi::String, ffi::Any> rope_scaling, Target target,
         int64_t page_size) -> tir::PrimFunc {
        return AttentionDecode(num_kv_heads, num_qo_heads, head_dim, std::string(qkv_dtype),
                               sliding_window, rope_scaling, target, page_size);
      });
}

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
