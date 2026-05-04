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
 * \file src/relax/frontend/nn/llm/kv_cache_attention_prefill_cpu.cc
 * \brief CPU TIR kernel for batched prefill with paged KV cache.
 *
 * Implements AttentionPrefillCpu (_attention_prefill_cpu in Python).
 *
 * Exact expected structure (GQA h_kv=4, h_q=32, d=128, float16, no rope):
 * ---------------------------------------------------------------------------
 * for h_qo, b_idx in T.grid(32, batch_size):
 *   with T.sblock("attn"):
 *     T.reads(...); T.writes(...)
 *     O_local = T.alloc_buffer((128,))
 *     ...
 *     cur_page_indptr_begin: T.int32 = page_indptr[b_idx]
 *     cur_page_indptr_end:   T.int32 = page_indptr[b_idx + 1]
 *     kv_chunk_len[0] = T.if_then_else(begin != end, (end-begin-1)*16 + length_info[b], 0)
 *     for q_idx in range(q_indptr[b_idx+1] - q_indptr[b_idx]):
 *       m_val[0] = -50000; d_val[0] = 1.0
 *       for d_idx in range(128): O_local[d_idx] = 0.0
 *       curl_q: T.int32 = q_indptr[b_idx] + q_idx
 *       for d_idx in range(128):
 *         freq = T.float32()
 *         Q_local[d_idx] = cast<f32>(if_then_else(rotary_mode==1, Let(...), q[curl_q,h_qo,d_idx]))
 *       for row_idx in range(max_num_pages * 16):
 *         if row_idx < kv_chunk_len[0]:
 *           page_no: T.int32(is_size_var=True) = page_values[begin + row_idx//16]
 *           page_offset: T.int32(is_size_var=True) = row_idx % 16
 *           for d_idx in range(128):
 *             freq = T.float32()
 *             K_local[d_idx] = cast<f32>(if_then_else(rotary_mode==1, Let(...), pages[...]))
 *             V_local[d_idx] = cast<f32>(pages[page_no, 1, h_qo//group, page_offset, d_idx])
 *           S_val[0] = 0.0
 *           for d_idx in range(128): S_val[0] += Q_local[d_idx] * K_local[d_idx]
 *           S_val[0] *= sm_scale * log2e
 *           if causal_cond: new_m[0] = max(m_val[0], S_val[0])
 *           else:           S_val[0] = -50000
 *           d_val[0] = d_val[0]*exp2(m-new_m) + exp2(S-new_m)
 *           scale_O[0] = exp2(m-new_m); m_val[0] = new_m[0]
 *           factor[0] = exp2(S-m)
 *           for d_idx: O_local[d_idx] *= scale_O[d_idx]
 *           for d_idx: O_local[d_idx] += V_local[d_idx] * factor[0]
 *       for d_idx: O_local[d_idx] /= d_val[0]; output[curl_q,h_qo,d_idx] = cast(O_local[d_idx])
 *       lse[curl_q, h_qo] = m_val[0] + log2(d_val[0])
 * ---------------------------------------------------------------------------
 */

#include "../../../../tir/ir/script/script_complete.h"
#include "kv_cache.h"
#include "kv_cache_attn_common.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

using namespace tvm::tir;

// ---------------------------------------------------------------------------
// Helper: build RoPE Let-expression matching TVMScript pattern exactly.
//
// For float16:
//   T.if_then_else(rotary_mode == 1,
//     T.Let(cast<float16>(cos(freq)*cast<f32>(elem) + sin(freq)*cast<f32>(partner)),
//           where={freq: cast<f32>(pos)*rope_scale / pow(rope_theta, cast<f32>(d*2%D)/D)}),
//     elem)
//
// For float32:
//   T.if_then_else(rotary_mode == 1,
//     T.Let(cos(freq)*elem + sin(freq)*partner,
//           where={freq: cast<f32>(pos)*rope_scale / pow(rope_theta, cast<f32>(d*2%D)/D)}),
//     elem)
// ---------------------------------------------------------------------------
static PrimExpr BuildRopeLetExpr(PrimExpr elem, PrimExpr partner, PrimExpr pos_i32, tir::Var d_var,
                                 int64_t D, tir::Var rope_scale, tir::Var rope_theta,
                                 tir::Var rotary_mode, const std::string& dtype,
                                 const ffi::Map<ffi::String, ffi::Any>& rope_scaling) {
  bool is_f16 = (dtype == "float16");
  (void)DataType(runtime::StringToDLDataType(dtype));

  RopeFreqFunc rope_freq_func = SwitchRopeFreqFunc(rope_scaling);
  PrimExpr pos_f32 = CastTo(pos_i32, "float32") * rope_scale;
  // Always compute cos/sin in float32 to match Python behavior
  auto freq_result = rope_freq_func(pos_f32, d_var, D, rope_theta, "float32", rope_scaling);

  PrimExpr rope_val;
  if (is_f16) {
    // cos/sin are float32; cast result to dtype (float16)
    rope_val = CastTo(freq_result.cos_freq * CastTo(elem, "float32") +
                          freq_result.sin_freq * CastTo(partner, "float32"),
                      dtype);
  } else {
    rope_val = freq_result.cos_freq * elem + freq_result.sin_freq * partner;
  }

  // Wrap in Let bindings (innermost first)
  for (auto it = freq_result.var_map.rbegin(); it != freq_result.var_map.rend(); ++it) {
    rope_val = tir::Let(it->first, it->second, rope_val);
  }

  return tvm::if_then_else(rotary_mode == I32(1), rope_val, elem);
}

// ---------------------------------------------------------------------------
// Helper: build partner element for standard RoPE
//   if d < D/2: buf[..., d + D/2] * (-1)  else: buf[..., d - D/2]
// ---------------------------------------------------------------------------
static PrimExpr BuildPartner(tir::Buffer buf, ffi::Array<PrimExpr> base_idx, tir::Var d_var,
                             int64_t D, const std::string& dtype) {
  DataType dt = DataType(runtime::StringToDLDataType(dtype));
  int64_t half = D / 2;
  PrimExpr neg_one = tir::make_const(dt, -1.0);

  ffi::Array<PrimExpr> idx_plus = base_idx;
  idx_plus.push_back(d_var + I32(half));
  ffi::Array<PrimExpr> idx_minus = base_idx;
  idx_minus.push_back(d_var - I32(half));

  return tvm::if_then_else(d_var < I32(half), tir::BufferLoad(buf, idx_plus) * neg_one,
                           tir::BufferLoad(buf, idx_minus));
}

// ---------------------------------------------------------------------------
// AttentionPrefillCpu
// ---------------------------------------------------------------------------

tir::PrimFunc AttentionPrefillCpu(int64_t h_kv, int64_t h_q, int64_t d, const std::string& dtype,
                                  bool sliding_window,
                                  const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
                                  int64_t page_size) {
  bool is_f16 = (dtype == "float16");
  DataType dt = DataType(runtime::StringToDLDataType(dtype));
  int64_t group = h_q / h_kv;

  // ── symbolic vars ──────────────────────────────────────────────────────────
  tir::SizeVar total_len("total_len", DataType::Int(32));
  tir::SizeVar batch_size("batch_size", DataType::Int(32));
  tir::SizeVar nnz_pages("nnz_pages", DataType::Int(32));
  tir::SizeVar max_num_pages("max_num_pages", DataType::Int(32));
  tir::SizeVar q_indptr_eo("q_indptr_elem_offset", DataType::Int(32));
  tir::SizeVar page_indptr_eo("page_indptr_elem_offset", DataType::Int(32));
  tir::SizeVar page_values_eo("page_values_elem_offset", DataType::Int(32));
  tir::SizeVar k_rope_pos_offset_eo("k_rope_pos_offset_elem_offset", DataType::Int(32));
  tir::SizeVar q_rope_position_eo("q_rope_position_elem_offset", DataType::Int(32));
  tir::SizeVar length_info_eo("length_info_elem_offset", DataType::Int(32));

  // ── handle params ──────────────────────────────────────────────────────────
  tir::Var h_q_hdl("var_q", DataType::Handle());
  tir::Var h_q_indptr("var_q_indptr", DataType::Handle());
  tir::Var h_pages("var_pages", DataType::Handle());
  tir::Var h_page_indptr("var_page_indptr", DataType::Handle());
  tir::Var h_page_values("var_page_values", DataType::Handle());
  tir::Var h_length_info("var_length_info", DataType::Handle());
  tir::Var h_k_rope_pos_offset("var_k_rope_pos_offset", DataType::Handle());
  tir::Var h_q_rope_position("var_q_rope_position", DataType::Handle());
  tir::Var h_output("var_output", DataType::Handle());
  tir::Var h_lse("var_lse", DataType::Handle());
  tir::Var causal("causal", DataType::Int(32));
  tir::Var rotary_mode("rotary_mode", DataType::Int(32));
  tir::Var rope_scale("rope_scale", DataType::Float(32));
  tir::Var rope_theta("rope_theta", DataType::Float(32));
  tir::Var sm_scale("sm_scale", DataType::Float(32));

  // ── buffers ────────────────────────────────────────────────────────────────
  tir::Buffer q_buf = tir::decl_buffer({total_len, I32(h_q), I32(d)}, dt, "q");

  tir::Buffer q_indptr_buf = tir::Buffer(
      tir::decl_buffer({batch_size + I32(1)}, DataType::Int(32), "q_indptr")->data,
      DataType::Int(32), {batch_size + I32(1)}, {}, q_indptr_eo, "q_indptr", 0, 0, tir::kDefault);

  // pages: no elem_offset for CPU
  tir::Buffer pages_buf =
      tir::decl_buffer({max_num_pages, I32(2), I32(h_kv), I32(page_size), I32(d)}, dt, "pages");

  tir::Buffer page_indptr_buf =
      tir::Buffer(tir::decl_buffer({batch_size + I32(1)}, DataType::Int(32), "page_indptr")->data,
                  DataType::Int(32), {batch_size + I32(1)}, {}, page_indptr_eo, "page_indptr", 0, 0,
                  tir::kDefault);

  tir::Buffer page_values_buf = tir::Buffer(
      tir::decl_buffer({nnz_pages}, DataType::Int(32), "page_values")->data, DataType::Int(32),
      {nnz_pages}, {}, page_values_eo, "page_values", 0, 0, tir::kDefault);

  tir::Buffer length_info_buf = tir::Buffer(
      tir::decl_buffer({batch_size}, DataType::Int(32), "length_info")->data, DataType::Int(32),
      {batch_size}, {}, length_info_eo, "length_info", 0, 0, tir::kDefault);

  tir::Buffer k_rope_pos_offset_buf =
      tir::Buffer(tir::decl_buffer({batch_size}, DataType::Int(32), "k_rope_pos_offset")->data,
                  DataType::Int(32), {batch_size}, {}, k_rope_pos_offset_eo, "k_rope_pos_offset", 0,
                  0, tir::kDefault);

  tir::Buffer q_rope_position_buf = tir::Buffer(
      tir::decl_buffer({total_len}, DataType::Int(32), "q_rope_position")->data, DataType::Int(32),
      {total_len}, {}, q_rope_position_eo, "q_rope_position", 0, 0, tir::kDefault);

  tir::Buffer output_buf = tir::decl_buffer({total_len, I32(h_q), I32(d)}, dt, "output");
  tir::Buffer lse_buf = tir::decl_buffer({total_len, I32(h_q)}, DataType::Float(32), "lse");

  // ── outer loop vars ────────────────────────────────────────────────────────
  tir::Var h_qo("h_qo", DataType::Int(32));
  tir::Var b_idx("b_idx", DataType::Int(32));

  // ── sblock alloc buffers ───────────────────────────────────────────────────
  tir::Buffer O_local = tir::decl_buffer({I32(d)}, DataType::Float(32), "O_local");
  tir::Buffer Q_local = tir::decl_buffer({I32(d)}, DataType::Float(32), "Q_local");
  tir::Buffer K_local = tir::decl_buffer({I32(d)}, DataType::Float(32), "K_local");
  tir::Buffer V_local = tir::decl_buffer({I32(d)}, DataType::Float(32), "V_local");
  tir::Buffer kv_chunk_len_buf = tir::decl_buffer({I32(1)}, DataType::Int(32), "kv_chunk_len");
  tir::Buffer m_val_buf = tir::decl_buffer({I32(1)}, DataType::Float(32), "m_val");
  tir::Buffer new_m_buf = tir::decl_buffer({I32(1)}, DataType::Float(32), "new_m");
  tir::Buffer d_val_buf = tir::decl_buffer({I32(1)}, DataType::Float(32), "d_val");
  tir::Buffer S_val_buf = tir::decl_buffer({I32(1)}, DataType::Float(32), "S_val");
  tir::Buffer scale_O_buf = tir::decl_buffer({I32(1)}, DataType::Float(32), "scale_O");
  tir::Buffer factor_buf = tir::decl_buffer({I32(1)}, DataType::Float(32), "factor");

  auto ld0 = [](tir::Buffer buf) { return tir::BufferLoad(buf, {I32(0)}); };
  auto st0 = [](tir::Buffer buf, PrimExpr val) { return tir::BufferStore(buf, val, {I32(0)}); };
  auto ldi = [](tir::Buffer buf, PrimExpr i) { return tir::BufferLoad(buf, {i}); };
  auto sti = [](tir::Buffer buf, PrimExpr i, PrimExpr val) {
    return tir::BufferStore(buf, val, {i});
  };

  auto m_val = [&]() { return ld0(m_val_buf); };
  auto new_m = [&]() { return ld0(new_m_buf); };
  auto d_val = [&]() { return ld0(d_val_buf); };
  auto S_val = [&]() { return ld0(S_val_buf); };
  auto kv_len = [&]() { return ld0(kv_chunk_len_buf); };

  // ── let-bound vars inside sblock ──────────────────────────────────────────
  tir::Var cur_begin("cur_page_indptr_begin", DataType::Int(32));
  tir::Var cur_end("cur_page_indptr_end", DataType::Int(32));

  PrimExpr kv_len_expr = tvm::if_then_else(
      cur_begin != cur_end,
      (cur_end - cur_begin - I32(1)) * I32(page_size) + tir::BufferLoad(length_info_buf, {b_idx}),
      I32(0));

  // ── inner loop vars ────────────────────────────────────────────────────────
  tir::Var q_idx("q_idx", DataType::Int(32));
  tir::Var row_idx("row_idx", DataType::Int(32));
  tir::Var curl_q("curl_q", DataType::Int(32));
  tir::SizeVar page_no("page_no", DataType::Int(32));
  tir::SizeVar page_offset("page_offset", DataType::Int(32));

  PrimExpr h_kv_idx = floordiv(h_qo, I32(group));

  // ── Q load loop ────────────────────────────────────────────────────────────
  auto build_q_val = [&](tir::Var dv) -> PrimExpr {
    PrimExpr elem = tir::BufferLoad(q_buf, {curl_q, h_qo, dv});
    PrimExpr partner = BuildPartner(q_buf, {curl_q, h_qo}, dv, d, dtype);
    PrimExpr pos = tir::BufferLoad(q_rope_position_buf, {curl_q});
    PrimExpr rope_expr = BuildRopeLetExpr(elem, partner, pos, dv, d, rope_scale, rope_theta,
                                          rotary_mode, dtype, rope_scaling);
    return is_f16 ? CastTo(rope_expr, "float32") : rope_expr;
  };

  tir::Var dq("d_idx", DataType::Int(32));
  Stmt q_load_loop =
      tir::For(dq, I32(0), I32(d), tir::ForKind::kSerial, sti(Q_local, dq, build_q_val(dq)));

  // ── K/V load loop ──────────────────────────────────────────────────────────
  auto build_k_val = [&](tir::Var dv) -> PrimExpr {
    PrimExpr elem = tir::BufferLoad(pages_buf, {page_no, I32(0), h_kv_idx, page_offset, dv});
    PrimExpr partner =
        BuildPartner(pages_buf, {page_no, I32(0), h_kv_idx, page_offset}, dv, d, dtype);
    PrimExpr pos = tir::BufferLoad(k_rope_pos_offset_buf, {b_idx}) + row_idx;
    PrimExpr rope_expr = BuildRopeLetExpr(elem, partner, pos, dv, d, rope_scale, rope_theta,
                                          rotary_mode, dtype, rope_scaling);
    return is_f16 ? CastTo(rope_expr, "float32") : rope_expr;
  };

  tir::Var dk("d_idx", DataType::Int(32));
  Stmt kv_load_loop = tir::For(
      dk, I32(0), I32(d), tir::ForKind::kSerial,
      tir::SeqStmt({sti(K_local, dk, build_k_val(dk)),
                    sti(V_local, dk,
                        is_f16 ? CastTo(tir::BufferLoad(pages_buf, {page_no, I32(1), h_kv_idx,
                                                                    page_offset, dk}),
                                        "float32")
                               : tir::BufferLoad(pages_buf,
                                                 {page_no, I32(1), h_kv_idx, page_offset, dk}))}));

  // ── S dot product ──────────────────────────────────────────────────────────
  tir::Var ds("d_idx", DataType::Int(32));
  Stmt s_dot = tir::For(ds, I32(0), I32(d), tir::ForKind::kSerial,
                        st0(S_val_buf, S_val() + ldi(Q_local, ds) * ldi(K_local, ds)));

  const double log2e = 1.4426950408889634;
  Stmt s_scale_stmt = st0(S_val_buf, S_val() * (sm_scale * F32(log2e)));

  // ── causal condition ───────────────────────────────────────────────────────
  PrimExpr q_len =
      tir::BufferLoad(q_indptr_buf, {b_idx + I32(1)}) - tir::BufferLoad(q_indptr_buf, {b_idx});
  PrimExpr causal_cond = tvm::if_then_else(
      causal > I32(0), row_idx < kv_len() - q_len + q_idx + I32(1), row_idx < kv_len());

  // ── softmax update ─────────────────────────────────────────────────────────
  Stmt update_new_m = tir::IfThenElse(causal_cond, st0(new_m_buf, tvm::max(m_val(), S_val())),
                                      st0(S_val_buf, F32(-50000.0)));
  Stmt update_d1 = st0(d_val_buf, d_val() * tvm::exp2(m_val() - new_m()));
  Stmt update_d2 = st0(d_val_buf, d_val() + tvm::exp2(S_val() - new_m()));
  Stmt update_scale_O = st0(scale_O_buf, tvm::exp2(m_val() - new_m()));
  Stmt update_m = st0(m_val_buf, new_m());
  Stmt update_factor = st0(factor_buf, tvm::exp2(S_val() - m_val()));

  // O_local[d] *= scale_O[d]  (scale_O indexed by d, matching TVMScript)
  tir::Var do1("d_idx", DataType::Int(32));
  Stmt o_scale = tir::For(do1, I32(0), I32(d), tir::ForKind::kSerial,
                          sti(O_local, do1, ldi(O_local, do1) * ldi(scale_O_buf, do1)));

  tir::Var do2("d_idx", DataType::Int(32));
  Stmt o_update =
      tir::For(do2, I32(0), I32(d), tir::ForKind::kSerial,
               sti(O_local, do2, ldi(O_local, do2) + ldi(V_local, do2) * ld0(factor_buf)));

  // ── row_idx body ───────────────────────────────────────────────────────────
  Stmt row_body = tir::IfThenElse(
      row_idx < kv_len(),
      tir::LetStmt(
          page_no,
          tir::BufferLoad(page_values_buf, {cur_begin + floordiv(row_idx, I32(page_size))}),
          tir::LetStmt(page_offset, floormod(row_idx, I32(page_size)),
                       tir::SeqStmt({kv_load_loop, st0(S_val_buf, F32(0.0)), s_dot, s_scale_stmt,
                                     update_new_m, update_d1, update_d2, update_scale_O, update_m,
                                     update_factor, o_scale, o_update}))));

  Stmt row_loop =
      tir::For(row_idx, I32(0), max_num_pages * I32(page_size), tir::ForKind::kSerial, row_body);

  // ── output write ───────────────────────────────────────────────────────────
  tir::Var dout("d_idx", DataType::Int(32));
  Stmt out_loop = tir::For(
      dout, I32(0), I32(d), tir::ForKind::kSerial,
      tir::SeqStmt({sti(O_local, dout, tvm::div(ldi(O_local, dout), d_val())),
                    tir::BufferStore(
                        output_buf, is_f16 ? CastTo(ldi(O_local, dout), dtype) : ldi(O_local, dout),
                        {curl_q, h_qo, dout})}));

  Stmt lse_store = tir::BufferStore(lse_buf, m_val() + tvm::log2(d_val()), {curl_q, h_qo});

  // ── q_idx body ─────────────────────────────────────────────────────────────
  tir::Var dinit("d_idx", DataType::Int(32));
  Stmt q_body = tir::SeqStmt(
      {st0(m_val_buf, F32(-50000.0)), st0(d_val_buf, F32(1.0)),
       tir::For(dinit, I32(0), I32(d), tir::ForKind::kSerial, sti(O_local, dinit, F32(0.0))),
       tir::LetStmt(curl_q, tir::BufferLoad(q_indptr_buf, {b_idx}) + q_idx,
                    tir::SeqStmt({q_load_loop, row_loop, out_loop, lse_store}))});

  Stmt q_loop = tir::For(
      q_idx, I32(0),
      tir::BufferLoad(q_indptr_buf, {b_idx + I32(1)}) - tir::BufferLoad(q_indptr_buf, {b_idx}),
      tir::ForKind::kSerial, q_body);

  // ── sblock body ────────────────────────────────────────────────────────────
  ffi::Array<tir::Buffer> alloc_bufs = {O_local,          Q_local,     K_local,   V_local,
                                        kv_chunk_len_buf, m_val_buf,   new_m_buf, d_val_buf,
                                        S_val_buf,        scale_O_buf, factor_buf};

  ffi::Map<ffi::String, ffi::Any> sblock_annots;
  sblock_annots.Set("tir.script_parsing_detect_access", IntImm(DataType::Int(32), 3));

  Stmt sblock_body = tir::LetStmt(
      cur_begin, tir::BufferLoad(page_indptr_buf, {b_idx}),
      tir::LetStmt(
          cur_end, tir::BufferLoad(page_indptr_buf, {b_idx + I32(1)}),
          tir::SeqStmt({tir::BufferStore(kv_chunk_len_buf, kv_len_expr, {I32(0)}), q_loop})));

  Stmt sblock = tir::SBlockRealize(
      {}, tir::const_true(),
      tir::SBlock({}, {}, {}, "attn", sblock_body, std::nullopt, alloc_bufs, {}, sblock_annots));

  // ── outer loops ────────────────────────────────────────────────────────────
  Stmt body = sblock;
  body = tir::For(b_idx, I32(0), batch_size, tir::ForKind::kSerial, body);
  body = tir::For(h_qo, I32(0), I32(h_q), tir::ForKind::kSerial, body);

  // ── buffer map ─────────────────────────────────────────────────────────────
  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_q_hdl, q_buf);
  buf_map.Set(h_q_indptr, q_indptr_buf);
  buf_map.Set(h_pages, pages_buf);
  buf_map.Set(h_page_indptr, page_indptr_buf);
  buf_map.Set(h_page_values, page_values_buf);
  buf_map.Set(h_length_info, length_info_buf);
  buf_map.Set(h_k_rope_pos_offset, k_rope_pos_offset_buf);
  buf_map.Set(h_q_rope_position, q_rope_position_buf);
  buf_map.Set(h_output, output_buf);
  buf_map.Set(h_lse, lse_buf);

  ffi::Array<tir::Var> params = {
      h_q_hdl,       h_q_indptr,          h_pages,           h_page_indptr, h_page_values,
      h_length_info, h_k_rope_pos_offset, h_q_rope_position, h_output,      h_lse,
      causal,        rotary_mode,         rope_scale,        rope_theta,    sm_scale};

  std::string cpu_global_symbol = "batch_prefill_paged_kv_cpu";
  if (sliding_window) cpu_global_symbol += "_sliding_window";
  tir::PrimFunc fn(params, body, VoidType(), buf_map);
  fn = WithAttr(fn, "global_symbol", ffi::Any(ffi::String(cpu_global_symbol)));
  fn = tir::ScriptComplete(fn, {});
  return fn;
}

// ---------------------------------------------------------------------------
// FFI registration
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.frontend.nn.llm.kv_cache.attention_prefill_cpu",
      [](int64_t h_kv, int64_t h_q, int64_t d, ffi::String dtype, bool sliding_window,
         ffi::Map<ffi::String, ffi::Any> rope_scaling, int64_t page_size) -> tir::PrimFunc {
        return AttentionPrefillCpu(h_kv, h_q, d, std::string(dtype), sliding_window, rope_scaling,
                                   page_size);
      });
}

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
