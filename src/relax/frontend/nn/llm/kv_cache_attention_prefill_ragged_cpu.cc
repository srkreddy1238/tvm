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
 * \file src/relax/frontend/nn/llm/kv_cache_attention_prefill_ragged_cpu.cc
 * \brief CPU TIR kernel for ragged prefill (non-paged KV).
 *        C++ port of _attention_prefill_ragged_cpu() in kv_cache.py.
 *
 * CPU implementation: no shared memory, no thread binding.
 * Loops: batch -> head -> q_tile -> kv_tile -> d.
 * Uses float32 accumulators throughout.
 */

#include "kv_cache.h"
#include <tvm/s_tir/stmt.h>
#include "kv_cache_attention_prefill_gpu_helpers.h"
#include "../../../../tir/ir/script/script_complete.h"

#include <tvm/tir/op.h>

#include <cmath>
#include <string>

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

using namespace tvm::tir;

tir::PrimFunc AttentionPrefillRaggedCpu(int64_t h_kv, int64_t h_q, int64_t d_qk, int64_t d_v,
                                        const std::string& dtype,
                                        const ffi::Map<ffi::String, ffi::Any>& rope_scaling) {
  int64_t group_size = h_q / h_kv;
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

  // ---- local alloc buffers ----
  tir::Buffer S_buf = tir::decl_buffer({kv_len}, DataType::Float(32), "S", "local");
  tir::Buffer O_buf = tir::decl_buffer({I32(d_v)}, DataType::Float(32), "O", "local");

  // ---- loop vars ----
  tir::Var b("b", DataType::Int(32));
  tir::Var h("h", DataType::Int(32));
  tir::Var q_pos("q_pos", DataType::Int(32));
  tir::Var kv_pos("kv_pos", DataType::Int(32));
  tir::Var d_q("d_q", DataType::Int(32));
  tir::Var d_v_var("d_v", DataType::Int(32));

  // Derived
  tir::Var q_base("q_base", DataType::Int(32));
  tir::Var kv_base("kv_base", DataType::Int(32));
  tir::Var q_len("q_len", DataType::Int(32));
  tir::Var kv_chunk("kv_chunk", DataType::Int(32));
  tir::Var h_kv_var("h_kv", DataType::Int(32));

  // ---- causal mask ----
  auto causal_mask = [&](PrimExpr row, PrimExpr col, PrimExpr kv_len_e, PrimExpr qo_len_e) {
    return tvm::if_then_else(causal > I32(0),
                              col < kv_len_e - qo_len_e + row + I32(1),
                              col < kv_len_e);
  };

  // ---- S computation: S[kv_pos] = sum_d q[q_base+q_pos, h, d] * k[kv_base+kv_pos, h_kv, d] ----
  tir::Var d_s("d", DataType::Int(32));
  PrimExpr q_elem_s = BuildPrefillRopeExpr(q_buf, {q_base + q_pos, h}, d_s, d_qk,
                                            tir::BufferLoad(q_rope_pos_buf, {q_base + q_pos}),
                                            rope_scale, rope_theta, rotary_mode, dtype, rope_scaling);
  PrimExpr k_elem_s = BuildPrefillRopeExpr(k_buf, {kv_base + kv_pos, h_kv_var}, d_s, d_qk,
                                            tir::BufferLoad(k_rope_pos_buf, {b}) + kv_pos,
                                            rope_scale, rope_theta, rotary_mode, dtype, rope_scaling);
  PrimExpr q_f32 = is_f16 ? CastTo(q_elem_s, "float32") : q_elem_s;
  PrimExpr k_f32 = is_f16 ? CastTo(k_elem_s, "float32") : k_elem_s;

  Stmt s_init = tir::BufferStore(S_buf, F32(0.0), {kv_pos});
  Stmt s_accum = tir::For(d_s, I32(0), I32(d_qk), tir::ForKind::kSerial,
      tir::BufferStore(S_buf,
          tir::BufferLoad(S_buf, {kv_pos}) + q_f32 * k_f32 * sm_scale * F32(log2e),
          {kv_pos}));
  Stmt s_mask = tir::IfThenElse(
      logical_not(causal_mask(q_pos, kv_pos, kv_chunk, q_len)),
      tir::BufferStore(S_buf, F32(-50000.0), {kv_pos}));
  Stmt s_loop = tir::For(kv_pos, I32(0), kv_chunk, tir::ForKind::kSerial,
      tir::SeqStmt({s_init, s_accum, s_mask}));

  // ---- softmax: m = max(S), d = sum(exp2(S - m)), S = exp2(S - m) / d ----
  tir::Var m_var("m", DataType::Float(32));
  tir::Var d_var("d_sum", DataType::Float(32));
  tir::Var kv_pos2("kv_pos", DataType::Int(32));
  tir::Var kv_pos3("kv_pos", DataType::Int(32));
  tir::Var kv_pos4("kv_pos", DataType::Int(32));

  // m = max over S
  Stmt m_init = tir::LetStmt(m_var, F32(-50000.0),
      tir::For(kv_pos2, I32(0), kv_chunk, tir::ForKind::kSerial,
          tir::LetStmt(m_var, tvm::max(m_var, tir::BufferLoad(S_buf, {kv_pos2})),
              tir::Evaluate(m_var))));
  // Actually we need mutable m — use a local buffer
  tir::Buffer m_buf = tir::decl_buffer({I32(1)}, DataType::Float(32), "m_local", "local");
  tir::Buffer d_buf = tir::decl_buffer({I32(1)}, DataType::Float(32), "d_local", "local");

  Stmt m_init2 = tir::BufferStore(m_buf, F32(-50000.0), {I32(0)});
  Stmt m_loop = tir::For(kv_pos2, I32(0), kv_chunk, tir::ForKind::kSerial,
      tir::BufferStore(m_buf,
          tvm::max(tir::BufferLoad(m_buf, {I32(0)}), tir::BufferLoad(S_buf, {kv_pos2})),
          {I32(0)}));
  Stmt d_init = tir::BufferStore(d_buf, F32(0.0), {I32(0)});
  Stmt softmax_loop = tir::For(kv_pos3, I32(0), kv_chunk, tir::ForKind::kSerial,
      tir::SeqStmt({
          tir::BufferStore(S_buf,
              tvm::exp2(tir::BufferLoad(S_buf, {kv_pos3}) - tir::BufferLoad(m_buf, {I32(0)})),
              {kv_pos3}),
          tir::BufferStore(d_buf,
              tir::BufferLoad(d_buf, {I32(0)}) + tir::BufferLoad(S_buf, {kv_pos3}),
              {I32(0)})}));

  // ---- O computation: O[d] = sum_kv S[kv] * v[kv_base+kv, h_kv, d] ----
  tir::Var d_o("d", DataType::Int(32));
  tir::Var kv_pos5("kv_pos", DataType::Int(32));
  Stmt o_init = tir::For(d_o, I32(0), I32(d_v), tir::ForKind::kSerial,
      tir::BufferStore(O_buf, F32(0.0), {d_o}));
  tir::Var d_o2("d", DataType::Int(32));
  PrimExpr v_elem = is_f16
      ? CastTo(tir::BufferLoad(v_buf, ffi::Array<PrimExpr>{kv_base + kv_pos5, h_kv_var, d_o2}), "float32")
      : tir::BufferLoad(v_buf, ffi::Array<PrimExpr>{kv_base + kv_pos5, h_kv_var, d_o2});
  Stmt o_accum = tir::For(kv_pos5, I32(0), kv_chunk, tir::ForKind::kSerial,
      tir::For(d_o2, I32(0), I32(d_v), tir::ForKind::kSerial,
          tir::BufferStore(O_buf,
              tir::BufferLoad(O_buf, {d_o2}) +
              tir::BufferLoad(S_buf, {kv_pos5}) * v_elem,
              {d_o2})));

  // ---- store output ----
  tir::Var d_out("d", DataType::Int(32));
  Stmt store_output = tir::For(d_out, I32(0), I32(d_v), tir::ForKind::kSerial,
      tir::BufferStore(output_buf,
          is_f16
          ? CastTo(tvm::div(tir::BufferLoad(O_buf, {d_out}), tir::BufferLoad(d_buf, {I32(0)})), dtype)
          : tvm::div(tir::BufferLoad(O_buf, {d_out}), tir::BufferLoad(d_buf, {I32(0)})), ffi::Array<PrimExpr>{q_base + q_pos, h, d_out}));
  Stmt store_lse = tir::BufferStore(lse_buf,
      tir::BufferLoad(m_buf, {I32(0)}) + tvm::log2(tir::BufferLoad(d_buf, {I32(0)})), ffi::Array<PrimExpr>{q_base + q_pos, h});

  // ---- per-query body ----
  Stmt per_query = tir::SeqStmt({
      s_loop, m_init2, m_loop, d_init, softmax_loop, o_init, o_accum, store_output, store_lse});

  // ---- head loop ----
  Stmt head_body = tir::LetStmt(h_kv_var, floordiv(h, I32(group_size)),
      tir::For(q_pos, I32(0), q_len, tir::ForKind::kSerial, per_query));
  Stmt head_loop = tir::For(h, I32(0), I32(h_q), tir::ForKind::kSerial, head_body);

  // ---- batch loop ----
  Stmt batch_body = tir::LetStmt(q_base, tir::BufferLoad(q_indptr_buf, {b}),
      tir::LetStmt(kv_base, tir::BufferLoad(kv_indptr_buf, {b}),
          tir::LetStmt(q_len,
              tir::BufferLoad(q_indptr_buf, {b + I32(1)}) -
              tir::BufferLoad(q_indptr_buf, {b}),
              tir::LetStmt(kv_chunk,
                  tir::BufferLoad(kv_indptr_buf, {b + I32(1)}) -
                  tir::BufferLoad(kv_indptr_buf, {b}),
                  head_loop))));
  Stmt batch_loop = tir::For(b, I32(0), batch_size, tir::ForKind::kSerial, batch_body);

  // ---- root sblock ----
  ffi::Array<tir::Buffer> alloc_bufs = {S_buf, O_buf, m_buf, d_buf};
  ffi::Map<ffi::String, ffi::Any> attn_ann;
  attn_ann.Set(s_tir::attr::script_parsing_detect_access, ffi::Any(IntImm(DataType::Int(32), 3)));

  Stmt body = tir::SBlockRealize({}, tir::const_true(),
      tir::SBlock({}, {}, {}, "root",
          tir::SBlockRealize({}, tir::const_true(),
              tir::SBlock({}, {}, {}, "attn",
                  batch_loop,
                  std::nullopt, alloc_bufs, {}, attn_ann))));

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
  fn = WithAttr(fn, "global_symbol", ffi::Any(ffi::String("batch_prefill_ragged_kv_cpu")));
  fn = tir::ScriptComplete(fn, {});
  return fn;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.frontend.nn.llm.kv_cache.attention_prefill_ragged_cpu",
      [](int64_t h_kv, int64_t h_q, int64_t d_qk, int64_t d_v, ffi::String dtype,
         ffi::Map<ffi::String, ffi::Any> rope_scaling) -> tir::PrimFunc {
        return AttentionPrefillRaggedCpu(h_kv, h_q, d_qk, d_v, std::string(dtype), rope_scaling);
      });
}

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
