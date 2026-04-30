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
 * \file src/relax/frontend/nn/llm/kv_cache_transpose_append.cc
 * \brief TIR kernel generators for KV cache transpose-append operations.
 *
 * Implements:
 *   - KVCacheTransposeAppend   (_kv_cache_transpose_append in Python)
 *   - KVCacheTransposeAppendMLA (_kv_cache_transpose_append_mla in Python)
 *
 * Expected TVMScript (h=4, d=128, float16, page_size=16):
 * ---------------------------------------------------------------------------
 * @T.prim_func
 * def tir_kv_cache_transpose_append(var_pages, var_k_data, var_v_data, var_position_map):
 *     T.func_attr({"tir.noalias": True})
 *     ntoken = T.SizeVar("num_tokens_excluding_cache", "int64")
 *     num_pages = T.int64()
 *     pages_elem_offset = T.int64()
 *     pages = T.match_buffer(var_pages, (num_pages, 2, 4, 16, 128), "float16",
 *                            elem_offset=pages_elem_offset)
 *     k_data = T.match_buffer(var_k_data, (ntoken, 4, 128), "float16")
 *     v_data = T.match_buffer(var_v_data, (ntoken, 4, 128), "float16")
 *     position_map_elem_offset = T.int32()
 *     position_map = T.match_buffer(var_position_map, (ntoken,), "int32",
 *                                   elem_offset=position_map_elem_offset)
 *     for global_pos, h, f in T.grid(ntoken, 4, 128):
 *         if position_map[global_pos] != T.int32(-1):
 *             with T.sblock("k_transpose_append"):
 *                 vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
 *                 ...
 *                 position: T.int32 = position_map[vgpos]
 *                 pages[position // 16, 0, vh, position % 16, vf] = k_data[vgpos, vh, vf]
 *             with T.sblock("v_transpose_append"):
 *                 ...
 *                 pages[position // 16, 1, vh, position % 16, vf] = v_data[vgpos, vh, vf]
 * ---------------------------------------------------------------------------
 */

#include "kv_cache.h"
#include "kv_cache_common.h"

#include <tvm/tir/stmt.h>
#include <tvm/tir/op.h>

#include "../../../../tir/ir/script/script_complete.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

using namespace tvm::tir;

// ---------------------------------------------------------------------------
// KVCacheTransposeAppend
// ---------------------------------------------------------------------------

tir::PrimFunc KVCacheTransposeAppend(int64_t num_key_value_heads, int64_t head_dim,
                                     const std::string& dtype, int64_t page_size) {
  DataType dt = DataType(runtime::StringToDLDataType(dtype));

  // Dynamic symbolic vars
  // ntoken: int64 SizeVar named "num_tokens_excluding_cache"
  tir::SizeVar ntoken("num_tokens_excluding_cache", DataType::Int(64));
  tir::Var num_pages("num_pages", DataType::Int(64));
  tir::Var pages_elem_offset("pages_elem_offset", DataType::Int(64));
  tir::Var position_map_elem_offset("position_map_elem_offset", DataType::Int(32));

  // Plain handle params
  tir::Var h_pages("var_pages", DataType::Handle());
  tir::Var h_k("var_k_data", DataType::Handle());
  tir::Var h_v("var_v_data", DataType::Handle());
  tir::Var h_pos("var_position_map", DataType::Handle());

  // pages buffer: (num_pages, 2, h, page_size, d)
  tir::Buffer pages_buf = tir::Buffer(
      tir::decl_buffer({num_pages, I64(2), I64(num_key_value_heads), I64(page_size), I64(head_dim)},
                       dt, "pages")->data,
      dt,
      {num_pages, I64(2), I64(num_key_value_heads), I64(page_size), I64(head_dim)},
      {},
      pages_elem_offset,
      "pages", 0, 0, tir::kDefault);

  // k_data: (ntoken, h, d)
  tir::Buffer k_buf = tir::decl_buffer({ntoken, I64(num_key_value_heads), I64(head_dim)}, dt,
                                       "k_data");
  // v_data: (ntoken, h, d)
  tir::Buffer v_buf = tir::decl_buffer({ntoken, I64(num_key_value_heads), I64(head_dim)}, dt,
                                       "v_data");
  // position_map: (ntoken,) int32, elem_offset=position_map_elem_offset
  tir::Buffer pos_buf = tir::Buffer(
      tir::decl_buffer({ntoken}, DataType::Int(32), "position_map")->data,
      DataType::Int(32), {ntoken}, {}, position_map_elem_offset,
      "position_map", 0, 0, tir::kDefault);

  // Loop vars: global_pos (int64), h (int64), f (int64)
  tir::Var gpos("global_pos", DataType::Int(64));
  tir::Var h_var("h", DataType::Int(64));
  tir::Var f_var("f", DataType::Int(64));

  // Axis vars for sblocks: vgpos, vh, vf (int64)
  tir::Var vgpos("vgpos", DataType::Int(64));
  tir::Var vh("vh", DataType::Int(64));
  tir::Var vf("vf", DataType::Int(64));

  // position: T.int32 = position_map[vgpos]
  tir::Var pos_var("position", DataType::Int(32));

  // pages[position // page_size, 0, vh, position % page_size, vf] = k_data[vgpos, vh, vf]
  PrimExpr pos_i64 = CastTo(pos_var, "int64");
  PrimExpr page_idx = tir::FloorDiv(pos_i64, I64(page_size));
  PrimExpr page_off = tir::FloorMod(pos_i64, I64(page_size));

  // k sblock body
  Stmt k_let = tir::LetStmt(
      pos_var, tir::BufferLoad(pos_buf, {vgpos}),
      tir::BufferStore(pages_buf, tir::BufferLoad(k_buf, {vgpos, vh, vf}),
                       {page_idx, I64(0), vh, page_off, vf}));

  // v sblock body
  tir::Var pos_var2("position", DataType::Int(32));
  PrimExpr pos_i64_2 = CastTo(pos_var2, "int64");
  PrimExpr page_idx2 = tir::FloorDiv(pos_i64_2, I64(page_size));
  PrimExpr page_off2 = tir::FloorMod(pos_i64_2, I64(page_size));
  Stmt v_let = tir::LetStmt(
      pos_var2, tir::BufferLoad(pos_buf, {vgpos}),
      tir::BufferStore(pages_buf, tir::BufferLoad(v_buf, {vgpos, vh, vf}),
                       {page_idx2, I64(1), vh, page_off2, vf}));

  // k sblock iter vars
  ffi::Array<tir::IterVar> k_iter_vars = {
      tir::IterVar(Range::FromMinExtent(I64(0), ntoken), vgpos, tir::kDataPar, ""),
      tir::IterVar(Range::FromMinExtent(I64(0), I64(num_key_value_heads)), vh, tir::kDataPar, ""),
      tir::IterVar(Range::FromMinExtent(I64(0), I64(head_dim)), vf, tir::kDataPar, "")};

  ffi::Map<ffi::String, ffi::Any> sblock_annots;
  sblock_annots.Set("tir.script_parsing_detect_access", IntImm(DataType::Int(32), 3));

  Stmt k_sblock = tir::SBlockRealize(
      {PrimExpr(gpos), PrimExpr(h_var), PrimExpr(f_var)},
      tir::const_true(),
      tir::SBlock(k_iter_vars, {}, {}, "k_transpose_append", k_let,
                  std::nullopt, {}, {}, sblock_annots));

  // v sblock iter vars (fresh vars)
  tir::Var vgpos2("vgpos", DataType::Int(64));
  tir::Var vh2("vh", DataType::Int(64));
  tir::Var vf2("vf", DataType::Int(64));
  // Rebuild v_let with vgpos2, vh2, vf2
  tir::Var pos_var3("position", DataType::Int(32));
  PrimExpr pos_i64_3 = CastTo(pos_var3, "int64");
  PrimExpr page_idx3 = tir::FloorDiv(pos_i64_3, I64(page_size));
  PrimExpr page_off3 = tir::FloorMod(pos_i64_3, I64(page_size));
  Stmt v_let2 = tir::LetStmt(
      pos_var3, tir::BufferLoad(pos_buf, {vgpos2}),
      tir::BufferStore(pages_buf, tir::BufferLoad(v_buf, {vgpos2, vh2, vf2}),
                       {page_idx3, I64(1), vh2, page_off3, vf2}));

  ffi::Array<tir::IterVar> v_iter_vars = {
      tir::IterVar(Range::FromMinExtent(I64(0), ntoken), vgpos2, tir::kDataPar, ""),
      tir::IterVar(Range::FromMinExtent(I64(0), I64(num_key_value_heads)), vh2, tir::kDataPar, ""),
      tir::IterVar(Range::FromMinExtent(I64(0), I64(head_dim)), vf2, tir::kDataPar, "")};

  Stmt v_sblock = tir::SBlockRealize(
      {PrimExpr(gpos), PrimExpr(h_var), PrimExpr(f_var)},
      tir::const_true(),
      tir::SBlock(v_iter_vars, {}, {}, "v_transpose_append", v_let2,
                  std::nullopt, {}, {}, sblock_annots));

  // if position_map[global_pos] != int32(-1): k_sblock; v_sblock
  PrimExpr cond = tir::NE(tir::BufferLoad(pos_buf, {gpos}),
                           IntImm(DataType::Int(32), -1));
  Stmt if_body = tir::SeqStmt({k_sblock, v_sblock});
  Stmt if_stmt = tir::IfThenElse(cond, if_body);

  // for global_pos, h, f in T.grid(ntoken, h, d):
  Stmt loop = if_stmt;
  loop = tir::For(f_var, I64(0), I64(head_dim), tir::ForKind::kSerial, loop);
  loop = tir::For(h_var, I64(0), I64(num_key_value_heads), tir::ForKind::kSerial, loop);
  loop = tir::For(gpos, I64(0), ntoken, tir::ForKind::kSerial, loop);

  // Buffer map
  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_pages, pages_buf);
  buf_map.Set(h_k, k_buf);
  buf_map.Set(h_v, v_buf);
  buf_map.Set(h_pos, pos_buf);

  ffi::Array<tir::Var> params = {h_pages, h_k, h_v, h_pos};
  tir::PrimFunc fn(params, loop, VoidType(), buf_map);
  fn = WithAttr(fn, "tir.noalias", ffi::Any(true));
  fn = tir::ScriptComplete(fn, {});
  return fn;
}

// ---------------------------------------------------------------------------
// KVCacheTransposeAppendMLA
// ---------------------------------------------------------------------------

tir::PrimFunc KVCacheTransposeAppendMLA(int64_t d_qk, const std::string& dtype,
                                        int64_t page_size) {
  DataType dt = DataType(runtime::StringToDLDataType(dtype));

  tir::SizeVar ntoken("num_tokens_excluding_cache", DataType::Int(64));
  tir::Var num_pages("num_pages", DataType::Int(64));
  tir::Var pages_elem_offset("pages_elem_offset", DataType::Int(64));
  tir::Var position_map_elem_offset("position_map_elem_offset", DataType::Int(32));

  tir::Var h_pages("var_pages", DataType::Handle());
  tir::Var h_kv("var_kv_data", DataType::Handle());
  tir::Var h_pos("var_position_map", DataType::Handle());

  // pages: (num_pages, page_size, d_qk)
  tir::Buffer pages_buf = tir::Buffer(
      tir::decl_buffer({num_pages, I64(page_size), I64(d_qk)}, dt, "pages")->data,
      dt, {num_pages, I64(page_size), I64(d_qk)}, {},
      pages_elem_offset, "pages", 0, 0, tir::kDefault);

  // kv_data: (ntoken, d_qk)
  tir::Buffer kv_buf = tir::decl_buffer({ntoken, I64(d_qk)}, dt, "kv_data");

  // position_map: (ntoken,) int32
  tir::Buffer pos_buf = tir::Buffer(
      tir::decl_buffer({ntoken}, DataType::Int(32), "position_map")->data,
      DataType::Int(32), {ntoken}, {}, position_map_elem_offset,
      "position_map", 0, 0, tir::kDefault);

  // Loop vars
  tir::Var gpos("global_pos", DataType::Int(64));
  tir::Var f_var("f", DataType::Int(64));

  // Axis vars
  tir::Var vgpos("vgpos", DataType::Int(64));
  tir::Var vf("vf", DataType::Int(64));

  tir::Var pos_var("position", DataType::Int(32));
  PrimExpr pos_i64 = CastTo(pos_var, "int64");
  PrimExpr page_idx = tir::FloorDiv(pos_i64, I64(page_size));
  PrimExpr page_off = tir::FloorMod(pos_i64, I64(page_size));

  Stmt k_let = tir::LetStmt(
      pos_var, tir::BufferLoad(pos_buf, {vgpos}),
      tir::BufferStore(pages_buf, tir::BufferLoad(kv_buf, {vgpos, vf}),
                       {page_idx, page_off, vf}));

  ffi::Array<tir::IterVar> iter_vars = {
      tir::IterVar(Range::FromMinExtent(I64(0), ntoken), vgpos, tir::kDataPar, ""),
      tir::IterVar(Range::FromMinExtent(I64(0), I64(d_qk)), vf, tir::kDataPar, "")};

  ffi::Map<ffi::String, ffi::Any> sblock_annots;
  sblock_annots.Set("tir.script_parsing_detect_access", IntImm(DataType::Int(32), 3));

  Stmt k_sblock = tir::SBlockRealize(
      {PrimExpr(gpos), PrimExpr(f_var)},
      tir::const_true(),
      tir::SBlock(iter_vars, {}, {}, "k_transpose_append", k_let,
                  std::nullopt, {}, {}, sblock_annots));

  PrimExpr cond = tir::NE(tir::BufferLoad(pos_buf, {gpos}),
                           IntImm(DataType::Int(32), -1));
  Stmt if_stmt = tir::IfThenElse(cond, k_sblock);

  Stmt loop = if_stmt;
  loop = tir::For(f_var, I64(0), I64(d_qk), tir::ForKind::kSerial, loop);
  loop = tir::For(gpos, I64(0), ntoken, tir::ForKind::kSerial, loop);

  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_pages, pages_buf);
  buf_map.Set(h_kv, kv_buf);
  buf_map.Set(h_pos, pos_buf);

  ffi::Array<tir::Var> params = {h_pages, h_kv, h_pos};
  tir::PrimFunc fn(params, loop, VoidType(), buf_map);
  fn = WithAttr(fn, "tir.noalias", ffi::Any(true));
  fn = tir::ScriptComplete(fn, {});
  return fn;
}

// ---------------------------------------------------------------------------
// FFI registrations
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.frontend.nn.llm.kv_cache.kv_cache_transpose_append",
           [](int64_t num_key_value_heads, int64_t head_dim, ffi::String dtype,
              int64_t page_size) -> tir::PrimFunc {
             return KVCacheTransposeAppend(num_key_value_heads, head_dim, std::string(dtype),
                                          page_size);
           })
      .def("relax.frontend.nn.llm.kv_cache.kv_cache_transpose_append_mla",
           [](int64_t d_qk, ffi::String dtype, int64_t page_size) -> tir::PrimFunc {
             return KVCacheTransposeAppendMLA(d_qk, std::string(dtype), page_size);
           });
}

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
