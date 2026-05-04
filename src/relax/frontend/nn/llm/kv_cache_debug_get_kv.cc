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
 * \file src/relax/frontend/nn/llm/kv_cache_debug_get_kv.cc
 * \brief TIR kernel generators for KV cache debug get-KV operations.
 *
 * Implements:
 *   - KVCacheDebugGetKV    (_kv_cache_debug_get_kv in Python)
 *   - KVCacheDebugGetKVMLA (_kv_cache_debug_get_kv_mla in Python)
 *
 * Expected TVMScript (num_layers=4, h=4, d=128, float16):
 * ---------------------------------------------------------------------------
 * @T.prim_func
 * def tir_kv_cache_debug_get_kv(var_pages, var_position_map, var_k_data, var_v_data,
 *                                layer_id: T.int64):
 *     T.func_attr({"tir.noalias": True})
 *     num_pages, page_size = T.int64(), T.int64(is_size_var=True)
 *     pages = T.match_buffer(var_pages, (num_pages, 2, 4, page_size, 128), "float16",
 *                            offset_factor=1)
 *     seqlen = T.int64(is_size_var=True)
 *     position_map = T.match_buffer(var_position_map, (seqlen,), "int32", offset_factor=1)
 *     k_data = T.match_buffer(var_k_data, (4, seqlen, 4, 128), "float16")
 *     v_data = T.match_buffer(var_v_data, (4, seqlen, 4, 128), "float16")
 *     for p, h, d in T.grid(seqlen, 4, 128):
 *         with T.sblock("copy0"):
 *             vp, vh, vd = T.axis.remap("SSS", [p, h, d])
 *             ...
 *             position: T.int32 = position_map[vp]
 *             k_data[layer_id, vp, vh, vd] = pages[cast(position)//page_size, 0, vh,
 *                                                   cast(position)%page_size, vd]
 *             v_data[layer_id, vp, vh, vd] = pages[cast(position)//page_size, 1, vh,
 *                                                   cast(position)%page_size, vd]
 * ---------------------------------------------------------------------------
 */

#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include "../../../../tir/ir/script/script_complete.h"
#include "kv_cache.h"
#include "kv_cache_common.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

using namespace tvm::tir;

// ---------------------------------------------------------------------------
// KVCacheDebugGetKV
// ---------------------------------------------------------------------------

tir::PrimFunc KVCacheDebugGetKV(int64_t num_hidden_layers, int64_t num_key_value_heads,
                                int64_t head_dim, const std::string& dtype) {
  DataType dt = DataType(runtime::StringToDLDataType(dtype));

  // Dynamic symbolic vars
  tir::Var num_pages("num_pages", DataType::Int(64));
  tir::SizeVar page_size("page_size", DataType::Int(64));
  tir::SizeVar seqlen("seqlen", DataType::Int(64));

  // Plain handle params
  tir::Var h_pages("var_pages", DataType::Handle());
  tir::Var h_pos("var_position_map", DataType::Handle());
  tir::Var h_k("var_k_data", DataType::Handle());
  tir::Var h_v("var_v_data", DataType::Handle());
  tir::Var layer_id("layer_id", DataType::Int(64));

  // pages: (num_pages, 2, h, page_size, d), offset_factor=1
  tir::Buffer pages_buf = MakeOffsetFactor1Buffer(
      "pages", {num_pages, I64(2), I64(num_key_value_heads), page_size, I64(head_dim)}, dtype);

  // position_map: (seqlen,) int32, offset_factor=1
  tir::Buffer pos_buf = MakeOffsetFactor1Buffer("position_map", {seqlen}, "int32");

  // k_data: (num_hidden_layers, seqlen, h, d)
  tir::Buffer k_buf = tir::decl_buffer(
      {I64(num_hidden_layers), seqlen, I64(num_key_value_heads), I64(head_dim)}, dt, "k_data");

  // v_data: (num_hidden_layers, seqlen, h, d)
  tir::Buffer v_buf = tir::decl_buffer(
      {I64(num_hidden_layers), seqlen, I64(num_key_value_heads), I64(head_dim)}, dt, "v_data");

  // Loop vars
  tir::Var p_var("p", DataType::Int(64));
  tir::Var h_var("h", DataType::Int(64));
  tir::Var d_var("d", DataType::Int(64));

  // Axis vars
  tir::Var vp("vp", DataType::Int(64));
  tir::Var vh("vh", DataType::Int(64));
  tir::Var vd("vd", DataType::Int(64));

  // position: T.int32 = position_map[vp]
  tir::Var pos_var("position", DataType::Int(32));
  PrimExpr pos_i64 = CastTo(pos_var, "int64");
  PrimExpr page_idx = tir::FloorDiv(pos_i64, page_size);
  PrimExpr page_off = tir::FloorMod(pos_i64, page_size);

  // k_data[layer_id, vp, vh, vd] = pages[page_idx, 0, vh, page_off, vd]
  // v_data[layer_id, vp, vh, vd] = pages[page_idx, 1, vh, page_off, vd]
  Stmt body = tir::LetStmt(
      pos_var, tir::BufferLoad(pos_buf, {vp}),
      tir::SeqStmt({
          tir::BufferStore(k_buf, tir::BufferLoad(pages_buf, {page_idx, I64(0), vh, page_off, vd}),
                           {layer_id, vp, vh, vd}),
          tir::BufferStore(v_buf, tir::BufferLoad(pages_buf, {page_idx, I64(1), vh, page_off, vd}),
                           {layer_id, vp, vh, vd}),
      }));

  ffi::Array<tir::IterVar> iter_vars = {
      tir::IterVar(Range::FromMinExtent(I64(0), seqlen), vp, tir::kDataPar, ""),
      tir::IterVar(Range::FromMinExtent(I64(0), I64(num_key_value_heads)), vh, tir::kDataPar, ""),
      tir::IterVar(Range::FromMinExtent(I64(0), I64(head_dim)), vd, tir::kDataPar, "")};

  ffi::Map<ffi::String, ffi::Any> sblock_annots;
  sblock_annots.Set("tir.script_parsing_detect_access", IntImm(DataType::Int(32), 3));

  Stmt sblock = tir::SBlockRealize(
      {PrimExpr(p_var), PrimExpr(h_var), PrimExpr(d_var)}, tir::const_true(),
      tir::SBlock(iter_vars, {}, {}, "copy0", body, std::nullopt, {}, {}, sblock_annots));

  // for p, h, d in T.grid(seqlen, h, d):
  Stmt loop = sblock;
  loop = tir::For(d_var, I64(0), I64(head_dim), tir::ForKind::kSerial, loop);
  loop = tir::For(h_var, I64(0), I64(num_key_value_heads), tir::ForKind::kSerial, loop);
  loop = tir::For(p_var, I64(0), seqlen, tir::ForKind::kSerial, loop);

  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_pages, pages_buf);
  buf_map.Set(h_pos, pos_buf);
  buf_map.Set(h_k, k_buf);
  buf_map.Set(h_v, v_buf);

  ffi::Array<tir::Var> params = {h_pages, h_pos, h_k, h_v, layer_id};
  tir::PrimFunc fn(params, loop, VoidType(), buf_map);
  fn = WithAttr(fn, "tir.noalias", ffi::Any(true));
  fn = tir::ScriptComplete(fn, {});
  return fn;
}

// ---------------------------------------------------------------------------
// KVCacheDebugGetKVMLA
// ---------------------------------------------------------------------------

tir::PrimFunc KVCacheDebugGetKVMLA(int64_t num_hidden_layers, int64_t d_qk,
                                   const std::string& dtype) {
  DataType dt = DataType(runtime::StringToDLDataType(dtype));

  tir::Var num_pages("num_pages", DataType::Int(64));
  tir::SizeVar page_size("page_size", DataType::Int(64));
  tir::SizeVar seqlen("seqlen", DataType::Int(64));

  tir::Var h_pages("var_pages", DataType::Handle());
  tir::Var h_pos("var_position_map", DataType::Handle());
  tir::Var h_kv("var_compressed_kv_with_k_pe_data", DataType::Handle());
  tir::Var layer_id("layer_id", DataType::Int(64));

  // pages: (num_pages, page_size, d_qk), offset_factor=1
  tir::Buffer pages_buf =
      MakeOffsetFactor1Buffer("pages", {num_pages, page_size, I64(d_qk)}, dtype);

  // position_map: (seqlen,) int32, offset_factor=1
  tir::Buffer pos_buf = MakeOffsetFactor1Buffer("position_map", {seqlen}, "int32");

  // compressed_kv_with_k_pe_data: (num_hidden_layers, seqlen, d_qk)
  tir::Buffer kv_buf = tir::decl_buffer({I64(num_hidden_layers), seqlen, I64(d_qk)}, dt,
                                        "compressed_kv_with_k_pe_data");

  // Loop vars
  tir::Var p_var("p", DataType::Int(64));
  tir::Var d_var("d", DataType::Int(64));

  // Axis vars
  tir::Var vp("vp", DataType::Int(64));
  tir::Var vd("vd", DataType::Int(64));

  tir::Var pos_var("position", DataType::Int(32));
  PrimExpr pos_i64 = CastTo(pos_var, "int64");
  PrimExpr page_idx = tir::FloorDiv(pos_i64, page_size);
  PrimExpr page_off = tir::FloorMod(pos_i64, page_size);

  Stmt body =
      tir::LetStmt(pos_var, tir::BufferLoad(pos_buf, {vp}),
                   tir::BufferStore(kv_buf, tir::BufferLoad(pages_buf, {page_idx, page_off, vd}),
                                    {layer_id, vp, vd}));

  ffi::Array<tir::IterVar> iter_vars = {
      tir::IterVar(Range::FromMinExtent(I64(0), seqlen), vp, tir::kDataPar, ""),
      tir::IterVar(Range::FromMinExtent(I64(0), I64(d_qk)), vd, tir::kDataPar, "")};

  ffi::Map<ffi::String, ffi::Any> sblock_annots;
  sblock_annots.Set("tir.script_parsing_detect_access", IntImm(DataType::Int(32), 3));

  Stmt sblock = tir::SBlockRealize(
      {PrimExpr(p_var), PrimExpr(d_var)}, tir::const_true(),
      tir::SBlock(iter_vars, {}, {}, "copy0", body, std::nullopt, {}, {}, sblock_annots));

  Stmt loop = sblock;
  loop = tir::For(d_var, I64(0), I64(d_qk), tir::ForKind::kSerial, loop);
  loop = tir::For(p_var, I64(0), seqlen, tir::ForKind::kSerial, loop);

  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_pages, pages_buf);
  buf_map.Set(h_pos, pos_buf);
  buf_map.Set(h_kv, kv_buf);

  ffi::Array<tir::Var> params = {h_pages, h_pos, h_kv, layer_id};
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
      .def("relax.frontend.nn.llm.kv_cache.kv_cache_debug_get_kv",
           [](int64_t num_hidden_layers, int64_t num_key_value_heads, int64_t head_dim,
              ffi::String dtype) -> tir::PrimFunc {
             return KVCacheDebugGetKV(num_hidden_layers, num_key_value_heads, head_dim,
                                      std::string(dtype));
           })
      .def("relax.frontend.nn.llm.kv_cache.kv_cache_debug_get_kv_mla",
           [](int64_t num_hidden_layers, int64_t d_qk, ffi::String dtype) -> tir::PrimFunc {
             return KVCacheDebugGetKVMLA(num_hidden_layers, d_qk, std::string(dtype));
           });
}

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
