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
 * \file src/relax/frontend/nn/llm/kv_cache_compact_kv_copy.cc
 * \brief TIR kernel generators for KV cache compact-copy operations.
 *
 * Implements:
 *   - CompactKVCopyCpu  (_compact_kv_copy_cpu in Python)
 *   - CompactKVCopy     (_compact_kv_copy GPU in Python)
 *
 * CPU pattern (h=4, d=128, float16):
 * ---------------------------------------------------------------------------
 * @T.prim_func
 * def compact_kv_copy_cpu(var_pages, var_copy_length_indptr, var_copy_src_dst_pos,
 *                          batch_size: T.int32):
 *     T.func_attr({"tir.is_scheduled": True})
 *     num_pages = T.int32()
 *     pages = T.match_buffer(var_pages, (num_pages, 2, 4, 16, 128), "float16")
 *     copy_length_indptr = T.match_buffer(var_copy_length_indptr, (batch_size+1,), "int32",
 *                                          offset_factor=1)
 *     total_copy_length = T.int32()
 *     copy_src_dst_pos = T.match_buffer(var_copy_src_dst_pos, (2, total_copy_length), "int32",
 *                                        offset_factor=1)
 *     with T.sblock("root"):
 *         T.reads(); T.writes()
 *         for bhd_o, bhd_i in T.grid(batch_size * 64, 8):
 *             b = (bhd_o*8+bhd_i) // 512
 *             h = (bhd_o*8+bhd_i) // 128 % 4
 *             d = (bhd_o*8+bhd_i) % 128
 *             if bhd_o*8+bhd_i < batch_size * 4 * 128:
 *                 for i in range(copy_length_indptr[b+1] - copy_length_indptr[b]):
 *                     src_pos = copy_src_dst_pos[0, copy_length_indptr[b]+i]
 *                     dst_pos = copy_src_dst_pos[1, copy_length_indptr[b]+i]
 *                     pages[dst_pos//16, 0, h, dst_pos%16, d] = pages[src_pos//16, 0, h, ...]
 *                     pages[dst_pos//16, 1, h, dst_pos%16, d] = pages[src_pos//16, 1, h, ...]
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
// Shared helper: build the inner copy body
// ---------------------------------------------------------------------------
static Stmt BuildCompactCopyBody(tir::Buffer pages_buf, tir::Buffer copy_indptr_buf,
                                 tir::Buffer copy_pos_buf, PrimExpr flat_idx, PrimExpr batch_size,
                                 int64_t h, int64_t d) {
  // b = flat_idx // (h * d)
  tir::Var b_var("b", DataType::Int(32));
  // h_var = flat_idx // d % h
  tir::Var h_var("h", DataType::Int(32));
  // d_var = flat_idx % d
  tir::Var d_var("d", DataType::Int(32));

  PrimExpr b_expr = CastTo(tir::FloorDiv(flat_idx, I32(h * d)), "int32");
  PrimExpr h_expr = CastTo(tir::FloorMod(tir::FloorDiv(flat_idx, I32(d)), I32(h)), "int32");
  PrimExpr d_expr = CastTo(tir::FloorMod(flat_idx, I32(d)), "int32");

  // Guard: flat_idx < batch_size * h * d
  PrimExpr guard = flat_idx < batch_size * I32(h) * I32(d);

  // Inner loop: for i in range(copy_length_indptr[b+1] - copy_length_indptr[b])
  tir::Var i_var("i", DataType::Int(32));
  tir::Var src_pos("src_pos", DataType::Int(32));
  tir::Var dst_pos("dst_pos", DataType::Int(32));

  PrimExpr indptr_b = tir::BufferLoad(copy_indptr_buf, {b_var});
  PrimExpr indptr_b1 = tir::BufferLoad(copy_indptr_buf, {b_var + I32(1)});
  PrimExpr loop_extent = indptr_b1 - indptr_b;

  PrimExpr src_pos_expr = tir::BufferLoad(copy_pos_buf, {I32(0), indptr_b + i_var});
  PrimExpr dst_pos_expr = tir::BufferLoad(copy_pos_buf, {I32(1), indptr_b + i_var});

  // pages[dst//16, 0, h, dst%16, d] = pages[src//16, 0, h, src%16, d]
  Stmt k_copy = tir::BufferStore(
      pages_buf,
      tir::BufferLoad(pages_buf, {floordiv(src_pos, I32(16)), I32(0), h_var,
                                  floormod(src_pos, I32(16)), d_var}),
      {floordiv(dst_pos, I32(16)), I32(0), h_var, floormod(dst_pos, I32(16)), d_var});
  Stmt v_copy = tir::BufferStore(
      pages_buf,
      tir::BufferLoad(pages_buf, {floordiv(src_pos, I32(16)), I32(1), h_var,
                                  floormod(src_pos, I32(16)), d_var}),
      {floordiv(dst_pos, I32(16)), I32(1), h_var, floormod(dst_pos, I32(16)), d_var});

  Stmt inner_body = tir::SeqStmt({tir::LetStmt(
      src_pos, src_pos_expr, tir::LetStmt(dst_pos, dst_pos_expr, tir::SeqStmt({k_copy, v_copy})))});

  Stmt inner_loop = tir::For(i_var, I32(0), loop_extent, tir::ForKind::kSerial, inner_body);

  // Bind b, h, d as let-stmts then guard
  Stmt guarded = tir::IfThenElse(guard, inner_loop);
  Stmt with_vars = tir::LetStmt(d_var, d_expr,
                                tir::LetStmt(h_var, h_expr, tir::LetStmt(b_var, b_expr, guarded)));
  return with_vars;
}

// ---------------------------------------------------------------------------
// CompactKVCopyCpu
// ---------------------------------------------------------------------------

tir::PrimFunc CompactKVCopyCpu(int64_t num_key_value_heads, int64_t head_dim,
                               const std::string& dtype) {
  DataType dt = DataType(runtime::StringToDLDataType(dtype));

  tir::Var num_pages("num_pages", DataType::Int(32));
  tir::Var batch_size("batch_size", DataType::Int(32));
  tir::Var total_copy_length("total_copy_length", DataType::Int(32));

  tir::Var h_pages("var_pages", DataType::Handle());
  tir::Var h_indptr("var_copy_length_indptr", DataType::Handle());
  tir::Var h_pos("var_copy_src_dst_pos", DataType::Handle());

  // pages: (num_pages, 2, h, 16, d) — no elem_offset for CPU
  tir::Buffer pages_buf = tir::decl_buffer(
      {num_pages, I32(2), I32(num_key_value_heads), I32(16), I32(head_dim)}, dt, "pages");

  // copy_length_indptr: (batch_size+1,) int32, offset_factor=1
  tir::Buffer indptr_buf =
      MakeOffsetFactor1Buffer("copy_length_indptr", {batch_size + I32(1)}, "int32");

  // copy_src_dst_pos: (2, total_copy_length) int32, offset_factor=1
  tir::Buffer pos_buf =
      MakeOffsetFactor1Buffer("copy_src_dst_pos", {I32(2), total_copy_length}, "int32");

  // Loop vars: bhd_o (int32), bhd_i (int32)
  tir::Var bhd_o("bhd_o", DataType::Int(32));
  tir::Var bhd_i("bhd_i", DataType::Int(32));

  PrimExpr flat_idx = bhd_o * I32(8) + bhd_i;

  Stmt body = BuildCompactCopyBody(pages_buf, indptr_buf, pos_buf, flat_idx, batch_size,
                                   num_key_value_heads, head_dim);

  // for bhd_o, bhd_i in T.grid(batch_size * (h*d/8), 8):
  // From test: T.grid(batch_size * 64, 8) where h=4, d=128 → h*d/8 = 512/8 = 64
  PrimExpr outer_extent = batch_size * I32(num_key_value_heads * head_dim / 8);
  PrimExpr inner_extent = I32(8);

  Stmt loop = body;
  loop = tir::For(bhd_i, I32(0), inner_extent, tir::ForKind::kSerial, loop);
  loop = tir::For(bhd_o, I32(0), outer_extent, tir::ForKind::kSerial, loop);

  // Root sblock with empty reads/writes
  Stmt root_block =
      tir::SBlockRealize({}, tir::const_true(), tir::SBlock({}, {}, {}, "root", loop));

  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_pages, pages_buf);
  buf_map.Set(h_indptr, indptr_buf);
  buf_map.Set(h_pos, pos_buf);

  ffi::Array<tir::Var> params = {h_pages, h_indptr, h_pos, batch_size};
  tir::PrimFunc fn(params, root_block, VoidType(), buf_map);
  fn = WithAttr(fn, "tir.is_scheduled", ffi::Any(true));
  return fn;
}

// ---------------------------------------------------------------------------
// CompactKVCopy (GPU)
// ---------------------------------------------------------------------------

tir::PrimFunc CompactKVCopy(int64_t num_key_value_heads, int64_t head_dim, const std::string& dtype,
                            Target target) {
  DataType dt = DataType(runtime::StringToDLDataType(dtype));
  const int64_t threads_per_block = 1024;

  tir::Var num_pages("num_pages", DataType::Int(32));
  tir::Var pages_elem_offset("pages_elem_offset", DataType::Int(64));
  tir::Var batch_size("batch_size", DataType::Int(32));
  tir::Var total_copy_length("total_copy_length", DataType::Int(32));
  tir::Var indptr_elem_offset("copy_length_indptr_elem_offset", DataType::Int(32));
  tir::Var pos_elem_offset("copy_src_dst_pos_elem_offset", DataType::Int(32));

  tir::Var h_pages("var_pages", DataType::Handle());
  tir::Var h_indptr("var_copy_length_indptr", DataType::Handle());
  tir::Var h_pos("var_copy_src_dst_pos", DataType::Handle());

  // pages with elem_offset
  tir::Buffer pages_buf = tir::Buffer(
      tir::decl_buffer({num_pages, I32(2), I32(num_key_value_heads), I32(16), I32(head_dim)}, dt,
                       "pages")
          ->data,
      dt, {num_pages, I32(2), I32(num_key_value_heads), I32(16), I32(head_dim)}, {},
      pages_elem_offset, "pages", 0, 0, tir::kDefault);

  // copy_length_indptr with elem_offset
  tir::Buffer indptr_buf = tir::Buffer(
      tir::decl_buffer({batch_size + I32(1)}, DataType::Int(32), "copy_length_indptr")->data,
      DataType::Int(32), {batch_size + I32(1)}, {}, indptr_elem_offset, "copy_length_indptr", 0, 0,
      tir::kDefault);

  // copy_src_dst_pos with elem_offset
  tir::Buffer pos_buf = tir::Buffer(
      tir::decl_buffer({I32(2), total_copy_length}, DataType::Int(32), "copy_src_dst_pos")->data,
      DataType::Int(32), {I32(2), total_copy_length}, {}, pos_elem_offset, "copy_src_dst_pos", 0, 0,
      tir::kDefault);

  // Thread binding vars
  tir::Var bhd_o("bhd_o", DataType::Int(32));
  tir::Var bhd_i("bhd_i", DataType::Int(32));

  PrimExpr flat_idx = bhd_o * I32(threads_per_block) + bhd_i;

  Stmt body = BuildCompactCopyBody(pages_buf, indptr_buf, pos_buf, flat_idx, batch_size,
                                   num_key_value_heads, head_dim);

  // Grid: (batch_size * h * d + threads_per_block - 1) // threads_per_block
  PrimExpr total = batch_size * I32(num_key_value_heads * head_dim);
  PrimExpr grid_x = tir::FloorDiv(total + I32(threads_per_block - 1), I32(threads_per_block));

  IterVar bx_iv(Range::FromMinExtent(I32(0), grid_x), bhd_o, tir::kThreadIndex, "blockIdx.x");
  IterVar tx_iv(Range::FromMinExtent(I32(0), I32(threads_per_block)), bhd_i, tir::kThreadIndex,
                "threadIdx.x");

  Stmt inner =
      tir::For(bhd_i, I32(0), I32(threads_per_block), tir::ForKind::kThreadBinding, body, tx_iv);
  Stmt outer = tir::For(bhd_o, I32(0), grid_x, tir::ForKind::kThreadBinding, inner, bx_iv);

  // Root sblock
  Stmt root_block =
      tir::SBlockRealize({}, tir::const_true(), tir::SBlock({}, {}, {}, "root", outer));

  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_pages, pages_buf);
  buf_map.Set(h_indptr, indptr_buf);
  buf_map.Set(h_pos, pos_buf);

  ffi::Array<tir::Var> params = {h_pages, h_indptr, h_pos, batch_size};
  tir::PrimFunc fn(params, root_block, VoidType(), buf_map);
  fn = WithAttr(fn, "tir.is_scheduled", ffi::Any(true));
  return fn;
}

// ---------------------------------------------------------------------------
// FFI registrations
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.frontend.nn.llm.kv_cache.compact_kv_copy_cpu",
           [](int64_t num_key_value_heads, int64_t head_dim, ffi::String dtype) -> tir::PrimFunc {
             return CompactKVCopyCpu(num_key_value_heads, head_dim, std::string(dtype));
           })
      .def("relax.frontend.nn.llm.kv_cache.compact_kv_copy",
           [](int64_t num_key_value_heads, int64_t head_dim, ffi::String dtype,
              Target target) -> tir::PrimFunc {
             return CompactKVCopy(num_key_value_heads, head_dim, std::string(dtype), target);
           });
}

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
