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
 * \file src/relax/frontend/nn/llm/kv_cache_copy_single_page.cc
 * \brief TIR kernel generators for KV cache copy-single-page operations.
 *
 * Implements:
 *   - CopySinglePageCpu   (_copy_single_page_cpu in Python)
 *   - CopySinglePage      (_copy_single_page GPU in Python)
 *   - CopySinglePageMLA   (_copy_single_page_mla GPU in Python)
 *
 * CPU pattern (h=4, page_size=16, d=128, float16):
 * ---------------------------------------------------------------------------
 * @T.prim_func
 * def copy_single_page_cpu(var_pages, src_page_id: T.int64, tgt_page_id: T.int64,
 *                           copy_length: T.int64):
 *     T.func_attr({"tir.is_scheduled": True})
 *     num_pages = T.int32()
 *     pages = T.match_buffer(var_pages, (num_pages, 2, 4, 16, 128), "float16")
 *     for b, t in T.grid(copy_length * 512, 1):
 *         with T.sblock("copy"):
 *             vh = T.axis.spatial(4, cast((b+t)//(copy_length*128)))
 *             vp = T.axis.spatial(copy_length, (b+t)%(copy_length*128)//128)
 *             vd = T.axis.spatial(128, cast((b+t)%128))
 *             T.where(b+t < copy_length*4*128)
 *             pages[tgt_page_id, 0, vh, vp, vd] = pages[src_page_id, 0, vh, vp, vd]
 *             pages[tgt_page_id, 1, vh, vp, vd] = pages[src_page_id, 1, vh, vp, vd]
 * ---------------------------------------------------------------------------
 *
 * GPU pattern (h=4, page_size=16, d=128, float16):
 * ---------------------------------------------------------------------------
 * @T.prim_func
 * def copy_single_page(var_pages, src_page_id: T.int64, tgt_page_id: T.int64,
 *                       copy_length: T.int64):
 *     T.func_attr({"tir.is_scheduled": True})
 *     num_pages = T.int32()
 *     pages_elem_offset = T.int64()
 *     pages = T.match_buffer(var_pages, (num_pages, 2, 4, 16, 128), "float16",
 *                            elem_offset=pages_elem_offset)
 *     for b in T.thread_binding((copy_length*512+1023)//1024, thread="blockIdx.x"):
 *         for t in T.thread_binding(1024, thread="threadIdx.x"):
 *             with T.sblock("copy"):
 *                 vh = T.axis.spatial(4, cast((b*1024+t)//(copy_length*128)))
 *                 vp = T.axis.spatial(copy_length, (b*1024+t)%(copy_length*128)//128)
 *                 vd = T.axis.spatial(128, cast((b*1024+t)%128))
 *                 T.where(b*1024+t < copy_length*4*128)
 *                 pages[tgt_page_id, 0, vh, vp, vd] = pages[src_page_id, 0, vh, vp, vd]
 *                 pages[tgt_page_id, 1, vh, vp, vd] = pages[src_page_id, 1, vh, vp, vd]
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
// Shared helper: build the sblock body for copy_single_page
//
// The sblock computes:
//   vh  = cast<int32>((flat_idx) // (copy_length * d))
//   vp  = (flat_idx) % (copy_length * d) // d
//   vd  = cast<int32>((flat_idx) % d)
//   where: flat_idx < copy_length * h * d
//   pages[tgt, 0, vh, vp, vd] = pages[src, 0, vh, vp, vd]
//   pages[tgt, 1, vh, vp, vd] = pages[src, 1, vh, vp, vd]
// ---------------------------------------------------------------------------
struct CopySinglePageSblock {
  tir::Var vh, vp, vd;
  tir::Buffer pages_buf;
  tir::Var src_page_id, tgt_page_id, copy_length;
  int64_t h, d;

  // flat_idx = b_expr (already computed by caller as b*stride + t)
  Stmt Build(PrimExpr flat_idx, PrimExpr b_loop, PrimExpr t_loop) const {
    // vh = cast<int32>(flat_idx // (copy_length * d))
    PrimExpr vh_expr = CastTo(tir::FloorDiv(flat_idx, copy_length * I64(d)), "int32");
    // vp = flat_idx % (copy_length * d) // d
    PrimExpr vp_expr = tir::FloorDiv(tir::FloorMod(flat_idx, copy_length * I64(d)), I64(d));
    // vd = cast<int32>(flat_idx % d)
    PrimExpr vd_expr = CastTo(tir::FloorMod(flat_idx, I64(d)), "int32");

    // T.where: flat_idx < copy_length * h * d
    PrimExpr where_cond = flat_idx < copy_length * I64(h) * I64(d);

    // reads: pages[src, 0:2, vh, vp, vd]
    // writes: pages[tgt, 0:2, vh, vp, vd]
    Stmt k_store = tir::BufferStore(
        pages_buf,
        tir::BufferLoad(pages_buf, {src_page_id, I64(0), vh, vp, vd}),
        {tgt_page_id, I64(0), vh, vp, vd});
    Stmt v_store = tir::BufferStore(
        pages_buf,
        tir::BufferLoad(pages_buf, {src_page_id, I64(1), vh, vp, vd}),
        {tgt_page_id, I64(1), vh, vp, vd});
    Stmt body = tir::SeqStmt({k_store, v_store});

    // IterVars for the sblock
    ffi::Array<tir::IterVar> iter_vars = {
        tir::IterVar(Range::FromMinExtent(I32(0), I32(h)), vh, tir::kDataPar, ""),
        tir::IterVar(Range::FromMinExtent(I64(0), copy_length), vp, tir::kDataPar, ""),
        tir::IterVar(Range::FromMinExtent(I32(0), I32(d)), vd, tir::kDataPar, "")};

    // reads/writes annotations
    ffi::Array<tir::BufferRegion> reads = {
        tir::BufferRegion(pages_buf,
                          {Range::FromMinExtent(src_page_id, I64(1)),
                           Range::FromMinExtent(I64(0), I64(2)),
                           Range::FromMinExtent(vh, I32(1)),
                           Range::FromMinExtent(vp, I64(1)),
                           Range::FromMinExtent(vd, I32(1))})};
    ffi::Array<tir::BufferRegion> writes = {
        tir::BufferRegion(pages_buf,
                          {Range::FromMinExtent(tgt_page_id, I64(1)),
                           Range::FromMinExtent(I64(0), I64(2)),
                           Range::FromMinExtent(vh, I32(1)),
                           Range::FromMinExtent(vp, I64(1)),
                           Range::FromMinExtent(vd, I32(1))})};

    Stmt sblock = tir::SBlockRealize(
        {vh_expr, vp_expr, vd_expr},
        where_cond,
        tir::SBlock(iter_vars, reads, writes, "copy", body));

    return sblock;
  }
};

// ---------------------------------------------------------------------------
// CopySinglePageCpu
// ---------------------------------------------------------------------------

tir::PrimFunc CopySinglePageCpu(int64_t num_key_value_heads, int64_t page_size, int64_t head_dim,
                                const std::string& dtype) {
  DataType dt = DataType(runtime::StringToDLDataType(dtype));

  tir::Var num_pages("num_pages", DataType::Int(32));
  tir::Var h_pages("var_pages", DataType::Handle());
  tir::Var src_page_id("src_page_id", DataType::Int(64));
  tir::Var tgt_page_id("tgt_page_id", DataType::Int(64));
  tir::Var copy_length("copy_length", DataType::Int(64));

  // pages: (num_pages, 2, h, page_size, d)  — no elem_offset for CPU
  tir::Buffer pages_buf = tir::decl_buffer(
      {CastTo(num_pages, "int64"), I64(2), I64(num_key_value_heads), I64(page_size), I64(head_dim)},
      dt, "pages");

  // Loop vars: b (int64), t (int64)
  tir::Var b_var("b", DataType::Int(64));
  tir::Var t_var("t", DataType::Int(64));

  // Axis vars
  tir::Var vh("vh", DataType::Int(32));
  tir::Var vp("vp", DataType::Int(64));
  tir::Var vd("vd", DataType::Int(32));

  // flat_idx = b + cast<int64>(t)
  PrimExpr flat_idx = b_var + CastTo(t_var, "int64");

  CopySinglePageSblock helper{vh, vp, vd, pages_buf, src_page_id, tgt_page_id, copy_length,
                               num_key_value_heads, head_dim};
  Stmt sblock = helper.Build(flat_idx, b_var, t_var);

  // for b, t in T.grid(copy_length * h * d / d, 1):
  // = for b in range(copy_length * h * d / d), t in range(1)
  // Actually: copy_length * (h * d / d) = copy_length * h ... no.
  // From test: for b, t in T.grid(copy_length * 512, 1)  where h=4, d=128 → h*d=512
  PrimExpr outer_extent = copy_length * I64(num_key_value_heads * head_dim / head_dim);
  // Wait: h=4, d=128 → 4*128=512. So outer = copy_length * h * d / d = copy_length * h
  // But test shows copy_length * 512 = copy_length * h * d
  // Actually: copy_length * h * d / (threads_per_block=1) = copy_length * h * d
  // For CPU: outer = copy_length * h * d, inner = 1
  outer_extent = copy_length * I64(num_key_value_heads) * I64(head_dim) / I64(head_dim);
  // Hmm, let me re-read: test shows "copy_length * T.int64(512)" where 512 = h*d = 4*128
  // So outer = copy_length * h * d / d? No: copy_length * 512 = copy_length * 4 * 128
  // The grid is (copy_length * h * d, 1) but divided by... no.
  // Actually looking at the test more carefully:
  // for b, t in T.grid(copy_length * T.int64(512), 1):
  // 512 = 4 * 128 = h * d. So outer = copy_length * h * d, inner = 1.
  // But then flat_idx = b + t, and vh = flat_idx // (copy_length * d)
  // = (b + t) // (copy_length * 128). With b in [0, copy_length*512), t=0:
  // vh = b // (copy_length * 128) ∈ [0, 4) ✓
  // vp = b % (copy_length * 128) // 128 ∈ [0, copy_length) ✓
  // vd = b % 128 ∈ [0, 128) ✓
  // where: b < copy_length * 4 * 128 ✓ (always true since b < copy_length*512=copy_length*4*128)
  // So outer = copy_length * h * d, inner = 1. ✓
  outer_extent = copy_length * I64(num_key_value_heads * head_dim);

  Stmt loop = sblock;
  loop = tir::For(t_var, I64(0), I64(1), tir::ForKind::kSerial, loop);
  loop = tir::For(b_var, I64(0), outer_extent, tir::ForKind::kSerial, loop);

  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_pages, pages_buf);

  ffi::Array<tir::Var> params = {h_pages, src_page_id, tgt_page_id, copy_length};
  tir::PrimFunc fn(params, loop, VoidType(), buf_map);
  fn = WithAttr(fn, "tir.is_scheduled", ffi::Any(true));
  return fn;
}

// ---------------------------------------------------------------------------
// CopySinglePage (GPU)
// ---------------------------------------------------------------------------

tir::PrimFunc CopySinglePage(int64_t num_key_value_heads, int64_t page_size, int64_t head_dim,
                             const std::string& dtype, Target target) {
  DataType dt = DataType(runtime::StringToDLDataType(dtype));
  const int64_t threads_per_block = 1024;

  tir::Var num_pages("num_pages", DataType::Int(32));
  tir::Var pages_elem_offset("pages_elem_offset", DataType::Int(64));
  tir::Var h_pages("var_pages", DataType::Handle());
  tir::Var src_page_id("src_page_id", DataType::Int(64));
  tir::Var tgt_page_id("tgt_page_id", DataType::Int(64));
  tir::Var copy_length("copy_length", DataType::Int(64));

  // pages: (num_pages, 2, h, page_size, d) with elem_offset
  tir::Buffer pages_buf = tir::Buffer(
      tir::decl_buffer(
          {CastTo(num_pages, "int64"), I64(2), I64(num_key_value_heads), I64(page_size),
           I64(head_dim)},
          dt, "pages")->data,
      dt,
      {CastTo(num_pages, "int64"), I64(2), I64(num_key_value_heads), I64(page_size), I64(head_dim)},
      {}, pages_elem_offset, "pages", 0, 0, tir::kDefault);

  // Thread binding vars
  tir::Var b_var("b", DataType::Int(64));
  tir::Var t_var("t", DataType::Int(64));

  // Axis vars
  tir::Var vh("vh", DataType::Int(32));
  tir::Var vp("vp", DataType::Int(64));
  tir::Var vd("vd", DataType::Int(32));

  // flat_idx = b * threads_per_block + cast<int64>(t)
  PrimExpr flat_idx = b_var * I64(threads_per_block) + CastTo(t_var, "int64");

  CopySinglePageSblock helper{vh, vp, vd, pages_buf, src_page_id, tgt_page_id, copy_length,
                               num_key_value_heads, head_dim};
  Stmt sblock = helper.Build(flat_idx, b_var, t_var);

  // Grid: (copy_length * h * d + threads_per_block - 1) // threads_per_block
  PrimExpr total = copy_length * I64(num_key_value_heads * head_dim);
  PrimExpr grid_x = tir::FloorDiv(total + I64(threads_per_block - 1), I64(threads_per_block));

  // Thread binding loops
  IterVar bx_iv(Range::FromMinExtent(I64(0), grid_x), b_var, tir::kThreadIndex, "blockIdx.x");
  IterVar tx_iv(Range::FromMinExtent(I64(0), I64(threads_per_block)), t_var, tir::kThreadIndex,
                "threadIdx.x");

  Stmt inner = tir::For(t_var, I64(0), I64(threads_per_block), tir::ForKind::kThreadBinding,
                        sblock, tx_iv);
  Stmt outer = tir::For(b_var, I64(0), grid_x, tir::ForKind::kThreadBinding, inner, bx_iv);

  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_pages, pages_buf);

  ffi::Array<tir::Var> params = {h_pages, src_page_id, tgt_page_id, copy_length};
  tir::PrimFunc fn(params, outer, VoidType(), buf_map);
  fn = WithAttr(fn, "tir.is_scheduled", ffi::Any(true));
  return fn;
}

// ---------------------------------------------------------------------------
// CopySinglePageMLA (GPU)
// ---------------------------------------------------------------------------

tir::PrimFunc CopySinglePageMLA(int64_t page_size, int64_t d_qk, const std::string& dtype,
                                Target target) {
  DataType dt = DataType(runtime::StringToDLDataType(dtype));
  const int64_t threads_per_block = 1024;

  tir::Var num_pages("num_pages", DataType::Int(32));
  tir::Var pages_elem_offset("pages_elem_offset", DataType::Int(64));
  tir::Var h_pages("var_pages", DataType::Handle());
  tir::Var src_page_id("src_page_id", DataType::Int(64));
  tir::Var tgt_page_id("tgt_page_id", DataType::Int(64));
  tir::Var copy_length("copy_length", DataType::Int(64));

  // pages: (num_pages, page_size, d_qk) with elem_offset
  tir::Buffer pages_buf = tir::Buffer(
      tir::decl_buffer({CastTo(num_pages, "int64"), I64(page_size), I64(d_qk)}, dt, "pages")->data,
      dt, {CastTo(num_pages, "int64"), I64(page_size), I64(d_qk)}, {},
      pages_elem_offset, "pages", 0, 0, tir::kDefault);

  tir::Var b_var("b", DataType::Int(64));
  tir::Var t_var("t", DataType::Int(64));

  // Axis vars
  tir::Var vp("vp", DataType::Int(64));
  tir::Var vd("vd", DataType::Int(32));

  PrimExpr flat_idx = b_var * I64(threads_per_block) + CastTo(t_var, "int64");

  // vp = flat_idx // d_qk
  PrimExpr vp_expr = tir::FloorDiv(flat_idx, I64(d_qk));
  // vd = cast<int32>(flat_idx % d_qk)
  PrimExpr vd_expr = CastTo(tir::FloorMod(flat_idx, I64(d_qk)), "int32");

  PrimExpr where_cond = flat_idx < copy_length * I64(d_qk);

  Stmt body = tir::BufferStore(
      pages_buf,
      tir::BufferLoad(pages_buf, {src_page_id, vp, vd}),
      {tgt_page_id, vp, vd});

  ffi::Array<tir::IterVar> iter_vars = {
      tir::IterVar(Range::FromMinExtent(I64(0), copy_length), vp, tir::kDataPar, ""),
      tir::IterVar(Range::FromMinExtent(I32(0), I32(d_qk)), vd, tir::kDataPar, "")};

  ffi::Array<tir::BufferRegion> reads = {
      tir::BufferRegion(pages_buf,
                        {Range::FromMinExtent(src_page_id, I64(1)),
                         Range::FromMinExtent(vp, I64(1)),
                         Range::FromMinExtent(vd, I32(1))})};
  ffi::Array<tir::BufferRegion> writes = {
      tir::BufferRegion(pages_buf,
                        {Range::FromMinExtent(tgt_page_id, I64(1)),
                         Range::FromMinExtent(vp, I64(1)),
                         Range::FromMinExtent(vd, I32(1))})};

  Stmt sblock = tir::SBlockRealize(
      {vp_expr, vd_expr},
      where_cond,
      tir::SBlock(iter_vars, reads, writes, "copy", body));

  PrimExpr total = copy_length * I64(d_qk);
  PrimExpr grid_x = tir::FloorDiv(total + I64(threads_per_block - 1), I64(threads_per_block));

  IterVar bx_iv(Range::FromMinExtent(I64(0), grid_x), b_var, tir::kThreadIndex, "blockIdx.x");
  IterVar tx_iv(Range::FromMinExtent(I64(0), I64(threads_per_block)), t_var, tir::kThreadIndex,
                "threadIdx.x");

  Stmt inner = tir::For(t_var, I64(0), I64(threads_per_block), tir::ForKind::kThreadBinding,
                        sblock, tx_iv);
  Stmt outer = tir::For(b_var, I64(0), grid_x, tir::ForKind::kThreadBinding, inner, bx_iv);

  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_pages, pages_buf);

  ffi::Array<tir::Var> params = {h_pages, src_page_id, tgt_page_id, copy_length};
  tir::PrimFunc fn(params, outer, VoidType(), buf_map);
  fn = WithAttr(fn, "tir.is_scheduled", ffi::Any(true));
  return fn;
}

// ---------------------------------------------------------------------------
// FFI registrations
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.frontend.nn.llm.kv_cache.copy_single_page_cpu",
           [](int64_t num_key_value_heads, int64_t page_size, int64_t head_dim,
              ffi::String dtype) -> tir::PrimFunc {
             return CopySinglePageCpu(num_key_value_heads, page_size, head_dim,
                                     std::string(dtype));
           })
      .def("relax.frontend.nn.llm.kv_cache.copy_single_page",
           [](int64_t num_key_value_heads, int64_t page_size, int64_t head_dim, ffi::String dtype,
              Target target) -> tir::PrimFunc {
             return CopySinglePage(num_key_value_heads, page_size, head_dim, std::string(dtype),
                                   target);
           })
      .def("relax.frontend.nn.llm.kv_cache.copy_single_page_mla",
           [](int64_t page_size, int64_t d_qk, ffi::String dtype,
              Target target) -> tir::PrimFunc {
             return CopySinglePageMLA(page_size, d_qk, std::string(dtype), target);
           });
}

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
