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
 * \file src/dlight/adreno/tensor_intrin_adreno.cc
 * \brief TIR tensor intrinsics for Adreno cooperative-matrix (wmma) operations.
 *
 * Provides:
 *   - GetWmmaTileSizes()          – returns (M, N, K) tile sizes for a dtype pair
 *   - GetAdrenoWmmaIntrinGroup()  – builds and registers the five wmma intrinsics
 *                                    (fill, load_a, load_b, sync, store) for a
 *                                    given shape, dtype, and memory scope
 *
 * All TIR PrimFunc bodies are constructed programmatically using the
 * tir::SBlock / tir::For / tir::BufferStore / tir::Evaluate node constructors.
 *
 * Supported WMMA profiles (in_dtype, out_dtype) → (M, N, K):
 *   (float16, float16) → (64, 64, 16)
 *   (float16, float32) → (64, 64, 16)
 *   (float32, float32) → (64, 64,  8)
 *   (int8,    int32)   → (64, 64, 32)
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace dlight {
namespace adreno {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/*! \brief Supported WMMA profiles: (in_dtype, out_dtype) → (M, N, K). */
static const std::unordered_map<std::string, std::tuple<int, int, int>> kSupportedProfiles = {
    {"float16:float16", {64, 64, 16}},
    {"float16:float32", {64, 64, 16}},
    {"float32:float32", {64, 64, 8}},
    {"int8:int32", {64, 64, 32}},
};

/*!
 * \brief Return the (M, N, K) tile sizes for the given dtype combination.
 * Returns false when the combination is unsupported.
 */
bool GetWmmaTileSizes(const std::string& in_dtype, const std::string& out_dtype, int* tile_m,
                      int* tile_n, int* tile_k) {
  std::string key = in_dtype + ":" + out_dtype;
  auto it = kSupportedProfiles.find(key);
  if (it == kSupportedProfiles.end()) return false;
  *tile_m = std::get<0>(it->second);
  *tile_n = std::get<1>(it->second);
  *tile_k = std::get<2>(it->second);
  return true;
}

/*! \brief Return a shorthand suffix for a dtype string (e.g. "float16" → "f16"). */
static std::string ShorthandDtype(const std::string& dtype) {
  if (dtype == "float16") return "f16";
  if (dtype == "float32") return "f32";
  if (dtype == "int8") return "i8";
  if (dtype == "int32") return "i32";
  TVM_FFI_THROW(InternalError) << "Unsupported dtype for shorthand: " << dtype;
}

/*! \brief Convert a dtype string to a TVM DataType. */
static DataType DTypeFromStr(const std::string& dtype) {
  if (dtype == "float16") return DataType::Float(16);
  if (dtype == "float32") return DataType::Float(32);
  if (dtype == "int8") return DataType::Int(8);
  if (dtype == "int32") return DataType::Int(32);
  TVM_FFI_THROW(InternalError) << "Unsupported dtype string: " << dtype;
}

/*!
 * \brief Compute the wmma fragment index from a buffer's elem_offset and stride.
 *
 * Given a 2-D fragment buffer with shape [m_dim, n_dim] and a symbolic
 * stride variable, the fragment index is:
 *   frag_index_m = elem_offset // stride // m_dim
 *   frag_index_n = (elem_offset % stride) // n_dim
 *   num_fragments_per_row = stride // n_dim
 *   return frag_index_m * num_fragments_per_row + frag_index_n
 */
static PrimExpr WmmaFragmentIndex(const tir::Buffer& buf, const tir::Var& stride, int m_dim,
                                  int n_dim) {
  PrimExpr elem_offset = buf->elem_offset;
  PrimExpr frag_index_m = floordiv(floordiv(elem_offset, stride), IntImm(DataType::Int(32), m_dim));
  PrimExpr frag_index_n = floordiv(floormod(elem_offset, stride), IntImm(DataType::Int(32), n_dim));
  PrimExpr num_frags_per_row = floordiv(stride, IntImm(DataType::Int(32), n_dim));
  return frag_index_m * num_frags_per_row + frag_index_n;
}

// ---------------------------------------------------------------------------
// Helper: build a Buffer with strides (for impl functions)
// ---------------------------------------------------------------------------

/*!
 * \brief Declare a buffer with explicit stride variables and a symbolic elem_offset.
 *
 * The returned buffer has strides = {stride1, stride0} and a free Var
 * elem_offset so that WmmaFragmentIndex produces a symbolic expression
 * that the tensorize lowering substitutes with the actual offset.
 */
static tir::Buffer DeclBufferWithStrides(const std::string& name, const ffi::Array<PrimExpr>& shape,
                                         const DataType& dtype, const std::string& scope,
                                         int data_alignment, int offset_factor,
                                         const ffi::Array<PrimExpr>& strides) {
  Type elem_type = PrimType(dtype);
  Type ptr_type = PointerType(elem_type, scope);
  tir::Var data_var(name + "_data", ptr_type);
  // Use a symbolic elem_offset variable so that WmmaFragmentIndex produces
  // a non-trivial symbolic expression (e.g. elem_offset // stride // m * ...)
  // rather than the constant 0.
  tir::Var elem_offset_var(name + "_elem_offset", DataType::Int(32));
  return tir::Buffer(data_var, dtype, shape, strides, elem_offset_var, name, data_alignment,
                     offset_factor, tir::kDefault);
}

/*!
 * \brief Declare a buffer without strides (contiguous).
 */
static tir::Buffer DeclBufferContiguous(const std::string& name, const ffi::Array<PrimExpr>& shape,
                                        const DataType& dtype, const std::string& scope,
                                        int data_alignment, int offset_factor) {
  return tir::BufferWithOffsetAlignment(shape, dtype, name, data_alignment, offset_factor,
                                        /*compact=*/false, scope);
}

// ---------------------------------------------------------------------------
// Helper: build a 2-D nested For loop with a BufferStore body
// ---------------------------------------------------------------------------

/*!
 * \brief Build:
 *   for i in range(ext_i):
 *     for j in range(ext_j):
 *       body(i_var, j_var)
 */
static tir::Stmt NestedFor2D(const tir::Var& i_var, const tir::Var& j_var, int ext_i, int ext_j,
                             tir::Stmt inner_body) {
  tir::Stmt j_loop = tir::For(j_var, IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), ext_j),
                              tir::ForKind::kSerial, inner_body);
  return tir::For(i_var, IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), ext_i),
                  tir::ForKind::kSerial, j_loop);
}

// ---------------------------------------------------------------------------
// Helper: build a 1-D For loop
// ---------------------------------------------------------------------------

static tir::Stmt For1D(const tir::Var& v, int extent, tir::Stmt body) {
  return tir::For(v, IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), extent),
                  tir::ForKind::kSerial, body);
}

// ---------------------------------------------------------------------------
// Helper: build a thread-binding For loop (threadIdx.x)
// ---------------------------------------------------------------------------

static tir::Stmt ThreadBindingFor(const tir::Var& v, int extent, const std::string& thread_tag,
                                  tir::Stmt body) {
  tir::IterVar thread_iv(Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), extent)),
                         tir::Var(thread_tag, DataType::Int(32)), tir::kThreadIndex, thread_tag);
  return tir::For(v, IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), extent),
                  tir::ForKind::kThreadBinding, body, thread_iv);
}

// ---------------------------------------------------------------------------
// Helper: build a root SBlock wrapping a body Stmt
// ---------------------------------------------------------------------------

static tir::Stmt RootSBlock(const ffi::Array<tir::BufferRegion>& reads,
                            const ffi::Array<tir::BufferRegion>& writes, tir::Stmt body) {
  return tir::SBlockRealize(
      /*iter_values=*/{}, /*predicate=*/IntImm(DataType::Bool(), 1),
      tir::SBlock(/*iter_vars=*/{}, reads, writes, "root", body));
}

// ---------------------------------------------------------------------------
// Helper: build an inner SBlock (e.g. "init", "load", "store", "")
// ---------------------------------------------------------------------------

static tir::Stmt InnerSBlock(const std::string& name, const ffi::Array<tir::IterVar>& iter_vars,
                             const ffi::Array<PrimExpr>& iter_values,
                             const ffi::Array<tir::BufferRegion>& reads,
                             const ffi::Array<tir::BufferRegion>& writes, tir::Stmt body) {
  return tir::SBlockRealize(iter_values, /*predicate=*/IntImm(DataType::Bool(), 1),
                            tir::SBlock(iter_vars, reads, writes, name, body));
}

// ---------------------------------------------------------------------------
// Helper: build a point BufferRegion from iter vars [v0, v1, ...]
// Each dimension is Range(v, v+1) — a single-element region.
// This matches what Python produces for inner block writes like C[vii, vjj].
// ---------------------------------------------------------------------------

static tir::BufferRegion PointRegion2D(const tir::Buffer& buf, const tir::Var& v0,
                                       const tir::Var& v1) {
  return tir::BufferRegion(buf, {Range(PrimExpr(v0), PrimExpr(v0) + IntImm(DataType::Int(32), 1)),
                                 Range(PrimExpr(v1), PrimExpr(v1) + IntImm(DataType::Int(32), 1))});
}

static tir::BufferRegion PointRegion1D(const tir::Buffer& buf, const tir::Var& v0) {
  return tir::BufferRegion(buf, {Range(PrimExpr(v0), PrimExpr(v0) + IntImm(DataType::Int(32), 1))});
}

// ---------------------------------------------------------------------------
// Helper: build a full 2-D BufferRegion [0:dim0, 0:dim1]
// ---------------------------------------------------------------------------

static tir::BufferRegion FullRegion2D(const tir::Buffer& buf, int dim0, int dim1) {
  return tir::BufferRegion(buf,
                           {Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), dim0)),
                            Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), dim1))});
}

// ---------------------------------------------------------------------------
// Helper: build a full 1-D BufferRegion [0:dim0]
// ---------------------------------------------------------------------------

static tir::BufferRegion FullRegion1D(const tir::Buffer& buf, int dim0) {
  return tir::BufferRegion(buf,
                           {Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), dim0))});
}

// ===========================================================================
// get_wmma_fill_intrin
// ===========================================================================

/*!
 * \brief Build the wmma fill (accumulator initialisation) intrinsic.
 *
 * desc: nested loops over (m_dim, n_dim) that zero-initialise the
 *       wmma.accumulator fragment element by element.
 * impl: a single tvm_fill_fragment call that initialises the entire fragment.
 */
static std::pair<tir::PrimFunc, tir::PrimFunc> GetWmmaFillIntrin(int m_dim, int n_dim, int k_dim,
                                                                 const std::string& dtype) {
  DataType dt = DTypeFromStr(dtype);
  int offset_factor = n_dim;
  const std::string scope = "wmma.accumulator";

  // ---- desc ----
  tir::PrimFunc desc = [&]() -> tir::PrimFunc {
    tir::Var c_param("c", DataType::Handle());
    tir::Buffer C = DeclBufferContiguous(
        "C", {IntImm(DataType::Int(32), m_dim), IntImm(DataType::Int(32), n_dim)}, dt, scope, 64,
        offset_factor);

    tir::Var vi("vii", DataType::Int(32)), vj("vjj", DataType::Int(32));
    tir::IterVar iv_i(Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), m_dim)), vi,
                      tir::kDataPar);
    tir::IterVar iv_j(Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), n_dim)), vj,
                      tir::kDataPar);

    // Loop vars must be declared before the block that uses them as iter_values.
    // iter_values = [i, j] (loop vars). Inner block writes are POINT regions.
    tir::Var i("i", DataType::Int(32)), j("j", DataType::Int(32));

    PrimExpr zero = tir::make_const(dt, 0);
    tir::Stmt init_body = tir::BufferStore(C, zero, {vi, vj});
    tir::Stmt init_block = InnerSBlock("init", {iv_i, iv_j}, {PrimExpr(i), PrimExpr(j)}, {},
                                       {PointRegion2D(C, vi, vj)}, init_body);

    tir::Stmt loops = NestedFor2D(i, j, m_dim, n_dim, init_block);

    tir::Stmt root_body = RootSBlock({}, {FullRegion2D(C, m_dim, n_dim)}, loops);

    ffi::Map<tir::Var, tir::Buffer> buf_map;
    buf_map.Set(c_param, C);
    return tir::PrimFunc({c_param}, root_body, VoidType(), buf_map);
  }();

  // ---- impl ----
  tir::PrimFunc impl = [&]() -> tir::PrimFunc {
    tir::Var c_param("c", DataType::Handle());
    tir::Var d1("d1", DataType::Int(32)), d0("d0", DataType::Int(32));
    tir::Buffer C = DeclBufferWithStrides(
        "C", {IntImm(DataType::Int(32), m_dim), IntImm(DataType::Int(32), n_dim)}, dt, scope, 64,
        offset_factor, {d1, d0});

    // tvm_fill_fragment(C.data, m, n, k, frag_idx, 0.0)
    PrimExpr frag_idx = WmmaFragmentIndex(C, d1, m_dim, n_dim);
    PrimExpr fill_call =
        tir::Call(DataType::Handle(), tir::builtin::tvm_fill_fragment(),
                  {C->data, IntImm(DataType::Int(32), m_dim), IntImm(DataType::Int(32), n_dim),
                   IntImm(DataType::Int(32), k_dim), frag_idx, FloatImm(DataType::Float(32), 0.0)});
    tir::Stmt root_body = RootSBlock({}, {FullRegion2D(C, m_dim, n_dim)}, tir::Evaluate(fill_call));

    ffi::Map<tir::Var, tir::Buffer> buf_map;
    buf_map.Set(c_param, C);
    return tir::PrimFunc({c_param}, root_body, VoidType(), buf_map);
  }();

  return {desc, impl};
}

// ===========================================================================
// get_wmma_load_intrin
// ===========================================================================

/*!
 * \brief Build the wmma load intrinsic for matrix A or B.
 *
 * desc: nested loops over the fragment shape that copy element by element
 *       from the source buffer (shared_scope) to the wmma fragment.
 * impl: a thread-binding loop over 64 threads that issues a single
 *       tvm_load_matrix_sync call.
 */
static std::pair<tir::PrimFunc, tir::PrimFunc> GetWmmaLoadIntrin(int m_dim, int n_dim, int k_dim,
                                                                 const std::string& dtype,
                                                                 const std::string& shared_scope,
                                                                 bool is_b, bool is_col_major) {
  DataType dt = DTypeFromStr(dtype);
  std::string wmma_scope = std::string("wmma.matrix_") + (is_b ? "b" : "a");
  std::string layout = is_col_major ? "col_major" : "row_major";

  int frag_m = is_b ? k_dim : m_dim;
  int frag_n = is_b ? n_dim : k_dim;
  if (is_col_major) std::swap(frag_m, frag_n);
  int offset_factor = frag_n;

  // ---- desc ----
  tir::PrimFunc desc = [&]() -> tir::PrimFunc {
    tir::Var a_param("a", DataType::Handle()), c_param("c", DataType::Handle());
    tir::Buffer A = DeclBufferContiguous(
        "A", {IntImm(DataType::Int(32), frag_m), IntImm(DataType::Int(32), frag_n)}, dt,
        shared_scope, 64, offset_factor);
    tir::Buffer C = DeclBufferContiguous(
        "C", {IntImm(DataType::Int(32), frag_m), IntImm(DataType::Int(32), frag_n)}, dt, wmma_scope,
        64, offset_factor);

    tir::Var vi("vii", DataType::Int(32)), vj("vjj", DataType::Int(32));
    tir::IterVar iv_i(Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), frag_m)), vi,
                      tir::kDataPar);
    tir::IterVar iv_j(Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), frag_n)), vj,
                      tir::kDataPar);

    // Loop vars declared before the block; iter_values = [i, j] (loop vars).
    // Inner block read/write regions are POINT regions (extent=1 per dim).
    tir::Var i("i", DataType::Int(32)), j("j", DataType::Int(32));

    tir::Stmt load_body = tir::BufferStore(C, tir::BufferLoad(A, {vi, vj}), {vi, vj});
    tir::Stmt load_block =
        InnerSBlock("load", {iv_i, iv_j}, {PrimExpr(i), PrimExpr(j)}, {PointRegion2D(A, vi, vj)},
                    {PointRegion2D(C, vi, vj)}, load_body);

    tir::Stmt loops = NestedFor2D(i, j, frag_m, frag_n, load_block);

    tir::Stmt root_body =
        RootSBlock({FullRegion2D(A, frag_m, frag_n)}, {FullRegion2D(C, frag_m, frag_n)}, loops);

    ffi::Map<tir::Var, tir::Buffer> buf_map;
    buf_map.Set(a_param, A);
    buf_map.Set(c_param, C);
    return tir::PrimFunc({a_param, c_param}, root_body, VoidType(), buf_map);
  }();

  // ---- impl ----
  tir::PrimFunc impl = [&]() -> tir::PrimFunc {
    tir::Var a_param("a", DataType::Handle()), c_param("c", DataType::Handle());
    tir::Var s1("s1", DataType::Int(32)), s0("s0", DataType::Int(32));
    tir::Var d1("d1", DataType::Int(32)), d0("d0", DataType::Int(32));

    tir::Buffer A = DeclBufferWithStrides(
        "A", {IntImm(DataType::Int(32), frag_m), IntImm(DataType::Int(32), frag_n)}, dt,
        shared_scope, 64, offset_factor, {s1, s0});
    tir::Buffer C = DeclBufferWithStrides(
        "C", {IntImm(DataType::Int(32), frag_m), IntImm(DataType::Int(32), frag_n)}, dt, wmma_scope,
        64, offset_factor, {d1, d0});

    PrimExpr frag_idx = WmmaFragmentIndex(C, d1, frag_m, frag_n);
    PrimExpr access_ptr = A.access_ptr(/*access_mask=*/1);  // "r" = 1
    PrimExpr load_call = tir::Call(
        DataType::Handle(), tir::builtin::tvm_load_matrix_sync(),
        {C->data, IntImm(DataType::Int(32), m_dim), IntImm(DataType::Int(32), n_dim),
         IntImm(DataType::Int(32), k_dim), frag_idx, access_ptr, s1, tir::StringImm(layout)});

    tir::Var thread_var("_", DataType::Int(32));
    tir::Stmt inner = tir::Evaluate(load_call);
    tir::Stmt thread_loop = ThreadBindingFor(thread_var, 64, "threadIdx.x", inner);

    tir::Stmt root_body = RootSBlock({FullRegion2D(A, frag_m, frag_n)},
                                     {FullRegion2D(C, frag_m, frag_n)}, thread_loop);

    ffi::Map<tir::Var, tir::Buffer> buf_map;
    buf_map.Set(a_param, A);
    buf_map.Set(c_param, C);
    return tir::PrimFunc({a_param, c_param}, root_body, VoidType(), buf_map);
  }();

  return {desc, impl};
}

// ===========================================================================
// get_wmma_store_intrin
// ===========================================================================

/*!
 * \brief Build the wmma store intrinsic.
 *
 * desc: nested loops over (m_dim, n_dim) that copy element by element
 *       from the wmma.accumulator fragment to the output buffer.
 * impl: a single tvm_store_matrix_sync call.
 */
static std::pair<tir::PrimFunc, tir::PrimFunc> GetWmmaStoreIntrin(int m_dim, int n_dim, int k_dim,
                                                                  const std::string& dtype,
                                                                  const std::string& scope) {
  DataType dt = DTypeFromStr(dtype);
  int offset_factor = n_dim;
  const std::string wmma_scope = "wmma.accumulator";

  // ---- desc ----
  tir::PrimFunc desc = [&]() -> tir::PrimFunc {
    tir::Var a_param("a", DataType::Handle()), c_param("c", DataType::Handle());
    tir::Buffer A = DeclBufferContiguous(
        "A", {IntImm(DataType::Int(32), m_dim), IntImm(DataType::Int(32), n_dim)}, dt, wmma_scope,
        64, offset_factor);
    tir::Buffer C = DeclBufferContiguous(
        "C", {IntImm(DataType::Int(32), m_dim), IntImm(DataType::Int(32), n_dim)}, dt, scope, 64,
        offset_factor);

    tir::Var vi("vii", DataType::Int(32)), vj("vjj", DataType::Int(32));
    tir::IterVar iv_i(Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), m_dim)), vi,
                      tir::kDataPar);
    tir::IterVar iv_j(Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), n_dim)), vj,
                      tir::kDataPar);

    // Loop vars declared before the block; iter_values = [i, j] (loop vars).
    // Inner block read/write regions are POINT regions (extent=1 per dim).
    tir::Var i("i", DataType::Int(32)), j("j", DataType::Int(32));

    tir::Stmt store_body = tir::BufferStore(C, tir::BufferLoad(A, {vi, vj}), {vi, vj});
    tir::Stmt store_block =
        InnerSBlock("store", {iv_i, iv_j}, {PrimExpr(i), PrimExpr(j)}, {PointRegion2D(A, vi, vj)},
                    {PointRegion2D(C, vi, vj)}, store_body);

    tir::Stmt loops = NestedFor2D(i, j, m_dim, n_dim, store_block);

    tir::Stmt root_body =
        RootSBlock({FullRegion2D(A, m_dim, n_dim)}, {FullRegion2D(C, m_dim, n_dim)}, loops);

    ffi::Map<tir::Var, tir::Buffer> buf_map;
    buf_map.Set(a_param, A);
    buf_map.Set(c_param, C);
    return tir::PrimFunc({a_param, c_param}, root_body, VoidType(), buf_map);
  }();

  // ---- impl ----
  tir::PrimFunc impl = [&]() -> tir::PrimFunc {
    tir::Var a_param("a", DataType::Handle()), c_param("c", DataType::Handle());
    tir::Var s1("s1", DataType::Int(32)), s0("s0", DataType::Int(32));
    tir::Var d1("d1", DataType::Int(32)), d0("d0", DataType::Int(32));

    // A is wmma.accumulator with strides [d1, d0]
    tir::Buffer A = DeclBufferWithStrides(
        "A", {IntImm(DataType::Int(32), m_dim), IntImm(DataType::Int(32), n_dim)}, dt, wmma_scope,
        64, offset_factor, {d1, d0});
    // C is output with strides [s1, s0]
    tir::Buffer C = DeclBufferWithStrides(
        "C", {IntImm(DataType::Int(32), m_dim), IntImm(DataType::Int(32), n_dim)}, dt, scope, 64,
        offset_factor, {s1, s0});

    PrimExpr frag_idx = WmmaFragmentIndex(A, d1, m_dim, n_dim);
    PrimExpr access_ptr = C.access_ptr(/*access_mask=*/2);  // "w" = 2
    PrimExpr store_call = tir::Call(
        DataType::Handle(), tir::builtin::tvm_store_matrix_sync(),
        {A->data, IntImm(DataType::Int(32), m_dim), IntImm(DataType::Int(32), n_dim),
         IntImm(DataType::Int(32), k_dim), frag_idx, access_ptr, s1, tir::StringImm("row_major")});

    tir::Stmt root_body = RootSBlock({FullRegion2D(A, m_dim, n_dim)},
                                     {FullRegion2D(C, m_dim, n_dim)}, tir::Evaluate(store_call));

    ffi::Map<tir::Var, tir::Buffer> buf_map;
    buf_map.Set(a_param, A);
    buf_map.Set(c_param, C);
    return tir::PrimFunc({a_param, c_param}, root_body, VoidType(), buf_map);
  }();

  return {desc, impl};
}

// ===========================================================================
// get_wmma_sync_intrin
// ===========================================================================

/*!
 * \brief Build the wmma sync (matrix multiply-accumulate) intrinsic.
 *
 * desc: a 3-D loop nest over (m_dim, n_dim, k_dim) that performs the
 *       scalar fused multiply-add C[i,j] += A[...] * B[...].
 * impl: a single tvm_mma_sync call.
 */
static std::pair<tir::PrimFunc, tir::PrimFunc> GetWmmaSyncIntrin(int m_dim, int n_dim, int k_dim,
                                                                 const std::string& in_dtype,
                                                                 const std::string& out_dtype,
                                                                 bool a_transposed,
                                                                 bool b_transposed) {
  DataType in_dt = DTypeFromStr(in_dtype);
  DataType out_dt = DTypeFromStr(out_dtype);

  auto maybe_swap = [](int x, int y, bool trans) -> std::pair<int, int> {
    return trans ? std::make_pair(y, x) : std::make_pair(x, y);
  };

  auto [a_shape_0, a_shape_1] = maybe_swap(m_dim, k_dim, a_transposed);
  auto [b_shape_0, b_shape_1] = maybe_swap(k_dim, n_dim, b_transposed);

  int A_offset_factor = k_dim;
  int B_offset_factor = b_shape_1;
  int out_offset_factor = n_dim;

  // ---- desc ----
  tir::PrimFunc desc = [&]() -> tir::PrimFunc {
    tir::Var a_param("a", DataType::Handle()), b_param("b", DataType::Handle()),
        c_param("c", DataType::Handle());

    tir::Buffer A = DeclBufferContiguous(
        "A", {IntImm(DataType::Int(32), a_shape_0), IntImm(DataType::Int(32), a_shape_1)}, in_dt,
        "wmma.matrix_a", 64, A_offset_factor);
    tir::Buffer B = DeclBufferContiguous(
        "B", {IntImm(DataType::Int(32), b_shape_0), IntImm(DataType::Int(32), b_shape_1)}, in_dt,
        "wmma.matrix_b", 64, B_offset_factor);
    tir::Buffer C = DeclBufferContiguous(
        "C", {IntImm(DataType::Int(32), m_dim), IntImm(DataType::Int(32), n_dim)}, out_dt,
        "wmma.accumulator", 64, out_offset_factor);

    tir::Var vi("vii", DataType::Int(32)), vj("vjj", DataType::Int(32)),
        vk("vkk", DataType::Int(32));
    tir::IterVar iv_i(Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), m_dim)), vi,
                      tir::kDataPar);
    tir::IterVar iv_j(Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), n_dim)), vj,
                      tir::kDataPar);
    tir::IterVar iv_k(Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), k_dim)), vk,
                      tir::kCommReduce);

    // A_index = maybe_swap(vii, vkk, a_transposed)
    auto [ai0, ai1] = a_transposed ? std::make_pair(vk, vi) : std::make_pair(vi, vk);
    // B_index = maybe_swap(vkk, vjj, b_transposed)
    auto [bi0, bi1] = b_transposed ? std::make_pair(vj, vk) : std::make_pair(vk, vj);

    PrimExpr a_val = tir::BufferLoad(A, {PrimExpr(ai0), PrimExpr(ai1)});
    PrimExpr b_val = tir::BufferLoad(B, {PrimExpr(bi0), PrimExpr(bi1)});

    // maybe_cast: if in_dtype != out_dtype, cast to out_dtype
    if (in_dtype != out_dtype) {
      a_val = tir::Cast(out_dt, {a_val});
      b_val = tir::Cast(out_dt, {b_val});
    }

    PrimExpr c_old = tir::BufferLoad(C, {vi, vj});
    tir::Stmt inner_body = tir::BufferStore(C, c_old + a_val * b_val, {vi, vj});

    // Loop vars declared before the block; iter_values = [i, j, k] (loop vars).
    // Inner block read/write regions are POINT regions (extent=1 per dim).
    tir::Var i("i", DataType::Int(32)), j("j", DataType::Int(32)), k("k", DataType::Int(32));

    tir::Stmt inner_block = InnerSBlock(
        "", {iv_i, iv_j, iv_k}, {PrimExpr(i), PrimExpr(j), PrimExpr(k)},
        {PointRegion2D(C, vi, vj), PointRegion2D(A, ai0, ai1), PointRegion2D(B, bi0, bi1)},
        {PointRegion2D(C, vi, vj)}, inner_body);

    // Build 3-D loop nest
    tir::Stmt k_loop = For1D(k, k_dim, inner_block);
    tir::Stmt j_loop = For1D(j, n_dim, k_loop);
    tir::Stmt i_loop = For1D(i, m_dim, j_loop);

    tir::Stmt root_body =
        RootSBlock({FullRegion2D(C, m_dim, n_dim), FullRegion2D(A, a_shape_0, a_shape_1),
                    FullRegion2D(B, b_shape_0, b_shape_1)},
                   {FullRegion2D(C, m_dim, n_dim)}, i_loop);

    ffi::Map<tir::Var, tir::Buffer> buf_map;
    buf_map.Set(a_param, A);
    buf_map.Set(b_param, B);
    buf_map.Set(c_param, C);
    return tir::PrimFunc({a_param, b_param, c_param}, root_body, VoidType(), buf_map);
  }();

  // ---- impl ----
  tir::PrimFunc impl = [&]() -> tir::PrimFunc {
    tir::Var a_param("a", DataType::Handle()), b_param("b", DataType::Handle()),
        c_param("c", DataType::Handle());
    tir::Var a1("a1", DataType::Int(32)), a0("a0", DataType::Int(32));
    tir::Var b1("b1", DataType::Int(32)), b0("b0", DataType::Int(32));
    tir::Var c1("c1", DataType::Int(32)), c0("c0", DataType::Int(32));

    tir::Buffer A = DeclBufferWithStrides(
        "A", {IntImm(DataType::Int(32), a_shape_0), IntImm(DataType::Int(32), a_shape_1)}, in_dt,
        "wmma.matrix_a", 64, A_offset_factor, {a1, a0});
    tir::Buffer B = DeclBufferWithStrides(
        "B", {IntImm(DataType::Int(32), b_shape_0), IntImm(DataType::Int(32), b_shape_1)}, in_dt,
        "wmma.matrix_b", 64, B_offset_factor, {b1, b0});
    tir::Buffer C = DeclBufferWithStrides(
        "C", {IntImm(DataType::Int(32), m_dim), IntImm(DataType::Int(32), n_dim)}, out_dt,
        "wmma.accumulator", 64, out_offset_factor, {c1, c0});

    PrimExpr a_frag_idx = WmmaFragmentIndex(A, a1, a_shape_0, a_shape_1);
    PrimExpr b_frag_idx = WmmaFragmentIndex(B, b1, b_shape_0, b_shape_1);
    PrimExpr c_frag_idx = WmmaFragmentIndex(C, c1, m_dim, n_dim);

    PrimExpr mma_call = tir::Call(
        DataType::Handle(), tir::builtin::tvm_mma_sync(),
        {C->data, c_frag_idx, A->data, a_frag_idx, B->data, b_frag_idx, C->data, c_frag_idx});

    tir::Stmt root_body =
        RootSBlock({FullRegion2D(C, m_dim, n_dim), FullRegion2D(A, a_shape_0, a_shape_1),
                    FullRegion2D(B, b_shape_0, b_shape_1)},
                   {FullRegion2D(C, m_dim, n_dim)}, tir::Evaluate(mma_call));

    ffi::Map<tir::Var, tir::Buffer> buf_map;
    buf_map.Set(a_param, A);
    buf_map.Set(b_param, B);
    buf_map.Set(c_param, C);
    return tir::PrimFunc({a_param, b_param, c_param}, root_body, VoidType(), buf_map);
  }();

  return {desc, impl};
}

// ===========================================================================
// get_wmma_qcom_intrin
// ===========================================================================

/*!
 * \brief Build the QCOM cooperative matrix construct/deconstruct intrinsic.
 *
 * Used for QCOM-specific local-memory ↔ wmma fragment transfers.
 * desc: a 1-D loop over frag_n that copies element by element.
 * impl: a single tvm_construct_coopmat_qcom or tvm_deconstruct_coopmat_qcom call.
 */
static std::pair<tir::PrimFunc, tir::PrimFunc> GetWmmaQcomIntrin(int m_dim, int n_dim, int k_dim,
                                                                 const std::string& dtype,
                                                                 bool is_b, bool is_col_major,
                                                                 bool is_load) {
  DataType dt = DTypeFromStr(dtype);
  std::string wmma_scope = std::string("wmma.matrix_") + (is_b ? "b" : "a");
  if (!is_load) wmma_scope = "wmma.accumulator";

  int frag_n;
  if (is_load) {
    frag_n = is_b ? k_dim : k_dim;  // both A and B use k_dim as the fragment inner dimension
    // More precisely:
    if (is_b)
      frag_n = k_dim;
    else
      frag_n = k_dim;
  } else {
    frag_n = n_dim;  // store fragment inner dimension is n_dim
  }
  int offset_factor = frag_n;

  std::string scope_a = is_load ? "local" : wmma_scope;
  std::string scope_c = is_load ? wmma_scope : "local";

  const Op& intrin_func = is_load ? tir::builtin::tvm_construct_coopmat_qcom()
                                  : tir::builtin::tvm_deconstruct_coopmat_qcom();

  // ---- desc ----
  tir::PrimFunc desc = [&]() -> tir::PrimFunc {
    tir::Var a_param("a", DataType::Handle()), c_param("c", DataType::Handle());
    tir::Buffer A = DeclBufferContiguous("A", {IntImm(DataType::Int(32), frag_n)}, dt, scope_a, 0,
                                         offset_factor);
    tir::Buffer C = DeclBufferContiguous("C", {IntImm(DataType::Int(32), frag_n)}, dt, scope_c, 0,
                                         offset_factor);

    tir::Var vj("vjj", DataType::Int(32));
    tir::IterVar iv_j(Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), frag_n)), vj,
                      tir::kDataPar);

    // Loop var declared before the block; iter_values = [j] (loop var).
    // Inner block read/write regions are POINT regions (extent=1).
    tir::Var j("j", DataType::Int(32));

    tir::Stmt load_body = tir::BufferStore(C, tir::BufferLoad(A, {vj}), {vj});
    tir::Stmt load_block = InnerSBlock("load", {iv_j}, {PrimExpr(j)}, {PointRegion1D(A, vj)},
                                       {PointRegion1D(C, vj)}, load_body);

    tir::Stmt loop = For1D(j, frag_n, load_block);

    tir::Stmt root_body = RootSBlock({FullRegion1D(A, frag_n)}, {FullRegion1D(C, frag_n)}, loop);

    ffi::Map<tir::Var, tir::Buffer> buf_map;
    buf_map.Set(a_param, A);
    buf_map.Set(c_param, C);
    return tir::PrimFunc({a_param, c_param}, root_body, VoidType(), buf_map);
  }();

  // ---- impl ----
  tir::PrimFunc impl = [&]() -> tir::PrimFunc {
    tir::Var a_param("a", DataType::Handle()), c_param("c", DataType::Handle());
    tir::Buffer A = DeclBufferContiguous("A", {IntImm(DataType::Int(32), frag_n)}, dt, scope_a, 0,
                                         offset_factor);
    tir::Buffer C = DeclBufferContiguous("C", {IntImm(DataType::Int(32), frag_n)}, dt, scope_c, 0,
                                         offset_factor);

    // intrin_func(C.data if is_load else A.data, m, n, k, A.data if is_load else C.data)
    PrimExpr first_arg = is_load ? C->data : A->data;
    PrimExpr last_arg = is_load ? A->data : C->data;
    PrimExpr call =
        tir::Call(DataType::Handle(), intrin_func,
                  {first_arg, IntImm(DataType::Int(32), m_dim), IntImm(DataType::Int(32), n_dim),
                   IntImm(DataType::Int(32), k_dim), last_arg});

    tir::Stmt root_body =
        RootSBlock({FullRegion1D(A, frag_n)}, {FullRegion1D(C, frag_n)}, tir::Evaluate(call));

    ffi::Map<tir::Var, tir::Buffer> buf_map;
    buf_map.Set(a_param, A);
    buf_map.Set(c_param, C);
    return tir::PrimFunc({a_param, c_param}, root_body, VoidType(), buf_map);
  }();

  return {desc, impl};
}

// ===========================================================================
// GetAdrenoWmmaIntrinGroup
// ===========================================================================

/*!
 * \brief Build and register the five wmma intrinsics for the given parameters,
 *        then return a name map {"init", "load_a", "load_b", "compute", "store"}.
 *
 * Intrinsics are registered lazily (only if not already present).
 */
ffi::Map<ffi::String, ffi::String> GetAdrenoWmmaIntrinGroup(
    int m, int n, int k, const std::string& load_scope_a, const std::string& load_scope_b,
    const std::string& store_scope, bool trans_a, bool trans_b, const std::string& dtype,
    const std::string& out_dtype) {
  // Validate profile
  int exp_m, exp_n, exp_k;
  if (!GetWmmaTileSizes(dtype, out_dtype, &exp_m, &exp_n, &exp_k)) {
    TVM_FFI_THROW(InternalError) << "Unsupported dtype profile: (" << dtype << ", " << dtype << ", "
                                 << out_dtype << ")";
  }
  if (m != exp_m || n != exp_n || k != exp_k) {
    TVM_FFI_THROW(InternalError) << "Unsupported shape (" << m << ", " << n << ", " << k
                                 << ") for dtype " << dtype;
  }

  std::string dtype_suffix = ShorthandDtype(dtype);
  std::string out_dtype_suffix = ShorthandDtype(out_dtype);

  std::string a_suffix = trans_a ? "_a_trans" : "_a";
  std::string b_suffix = trans_b ? "_b_trans" : "_b";

  std::string shape_str = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);

  std::string load_a_intrin =
      "wmma_load_" + shape_str + "_" + dtype_suffix + a_suffix + "_" + load_scope_a;
  std::string load_b_intrin =
      "wmma_load_" + shape_str + "_" + dtype_suffix + b_suffix + "_" + load_scope_b;
  std::string compute_intrin = "wmma_sync_" + shape_str + "_" + dtype_suffix + dtype_suffix +
                               a_suffix + b_suffix + "_" + out_dtype;
  std::string init_intrin = "wmma_fill_" + shape_str + "_" + out_dtype_suffix;
  std::string store_intrin = "wmma_store_" + shape_str + "_" + out_dtype_suffix + "_" + store_scope;

  // Lazy registration
  auto register_if_missing = [](const std::string& name, tir::PrimFunc desc, tir::PrimFunc impl) {
    if (!tir::TensorIntrin::Get(name, /*allow_missing=*/true).defined()) {
      tir::TensorIntrin::Register(name, tir::TensorIntrin(desc, impl));
    }
  };

  // init
  {
    auto [desc, impl] = GetWmmaFillIntrin(m, n, k, out_dtype);
    register_if_missing(init_intrin, desc, impl);
  }

  // load_a
  {
    if (load_scope_a == "local") {
      auto [desc, impl] = GetWmmaQcomIntrin(m, n, k, dtype, /*is_b=*/false,
                                            /*is_col_major=*/false, /*is_load=*/true);
      register_if_missing(load_a_intrin, desc, impl);
    } else {
      auto [desc, impl] = GetWmmaLoadIntrin(m, n, k, dtype, load_scope_a, /*is_b=*/false, trans_a);
      register_if_missing(load_a_intrin, desc, impl);
    }
  }

  // load_b
  {
    if (load_scope_b == "local") {
      auto [desc, impl] = GetWmmaQcomIntrin(m, n, k, dtype, /*is_b=*/true,
                                            /*is_col_major=*/false, /*is_load=*/true);
      register_if_missing(load_b_intrin, desc, impl);
    } else {
      auto [desc, impl] = GetWmmaLoadIntrin(m, n, k, dtype, load_scope_b, /*is_b=*/true, trans_b);
      register_if_missing(load_b_intrin, desc, impl);
    }
  }

  // compute
  {
    auto [desc, impl] = GetWmmaSyncIntrin(m, n, k, dtype, out_dtype, trans_a, trans_b);
    register_if_missing(compute_intrin, desc, impl);
  }

  // store
  {
    if (store_scope == "local") {
      auto [desc, impl] = GetWmmaQcomIntrin(m, n, k, out_dtype, /*is_b=*/false,
                                            /*is_col_major=*/false, /*is_load=*/false);
      register_if_missing(store_intrin, desc, impl);
    } else {
      auto [desc, impl] = GetWmmaStoreIntrin(m, n, k, out_dtype, store_scope);
      register_if_missing(store_intrin, desc, impl);
    }
  }

  ffi::Map<ffi::String, ffi::String> result;
  result.Set("init", init_intrin);
  result.Set("load_a", load_a_intrin);
  result.Set("load_b", load_b_intrin);
  result.Set("compute", compute_intrin);
  result.Set("store", store_intrin);
  return result;
}

// ---------------------------------------------------------------------------
// Global function registrations (callable from Python / other C++ modules)
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::GlobalDef().def(
      "tir.tensor_intrin.adreno.GetWmmaTileSizes",
      [](const std::string& in_dtype, const std::string& out_dtype) -> ffi::Array<Integer> {
        int tile_m, tile_n, tile_k;
        if (!GetWmmaTileSizes(in_dtype, out_dtype, &tile_m, &tile_n, &tile_k)) {
          return {};
        }
        return {Integer(tile_m), Integer(tile_n), Integer(tile_k)};
      });

  refl::GlobalDef().def(
      "tir.tensor_intrin.adreno.GetAdrenoWmmaIntrinGroup",
      [](int m, int n, int k, ffi::Array<ffi::String> load_scope, const std::string& store_scope,
         bool trans_a, bool trans_b, const std::string& dtype,
         const std::string& out_dtype) -> ffi::Map<ffi::String, ffi::String> {
        std::string ls_a = load_scope.size() >= 1 ? std::string(load_scope[0]) : "global";
        std::string ls_b = load_scope.size() >= 2 ? std::string(load_scope[1]) : "global";
        return GetAdrenoWmmaIntrinGroup(m, n, k, ls_a, ls_b, store_scope, trans_a, trans_b, dtype,
                                        out_dtype);
      });
}

}  // namespace adreno
}  // namespace dlight
}  // namespace tvm
