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
 * \file topi/scatter.cc
 * \brief C++ implementations of scatter_nd, scatter_elements, and slice_scatter.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/topi/transform.h>

#include <string>

namespace tvm {
namespace topi {

static PrimExpr ShapeProduct(const ffi::Array<PrimExpr>& shape, DataType dtype) {
  PrimExpr result = tvm::tir::make_const(dtype, 1);
  for (const auto& dim : shape) {
    result = result * tvm::cast(dtype, dim);
  }
  return result;
}

static tir::Buffer MakeFlatBuffer(const te::Tensor& t, const std::string& name) {
  return tir::decl_buffer(t->shape, t->dtype, name);
}

static ffi::Array<PrimExpr> FlatToMultiIndex(PrimExpr flat_idx, const ffi::Array<PrimExpr>& shape) {
  int ndim = static_cast<int>(shape.size());
  ffi::Array<PrimExpr> indices;
  indices.resize(ndim);
  PrimExpr remaining = flat_idx;
  for (int i = ndim - 1; i >= 0; --i) {
    PrimExpr dim = tvm::cast(flat_idx.dtype(), shape[i]);
    indices.Set(i, tvm::tir::FloorMod(remaining, dim));
    remaining = tvm::tir::FloorDiv(remaining, dim);
  }
  return indices;
}

/*!
 * \brief Scatter updates into data at multi-dimensional indices.
 * \param data    Source tensor (output is initialised from this).
 * \param indices Integer index tensor, shape (M, ...).
 * \param updates Values to scatter.
 * \param mode    Reduction mode: "update", "add", "mul", "min", "max".
 * \return Output tensor with same shape as data.
 */
te::Tensor scatter_nd(const te::Tensor& data, const te::Tensor& indices, const te::Tensor& updates,
                      const std::string& mode) {
  const DataType idx_dtype = DataType::Int(64);
  arith::Analyzer analyzer;

  const int64_t M = indices->shape[0].as<IntImmNode>()->value;
  const int ndim_data = static_cast<int>(data->shape.size());
  TVM_FFI_ICHECK(M <= ndim_data) << "scatter_nd: indices.shape[0] (" << M
                                 << ") must be <= data.ndim (" << ndim_data << ")";

  PrimExpr fused_indices_dim = tvm::tir::make_const(idx_dtype, 1);
  for (size_t i = 1; i < indices->shape.size(); ++i) {
    fused_indices_dim = fused_indices_dim * tvm::cast(idx_dtype, indices->shape[i]);
  }
  fused_indices_dim = analyzer.Simplify(fused_indices_dim);

  const int updates_offset = static_cast<int>(indices->shape.size()) - 1;
  PrimExpr fused_updates_dim = tvm::tir::make_const(idx_dtype, 1);
  for (int i = updates_offset; i < static_cast<int>(updates->shape.size()); ++i) {
    fused_updates_dim = fused_updates_dim * tvm::cast(idx_dtype, updates->shape[i]);
  }
  fused_updates_dim = analyzer.Simplify(fused_updates_dim);

  PrimExpr fused_shape = ShapeProduct(data->shape, idx_dtype);
  fused_shape = analyzer.Simplify(fused_shape);

  tir::Buffer data_buf = MakeFlatBuffer(data, "data");
  tir::Buffer indices_buf = MakeFlatBuffer(indices, "indices");
  tir::Buffer updates_buf = MakeFlatBuffer(updates, "updates");
  tir::Buffer out_buf = tir::decl_buffer(data->shape, data->dtype, "out_buf");

  tir::Var i("i", idx_dtype);
  tir::Var j("j", idx_dtype);

  auto flat_load = [&](const tir::Buffer& buf, PrimExpr flat_idx) -> PrimExpr {
    return tir::BufferLoad(buf, FlatToMultiIndex(flat_idx, buf->shape));
  };
  auto flat_store = [&](const tir::Buffer& buf, PrimExpr flat_idx, PrimExpr val) -> tir::Stmt {
    return tir::BufferStore(buf, val, FlatToMultiIndex(flat_idx, buf->shape));
  };

  tir::Var copy_i("copy_i", idx_dtype);
  tir::Stmt copy_body = flat_store(out_buf, copy_i, flat_load(data_buf, copy_i));
  tir::Stmt copy_loop = tir::For(copy_i, tvm::tir::make_const(idx_dtype, 0), fused_shape,
                                 tir::ForKind::kSerial, copy_body);

  PrimExpr offset = fused_updates_dim;
  PrimExpr index = PrimExpr(j);
  for (int l = M - 1; l >= 0; --l) {
    PrimExpr ind_flat = i + tvm::tir::make_const(idx_dtype, l) * fused_indices_dim;
    PrimExpr ind_val = tvm::cast(idx_dtype, flat_load(indices_buf, ind_flat));
    index = index + offset * ind_val;
    offset = offset * tvm::cast(idx_dtype, data->shape[l]);
  }
  index = analyzer.Simplify(index);

  PrimExpr upd_flat = i * fused_updates_dim + j;
  PrimExpr upd_val = flat_load(updates_buf, upd_flat);
  PrimExpr cur_val = flat_load(out_buf, index);

  tir::Stmt update_stmt;
  if (mode == "update") {
    update_stmt = flat_store(out_buf, index, upd_val);
  } else if (mode == "add") {
    update_stmt = flat_store(out_buf, index, cur_val + upd_val);
  } else if (mode == "mul") {
    update_stmt = flat_store(out_buf, index, cur_val * upd_val);
  } else if (mode == "min") {
    update_stmt = flat_store(out_buf, index, tvm::min(cur_val, upd_val));
  } else if (mode == "max") {
    update_stmt = flat_store(out_buf, index, tvm::max(cur_val, upd_val));
  } else {
    TVM_FFI_THROW(InternalError) << "scatter_nd: unsupported mode '" << mode
                                 << "'. Expected one of: update, add, mul, min, max.";
  }

  tir::Stmt inner_loop = tir::For(j, tvm::tir::make_const(idx_dtype, 0), fused_updates_dim,
                                  tir::ForKind::kParallel, update_stmt);
  tir::Stmt outer_loop = tir::For(i, tvm::tir::make_const(idx_dtype, 0), fused_indices_dim,
                                  tir::ForKind::kSerial, inner_loop);

  tir::Stmt body = tir::SeqStmt::Flatten(copy_loop, outer_loop);

  return te::ExternOp("scatter_nd.generic", "scatter_nd.generic",
                      /*attrs=*/{},
                      /*inputs=*/{data, indices, updates},
                      /*input_placeholders=*/{data_buf, indices_buf, updates_buf},
                      /*output_placeholders=*/{out_buf}, body)
      .output(0);
}

/*!
 * \brief Scatter elements from updates into data along a given axis.
 * \param data      Source tensor.
 * \param indices   Integer index tensor (same shape as updates).
 * \param updates   Values to scatter.
 * \param axis      Axis along which to scatter (may be negative).
 * \param reduction Reduction mode: "update", "add", "mul", "mean", "min", "max".
 * \return Output tensor with same shape as data.
 */
te::Tensor scatter_elements(const te::Tensor& data, const te::Tensor& indices,
                            const te::Tensor& updates, int axis, const std::string& reduction) {
  const DataType idx_dtype = DataType::Int(64);
  arith::Analyzer analyzer;

  const int ndim = static_cast<int>(data->shape.size());
  if (axis < 0) axis += ndim;
  TVM_FFI_ICHECK(axis >= 0 && axis < ndim)
      << "scatter_elements: axis " << axis << " out of range for ndim=" << ndim;

  PrimExpr axis_range = tvm::cast(idx_dtype, data->shape[axis]);

  PrimExpr full_range = ShapeProduct(data->shape, idx_dtype);
  full_range = analyzer.Simplify(full_range);

  PrimExpr after_axis_range = tvm::tir::make_const(idx_dtype, 1);
  for (int i = axis + 1; i < ndim; ++i) {
    after_axis_range = after_axis_range * tvm::cast(idx_dtype, data->shape[i]);
  }
  after_axis_range = analyzer.Simplify(after_axis_range);

  PrimExpr before_axis_stride = analyzer.Simplify(axis_range * after_axis_range);

  PrimExpr ind_axis_range = tvm::cast(idx_dtype, indices->shape[axis]);

  PrimExpr ind_after_axis_range = tvm::tir::make_const(idx_dtype, 1);
  for (int i = axis + 1; i < ndim; ++i) {
    ind_after_axis_range = ind_after_axis_range * tvm::cast(idx_dtype, indices->shape[i]);
  }
  ind_after_axis_range = analyzer.Simplify(ind_after_axis_range);

  PrimExpr ind_before_axis_range = tvm::tir::make_const(idx_dtype, 1);
  for (int i = 0; i < axis; ++i) {
    ind_before_axis_range = ind_before_axis_range * tvm::cast(idx_dtype, indices->shape[i]);
  }
  ind_before_axis_range = analyzer.Simplify(ind_before_axis_range);

  PrimExpr ind_before_axis_stride = analyzer.Simplify(ind_axis_range * ind_after_axis_range);

  tir::Buffer data_buf = MakeFlatBuffer(data, "data");
  tir::Buffer indices_buf = MakeFlatBuffer(indices, "indices");
  tir::Buffer updates_buf = MakeFlatBuffer(updates, "updates");
  tir::Buffer out_buf = tir::decl_buffer(data->shape, data->dtype, "out_buf");

  auto flat_load = [&](const tir::Buffer& buf, PrimExpr flat_idx) -> PrimExpr {
    return tir::BufferLoad(buf, FlatToMultiIndex(flat_idx, buf->shape));
  };
  auto flat_store = [&](const tir::Buffer& buf, PrimExpr flat_idx, PrimExpr val) -> tir::Stmt {
    return tir::BufferStore(buf, val, FlatToMultiIndex(flat_idx, buf->shape));
  };

  tir::Var ci("ci", idx_dtype);
  tir::Stmt copy_body = flat_store(out_buf, ci, flat_load(data_buf, ci));
  tir::Stmt copy_loop = tir::For(ci, tvm::tir::make_const(idx_dtype, 0), full_range,
                                 tir::ForKind::kParallel, copy_body);

  tir::Var fused("fused", idx_dtype);
  tir::Var k("k", idx_dtype);

  PrimExpr i_var = tvm::tir::FloorDiv(fused, ind_after_axis_range);
  PrimExpr j_var = tvm::tir::FloorMod(fused, ind_after_axis_range);

  PrimExpr pre_index1 = i_var * ind_before_axis_stride + j_var;
  PrimExpr index1 = analyzer.Simplify(pre_index1 + k * ind_after_axis_range);

  PrimExpr k_new = tvm::cast(idx_dtype, flat_load(indices_buf, index1));
  PrimExpr shifted_index = analyzer.Simplify(
      k_new + tvm::cast(idx_dtype,
                        tvm::cast(DataType::Bool(), k_new < tvm::tir::make_const(idx_dtype, 0))) *
                  axis_range);

  PrimExpr pre_index2 = i_var * before_axis_stride + j_var;
  PrimExpr index2 = analyzer.Simplify(pre_index2 + shifted_index * after_axis_range);

  PrimExpr upd_val = flat_load(updates_buf, index1);
  PrimExpr cur_val = flat_load(out_buf, index2);

  tir::Stmt update_stmt;
  if (reduction == "update") {
    update_stmt = flat_store(out_buf, index2, upd_val);
  } else if (reduction == "add") {
    update_stmt = flat_store(out_buf, index2, cur_val + upd_val);
  } else if (reduction == "mul") {
    update_stmt = flat_store(out_buf, index2, cur_val * upd_val);
  } else if (reduction == "mean") {
    update_stmt = flat_store(
        out_buf, index2,
        (cur_val + upd_val) / tvm::cast(data->dtype, tvm::tir::make_const(idx_dtype, 2)));
  } else if (reduction == "min") {
    update_stmt = flat_store(out_buf, index2, tvm::min(cur_val, upd_val));
  } else if (reduction == "max") {
    update_stmt = flat_store(out_buf, index2, tvm::max(cur_val, upd_val));
  } else {
    TVM_FFI_THROW(InternalError) << "scatter_elements: unsupported reduction '" << reduction
                                 << "'. Expected one of: update, add, mul, mean, min, max.";
  }

  tir::Stmt inner_loop = tir::For(k, tvm::tir::make_const(idx_dtype, 0), ind_axis_range,
                                  tir::ForKind::kSerial, update_stmt);

  PrimExpr outer_extent = analyzer.Simplify(ind_before_axis_range * ind_after_axis_range);
  tir::Stmt outer_loop = tir::For(fused, tvm::tir::make_const(idx_dtype, 0), outer_extent,
                                  tir::ForKind::kParallel, inner_loop);

  tir::Stmt body = tir::SeqStmt::Flatten(copy_loop, outer_loop);

  return te::ExternOp("scatter_elements.generic", "scatter_elements.generic",
                      /*attrs=*/{},
                      /*inputs=*/{data, indices, updates},
                      /*input_placeholders=*/{data_buf, indices_buf, updates_buf},
                      /*output_placeholders=*/{out_buf}, body)
      .output(0);
}

/*!
 * \brief Scatter a slice of src into input along a given axis (SSA form).
 * \param input  Source tensor (provides background values).
 * \param src    Values to scatter into the slice.
 * \param start  Start index of the slice along \p axis.
 * \param end    End index (exclusive) of the slice along \p axis.
 * \param step   Step size of the slice.
 * \param axis   Axis along which to scatter.
 * \return Output tensor with same shape as input.
 */
te::Tensor slice_scatter(const te::Tensor& input, const te::Tensor& src, int start, int end,
                         int step, int axis) {
  const int ndim = static_cast<int>(input->shape.size());
  if (axis < 0) axis += ndim;
  TVM_FFI_ICHECK(axis >= 0 && axis < ndim)
      << "slice_scatter: axis " << axis << " out of range for ndim=" << ndim;

  arith::Analyzer analyzer;
  const DataType idx_dtype = DataType::Int(64);

  const int64_t dim_size = input->shape[axis].as<IntImmNode>()->value;

  if (start == 0 && end == dim_size && step == 1) {
    return te::compute(
        input->shape, [&](const ffi::Array<tir::Var>& idx) { return src(idx); },
        "slice_scatter_identity");
  }

  return te::compute(
      input->shape,
      [&](const ffi::Array<tir::Var>& out_idx) -> PrimExpr {
        PrimExpr pos = tvm::cast(idx_dtype, out_idx[axis]);

        PrimExpr in_slice = tvm::tir::make_const(DataType::Bool(), true);

        if (start != 0) {
          in_slice = tvm::logical_and(in_slice, pos >= tvm::tir::make_const(idx_dtype, start));
        }
        if (end != dim_size) {
          in_slice = tvm::logical_and(in_slice, pos < tvm::tir::make_const(idx_dtype, end));
        }
        if (step != 1) {
          PrimExpr rem = tvm::floormod(pos - tvm::tir::make_const(idx_dtype, start),
                                       tvm::tir::make_const(idx_dtype, step));
          in_slice = tvm::logical_and(in_slice, rem == tvm::tir::make_const(idx_dtype, 0));
        }
        in_slice = analyzer.Simplify(in_slice);

        PrimExpr idx_new_pre = pos - tvm::tir::make_const(idx_dtype, start) +
                               tvm::tir::make_const(idx_dtype, step - 1);
        PrimExpr idx_new_div =
            tvm::tir::FloorDiv(idx_new_pre, tvm::tir::make_const(idx_dtype, step));
        PrimExpr idx_new =
            tvm::max(tvm::tir::make_const(idx_dtype, 0),
                     tvm::min(idx_new_div, tvm::tir::make_const(idx_dtype, dim_size - 1)));
        idx_new = analyzer.Simplify(idx_new);

        ffi::Array<PrimExpr> src_idx;
        for (int d = 0; d < ndim; ++d) {
          if (d == axis) {
            src_idx.push_back(idx_new);
          } else {
            src_idx.push_back(out_idx[d]);
          }
        }

        return tvm::if_then_else(in_slice, src(src_idx), input(out_idx));
      },
      "slice_scatter");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("topi.scatter_nd",
           [](te::Tensor data, te::Tensor indices, te::Tensor updates, std::string mode) {
             return scatter_nd(data, indices, updates, mode);
           })
      .def("topi.scatter_elements",
           [](te::Tensor data, te::Tensor indices, te::Tensor updates, int axis,
              std::string reduction) {
             return scatter_elements(data, indices, updates, axis, reduction);
           })
      .def("topi.slice_scatter",
           [](te::Tensor input, te::Tensor src, int start, int end, int step, int axis) {
             return slice_scatter(input, src, start, end, step, axis);
           });
}

}  // namespace topi
}  // namespace tvm
