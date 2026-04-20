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
 * \file tvm/relax/transform/legalize_ops/vision.cc
 * \brief Legalize relax.vision.all_class_non_max_suppression to call_tir.
 */
#include <tvm/relax/attrs/vision.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/te/operation.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/topi/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

// Convenience: cast a PrimExpr to int32
static PrimExpr CastI32(PrimExpr e) { return tvm::cast(DataType::Int(32), e); }
// Convenience: cast a PrimExpr to int64
static PrimExpr CastI64(PrimExpr e) { return tvm::cast(DataType::Int(64), e); }
// Convenience: integer constant helpers
static PrimExpr I32(int64_t v) { return tvm::tir::make_const(DataType::Int(32), v); }
static PrimExpr I64(int64_t v) { return tvm::tir::make_const(DataType::Int(64), v); }
static PrimExpr F32(double v) { return tvm::tir::make_const(DataType::Float(32), v); }

/*!
 * \brief Build a 1-D Buffer declaration for use in ExternOp.
 *
 * \param size Number of elements.
 * \param dtype Element data type.
 * \param name Buffer name hint.
 * \return The declared 1-D buffer.
 */
static tir::Buffer DeclBuf1D(PrimExpr size, DataType dtype, const std::string& name) {
  return tir::decl_buffer({size}, dtype, name);
}
/*!
 * \brief Build a 2-D Buffer declaration for use in ExternOp.
 *
 * \param rows Number of rows.
 * \param cols Number of columns.
 * \param dtype Element data type.
 * \param name Buffer name hint.
 * \return The declared 2-D buffer.
 */
static tir::Buffer DeclBuf2D(PrimExpr rows, PrimExpr cols, DataType dtype,
                             const std::string& name) {
  return tir::decl_buffer({rows, cols}, dtype, name);
}

/*!
 * \brief Reshape a [B, C, N] scores tensor to [B*C, N].
 *
 * \param scores The input scores tensor.
 * \param batch_class The product B*C.
 * \param num_boxes The number of boxes N.
 * \return The reshaped 2-D tensor.
 */
static te::Tensor ReshapeScores(const te::Tensor& scores, PrimExpr batch_class,
                                PrimExpr num_boxes) {
  return topi::reshape(scores, {batch_class, num_boxes});
}

/*!
 * \brief Argsort each row of a 2-D tensor in descending order.
 *
 * Calls the `tvm.contrib.sort.argsort` packed function via an ExternOp.
 *
 * \param data The 2-D input tensor to sort.
 * \return A 2-D int32 tensor of sorted indices (same shape as `data`).
 */
static te::Tensor Argsort2D(const te::Tensor& data) {
  PrimExpr rows = data->shape[0];
  PrimExpr cols = data->shape[1];

  tir::Buffer data_buf = DeclBuf2D(rows, cols, data->dtype, "data_buf");
  tir::Buffer out_buf = DeclBuf2D(rows, cols, DataType::Int(32), "out_buf");

  auto make_dl_tensor = [&](const tir::Buffer& buf) -> PrimExpr {
    PrimExpr shape_ptr = tir::Call(DataType::Handle(), tir::builtin::tvm_stack_make_shape(),
                                   {buf->shape[0], buf->shape[1]});
    PrimExpr typed_zero = tvm::tir::make_const(buf->dtype, 0);
    return tir::Call(DataType::Handle(), tir::builtin::tvm_stack_make_array(),
                     {buf->data, shape_ptr,
                      /*strides=*/tvm::tir::make_zero(DataType::Handle()),
                      /*ndim=*/I32(2),
                      /*dtype_placeholder=*/typed_zero,
                      /*elem_offset=*/I32(0)});
  };

  PrimExpr data_dl = make_dl_tensor(data_buf);
  PrimExpr out_dl = make_dl_tensor(out_buf);

  tir::Stmt body = tir::Evaluate(tir::Call(DataType::Int(32), tir::builtin::tvm_call_packed(),
                                           {tir::StringImm("tvm.contrib.sort.argsort"), data_dl,
                                            out_dl, I32(1) /*axis*/, I32(0) /*is_ascend=False*/}));

  return te::ExternOp("argsort_cpu", "argsort_cpu", {},
                      /*inputs=*/{data},
                      /*input_placeholders=*/{data_buf},
                      /*output_placeholders=*/{out_buf},
                      /*body=*/body)
      .output(0);
}

/*!
 * \brief Gather rows of `scores` according to `sorted_indices`.
 *
 * \param scores The 2-D scores tensor.
 * \param sorted_indices The 2-D index tensor produced by Argsort2D.
 * \return The 2-D tensor of scores reordered by `sorted_indices`.
 */
static te::Tensor GatherScores(const te::Tensor& scores, const te::Tensor& sorted_indices) {
  return topi::gather(scores, 1, sorted_indices);
}

/*!
 * \brief For each row, binary-search for the first score <= score_threshold.
 *
 * Assumes each row of `sorted_scores` is sorted in descending order.
 * Returns a 1-D int32 tensor of length `batch_class` containing the
 * number of valid (above-threshold) boxes per row.
 *
 * \param sorted_scores The 2-D sorted scores tensor.
 * \param score_threshold_f32 The float32 score threshold.
 * \return A 1-D int32 tensor of valid-box counts.
 */
static te::Tensor SearchSorted(const te::Tensor& sorted_scores, PrimExpr score_threshold_f32) {
  PrimExpr batch_class = sorted_scores->shape[0];
  PrimExpr num_boxes = sorted_scores->shape[1];

  tir::Buffer scores_buf = DeclBuf2D(batch_class, num_boxes, DataType::Float(32), "scores_buf");
  tir::Buffer out_buf = DeclBuf1D(batch_class, DataType::Int(32), "searchsorted_buf");

  tir::Var y("y", DataType::Int(32));
  tir::Var lo_var("lo", DataType::Int(32));
  tir::Var hi_var("hi", DataType::Int(32));
  tir::Var mid_var("mid", DataType::Int(32));

  // Allocate lo and hi as local scalars.
  tir::Buffer lo_buf = tir::decl_buffer({I32(1)}, DataType::Int(32), "lo", "local");
  tir::Buffer hi_buf = tir::decl_buffer({I32(1)}, DataType::Int(32), "hi", "local");

  PrimExpr mid_expr = right_shift(
      hi_buf.vload({I32(0)}, DataType::Int(32)) + lo_buf.vload({I32(0)}, DataType::Int(32)),
      I32(1));

  tir::Stmt update_lo =
      tir::BufferStore(lo_buf, lo_buf.vload({I32(0)}, DataType::Int(32)) + I32(1), {I32(0)});
  tir::Stmt update_hi = tir::BufferStore(hi_buf, mid_expr, {I32(0)});
  tir::Stmt while_body = tir::IfThenElse(
      tir::GT(scores_buf.vload({CastI32(y), mid_expr}, DataType::Float(32)), score_threshold_f32),
      update_lo, update_hi);

  tir::Stmt while_loop = tir::While(
      tir::LT(lo_buf.vload({I32(0)}, DataType::Int(32)), hi_buf.vload({I32(0)}, DataType::Int(32))),
      while_body);

  tir::Stmt store_result =
      tir::BufferStore(out_buf, lo_buf.vload({I32(0)}, DataType::Int(32)), {CastI32(y)});

  tir::Stmt inner = tir::SeqStmt({
      tir::BufferStore(lo_buf, I32(0), {I32(0)}),
      tir::BufferStore(hi_buf, CastI32(num_boxes), {I32(0)}),
      while_loop,
      store_result,
  });
  tir::Stmt with_alloc = tir::Allocate(
      lo_buf->data, DataType::Int(32), {I32(1)}, tir::const_true(),
      tir::Allocate(hi_buf->data, DataType::Int(32), {I32(1)}, tir::const_true(), inner));

  tir::Stmt body = tir::For(y, I32(0), CastI32(batch_class), tir::ForKind::kParallel, with_alloc);

  return te::ExternOp("searchsorted", "searchsorted", {},
                      /*inputs=*/{sorted_scores},
                      /*input_placeholders=*/{scores_buf},
                      /*output_placeholders=*/{out_buf},
                      /*body=*/body)
      .output(0);
}

/*!
 * \brief Run all-class NMS and return selected box indices and per-class detection counts.
 *
 * \param boxes Input boxes tensor [B, N, 4] in float32.
 * \param sorted_scores Scores sorted in descending order [B*C, N].
 * \param sorted_indices Argsort indices corresponding to sorted_scores [B*C, N].
 * \param valid_count Number of above-threshold boxes per row [B*C].
 * \param iou_threshold_f32 IoU threshold for suppression.
 * \param max_output_size_i32 Maximum number of detections per class.
 * \param score_threshold_f32 Score threshold (used to skip already-suppressed boxes).
 * \param batch_class Product B*C.
 * \param num_class Number of classes C.
 * \param num_boxes Number of boxes N.
 * \return A pair (selected_indices [B*C, N], num_detections [B*C]).
 */
static std::pair<te::Tensor, te::Tensor> AllClassNMS(
    const te::Tensor& boxes, const te::Tensor& sorted_scores, const te::Tensor& sorted_indices,
    const te::Tensor& valid_count, PrimExpr iou_threshold_f32, PrimExpr max_output_size_i32,
    PrimExpr score_threshold_f32, PrimExpr batch_class, PrimExpr num_class, PrimExpr num_boxes) {
  PrimExpr num_class_i32 = CastI32(num_class);
  PrimExpr num_boxes_i32 = CastI32(num_boxes);
  PrimExpr batch_class_i32 = CastI32(batch_class);

  tir::Buffer boxes_buf = tir::decl_buffer({boxes->shape[0], boxes->shape[1], boxes->shape[2]},
                                           DataType::Float(32), "boxes_buf");
  tir::Buffer sorted_scores_buf =
      DeclBuf2D(batch_class, num_boxes, DataType::Float(32), "sorted_scores_buf");
  tir::Buffer sorted_indices_buf =
      DeclBuf2D(batch_class, num_boxes, DataType::Int(32), "sorted_indices_buf");
  tir::Buffer valid_count_buf = DeclBuf1D(batch_class, DataType::Int(32), "valid_count_buf");
  tir::Buffer box_indices_buf =
      DeclBuf2D(batch_class, num_boxes, DataType::Int(32), "all_class_nms0");
  tir::Buffer num_valid_buf = DeclBuf1D(batch_class, DataType::Int(32), "all_class_nms1");

  tir::Var i("i", DataType::Int(32));  // batch_class index
  tir::Var j("j", DataType::Int(32));  // box index (outer, valid boxes)
  tir::Var k("k", DataType::Int(32));  // box index (inner, suppression check)

  tir::Buffer nv_local_buf =
      tir::decl_buffer({I32(1)}, DataType::Int(32), "num_valid_boxes_local", "local");

  auto calc_iou = [&](PrimExpr bi, PrimExpr bj, PrimExpr bk) -> PrimExpr {
    PrimExpr batch_id = floordiv(bi, num_class_i32);
    PrimExpr base = batch_id * num_boxes_i32 * I32(4);
    auto load_box = [&](PrimExpr box_idx, int coord) -> PrimExpr {
      return boxes_buf.vload({CastI64(batch_id), CastI64(box_idx), I64(coord)},
                             DataType::Float(32));
    };
    PrimExpr sj = CastI32(sorted_indices_buf.vload({bi, bj}, DataType::Int(32)));
    PrimExpr sk = CastI32(sorted_indices_buf.vload({bi, bk}, DataType::Int(32)));

    PrimExpr aj_x1 = load_box(sj, 0), aj_y1 = load_box(sj, 1);
    PrimExpr aj_x2 = load_box(sj, 2), aj_y2 = load_box(sj, 3);
    PrimExpr ak_x1 = load_box(sk, 0), ak_y1 = load_box(sk, 1);
    PrimExpr ak_x2 = load_box(sk, 2), ak_y2 = load_box(sk, 3);

    PrimExpr a_l = tvm::min(aj_x1, aj_x2), a_t = tvm::min(aj_y1, aj_y2);
    PrimExpr a_r = tvm::max(aj_x1, aj_x2), a_b = tvm::max(aj_y1, aj_y2);
    PrimExpr b_l = tvm::min(ak_x1, ak_x2), b_t = tvm::min(ak_y1, ak_y2);
    PrimExpr b_r = tvm::max(ak_x1, ak_x2), b_b = tvm::max(ak_y1, ak_y2);

    PrimExpr w = tvm::max(F32(0.0), tvm::min(a_r, b_r) - tvm::max(a_l, b_l));
    PrimExpr h = tvm::max(F32(0.0), tvm::min(a_b, b_b) - tvm::max(a_t, b_t));
    PrimExpr area = w * h;
    PrimExpr u = (a_r - a_l) * (a_b - a_t) + (b_r - b_l) * (b_b - b_t) - area;
    return tir::Select(tir::LE(u, F32(0.0)), F32(0.0), area / u);
  };

  PrimExpr nkeep = valid_count_buf.vload({i}, DataType::Int(32));

  tir::Var k2("k2", DataType::Int(32));  // offset variable for parallel inner loop
  PrimExpr k_abs = j + I32(1) + k2;      // absolute box index
  PrimExpr num_to_check = nkeep - (j + I32(1));

  PrimExpr iou_val = calc_iou(i, j, k_abs);
  tir::Stmt suppress = tir::BufferStore(sorted_scores_buf, F32(-1.0f), {i, k_abs});
  tir::Stmt inner_if = tir::IfThenElse(
      tir::And(
          tir::And(tir::LT(k_abs, nkeep),
                   tir::GT(sorted_scores_buf.vload({i, k_abs}, DataType::Float(32)), F32(0.0f))),
          tir::GE(iou_val, iou_threshold_f32)),
      suppress);
  tir::Stmt inner_loop =
      tir::For(k2, I32(0), tvm::max(I32(0), num_to_check), tir::ForKind::kParallel, inner_if);

  PrimExpr cur_valid = nv_local_buf.vload({I32(0)}, DataType::Int(32));
  tir::Stmt record_box = tir::BufferStore(
      box_indices_buf, sorted_indices_buf.vload({i, j}, DataType::Int(32)), {i, cur_valid});
  tir::Stmt inc_valid = tir::BufferStore(nv_local_buf, cur_valid + I32(1), {I32(0)});

  PrimExpr score_j = sorted_scores_buf.vload({i, j}, DataType::Float(32));
  tir::Stmt j_body = tir::SeqStmt({record_box, inc_valid, inner_loop});
  tir::Stmt j_if = tir::IfThenElse(
      tir::And(
          tir::And(tir::GT(score_j, F32(-1.0f)),
                   tir::LT(nv_local_buf.vload({I32(0)}, DataType::Int(32)), max_output_size_i32)),
          tir::GT(score_j, score_threshold_f32)),
      j_body);
  tir::Stmt j_loop = tir::For(j, I32(0), nkeep, tir::ForKind::kSerial, j_if);

  tir::Stmt store_nv =
      tir::BufferStore(num_valid_buf, nv_local_buf.vload({I32(0)}, DataType::Int(32)), {i});

  tir::Stmt i_body_inner = tir::SeqStmt({
      tir::BufferStore(nv_local_buf, I32(0), {I32(0)}),
      j_loop,
      store_nv,
  });
  tir::Stmt i_body_with_alloc = tir::Allocate(nv_local_buf->data, DataType::Int(32), {I32(1)},
                                              tir::const_true(), i_body_inner);

  tir::Stmt zero_nv = tir::BufferStore(num_valid_buf, I32(0), {i});
  tir::Stmt i_body =
      tir::IfThenElse(tir::And(tir::GT(iou_threshold_f32, F32(0.0f)),
                               tir::GT(valid_count_buf.vload({i}, DataType::Int(32)), I32(0))),
                      i_body_with_alloc, zero_nv);

  tir::Stmt body = tir::For(i, I32(0), batch_class_i32, tir::ForKind::kSerial, i_body);

  te::ExternOp nms_op(
      "all_class_nms", "all_class_nms", {},
      /*inputs=*/{boxes, sorted_scores, sorted_indices, valid_count},
      /*input_placeholders=*/{boxes_buf, sorted_scores_buf, sorted_indices_buf, valid_count_buf},
      /*output_placeholders=*/{box_indices_buf, num_valid_buf},
      /*body=*/body);
  te::Tensor out_indices = nms_op.output(0);
  te::Tensor out_num_det = nms_op.output(1);
  return {out_indices, out_num_det};
}

/*!
 * \brief Compute the exclusive prefix-sum of a 1-D int32 tensor.
 *
 * The output dtype is int64. Element `i` of the output equals the sum of
 * the first `i` elements of the input.
 *
 * \param num_detections A 1-D int32 tensor of per-class detection counts.
 * \return A 1-D int64 tensor of row offsets.
 */
static te::Tensor ExclusiveCumsum(const te::Tensor& num_detections) {
  PrimExpr n = num_detections->shape[0];
  tir::Buffer in_buf = DeclBuf1D(n, DataType::Int(32), "cumsum_in");
  tir::Buffer out_buf = DeclBuf1D(n, DataType::Int(64), "cumsum_out");

  tir::Var idx("idx", DataType::Int(32));

  tir::Stmt init = tir::BufferStore(out_buf, I64(0), {I32(0)});
  PrimExpr prev_out = out_buf.vload({idx - I32(1)}, DataType::Int(64));
  PrimExpr prev_in = CastI64(in_buf.vload({idx - I32(1)}, DataType::Int(32)));
  tir::Stmt update = tir::BufferStore(out_buf, prev_out + prev_in, {idx});
  tir::Stmt loop = tir::For(idx, I32(1), CastI32(n), tir::ForKind::kSerial, update);
  tir::Stmt body = tir::SeqStmt({init, loop});

  return te::ExternOp("cumsum_generic", "cumsum_generic", {},
                      /*inputs=*/{num_detections},
                      /*input_placeholders=*/{in_buf},
                      /*output_placeholders=*/{out_buf},
                      /*body=*/body)
      .output(0);
}

/*!
 * \brief Sum a 1-D int32 tensor with each element clamped to max_boxes.
 *
 * Returns a scalar int64 tensor containing the total number of output rows.
 *
 * \param num_detections A 1-D int32 tensor of per-class detection counts.
 * \param max_boxes_i32 The per-class maximum box count.
 * \return A scalar int64 tensor.
 */
static te::Tensor SumClamped(const te::Tensor& num_detections, PrimExpr max_boxes_i32) {
  PrimExpr n = num_detections->shape[0];
  tir::Buffer in_buf = DeclBuf1D(n, DataType::Int(32), "sum_in");
  tir::Buffer out_buf = DeclBuf1D(I32(1), DataType::Int(64), "sum_out");

  tir::Var idx("idx", DataType::Int(32));
  tir::Buffer acc_buf = tir::decl_buffer({I32(1)}, DataType::Int(64), "acc", "local");

  tir::Stmt init_acc = tir::BufferStore(acc_buf, I64(0), {I32(0)});
  PrimExpr clamped = CastI64(tvm::min(in_buf.vload({idx}, DataType::Int(32)), max_boxes_i32));
  tir::Stmt add_step =
      tir::BufferStore(acc_buf, acc_buf.vload({I32(0)}, DataType::Int(64)) + clamped, {I32(0)});
  tir::Stmt loop = tir::For(idx, I32(0), CastI32(n), tir::ForKind::kSerial, add_step);
  tir::Stmt store_result =
      tir::BufferStore(out_buf, acc_buf.vload({I32(0)}, DataType::Int(64)), {I32(0)});
  tir::Stmt inner = tir::SeqStmt({init_acc, loop, store_result});
  tir::Stmt body =
      tir::Allocate(acc_buf->data, DataType::Int(64), {I32(1)}, tir::const_true(), inner);

  te::Tensor scalar_out = te::ExternOp("sum_clamped", "sum_clamped", {},
                                       /*inputs=*/{num_detections},
                                       /*input_placeholders=*/{in_buf},
                                       /*output_placeholders=*/{out_buf},
                                       /*body=*/body)
                              .output(0);
  return scalar_out;
}

/*!
 * \brief Collect (batch_id, class_id, box_id) triples for all selected boxes.
 *
 * Writes at most `min(num_detections[i], max_boxes)` triples per row `i`
 * into a flat output tensor of shape [out_rows, 3].
 *
 * \param selected_indices Selected box indices [B*C, N] from AllClassNMS.
 * \param num_detections Number of valid detections per row [B*C].
 * \param row_offsets Exclusive prefix-sum of num_detections [B*C].
 * \param num_class Number of classes C.
 * \param max_boxes_i32 Per-class maximum box count.
 * \param out_rows Total number of output rows (B*C*max_boxes).
 * \return A 2-D int64 tensor of shape [out_rows, 3].
 */
static te::Tensor CollectIndices(const te::Tensor& selected_indices,
                                 const te::Tensor& num_detections, const te::Tensor& row_offsets,
                                 PrimExpr num_class, PrimExpr max_boxes_i32, PrimExpr out_rows) {
  PrimExpr batch_class = selected_indices->shape[0];
  PrimExpr num_boxes = selected_indices->shape[1];

  tir::Buffer sel_buf = DeclBuf2D(batch_class, num_boxes, DataType::Int(32), "sel_indices_buf");
  tir::Buffer ndet_buf = DeclBuf1D(batch_class, DataType::Int(32), "ndet_buf");
  tir::Buffer roff_buf = DeclBuf1D(batch_class, DataType::Int(64), "roff_buf");
  tir::Buffer out_buf = tir::decl_buffer({out_rows, I32(3)}, DataType::Int(64), "collect_out");

  tir::Var bi("bi", DataType::Int(32));  // batch_class index
  tir::Var bj("bj", DataType::Int(32));  // box index within class

  tir::Var init_i("init_i", DataType::Int(32));
  tir::Var init_j("init_j", DataType::Int(32));
  tir::Stmt init_inner = tir::BufferStore(out_buf, I64(0), {init_i, init_j});
  tir::Stmt init_j_loop = tir::For(init_j, I32(0), I32(3), tir::ForKind::kSerial, init_inner);
  tir::Stmt init_loop =
      tir::For(init_i, I32(0), CastI32(out_rows), tir::ForKind::kSerial, init_j_loop);

  PrimExpr bi64 = CastI64(bi);
  PrimExpr batch_id = tir::FloorDiv(bi64, num_class);
  PrimExpr class_id = tir::FloorMod(bi64, num_class);
  PrimExpr limit = tvm::min(ndet_buf.vload({bi}, DataType::Int(32)), max_boxes_i32);
  PrimExpr row_base = roff_buf.vload({bi}, DataType::Int(64));
  PrimExpr box_global_idx = CastI64(sel_buf.vload({bi, bj}, DataType::Int(32)));

  tir::Stmt store0 = tir::BufferStore(out_buf, batch_id, {row_base + CastI64(bj), I64(0)});
  tir::Stmt store1 = tir::BufferStore(out_buf, class_id, {row_base + CastI64(bj), I64(1)});
  tir::Stmt store2 = tir::BufferStore(out_buf, box_global_idx, {row_base + CastI64(bj), I64(2)});
  tir::Stmt fill_body = tir::SeqStmt({store0, store1, store2});
  tir::Stmt bj_loop = tir::For(bj, I32(0), limit, tir::ForKind::kSerial, fill_body);
  tir::Stmt bi_loop = tir::For(bi, I32(0), CastI32(batch_class), tir::ForKind::kParallel, bj_loop);

  tir::Stmt body = tir::SeqStmt({init_loop, bi_loop});

  return te::ExternOp("collect_indices", "collect_indices", {},
                      /*inputs=*/{selected_indices, num_detections, row_offsets},
                      /*input_placeholders=*/{sel_buf, ndet_buf, roff_buf},
                      /*output_placeholders=*/{out_buf},
                      /*body=*/body)
      .output(0);
}

/*!
 * \brief TE handler for all-class non-maximum suppression.
 *
 * Orchestrates the full NMS pipeline: reshape scores, argsort, gather,
 * searchsorted, NMS, cumsum, sum, and collect-indices.
 *
 * \param args Packed argument list: [boxes, scores, max_output_boxes_per_class,
 *             iou_threshold, score_threshold].
 * \return A two-element array: [flat_indices [B*C*max_boxes, 3],
 *         num_total_detections [1]].
 */
static ffi::Array<te::Tensor> AllClassNMSHandler(const ffi::Array<ffi::Any>& args) {
  te::Tensor boxes = args[0].cast<te::Tensor>();      // [B, N, 4]  float32
  te::Tensor scores = args[1].cast<te::Tensor>();     // [B, C, N]  float32
  PrimExpr max_boxes_i64 = args[2].cast<PrimExpr>();  // int64 scalar
  PrimExpr iou_thr = args[3].cast<PrimExpr>();        // float32 scalar
  PrimExpr score_thr = args[4].cast<PrimExpr>();      // float32 scalar

  // Ensure scalar types are correct for the TIR bodies.
  PrimExpr iou_thr_f32 = tvm::cast(DataType::Float(32), iou_thr);
  PrimExpr score_thr_f32 = tvm::cast(DataType::Float(32), score_thr);
  PrimExpr max_boxes_i32 = tvm::cast(DataType::Int(32), max_boxes_i64);

  // Shape extraction
  PrimExpr batch = scores->shape[0];
  PrimExpr num_class = scores->shape[1];
  PrimExpr num_boxes = scores->shape[2];
  PrimExpr batch_class = batch * num_class;

  // Step 1: reshape scores [B, C, N] -> [B*C, N]
  te::Tensor scores_2d = ReshapeScores(scores, batch_class, num_boxes);

  // Step 2: argsort descending
  te::Tensor sorted_indices = Argsort2D(scores_2d);

  // Step 3: gather sorted scores
  te::Tensor sorted_scores = GatherScores(scores_2d, sorted_indices);

  // Step 4: searchsorted -> valid_count
  te::Tensor valid_count = SearchSorted(sorted_scores, score_thr_f32);

  // Step 5: all_class_nms
  auto [sel_indices, num_det] =
      AllClassNMS(boxes, sorted_scores, sorted_indices, valid_count, iou_thr_f32, max_boxes_i32,
                  score_thr_f32, batch_class, num_class, num_boxes);

  // Step 6: exclusive cumsum -> row_offsets
  te::Tensor row_offsets = ExclusiveCumsum(num_det);

  // Step 7: sum(clamp(num_det, max_boxes)) -> num_total_detections [1]
  te::Tensor num_total_det = SumClamped(num_det, max_boxes_i32);

  // Step 8: collect_indices -> flat_indices [B*C*max_boxes, 3]
  PrimExpr out_rows = batch_class * CastI64(max_boxes_i32);
  te::Tensor flat_indices =
      CollectIndices(sel_indices, num_det, row_offsets, num_class, max_boxes_i32, out_rows);

  return {flat_indices, num_total_det};
}

/*!
 * \brief Legalize relax.vision.all_class_non_max_suppression to call_tir.
 *
 * Only the ONNX output format is handled by this C++ path; other formats
 * fall back to the Python legalization. The function emits the full NMS
 * pipeline and trims the output via relax.dynamic_strided_slice.
 *
 * \param bb The block builder.
 * \param call The relax.vision.all_class_non_max_suppression call to legalize.
 * \return The legalized expression (a Tuple of trimmed indices and detection
 *         count), or the original call if the output format is not "onnx".
 */
Expr LegalizeAllClassNonMaxSuppression(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<AllClassNonMaximumSuppressionAttrs>();
  TVM_FFI_ICHECK_NOTNULL(attrs);
  // Only the ONNX output format is handled by this C++ path.
  // The tensorflow path is left to the Python fallback (lower priority).
  if (attrs->output_format != "onnx") {
    return call;  // signal: not legalized, let Python handle it
  }

  Expr boxes = call->args[0];                       // [B, N, 4]
  Expr scores = call->args[1];                      // [B, C, N]
  Expr max_output_boxes_per_class = call->args[2];  // scalar int64 tensor
  Expr iou_threshold = call->args[3];               // scalar float32 tensor
  Expr score_threshold = call->args[4];             // scalar float32 tensor

  // Extract max_boxes as a compile-time int64 constant.
  // The ONNX spec requires this to be a constant for static shape allocation.
  auto scores_sinfo = GetStructInfo(scores).as<TensorStructInfoNode>();
  TVM_FFI_ICHECK_NOTNULL(scores_sinfo);
  TVM_FFI_ICHECK(scores_sinfo->shape.defined()) << "scores must have a known static shape";
  auto scores_shape = scores_sinfo->shape.as<ShapeExprNode>()->values;
  PrimExpr num_boxes_expr = scores_shape[2];  // scores is [B, C, N]

  int64_t max_boxes_val = -1;
  if (auto c = max_output_boxes_per_class.as<ConstantNode>()) {
    auto sinfo = GetStructInfo(max_output_boxes_per_class).as<TensorStructInfoNode>();
    if (sinfo && sinfo->dtype.is_int()) {
      if (sinfo->dtype.bits() == 64) {
        max_boxes_val = *reinterpret_cast<const int64_t*>(c->data.operator->()->data);
      } else if (sinfo->dtype.bits() == 32) {
        max_boxes_val =
            static_cast<int64_t>(*reinterpret_cast<const int32_t*>(c->data.operator->()->data));
      }
    }
  }
  if (max_boxes_val < 0) {
    if (auto imm = num_boxes_expr.as<IntImmNode>()) {
      max_boxes_val = imm->value;
    } else {
      max_boxes_val = 0;
    }
  }

  // Extract iou_threshold and score_threshold as compile-time float constants.
  // If they are runtime tensors we read them as scalar PrimExprs via TryConvertToScalarConst.
  auto to_f32_prim = [&](Expr e) -> PrimExpr {
    ffi::Any scalar = TryConvertToScalarConst(e);
    if (auto fimm = scalar.as<FloatImm>()) return fimm.value();
    if (auto iimm = scalar.as<IntImm>()) {
      return tvm::FloatImm(DataType::Float(32), static_cast<double>(iimm.value()->value));
    }
    // Fallback: 0.0 (conservative)
    return tvm::FloatImm(DataType::Float(32), 0.0);
  };

  PrimExpr iou_thr_prim = to_f32_prim(iou_threshold);
  PrimExpr score_thr_prim = to_f32_prim(score_threshold);
  PrimExpr max_boxes_prim = tvm::IntImm(DataType::Int(64), max_boxes_val);

  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> topi_args;
  topi_args.push_back(boxes);
  topi_args.push_back(scores);
  topi_args.push_back(max_boxes_prim);
  topi_args.push_back(iou_thr_prim);
  topi_args.push_back(score_thr_prim);

  Expr nms_result =
      m_te.Make(topi_args, FTOPIHandler(AllClassNMSHandler), std::string("all_class_nms"));
  nms_result = bb->Normalize(nms_result);

  // The handler returns [flat_indices, num_total_detections].
  Expr flat_indices = bb->Emit(TupleGetItem(nms_result, 0));
  Expr num_total_detections = bb->Emit(TupleGetItem(nms_result, 1));

  // Step 9 – dynamic_strided_slice: trim flat_indices to valid rows only.
  // begin = [0, 0]  (int64, shape [2])
  auto begin_handler = [](const ffi::Array<ffi::Any>&) -> ffi::Array<te::Tensor> {
    return {te::compute(
        {I64(2)}, [](const tir::Var&) { return tvm::tir::make_const(DataType::Int(64), 0); },
        "begin")};
  };

  // strides = [1, 1]  (int64, shape [2])
  auto strides_handler = [](const ffi::Array<ffi::Any>&) -> ffi::Array<te::Tensor> {
    return {te::compute(
        {I64(2)}, [](const tir::Var&) { return tvm::tir::make_const(DataType::Int(64), 1); },
        "strides")};
  };

  // end = [num_total_detections[0], 3]  (int64, shape [2])
  auto end_handler = [](const ffi::Array<ffi::Any>& args) -> ffi::Array<te::Tensor> {
    te::Tensor count = args[0].cast<te::Tensor>();  // shape [1], dtype int64
    return {te::compute(
        {I64(2)},
        [count](const tir::Var& i) -> PrimExpr {
          return tir::Select(tir::EQ(i, tvm::tir::make_const(DataType::Int(64), 0)),
                             count(tvm::tir::make_const(DataType::Int(64), 0)),
                             tvm::tir::make_const(DataType::Int(64), 3));
        },
        "end")};
  };

  // Materialise the three slice-parameter tensors via MakeCallTE.
  auto m_begin = MakeCallTE(bb, call);
  Expr begin_expr = m_begin.Make({}, FTOPIHandler(begin_handler), std::string("slice_begin"));
  begin_expr = bb->Emit(begin_expr);

  auto m_strides = MakeCallTE(bb, call);
  Expr strides_expr =
      m_strides.Make({}, FTOPIHandler(strides_handler), std::string("slice_strides"));
  strides_expr = bb->Emit(strides_expr);

  auto m_end = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> end_args;
  end_args.push_back(num_total_detections);
  Expr end_expr = m_end.Make(end_args, FTOPIHandler(end_handler), std::string("slice_end"));
  end_expr = bb->Emit(end_expr);

  Expr trimmed_indices =
      bb->Emit(Call(Op::Get("relax.dynamic_strided_slice"),
                    {flat_indices, begin_expr, end_expr, strides_expr}, tvm::Attrs(), {}));

  return relax::Tuple({trimmed_indices, num_total_detections});
}

TVM_REGISTER_OP("relax.vision.all_class_non_max_suppression")
    .set_attr<FLegalize>("FLegalize", LegalizeAllClassNonMaxSuppression, TVM_LEGALIZE_CPP_LEVEL);

}  // namespace relax
}  // namespace tvm
