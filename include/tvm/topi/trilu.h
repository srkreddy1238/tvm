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
 * \file tvm/topi/trilu.h
 * \brief C++ TOPI declaration for trilu (tril / triu).
 */
#ifndef TVM_TOPI_TRILU_H_
#define TVM_TOPI_TRILU_H_

#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/topi/tags.h>

namespace tvm {
namespace topi {

/*!
 * \brief Return the upper or lower triangular part of a matrix (or batch of matrices).
 * \param data  Input tensor of rank >= 2.
 * \param k     Diagonal offset (int32).  0 = main diagonal, positive = above, negative = below.
 * \param upper If true compute triu, otherwise tril.
 * \return Output tensor with same shape and dtype as \p data.
 */
inline te::Tensor trilu(const te::Tensor& data, const PrimExpr& k, bool upper) {
  // Ensure k is int32, matching the Python: `if k.dtype != "int32": k = Cast("int32", k)`
  PrimExpr k32 = (k.dtype() == DataType::Int(32)) ? k : tvm::cast(DataType::Int(32), k);

  return te::compute(
      data->shape,
      [&](const ffi::Array<tir::Var>& indices) -> PrimExpr {
        const int ndim = static_cast<int>(indices.size());

        // Last two indices are the row and column of the 2-D matrix slice.
        PrimExpr row_idx = indices[ndim - 2];
        PrimExpr col_idx = indices[ndim - 1];

        // Promote row and col to a common type if they differ, mirroring:
        PrimExpr row = row_idx;
        PrimExpr col = col_idx;
        if (row.dtype() != col.dtype()) {
          DataType target = (col + row).dtype();
          if (row.dtype() != target)
            row = tvm::cast(target, row);
          else
            col = tvm::cast(target, col);
        }

        // check_position:
        //   upper → row <= col - k   (tvm::tir::LE)
        //   lower → row >= col - k   (tvm::tir::GE)
        PrimExpr threshold = col - tvm::cast(col.dtype(), k32);
        PrimExpr keep = upper ? (row <= threshold) : (row >= threshold);

        // Gather all indices to index into data.
        ffi::Array<PrimExpr> data_idx;
        for (const auto& idx : indices) data_idx.push_back(idx);

        return tir::Select(keep, data(data_idx), tir::make_const(data->dtype, 0));
      },
      "trilu", kElementWise);
}

}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_TRILU_H_
