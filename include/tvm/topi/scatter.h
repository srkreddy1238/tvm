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
 * \file tvm/topi/scatter.h
 * \brief C++ TOPI declarations for scatter_nd, scatter_elements, and slice_scatter.
 */
#ifndef TVM_TOPI_SCATTER_H_
#define TVM_TOPI_SCATTER_H_

#include <tvm/te/operation.h>

#include <string>

namespace tvm {
namespace topi {

/*!
 * \brief Scatter updates into data at multi-dimensional indices.
 * \param data    Source tensor (output is initialised from this).
 * \param indices Integer index tensor, shape (M, ...).
 * \param updates Values to scatter.
 * \param mode    Reduction mode: "update", "add", "mul", "min", "max".
 * \return Output tensor with same shape as data.
 */
TVM_DLL te::Tensor scatter_nd(const te::Tensor& data, const te::Tensor& indices,
                              const te::Tensor& updates, const std::string& mode);

/*!
 * \brief Scatter elements from updates into data along a given axis.
 * data, indices, updates all have the same rank.
 * \param data      Source tensor.
 * \param indices   Integer index tensor (same shape as updates).
 * \param updates   Values to scatter.
 * \param axis      Axis along which to scatter (may be negative).
 * \param reduction Reduction mode: "update", "add", "mul", "mean", "min", "max".
 * \return Output tensor with same shape as data.
 */
TVM_DLL te::Tensor scatter_elements(const te::Tensor& data, const te::Tensor& indices,
                                    const te::Tensor& updates, int axis,
                                    const std::string& reduction);

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
TVM_DLL te::Tensor slice_scatter(const te::Tensor& input, const te::Tensor& src, int start, int end,
                                 int step, int axis);

}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_SCATTER_H_
