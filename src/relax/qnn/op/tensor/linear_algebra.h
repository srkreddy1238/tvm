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
 * \file dense.h
 * \brief QNN Dense operator header
 */
#ifndef TVM_RELAX_QNN_OP_TENSOR_LINEAR_ALGEBRA_H_
#define TVM_RELAX_QNN_OP_TENSOR_LINEAR_ALGEBRA_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/qnn/attrs.h>

#include "../../../op/op_common.h"
#include "../../../op/tensor/linear_algebra.h"

namespace tvm {
namespace relax {
namespace qnn {

/*!
 * \brief Quantized dense operator (TFLite-style API).
 *
 * Computes quantized dense without using floating point:
 * output_int32 = (data - input_zp) @ (weight - kernel_zp)
 *
 * The output is int32/int64 with:
 * - Effective scale: input_scale × kernel_scale
 * - Effective zero point: 0
 *
 * This does NOT apply output requantization. Use relax.qnn.requantize afterward.
 *
 * Note: Weight must already be transposed! For TFLite conversion, use
 * relax.op.permute_dims(weight, [1, 0]) before calling this function.
 *
 * \param data The input quantized tensor (int8/uint8)
 * \param weight The weight quantized tensor, already transposed (int8/uint8)
 * \param input_zero_point Zero point for input (scalar)
 * \param kernel_zero_point Zero point for weight/kernel (scalar)
 * \param input_scale Scale for input (scalar float)
 * \param kernel_scale Scale for weight/kernel (scalar float)
 * \param units Number of output units (-1 to infer from weight shape)
 * \param out_dtype Output data type ("int32" or "int64")
 * \return The quantized dense output (int32/int64)
 */
Expr dense(Expr data, Expr weight, Expr input_scale, Expr input_zero_point, Expr kernel_scale,
           Expr kernel_zero_point, DataType out_dtype);

}  // namespace qnn
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_QNN_OP_TENSOR_LINEAR_ALGEBRA_H_
