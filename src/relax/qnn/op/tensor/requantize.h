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
 * \file requantize.h
 * \brief The functions to make Relax neural network binary operator requantize...
 */

#ifndef TVM_RELAX_QNN_OP_TENSOR_REQUANTIZE_H_
#define TVM_RELAX_QNN_OP_TENSOR_REQUANTIZE_H_

#include <tvm/relax/attrs/qdq.h>

#include "../../../op/op_common.h"
#include "../../../transform/utils.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace qnn {

Expr requantize(Expr data, Expr input_scale, Expr input_zero_point, Expr output_scale,
                Expr output_zero_point, int axis, DataType out_dtype);

StructInfo InferStructInfoRequantize(const Call& call, const BlockBuilder& ctx);

InferLayoutOutput InferLayoutRequantize(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map);
}  // namespace qnn
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_QNN_OP_TENSOR_REQUANTIZE_H_
