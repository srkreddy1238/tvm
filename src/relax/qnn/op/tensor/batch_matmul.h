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
 * \file batch_matmul.h
 * \brief The functions to make Relax neural network binary operator batch matmul
 */

#ifndef TVM_RELAX_QNN_OP_TENSOR_BATCH_MATMUL_H_
#define TVM_RELAX_QNN_OP_TENSOR_BATCH_MATMUL_H_

#include <tvm/relax/qnn/attrs.h>

#include <string>
#include <utility>
#include <vector>

#include "../../../op/op_common.h"
#include "../../../op/tensor/linear_algebra.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace qnn {
Expr batch_matmul(Expr x, Expr y, Expr x_zero_point, Expr y_zero_point, Expr x_scale, Expr y_scale,
                  DataType out_dtype);

}  // namespace qnn
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_QNN_OP_TENSOR_BATCH_MATMUL_H_
