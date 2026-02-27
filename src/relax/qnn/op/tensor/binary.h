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
 * \file binary.h
 * \brief The functions to infer struct info for QNN binary operators
 */

#ifndef TVM_RELAX_QNN_OP_TENSOR_BINARY_H_
#define TVM_RELAX_QNN_OP_TENSOR_BINARY_H_

#include <string>
#include <utility>

#include "../../../op/op_common.h"

namespace tvm {
namespace relax {
namespace qnn {

/*
 * \brief Number of inputs for the QNN binary operator.
 */
static constexpr int kNumQnnBinaryOpInputs = 8;

// Struct info inference functions
StructInfo InferStructInfoQnnBinaryArith(const Call& call, const BlockBuilder& ctx);

}  // namespace qnn
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_QNN_OP_TENSOR_BINARY_H_
