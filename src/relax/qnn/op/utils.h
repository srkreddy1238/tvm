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

// utils.h (qnn needed helper functions)


#ifndef TVM_RELAX_QNN_OP_UTILS_H_
#define TVM_RELAX_QNN_OP_UTILS_H_

#include <string>
#include <utility>
#include <vector>

#include "../../op/op_common.h"
#include "../../transform/utils.h"


namespace tvm {
namespace relax {
namespace qnn {

// Checks that data has integer dtype (int or uint).
// Reports a fatal error on mismatch.
void CheckIntegerInputDtype(const Call& call, const BlockBuilder& ctx,
                            const TensorStructInfo& data_sinfo,
                            const std::string& op_name);

// Checks that scale is float (or unknown), otherwise fatal.
void CheckScaleDtype(const Call& call, const BlockBuilder& ctx,
                     const TensorStructInfo& sinfo, const std::string& op_name,
                     const char* name);

// Checks that zero point is int/uint (or unknown), otherwise fatal.
void CheckZeroPointDtype(const Call& call, const BlockBuilder& ctx,
                         const TensorStructInfo& sinfo, const std::string& op_name,
                         const char* name);

}  // namespace qnn
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_QNN_OP_UTILS_H_
