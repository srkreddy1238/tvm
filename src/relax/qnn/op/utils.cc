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

#include "./utils.h"

namespace tvm {
namespace relax {
namespace qnn {

void CheckIntegerInputDtype(const Call& call, const BlockBuilder& ctx,
                            const TensorStructInfo& data_sinfo,
                            const std::string& op_name) {
  if (!data_sinfo->IsUnknownDtype() && !data_sinfo->dtype.is_int() &&
      !data_sinfo->dtype.is_uint()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << op_name << ": Input data must be of integer type. "
                     << "However, the given dtype is " << data_sinfo->dtype);
  }
}

void CheckScaleDtype(const Call& call, const BlockBuilder& ctx,
                     const TensorStructInfo& sinfo, const std::string& op_name,
                     const char* name) {
  if (!sinfo->IsUnknownDtype() && !sinfo->dtype.is_float()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << op_name << ": " << name
                     << " must have a floating-point dtype. However, the given dtype is "
                     << sinfo->dtype);
  }
}

void CheckZeroPointDtype(const Call& call, const BlockBuilder& ctx,
                         const TensorStructInfo& sinfo, const std::string& op_name,
                         const char* name) {
  if (!sinfo->IsUnknownDtype() && !sinfo->dtype.is_int() && !sinfo->dtype.is_uint()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << op_name << ": " << name
                     << " must have an integer dtype. However, the given dtype is "
                     << sinfo->dtype);
  }
}

}  // namespace qnn
}  // namespace relax
}  // namespace tvm
