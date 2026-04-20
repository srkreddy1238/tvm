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
 * \file tvm/relax/transform/legalize_ops/datatype.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */
#include "utils.h"

namespace tvm {
namespace relax {

/*!
 * \brief Legalize relax.astype to call_tir via topi.cast.
 *
 * \note Priority is set to 9 (below the default Python level of 10) so that
 *       the Python legalization, which handles scalar casting via NumPy
 *       helpers, takes precedence.
 *
 * \param bb The block builder.
 * \param call The relax.astype call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeAsType(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(Downcast<TensorStructInfo>(GetStructInfo(call))->dtype);
  auto call_ret = m_te.Make(args, ffi::String("topi.cast"), std::string("tir_astype"));
  return call_ret;
}
/*
 * Don't raise the priority above 10 which lets py interface use this legalization
 * Python legalization does handle scalars casting via numpy helpers.
 */
TVM_REGISTER_OP("relax.astype").set_attr<FLegalize>("FLegalize", LegalizeAsType, 9);

}  // namespace relax
}  // namespace tvm
