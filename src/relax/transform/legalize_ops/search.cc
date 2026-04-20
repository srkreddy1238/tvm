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
 * \file tvm/relax/transform/legalize_ops/search.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */
#include <tvm/relax/attrs/search.h>
#include <tvm/topi/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

// ArgMin / ArgMax
#define TVM_LEGALIZE_ARGMIN_ARGMAX_OP(OpName, TopiHandler)                            \
  Expr MAKE_NAME(Legalize, OpName)(const BlockBuilder& bb, const Call& call) {        \
    const auto* attrs = call->attrs.as<ArgmaxArgminAttrs>();                          \
    auto m_te = MakeCallTE(bb, call);                                                 \
    tvm::ffi::Array<tvm::ffi::Any> args;                                              \
    args.push_back(call->args[0]);                                                    \
    if (attrs->axis.has_value()) {                                                    \
      args.push_back(attrs->axis.value());                                            \
    } else {                                                                          \
      args.push_back(nullptr);                                                        \
    }                                                                                 \
    args.push_back(attrs->keepdims);                                                  \
    args.push_back(false); /* select_last_index */                                    \
    auto call_ret = m_te.Make(args, ffi::String(#TopiHandler), std::string(#OpName)); \
    return call_ret;                                                                  \
  }                                                                                   \
  TVM_REGISTER_OP("relax." #OpName)                                                   \
      .set_attr<FLegalize>("FLegalize", MAKE_NAME(Legalize, OpName), TVM_LEGALIZE_CPP_LEVEL);

TVM_LEGALIZE_ARGMIN_ARGMAX_OP(argmin, topi.argmin);
TVM_LEGALIZE_ARGMIN_ARGMAX_OP(argmax, topi.argmax);

/*!
 * \brief Legalize relax.where to call_tir via topi.where.
 *
 * \param bb The block builder.
 * \param call The relax.where call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeWhere(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(call->args[2]);
  return m_te.Make(args, ffi::String("topi.where"), std::string("where"));
}
TVM_REGISTER_OP("relax.where")
    .set_attr<FLegalize>("FLegalize", LegalizeWhere, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Legalize relax.bucketize to call_tir via topi.searchsorted.
 *
 * \param bb The block builder.
 * \param call The relax.bucketize call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeBucketize(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<BucketizeAttrs>();
  auto m_te = MakeCallTE(bb, call);
  auto input_tensor = call->args[0];
  auto boundaries = call->args[1];
  auto out_dtype = attrs->out_int32 ? DataType::Int(32) : DataType::Int(64);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(boundaries);
  args.push_back(input_tensor);
  args.push_back(attrs->right);
  args.push_back(out_dtype);
  return m_te.Make(args, ffi::String("topi.searchsorted"), std::string("bucketize"));
}
TVM_REGISTER_OP("relax.bucketize")
    .set_attr<FLegalize>("FLegalize", LegalizeBucketize, TVM_LEGALIZE_CPP_LEVEL);

}  // namespace relax
}  // namespace tvm
