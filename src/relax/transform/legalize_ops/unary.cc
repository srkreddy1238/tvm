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
 * \file tvm/relax/transform/legalize_ops/unary.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */
#include "utils.h"

namespace tvm {
namespace relax {

#define TVM_LEGALIZE_UNARY_OP(OpName)                                               \
  Expr MAKE_NAME(UnaryLegalize, OpName)(const BlockBuilder& bb, const Call& call) { \
    auto m_te = MakeCallTE(bb, call);                                               \
    auto call_ret = m_te.Make(tvm::ffi::Array<tvm::ffi::Any>({call->args[0]}),      \
                              std::string("topi.") + std::string(#OpName),          \
                              std::string("tir_") + std::string(#OpName));          \
    return call_ret;                                                                \
  }                                                                                 \
  TVM_REGISTER_OP("relax." #OpName)                                                 \
      .set_attr<FLegalize>("FLegalize", MAKE_NAME(UnaryLegalize, OpName), 9);

TVM_LEGALIZE_UNARY_OP(acos);
TVM_LEGALIZE_UNARY_OP(acosh)
TVM_LEGALIZE_UNARY_OP(asin)
TVM_LEGALIZE_UNARY_OP(asinh)
TVM_LEGALIZE_UNARY_OP(atan)
TVM_LEGALIZE_UNARY_OP(atanh)
TVM_LEGALIZE_UNARY_OP(bitwise_not)
TVM_LEGALIZE_UNARY_OP(cos)
TVM_LEGALIZE_UNARY_OP(erf)
TVM_LEGALIZE_UNARY_OP(exp)
TVM_LEGALIZE_UNARY_OP(fast_erf)
TVM_LEGALIZE_UNARY_OP(fast_exp)
TVM_LEGALIZE_UNARY_OP(fast_tanh)
TVM_LEGALIZE_UNARY_OP(identity)
TVM_LEGALIZE_UNARY_OP(log)
TVM_LEGALIZE_UNARY_OP(log10)
TVM_LEGALIZE_UNARY_OP(log2)
TVM_LEGALIZE_UNARY_OP(logical_not)
TVM_LEGALIZE_UNARY_OP(negative)
TVM_LEGALIZE_UNARY_OP(rsqrt)
TVM_LEGALIZE_UNARY_OP(sigmoid)
TVM_LEGALIZE_UNARY_OP(sin)
TVM_LEGALIZE_UNARY_OP(sinh)
TVM_LEGALIZE_UNARY_OP(sqrt)
TVM_LEGALIZE_UNARY_OP(tan)
TVM_LEGALIZE_UNARY_OP(tanh)
TVM_LEGALIZE_UNARY_OP(cast)
TVM_LEGALIZE_UNARY_OP(reinterpret)
TVM_LEGALIZE_UNARY_OP(elementwise_sum)

Expr UnaryLegalizeClip(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  auto call_ret =
      m_te.Make(tvm::ffi::Array<tvm::ffi::Any>({call->args[0], call->args[1], call->args[2]}),
                std::string("topi.clip"), std::string("tir_clip"));
  return call_ret;
}
TVM_REGISTER_OP("relax.clip").set_attr<FLegalize>("FLegalize", UnaryLegalizeClip, 9);

}  // namespace relax
}  // namespace tvm
