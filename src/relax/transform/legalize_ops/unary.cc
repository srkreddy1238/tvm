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
#include <tvm/topi/elemwise.h>
#include <tvm/topi/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

#define TVM_LEGALIZE_UNARY_OP(OpName)                                                   \
  Expr MAKE_NAME(UnaryLegalize, OpName)(const BlockBuilder& bb, const Call& call) {     \
    auto m_te = MakeCallTE(bb, call);                                                   \
    auto call_ret = m_te.Make(tvm::ffi::Array<tvm::ffi::Any>({call->args[0]}),          \
                              ffi::String(std::string("topi.") + std::string(#OpName)), \
                              std::string("tir_") + std::string(#OpName));              \
    return call_ret;                                                                    \
  }                                                                                     \
  TVM_REGISTER_OP("relax." #OpName)                                                     \
      .set_attr<FLegalize>("FLegalize", MAKE_NAME(UnaryLegalize, OpName), TVM_LEGALIZE_CPP_LEVEL);

TVM_LEGALIZE_UNARY_OP(acos);
TVM_LEGALIZE_UNARY_OP(acosh)
TVM_LEGALIZE_UNARY_OP(asin)
TVM_LEGALIZE_UNARY_OP(asinh)
TVM_LEGALIZE_UNARY_OP(atan)
TVM_LEGALIZE_UNARY_OP(atanh)
TVM_LEGALIZE_UNARY_OP(abs)
TVM_LEGALIZE_UNARY_OP(bitwise_not)
TVM_LEGALIZE_UNARY_OP(ceil)
TVM_LEGALIZE_UNARY_OP(cos)
TVM_LEGALIZE_UNARY_OP(cosh)
TVM_LEGALIZE_UNARY_OP(exp)
TVM_LEGALIZE_UNARY_OP(fast_erf)
TVM_LEGALIZE_UNARY_OP(fast_exp)
TVM_LEGALIZE_UNARY_OP(fast_tanh)
TVM_LEGALIZE_UNARY_OP(floor)
TVM_LEGALIZE_UNARY_OP(identity)
TVM_LEGALIZE_UNARY_OP(log)
TVM_LEGALIZE_UNARY_OP(log10)
TVM_LEGALIZE_UNARY_OP(log2)
TVM_LEGALIZE_UNARY_OP(logical_not)
TVM_LEGALIZE_UNARY_OP(negative)
TVM_LEGALIZE_UNARY_OP(round)
TVM_LEGALIZE_UNARY_OP(rsqrt)
TVM_LEGALIZE_UNARY_OP(sigmoid)
TVM_LEGALIZE_UNARY_OP(sign)
TVM_LEGALIZE_UNARY_OP(sin)
TVM_LEGALIZE_UNARY_OP(sinh)
TVM_LEGALIZE_UNARY_OP(sqrt)
TVM_LEGALIZE_UNARY_OP(tan)
TVM_LEGALIZE_UNARY_OP(tanh)
TVM_LEGALIZE_UNARY_OP(trunc)
TVM_LEGALIZE_UNARY_OP(cast)
TVM_LEGALIZE_UNARY_OP(reinterpret)
TVM_LEGALIZE_UNARY_OP(elementwise_sum)

// square: x * x
ffi::Array<te::Tensor> SquareTE(const ffi::Array<ffi::Any> args) {
  auto x = args[0].cast<te::Tensor>();
  return {tvm::te::compute(
      x->shape, [&](const ffi::Array<tvm::tir::Var>& i) { return x(i) * x(i); }, "tir_square",
      topi::kElementWise)};
}
Expr UnaryLegalizeSquare(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  return m_te.Make(tvm::ffi::Array<tvm::ffi::Any>({call->args[0]}), FTOPIHandler(SquareTE),
                   std::string("tir_square"));
}
TVM_REGISTER_OP("relax.square")
    .set_attr<FLegalize>("FLegalize", UnaryLegalizeSquare, TVM_LEGALIZE_CPP_LEVEL);

// erf: float16 inputs are cast to float32, erf computed, then cast back.
ffi::Array<te::Tensor> ErfTE(const ffi::Array<ffi::Any> args) {
  auto x = args[0].cast<te::Tensor>();
  if (x->dtype == DataType::Float(16)) {
    auto x_f32 = topi::cast(x, DataType::Float(32));
    auto erf_f32 = topi::erf(x_f32);
    return {topi::cast(erf_f32, DataType::Float(16))};
  }
  return {topi::erf(x)};
}
Expr UnaryLegalizeErf(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  return m_te.Make(tvm::ffi::Array<tvm::ffi::Any>({call->args[0]}), FTOPIHandler(ErfTE),
                   std::string("tir_erf"));
}
TVM_REGISTER_OP("relax.erf")
    .set_attr<FLegalize>("FLegalize", UnaryLegalizeErf, TVM_LEGALIZE_CPP_LEVEL);

Expr UnaryLegalizeClip(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  auto call_ret =
      m_te.Make(tvm::ffi::Array<tvm::ffi::Any>({call->args[0], call->args[1], call->args[2]}),
                ffi::String("topi.clip"), std::string("tir_clip"));
  return call_ret;
}
TVM_REGISTER_OP("relax.clip")
    .set_attr<FLegalize>("FLegalize", UnaryLegalizeClip, TVM_LEGALIZE_CPP_LEVEL);

}  // namespace relax
}  // namespace tvm
