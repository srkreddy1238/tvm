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
 * \file tvm/relax/transform/legalize_ops/binary.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */
#include "utils.h"

namespace tvm {
namespace relax {

#define TVM_LEGALIZE_BINARY_OP(OpName, TopiHandler)                                  \
  Expr MAKE_NAME(BinaryLegalize, OpName)(const BlockBuilder& bb, const Call& call) { \
    auto m_te = MakeCallTE(bb, call);                                                \
    tvm::ffi::Array<tvm::ffi::Any> call_args;                                        \
    call_args.push_back(TryConvertToScalarConst(call->args[0]));                     \
    if (call_args[0].as<relax::Expr>()) {                                            \
      call_args.push_back(TryConvertToScalarConst(call->args[1]));                   \
    } else {                                                                         \
      call_args.push_back(call->args[1]);                                            \
    }                                                                                \
    auto call_ret = m_te.Make(call_args, ffi::String(#TopiHandler), #OpName);        \
    return call_ret;                                                                 \
  }                                                                                  \
  TVM_REGISTER_OP("relax." #OpName)                                                  \
      .set_attr<FLegalize>("FLegalize", MAKE_NAME(BinaryLegalize, OpName),           \
                           TVM_LEGALIZE_CPP_LEVEL);

TVM_LEGALIZE_BINARY_OP(add, topi.add);
TVM_LEGALIZE_BINARY_OP(subtract, topi.subtract);
TVM_LEGALIZE_BINARY_OP(divide, topi.divide);
TVM_LEGALIZE_BINARY_OP(floor_divide, topi.floor_divide);
TVM_LEGALIZE_BINARY_OP(log_add_exp, topi.log_add_exp);
TVM_LEGALIZE_BINARY_OP(multiply, topi.multiply);
TVM_LEGALIZE_BINARY_OP(power, topi.power);
TVM_LEGALIZE_BINARY_OP(equal, topi.equal);
TVM_LEGALIZE_BINARY_OP(mod, topi.mod);
TVM_LEGALIZE_BINARY_OP(floor_mod, topi.floor_mod);
TVM_LEGALIZE_BINARY_OP(greater, topi.greater);
TVM_LEGALIZE_BINARY_OP(greater_equal, topi.greater_equal);
TVM_LEGALIZE_BINARY_OP(less, topi.less);
TVM_LEGALIZE_BINARY_OP(less_equal, topi.less_equal);
TVM_LEGALIZE_BINARY_OP(not_equal, topi.not_equal);

TVM_LEGALIZE_BINARY_OP(maximum, topi.maximum);
TVM_LEGALIZE_BINARY_OP(minimum, topi.minimum);

// Bitwise
TVM_LEGALIZE_BINARY_OP(bitwise_and, topi.bitwise_and);
TVM_LEGALIZE_BINARY_OP(bitwise_or, topi.bitwise_or);
TVM_LEGALIZE_BINARY_OP(bitwise_xor, topi.bitwise_xor);
TVM_LEGALIZE_BINARY_OP(left_shift, topi.left_shift);
TVM_LEGALIZE_BINARY_OP(right_shift, topi.right_shift);

// Logical
TVM_LEGALIZE_BINARY_OP(logical_and, topi.logical_and);
TVM_LEGALIZE_BINARY_OP(logical_or, topi.logical_or);
TVM_LEGALIZE_BINARY_OP(logical_xor, topi.logical_xor);

}  // namespace relax
}  // namespace tvm
