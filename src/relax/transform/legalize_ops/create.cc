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
 * \file tvm/relax/transform/legalize_ops/create.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */
#include "utils.h"

namespace tvm {
namespace relax {

#define TVM_LEGALIZE_CREATE_OP(OpName, IsLike)                                       \
  Expr MAKE_NAME(CreateLegalize, OpName)(const BlockBuilder& bb, const Call& call) { \
    auto m_te = MakeCallTE(bb, call);                                                \
    tvm::ffi::Array<tvm::ffi::Any> args;                                             \
    if (IsLike) {                                                                    \
      args.push_back(call->args[0]);                                                 \
    } else {                                                                         \
      args.push_back(Downcast<ShapeExpr>(call->args[0]));                            \
      args.push_back(Downcast<TensorStructInfo>(GetStructInfo(call))->dtype);        \
    }                                                                                \
    args.push_back(call->args[1]);                                                   \
    auto call_ret = m_te.Make(args, std::string("topi.") + std::string(#OpName),     \
                              std::string("tir_") + std::string(#OpName));           \
    return call_ret;                                                                 \
  }                                                                                  \
  TVM_REGISTER_OP("relax." #OpName)                                                  \
      .set_attr<FLegalize>("FLegalize", MAKE_NAME(CreateLegalize, OpName), 9);

TVM_LEGALIZE_CREATE_OP(full, false);
TVM_LEGALIZE_CREATE_OP(full_like, true);

#define TVM_LEGALIZE_CREATE_OP_BY_VALUE(OpName, IsLike, Value)                                 \
  Expr MAKE_NAME(CreateLegalize, OpName)(const BlockBuilder& bb, const Call& call) {           \
    auto m_te = MakeCallTE(bb, call);                                                          \
    tvm::ffi::Array<tvm::ffi::Any> args;                                                       \
    auto call_dtype = Downcast<TensorStructInfo>(GetStructInfo(call))->dtype;                  \
    std::string topi_op_name = "topi.full";                                                    \
    if (IsLike) {                                                                              \
      args.push_back(call->args[0]);                                                           \
      topi_op_name = topi_op_name + "_like";                                                   \
    } else {                                                                                   \
      args.push_back(Downcast<ShapeExpr>(call->args[0]));                                      \
      args.push_back(call_dtype);                                                              \
    }                                                                                          \
    auto value_tensor = runtime::Tensor::Empty(ffi::Shape({}), call_dtype,                     \
                                               tvm::Device({kDLCPU, 0}), std::nullopt);        \
    if (call_dtype.code() == kDLInt) {                                                         \
      int i_val = Value;                                                                       \
      value_tensor.CopyFromBytes(static_cast<void*>(&i_val), 4);                               \
    } else if (call_dtype.code() == kDLFloat) {                                                \
      float f_val = Value;                                                                     \
      value_tensor.CopyFromBytes(static_cast<void*>(&f_val), 4);                               \
    } else {                                                                                   \
      LOG(FATAL) << "Dtype " << call_dtype << " not handled for op " << #OpName;               \
    }                                                                                          \
    auto V = relax::Constant(                                                                  \
        value_tensor,                                                                          \
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({})), call_dtype)); \
    args.push_back(V);                                                                         \
    auto call_ret = m_te.Make(args, topi_op_name, std::string("tir_") + std::string(#OpName)); \
    return call_ret;                                                                           \
  }                                                                                            \
  TVM_REGISTER_OP("relax." #OpName)                                                            \
      .set_attr<FLegalize>("FLegalize", MAKE_NAME(CreateLegalize, OpName), 9);

TVM_LEGALIZE_CREATE_OP_BY_VALUE(ones, false, 1.0);
TVM_LEGALIZE_CREATE_OP_BY_VALUE(ones_like, true, 1.0);
TVM_LEGALIZE_CREATE_OP_BY_VALUE(zeros, false, 0.0);
TVM_LEGALIZE_CREATE_OP_BY_VALUE(zeros_like, true, 0.0);

}  // namespace relax
}  // namespace tvm
