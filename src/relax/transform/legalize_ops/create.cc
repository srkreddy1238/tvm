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
#include <tvm/relax/attrs/create.h>
#include <tvm/runtime/tensor.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/topi/trilu.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "utils.h"

namespace tvm {
namespace relax {

#define TVM_LEGALIZE_CREATE_OP(OpName, IsLike, FName)                                         \
  Expr MAKE_NAME(CreateLegalize, OpName)(const BlockBuilder& bb, const Call& call) {          \
    auto m_te = MakeCallTE(bb, call);                                                         \
    tvm::ffi::Array<tvm::ffi::Any> args;                                                      \
    if (IsLike) {                                                                             \
      args.push_back(GetStructInfo(call->args[0]).as<TensorStructInfoNode>()->shape.value()); \
    } else {                                                                                  \
      args.push_back(Downcast<ShapeExpr>(call->args[0]));                                     \
    }                                                                                         \
    args.push_back(Downcast<TensorStructInfo>(GetStructInfo(call))->dtype);                   \
    args.push_back(TryConvertToScalarConst(call->args[1]));                                   \
    auto call_ret = m_te.Make(args, ffi::String("topi.full"), std::string(#FName));           \
    return call_ret;                                                                          \
  }                                                                                           \
  TVM_REGISTER_OP("relax." #OpName)                                                           \
      .set_attr<FLegalize>("FLegalize", MAKE_NAME(CreateLegalize, OpName),                    \
                           TVM_LEGALIZE_CPP_LEVEL);

TVM_LEGALIZE_CREATE_OP(full, false, full);
TVM_LEGALIZE_CREATE_OP(full_like, true, full);

#define TVM_LEGALIZE_CREATE_OP_BY_VALUE(OpName, IsLike, Value, FName)                         \
  Expr MAKE_NAME(CreateLegalize, OpName)(const BlockBuilder& bb, const Call& call) {          \
    auto m_te = MakeCallTE(bb, call);                                                         \
    tvm::ffi::Array<tvm::ffi::Any> args;                                                      \
    auto call_dtype = Downcast<TensorStructInfo>(GetStructInfo(call))->dtype;                 \
    std::string topi_op_name = "topi.full";                                                   \
    if (IsLike) {                                                                             \
      args.push_back(GetStructInfo(call->args[0]).as<TensorStructInfoNode>()->shape.value()); \
    } else {                                                                                  \
      args.push_back(Downcast<ShapeExpr>(call->args[0]));                                     \
    }                                                                                         \
    args.push_back(call_dtype);                                                               \
    auto value_tensor = runtime::Tensor::Empty(ffi::Shape({}), call_dtype,                    \
                                               tvm::Device({kDLCPU, 0}), std::nullopt);       \
    if (call_dtype.code() == kDLInt) {                                                        \
      args.push_back(tvm::IntImm(call_dtype, Value));                                         \
    } else if (call_dtype.code() == kDLFloat) {                                               \
      args.push_back(tvm::FloatImm(call_dtype, Value));                                       \
    } else {                                                                                  \
      LOG(FATAL) << "Dtype " << call_dtype << " not handled for op " << #OpName;              \
    }                                                                                         \
    auto call_ret = m_te.Make(args, ffi::String(topi_op_name), std::string(#FName));          \
    return call_ret;                                                                          \
  }                                                                                           \
  TVM_REGISTER_OP("relax." #OpName)                                                           \
      .set_attr<FLegalize>("FLegalize", MAKE_NAME(CreateLegalize, OpName),                    \
                           TVM_LEGALIZE_CPP_LEVEL);

TVM_LEGALIZE_CREATE_OP_BY_VALUE(ones, false, 1.0, ones);
TVM_LEGALIZE_CREATE_OP_BY_VALUE(ones_like, true, 1.0, ones);
TVM_LEGALIZE_CREATE_OP_BY_VALUE(zeros, false, 0.0, zeros);
TVM_LEGALIZE_CREATE_OP_BY_VALUE(zeros_like, true, 0.0, zeros);

// tril / triu
#define TVM_LEGALIZE_TRILU_OP(OpName, IsUpper)                                       \
  Expr MAKE_NAME(CreateLegalize, OpName)(const BlockBuilder& bb, const Call& call) { \
    auto m_te = MakeCallTE(bb, call);                                                \
    tvm::ffi::Array<tvm::ffi::Any> args;                                             \
    args.push_back(call->args[0]);                                                   \
    args.push_back(call->args[1]);                                                   \
    std::function<ffi::Array<te::Tensor>(const ffi::Array<ffi::Any>)> _te_trilu =    \
        [&](const ffi::Array<ffi::Any> te_args) -> ffi::Array<te::Tensor> {          \
      te::Tensor data = te_args[0].cast<te::Tensor>();                               \
      tvm::PrimExpr k_val = te_args[1].cast<tvm::PrimExpr>();                        \
      return {tvm::topi::trilu(data, k_val, IsUpper)};                               \
    };                                                                               \
    auto ret = m_te.Make(args, FTOPIHandler(_te_trilu), std::string(#OpName));       \
    return ret;                                                                      \
  }                                                                                  \
  TVM_REGISTER_OP("relax." #OpName)                                                  \
      .set_attr<FLegalize>("FLegalize", MAKE_NAME(CreateLegalize, OpName),           \
                           TVM_LEGALIZE_CPP_LEVEL);

TVM_LEGALIZE_TRILU_OP(tril, false);
TVM_LEGALIZE_TRILU_OP(triu, true);

/*!
 * \brief TE handler for the eye / eye_like operator.
 *
 * Produces an n×m matrix with ones on the k-th diagonal and zeros elsewhere.
 *
 * \param args Packed argument list: [n, m, k, dtype].
 * \return A single-element array containing the eye output tensor.
 */
static ffi::Array<te::Tensor> EyeTEHandler(const ffi::Array<ffi::Any>& args) {
  PrimExpr n = args[0].cast<PrimExpr>();
  PrimExpr m = args[1].cast<PrimExpr>();
  PrimExpr k = args[2].cast<PrimExpr>();
  DataType dtype = args[3].cast<DataType>();
  return {te::compute(
      {n, m},
      [k, dtype](const tir::Var& i, const tir::Var& j) -> PrimExpr {
        return tir::Select(tir::EQ(i, j - k), tvm::tir::make_const(dtype, 1),
                           tvm::tir::make_const(dtype, 0));
      },
      "eye")};
}

/*!
 * \brief Legalize relax.eye to call_tir via EyeTEHandler.
 *
 * \param bb The block builder.
 * \param call The relax.eye call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeEye(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<InitAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  // n
  args.push_back(TryConvertToScalarConst(call->args[0]));
  // m  (defaults to n if not provided)
  args.push_back(call->args.size() > 1 ? TryConvertToScalarConst(call->args[1])
                                       : TryConvertToScalarConst(call->args[0]));
  // k  (defaults to 0)
  args.push_back(call->args.size() > 2 ? TryConvertToScalarConst(call->args[2])
                                       : tvm::IntImm(DataType::Int(64), 0));
  args.push_back(attrs->dtype);
  return m_te.Make(args, FTOPIHandler(EyeTEHandler), std::string("eye"));
}
TVM_REGISTER_OP("relax.eye").set_attr<FLegalize>("FLegalize", LegalizeEye, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Legalize relax.eye_like to call_tir via EyeTEHandler.
 *
 * Derives n, m from the shape of the input tensor.
 *
 * \param bb The block builder.
 * \param call The relax.eye_like call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeEyeLike(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  auto sinfo = GetStructInfo(call->args[0]).as<TensorStructInfoNode>();
  TVM_FFI_ICHECK_NOTNULL(sinfo);
  auto shape = sinfo->shape.as<ShapeExprNode>()->values;
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(shape[0]);  // n
  args.push_back(shape[1]);  // m
  // k  (defaults to 0)
  args.push_back(call->args.size() > 1 ? TryConvertToScalarConst(call->args[1])
                                       : tvm::IntImm(DataType::Int(64), 0));
  args.push_back(sinfo->dtype);
  return m_te.Make(args, FTOPIHandler(EyeTEHandler), std::string("eye_like"));
}
TVM_REGISTER_OP("relax.eye_like")
    .set_attr<FLegalize>("FLegalize", LegalizeEyeLike, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Check whether a PrimExpr is a compile-time integer or float immediate.
 *
 * \param e The expression to test.
 * \return True if `e` is an IntImm or FloatImm.
 */
static bool IsConstScalar(const PrimExpr& e) {
  return e->IsInstance<IntImmNode>() || e->IsInstance<FloatImmNode>();
}

/*!
 * \brief Extract the numeric value of an IntImm or FloatImm as a double.
 *
 * \param e The scalar immediate to convert.
 * \return The value as a double.
 */
static double ScalarToDouble(const PrimExpr& e) {
  if (const auto* i = e.as<IntImmNode>()) return static_cast<double>(i->value);
  if (const auto* f = e.as<FloatImmNode>()) return f->value;
  LOG(FATAL) << "ScalarToDouble: not a scalar immediate";
}

/*!
 * \brief Portable float32 to float16 bit-pattern conversion.
 *
 * Implements IEEE 754 round-to-nearest half-precision conversion using only
 * integer arithmetic. Denormals are flushed to zero.
 *
 * \param v The float32 value to convert.
 * \return The float16 bit-pattern as a uint16_t.
 */
static uint16_t FloatToFloat16Bits(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
  uint32_t mantissa = bits & 0x7FFFFFu;
  if (exponent <= 0) {
    // Underflow to zero (or subnormal -- treat as zero for simplicity).
    return sign;
  } else if (exponent >= 31) {
    // Overflow to infinity.
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) |
                               static_cast<uint16_t>(mantissa >> 13));
}

/*!
 * \brief Eagerly evaluate arange(start, end, step) and return a relax::Constant.
 *
 * Returns a null Expr if the dtype is not supported for eager evaluation,
 * signalling the caller to fall back to the dynamic path.
 *
 * \param start The start value of the range.
 * \param end The end value of the range (exclusive).
 * \param step The step size.
 * \param dtype The output element data type.
 * \return A relax::Constant holding the arange result, or a null Expr.
 */
static Expr MakeArangeConstant(double start, double end, double step, DataType dtype) {
  const int64_t n = static_cast<int64_t>(std::max(0.0, std::ceil((end - start) / step)));

  runtime::Tensor data = runtime::Tensor::Empty(ffi::Shape({n}), dtype, Device{kDLCPU, 0});

  const int bits = dtype.bits();
  if (dtype.is_int() || dtype.is_uint()) {
    if (bits == 8) {
      auto* p = static_cast<int8_t*>(data->data);
      for (int64_t i = 0; i < n; ++i) p[i] = static_cast<int8_t>(start + i * step);
    } else if (bits == 16) {
      auto* p = static_cast<int16_t*>(data->data);
      for (int64_t i = 0; i < n; ++i) p[i] = static_cast<int16_t>(start + i * step);
    } else if (bits == 32) {
      auto* p = static_cast<int32_t*>(data->data);
      for (int64_t i = 0; i < n; ++i) p[i] = static_cast<int32_t>(start + i * step);
    } else if (bits == 64) {
      auto* p = static_cast<int64_t*>(data->data);
      for (int64_t i = 0; i < n; ++i) p[i] = static_cast<int64_t>(start + i * step);
    } else {
      return Expr();  // unsupported width -- fall back to dynamic path
    }
  } else if (dtype.is_float()) {
    if (bits == 16) {
      // float16: convert each value via a portable float32->fp16 bit conversion.
      auto* p = static_cast<uint16_t*>(data->data);
      for (int64_t i = 0; i < n; ++i)
        p[i] = FloatToFloat16Bits(static_cast<float>(start + i * step));
    } else if (bits == 32) {
      auto* p = static_cast<float*>(data->data);
      for (int64_t i = 0; i < n; ++i) p[i] = static_cast<float>(start + i * step);
    } else if (bits == 64) {
      auto* p = static_cast<double*>(data->data);
      for (int64_t i = 0; i < n; ++i) p[i] = start + i * step;
    } else {
      return Expr();  // unsupported width -- fall back to dynamic path
    }
  } else {
    return Expr();  // unsupported type code -- fall back to dynamic path
  }

  return Constant(data);
}

/*!
 * \brief Legalize relax.arange to call_tir via topi.arange.
 *
 * When all three arguments are compile-time constants the result is computed
 * eagerly and returned as a relax::Constant. Otherwise the dynamic TOPI path
 * is used.
 *
 * \param bb The block builder.
 * \param call The relax.arange call to legalize.
 * \return A relax::Constant for static inputs, or a call_tir expression.
 */
Expr LegalizeArange(const BlockBuilder& bb, const Call& call) {
  TVM_FFI_ICHECK_EQ(call->args.size(), 3);
  const auto* attrs = call->attrs.as<InitAttrs>();

  const PrimExpr start_expr = call->args[0].as<PrimValueNode>()->value;
  const PrimExpr end_expr = call->args[1].as<PrimValueNode>()->value;
  const PrimExpr step_expr = call->args[2].as<PrimValueNode>()->value;
  const DataType dtype = attrs->dtype;

  // Fast path: all three arguments are compile-time constants.
  // Compute the arange result eagerly and return a relax::Constant,
  if (IsConstScalar(start_expr) && IsConstScalar(end_expr) && IsConstScalar(step_expr)) {
    Expr constant = MakeArangeConstant(ScalarToDouble(start_expr), ScalarToDouble(end_expr),
                                       ScalarToDouble(step_expr), dtype);
    if (constant.defined()) return constant;
  }

  // Dynamic path
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(start_expr);
  args.push_back(end_expr);
  args.push_back(step_expr);
  args.push_back(dtype);
  return m_te.Make(args, ffi::String("topi.arange"), std::string("arange"));
}
TVM_REGISTER_OP("relax.arange")
    .set_attr<FLegalize>("FLegalize", LegalizeArange, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Legalize relax.hamming_window to call_tir via topi.hamming_window.
 *
 * \param bb The block builder.
 * \param call The relax.hamming_window call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeHammingWindow(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<InitAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  // args: window_size, periodic, alpha, beta (all PrimValue), dtype from attrs
  args.push_back(call->args[0].as<PrimValueNode>()->value);
  args.push_back(call->args[1].as<PrimValueNode>()->value);
  args.push_back(call->args[2].as<PrimValueNode>()->value);
  args.push_back(call->args[3].as<PrimValueNode>()->value);
  args.push_back(attrs->dtype);
  return m_te.Make(args, ffi::String("topi.hamming_window"), std::string("hamming_window"));
}
TVM_REGISTER_OP("relax.hamming_window")
    .set_attr<FLegalize>("FLegalize", LegalizeHammingWindow, TVM_LEGALIZE_CPP_LEVEL);

}  // namespace relax
}  // namespace tvm
