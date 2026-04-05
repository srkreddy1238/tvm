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
 * \file tvm/relax/transform/legalize_ops/index.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */
#include <tvm/relax/attrs/index.h>
#include <tvm/topi/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

// Take
Expr LegalizeTake(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<TakeAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(0);  // batch_dim
  if (attrs->axis.has_value()) {
    args.push_back(attrs->axis);
  }
  args.push_back(attrs->mode);
  auto call_ret = m_te.Make(args, ffi::String("topi.take"), std::string("take"));
  return call_ret;
}
TVM_REGISTER_OP("relax.take")
    .set_attr<FLegalize>("FLegalize", LegalizeTake, TVM_LEGALIZE_CPP_LEVEL);

// Strided slice
Expr LegalizeStridedSlice(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<StridedSliceAttrs>();
  auto m_te = MakeCallTE(bb, call);

  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);  // input
  args.push_back(call->args[2]);  // begin
  args.push_back(call->args[3]);  // end
  if (call->args.size() == 5) {
    args.push_back(call->args[4]);  // strides
  } else {
    ffi::Array<tvm::relax::Expr> strides;
    for (size_t i = 0; i < call->args[1].as<TupleNode>()->fields.size(); ++i) {
      strides.push_back(relax::PrimValue(tvm::PrimExpr(1)));
    }
    args.push_back(relax::Tuple(strides));
  }
  args.push_back(call->args[1]);  // axes
  args.push_back(ffi::String("end"));
  args.push_back(attrs->assume_inbound);

  auto call_ret = m_te.Make(args, ffi::String("topi.strided_slice"), std::string("strided_slice"));
  return call_ret;
}
TVM_REGISTER_OP("relax.strided_slice")
    .set_attr<FLegalize>("FLegalize", LegalizeStridedSlice, TVM_LEGALIZE_CPP_LEVEL);

// Dynamic strided slice
inline PrimExpr CanonicalizeIndex(PrimExpr index, PrimExpr extent, PrimExpr stride,
                                  DataType dtype) {
  PrimExpr begin_range =
      tvm::tir::Select(stride < 0, tvm::tir::make_const(dtype, -1), tvm::tir::make_const(dtype, 0));
  PrimExpr end_range = tvm::tir::Select(stride < tvm::tir::make_const(dtype, 0),
                                        extent - tvm::tir::make_const(dtype, 1), extent);

  index = tvm::tir::Select(index < tvm::tir::make_const(dtype, 0), index + extent, index);
  return tvm::tir::Min(tvm::tir::Max(index, begin_range), end_range);
}

inline PrimExpr GetLength(PrimExpr begin, PrimExpr end, PrimExpr stride, PrimExpr extent,
                          DataType dtype) {
  begin = CanonicalizeIndex(begin, extent, stride, dtype);
  end = CanonicalizeIndex(end, extent, stride, dtype);
  return tvm::tir::Select(stride < tvm::tir::make_const(dtype, 0),
                          ceildiv(begin - end, stride * tvm::tir::make_const(dtype, -1)),
                          ceildiv(end - begin, stride));
}

inline te::Tensor dynamic_strided_slice_shape(const te::Tensor& data, const te::Tensor& begin,
                                              const te::Tensor& end, const te::Tensor& strides) {
  DataType index_dtype = begin->shape[0]->dtype;
  const int64_t num_dynamic_axes = begin->shape[0].as<IntImmNode>()->value;
  TVM_FFI_ICHECK_EQ(end->shape[0].as<IntImmNode>()->value, num_dynamic_axes);
  TVM_FFI_ICHECK_EQ(strides->shape[0].as<IntImmNode>()->value, num_dynamic_axes);

  ffi::Array<PrimExpr> out_shape = {begin->shape[0]};

  return te::compute(
      out_shape,
      [&](const tvm::tir::Var& indice) {
        auto length = tvm::tir::make_const(index_dtype, -1);
        for (size_t idx = 0; idx < data->shape.size(); ++idx) {
          length = tvm::tir::Select(indice == tvm::tir::make_const(index_dtype, idx),
                                    data->shape[idx], length);
        }
        return GetLength(begin(indice), end(indice), strides(indice), length, index_dtype);
      },
      "T_dynamic_strided_slice_shape_func");
}

ffi::Array<te::Tensor> dynamic_strided_slice_shape_func(const ffi::Array<ffi::Any> args) {
  te::Tensor x = args[0].cast<te::Tensor>();
  te::Tensor begin = args[1].cast<te::Tensor>();
  te::Tensor end = args[2].cast<te::Tensor>();
  te::Tensor strides = args[3].cast<te::Tensor>();

  return {dynamic_strided_slice_shape(x, begin, end, strides)};
}

Expr LegalizeDynamicStridedSlice(const BlockBuilder& bb, const Call& call) {
  TVM_FFI_ICHECK(call->args.size() == 4) << "Expected 4 args to dynamic strided slice";
  auto m_te = MakeCallTE(bb, call);

  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);  // data
  args.push_back(call->args[1]);  // begin
  args.push_back(call->args[2]);  // end
  args.push_back(call->args[3]);  // strides

  Expr shape_out =
      m_te.Make(args, FTOPIHandler(dynamic_strided_slice_shape_func), std::string("shape_func"));

  shape_out = bb->Normalize(shape_out);
  int ndim = GetStructInfo(shape_out)
                 .as<TensorStructInfoNode>()
                 ->shape.as<ShapeExprNode>()
                 ->values[0]
                 .as<IntImmNode>()
                 ->value;
  Expr shape_expr_out =
      bb->Emit(Call(Op::Get("relax.tensor_to_shape"), {shape_out}, tvm::Attrs(), {}));

  ffi::Array<tvm::PrimExpr> output_shape_vars;
  ffi::Array<tvm::relax::Expr> output_shape_vars_expr;
  for (int ii = 0; ii < ndim; ++ii) {
    auto var_val = tvm::tir::Var("s", tvm::runtime::DataType::Int(64));
    output_shape_vars.push_back(var_val.as<tvm::PrimExpr>().value());
    output_shape_vars_expr.push_back(relax::PrimValue(output_shape_vars[ii]));
  }

  bb->EmitMatchCast(shape_expr_out, tvm::relax::ShapeStructInfo(output_shape_vars), "");

  m_te = MakeCallTE(bb, call);
  args.push_back(relax::Tuple(output_shape_vars_expr));  // Output shape
  auto call_ret = m_te.Make(args, ffi::String("topi.relax_dynamic_strided_slice"),
                            std::string("dynamic_strided_slice"));
  return call_ret;
}
TVM_REGISTER_OP("relax.dynamic_strided_slice")
    .set_attr<FLegalize>("FLegalize", LegalizeDynamicStridedSlice, TVM_LEGALIZE_CPP_LEVEL);

}  // namespace relax
}  // namespace tvm
