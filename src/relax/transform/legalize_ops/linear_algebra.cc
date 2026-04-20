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
 * \file tvm/relax/transform/legalize_ops/linear_algebra.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */
#include <tvm/relax/attrs/linear_algebra.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/te/operation.h>
#include <tvm/topi/einsum.h>
#include <tvm/topi/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

/*!
 * \brief Legalize relax.matmul to call_tir.
 *
 * Handles broadcasting batch dimensions and the 1-D vector cases by
 * temporarily prepending or appending a size-1 dimension, then removing
 * it from the output shape.
 *
 * \param bb The block builder.
 * \param call The relax.matmul call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeMatmul(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<MatmulAttrs>();
  auto m_te = MakeCallTE(bb, call);

  tvm::ffi::Array<tvm::ffi::Any> args;
  TVM_FFI_ICHECK(!GetStructInfo(call->args[0]).as<TensorStructInfoNode>()->IsUnknownDtype() &&
                 !GetStructInfo(call->args[1]).as<TensorStructInfoNode>()->IsUnknownDtype())
      << "To legalize R.matmul into R.call_tir, the dtype of both operands must be known";
  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(attrs->out_dtype);

  std::function<ffi::Array<te::Tensor>(const ffi::Array<ffi::Any>)> _te_matmul =
      [&](const ffi::Array<ffi::Any> te_args) -> ffi::Array<te::Tensor> {
    te::Tensor a = te_args[0].cast<te::Tensor>();
    te::Tensor b = te_args[1].cast<te::Tensor>();
    DataType out_dtype = te_args[2].cast<DataType>();

    std::vector<PrimExpr> a_shape(a->shape.begin(), a->shape.end());
    std::vector<PrimExpr> b_shape(b->shape.begin(), b->shape.end());

    bool a_prepended = false;
    bool b_appended = false;
    if (a_shape.size() == 1) {
      a_prepended = true;
      a_shape.insert(a_shape.begin(), tvm::IntImm(a_shape[0]->dtype, 1));
    }
    if (b_shape.size() == 1) {
      b_appended = true;
      b_shape.push_back(tvm::IntImm(b_shape[0]->dtype, 1));
    }

    bool is_a_larger = a_shape.size() > b_shape.size();
    size_t offset = is_a_larger ? a_shape.size() - b_shape.size() : b_shape.size() - a_shape.size();

    tvm::OpAttrMap<FInferStructInfo> op_map_infer_struct_info_ =
        Op::GetAttrMap<FInferStructInfo>("FInferStructInfo");
    auto* op_ptr = call->op.as<OpNode>();
    Op op = ffi::GetRef<Op>(op_ptr);
    TVM_FFI_ICHECK(op_map_infer_struct_info_.count(op))
        << "Cannot find the FInferStructInfo attribute registered to op: " << op->name;

    auto a_relax = relax::Var("x1", relax::TensorStructInfo(relax::ShapeExpr(a->shape), a->dtype));
    auto b_relax = relax::Var("x2", relax::TensorStructInfo(relax::ShapeExpr(b->shape), b->dtype));

    ffi::ObjectPtr<MatmulAttrs> t_attrs = ffi::make_object<MatmulAttrs>();
    t_attrs->out_dtype = out_dtype;

    auto t_call = relax::Call(op, {a_relax, b_relax}, tvm::Attrs(t_attrs), {});
    auto out_sinfo = op_map_infer_struct_info_[op](t_call, bb);
    auto out_shape = out_sinfo.as<TensorStructInfoNode>()->shape.as<ShapeExprNode>()->values;
    size_t out_ndim_final = out_shape.size();

    auto k = tvm::te::reduce_axis(tvm::Range(0, a_shape[a_shape.size() - 1]), "k");

    return {tvm::te::compute(
        out_shape,
        [&](const ffi::Array<tvm::tir::Var>& idx) {
          ffi::Array<PrimExpr> a_indices;
          ffi::Array<PrimExpr> b_indices;

          // Offset batch dims
          for (size_t i = 0; i < offset; ++i) {
            if (is_a_larger) {
              a_indices.push_back(idx[i]);
            } else {
              b_indices.push_back(idx[i]);
            }
          }
          // Shared batch dims (with broadcast)
          size_t spatial_end = out_ndim_final - (2 - (a_prepended ? 1 : 0) - (b_appended ? 1 : 0));
          for (size_t i = offset; i < spatial_end; ++i) {
            PrimExpr a_dim = a_shape[is_a_larger ? i : i - offset];
            PrimExpr b_dim = b_shape[is_a_larger ? i - offset : i];
            auto a_is_one = a_dim.as<IntImmNode>() && a_dim.as<IntImmNode>()->value == 1;
            auto b_is_one = b_dim.as<IntImmNode>() && b_dim.as<IntImmNode>()->value == 1;
            a_indices.push_back(a_is_one ? PrimExpr(tvm::IntImm(idx[i]->dtype, 0))
                                         : PrimExpr(idx[i]));
            b_indices.push_back(b_is_one ? PrimExpr(tvm::IntImm(idx[i]->dtype, 0))
                                         : PrimExpr(idx[i]));
          }
          // M index (row of a)
          if (!a_prepended) {
            a_indices.push_back(idx[out_ndim_final - 2 + (b_appended ? 1 : 0)]);
          }
          // Reduction index
          a_indices.push_back(k);
          b_indices.push_back(k);
          // N index (col of b)
          if (!b_appended) {
            b_indices.push_back(idx[out_ndim_final - 1]);
          }

          PrimExpr a_val = a(a_indices);
          PrimExpr b_val = b(b_indices);
          if (out_dtype != DataType::Void() && out_dtype != a->dtype) {
            a_val = tvm::cast(out_dtype, a_val);
            b_val = tvm::cast(out_dtype, b_val);
          }
          return tvm::sum(a_val * b_val, {k});
        },
        "matmul")};
  };

  return m_te.Make(args, FTOPIHandler(_te_matmul), std::string("matmul"));
}
TVM_REGISTER_OP("relax.matmul")
    .set_attr<FLegalize>("FLegalize", LegalizeMatmul, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief Legalize relax.einsum to call_tir via topi.einsum.
 *
 * Unpacks the tuple argument into individual tensors before forwarding
 * to the TOPI handler.
 *
 * \param bb The block builder.
 * \param call The relax.einsum call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeEinsum(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<EinsumAttrs>();
  auto m_te = MakeCallTE(bb, call);

  // Unpack the tuple argument into individual tensors
  relax::Expr t = call->args[0];
  size_t n_field = GetStructInfo(t).as<TupleStructInfoNode>()->fields.size();
  while (t.as<VarNode>()) {
    auto binding = bb->LookupBinding(Downcast<Var>(t));
    if (!(binding.as<TupleNode>() || binding.as<VarNode>())) break;
    t = binding.value();
  }

  ffi::Array<Expr> input_tensors;
  if (auto tpl = t.as<TupleNode>()) {
    for (auto field : tpl->fields) input_tensors.push_back(field);
  } else {
    for (size_t i = 0; i < n_field; ++i) {
      input_tensors.push_back(bb->Emit(TupleGetItem(t, i)));
    }
  }

  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(attrs->subscripts);  // args[0]: subscripts string
  args.push_back(input_tensors);      // args[1]: Array<Tensor>
  return m_te.Make(args, ffi::String("topi.einsum"), std::string("einsum"));
}
TVM_REGISTER_OP("relax.einsum")
    .set_attr<FLegalize>("FLegalize", LegalizeEinsum, TVM_LEGALIZE_CPP_LEVEL);

/*!
 * \brief TE handler for the outer product of two 1-D tensors.
 *
 * \param args Packed argument list: [a, b] where both are 1-D tensors.
 * \return A single-element array containing the 2-D outer-product tensor.
 */
ffi::Array<te::Tensor> OuterTE(const ffi::Array<ffi::Any> args) {
  te::Tensor a = args[0].cast<te::Tensor>();
  te::Tensor b = args[1].cast<te::Tensor>();
  TVM_FFI_ICHECK_EQ(a->shape.size(), 1) << "outer: a must be 1-D";
  TVM_FFI_ICHECK_EQ(b->shape.size(), 1) << "outer: b must be 1-D";
  return {tvm::te::compute(
      ffi::Array<PrimExpr>({a->shape[0], b->shape[0]}),
      [&](const ffi::Array<tvm::tir::Var>& idx) {
        return a(ffi::Array<PrimExpr>({idx[0]})) * b(ffi::Array<PrimExpr>({idx[1]}));
      },
      "outer")};
}

/*!
 * \brief Legalize relax.outer to call_tir via OuterTE.
 *
 * \param bb The block builder.
 * \param call The relax.outer call to legalize.
 * \return The legalized call_tir expression.
 */
Expr LegalizeOuter(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  return m_te.Make(args, FTOPIHandler(OuterTE), std::string("outer"));
}
TVM_REGISTER_OP("relax.outer")
    .set_attr<FLegalize>("FLegalize", LegalizeOuter, TVM_LEGALIZE_CPP_LEVEL);

}  // namespace relax
}  // namespace tvm
