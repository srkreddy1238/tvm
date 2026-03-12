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
 * \file tvm/relax/transform/fold_batchnorm_conv2d.cc
 * \brief Fold batchnorm weights into conv2d weights.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

namespace {
std::tuple<DFPattern, ffi::TypedFunction<Expr(Expr, ffi::Map<DFPattern, Expr>)>> CreatePatterns(
    const Function& func) {
  DFPattern input = Wildcard();
  DFPattern weight = IsConst();
  DFPattern bn_weight = IsConst();
  DFPattern bn_bias = IsConst();
  DFPattern bn_mean = IsConst();
  DFPattern bn_variance = IsConst();

  auto pat_conv2d = IsOp("relax.nn.conv2d")(input, weight);
  auto pat_bn = IsOp("relax.nn.batch_norm")(pat_conv2d, bn_weight, bn_bias, bn_mean, bn_variance);

  auto pat = TupleGetItemPattern(pat_bn, 0);

  auto rewriter = [=](Expr expr, ffi::Map<DFPattern, Expr> matches) -> Expr {
    auto expr_input = matches[input];
    auto expr_weight = matches[weight];
    auto expr_bn_weight = matches[bn_weight];
    auto expr_bn_bias = matches[bn_bias];
    auto expr_bn_mean = matches[bn_mean];
    auto expr_bn_variance = matches[bn_variance];

    auto conv_op = Downcast<Call>(matches[pat_conv2d]);
    auto bn_op = Downcast<Call>(matches[pat_bn]);

    const auto* conv_attrs = conv_op->attrs.as<Conv2DAttrs>();
    const auto* bn_attrs = bn_op->attrs.as<BatchNormAttrs>();

    Expr expr_bn_var =
        Call(Op::Get("relax.add"),
             {expr_bn_variance, relax::PrimValue(FloatImm(DataType::Float(32), bn_attrs->epsilon))},
             Attrs(), {});
    Expr dino = Call(Op::Get("relax.sqrt"), {expr_bn_var}, Attrs(), {});
    Expr new_weight = Call(Op::Get("relax.divide"), {expr_bn_weight, dino}, Attrs(), {});
    Expr bs_part = Call(Op::Get("relax.multiply"), {expr_bn_mean, new_weight}, Attrs(), {});
    Expr bs = Call(Op::Get("relax.subtract"), {expr_bn_bias, bs_part}, Attrs(), {});

    auto wt_shape = Downcast<TensorStructInfo>(GetStructInfo(expr_bn_weight))->GetShape().value();
    ffi::Array<PrimExpr> new_shape;
    if (conv_attrs->kernel_layout == "OIHW") {
      new_shape.push_back(wt_shape[0]);
      new_shape.push_back(IntImm(wt_shape[0].dtype(), 1));
    } else if (conv_attrs->kernel_layout == "IOHW") {
      new_shape.push_back(IntImm(wt_shape[0].dtype(), 1));
      new_shape.push_back(wt_shape[0]);
    } else {
      return expr;
    }

    new_shape.push_back(IntImm(wt_shape[0].dtype(), 1));
    new_shape.push_back(IntImm(wt_shape[0].dtype(), 1));
    new_weight = Call(Op::Get("relax.reshape"),
                      ffi::Array<Expr>({new_weight, ShapeExpr(new_shape)}), Attrs(), {});

    Expr wt_conv = Call(Op::Get("relax.multiply"), {expr_weight, new_weight}, Attrs(), {});
    auto bn_bias_shape =
        Downcast<TensorStructInfo>(GetStructInfo(expr_bn_bias))->GetShape().value();
    Expr bs_args =
        Call(Op::Get("relax.reshape"),
             ffi::Array<Expr>({bs, ShapeExpr({IntImm(bn_bias_shape[0].dtype(), 1), bn_bias_shape[0],
                                              IntImm(bn_bias_shape[0].dtype(), 1),
                                              IntImm(bn_bias_shape[0].dtype(), 1)})}),
             Attrs(), {});

    auto conv_out = Call(Op::Get("relax.nn.conv2d"), {expr_input, wt_conv}, conv_op->attrs, {});
    auto out = Call(Op::Get("relax.add"), {conv_out, bs_args}, Attrs(), {});

    return out;
  };

  return {pat, rewriter};
}

}  // namespace

namespace transform {
Pass FoldBatchnormToConv2D() {
  auto pass_func = [=](Function func, IRModule mod, PassContext pc) {
    auto [pattern, rewriter] = CreatePatterns(func);
    return RewriteCall(pattern, rewriter, func);
  };
  return CreateFunctionPass(pass_func, 1, "FoldBatchnormToConv2D", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.FoldBatchnormToConv2D", FoldBatchnormToConv2D);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
