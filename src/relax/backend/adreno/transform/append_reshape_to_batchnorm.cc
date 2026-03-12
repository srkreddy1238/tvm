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
 * \file src/relax/backend/adreno/transform/append_reshape_to_batchnorm.cc
 * \brief Pass for adding reshare at end of batchnorm follwoed by TupleGetItem
 *        to acommodate propoer graph partitions.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/backend/adreno/transform.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace relax {
namespace backend {
namespace adreno {

class AppendReshapeHandler : public ExprMutator {
 public:
  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) final {
    builder_->BeginDataflowBlock();
    for (Binding binding : block->bindings) {
      if (auto var_bind = binding.as<VarBindingNode>()) {
        if (auto call_node = var_bind->value.as<CallNode>()) {
          if (call_node->op == Op::Get("relax.nn.batch_norm")) {
            bn_vars.Set(var_bind->var, var_bind->value);
          }
        }
      }
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const TupleGetItemNode* op) override {
    auto tuple_value = op->tuple;
    if (tuple_value.as<VarNode>() && (bn_vars.find(Downcast<Var>(tuple_value)) != bn_vars.end())) {
      auto var_val = Downcast<Var>(tuple_value);
      if (op->index == 0) {
        auto bn_call = Downcast<Call>(bn_vars[var_val]);
        auto bn_out = TupleGetItem(bn_call, 0);
        auto in_shape =
            Downcast<TensorStructInfo>(GetStructInfo(bn_call->args[0]))->GetShape().value();

        return Call(Op::Get("relax.reshape"), ffi::Array<Expr>({bn_out, ShapeExpr(in_shape)}),
                    Attrs(), {});
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  ffi::Map<Var, Expr> bn_vars;
};

namespace transform {

Pass AppendReshapeToBatchnorm() {
  auto pass_func = [=](Function f, IRModule m, PassContext pc) {
    return Downcast<Function>(tvm::relax::RemoveAllUnused(AppendReshapeHandler()(f)));
  };
  return CreateFunctionPass(pass_func, 1, "AppendReshapeToBatchnorm", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.backend.adreno.transform.AppendReshapeToBatchnorm",
                        AppendReshapeToBatchnorm);
}

}  // namespace transform

}  // namespace adreno
}  // namespace backend
}  // namespace relax
}  // namespace tvm
