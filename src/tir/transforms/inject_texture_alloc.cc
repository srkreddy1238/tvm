/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file inject_texture_alloc.cc
 */

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../runtime/texture.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {
using runtime::ApplyTexture2DFlattening;
using runtime::DefaultTextureLayoutSeparator;
using runtime::IsTextureStorage;

/*!
 * \brief Inject Texture Alloc Intrensic right after AllocateNode are realized.
 */
class TextureAllocInjector : public arith::IRMutatorWithAnalyzer {
 public:
  static PrimFunc Inject(PrimFunc func) {
    arith::Analyzer ana;
    auto pass = TextureAllocInjector(&ana);
    auto writer = func.CopyOnWrite();
    pass.MarkBufferMapShapes(func);
    writer->body = pass.VisitStmt(func->body);
    return func;
  }

 private:
  using IRMutatorWithAnalyzer::VisitExpr;
  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt;
  using IRMutatorWithAnalyzer::VisitStmt_;

  explicit TextureAllocInjector(arith::Analyzer* ana) : IRMutatorWithAnalyzer(ana) {}

  Stmt VisitStmt_(const AllocateNode* op) final {
   // LOG(WARNING) << "**** SRK Inject InjectTextureAlloc ***";
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    std::string storage_scope = GetStorageScope(op->buffer_var);
    if (IsTextureStorage(storage_scope)) {
      op = stmt.as<AllocateNode>();
      ICHECK(op->extents.size() >= 3) << "Only 2D Array RGBA texture is currently supported";
      int vec_length = static_cast<int>(op->extents.back().as<IntImmNode>()->value);
      ICHECK(vec_length == 4 || vec_length == 1)
          << "Inner dimension of texture must be vector of length 1 or 4 (RGBA), was: "
          << vec_length;

      size_t axis = DefaultTextureLayoutSeparator(op->extents.size(), storage_scope);
      auto texture = ApplyTexture2DFlattening<PrimExpr>(op->extents, op->extents.size(), axis);
      Array<PrimExpr> args;
      args.push_back(StringImm(storage_scope));
      args.push_back(IntImm(DataType::Int(64), 2));  // 2d Array
      args.push_back(Call(DataType::Handle(), builtin::tvm_stack_make_shape(),
                          {texture.width, texture.height}));
      LOG(WARNING) << "**** SRK Inject InjectTextureAlloc - call builtin::nd_mem_alloc_with_scope";
      stmt =
          LetStmt(op->buffer_var,
                  Call(op->buffer_var.dtype(), builtin::nd_mem_alloc_with_scope(), args), op->body);
    }
    return stmt;
  }

 protected:
  std::string GetStorageScope(const Var& buffer_var) {
    auto* ptr = buffer_var->type_annotation.as<PointerTypeNode>();
    ICHECK(ptr) << "Buffer Var's type annotation must be of PointerType";
    return ptr->storage_scope;
  }
};

namespace transform {

Pass InjectTextureAlloc() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto ret = TextureAllocInjector::Inject(std::move(f));
    // LOG(WARNING) <<"Ret:" << ret;
    return ret;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectTextureAlloc", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectTextureAlloc").set_body_typed(InjectTextureAlloc);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
