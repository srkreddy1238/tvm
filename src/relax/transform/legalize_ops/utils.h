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

#ifndef TVM_RELAX_TRANSFORM_LEGALIZE_OPS_UTILS_H_
#define TVM_RELAX_TRANSFORM_LEGALIZE_OPS_UTILS_H_

#include <tvm/ir/attrs.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/distributed/struct_info.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/utils.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <typeinfo>

#include "../../../te/operation/create_primfunc.h"
#include "../../ir/emit_te.h"

namespace tvm {
namespace relax {

class MakeCallTE {
 public:
  /* Constructor */
  MakeCallTE(const BlockBuilder& bb, const Call& call) : bb_(bb), call_(call) {}

  // Convert given args to TE
  ffi::Array<tvm::ffi::Any> CallArgsToTE(const tvm::ffi::Array<Expr>& args);
  tvm::ffi::Any ConvertToTE(const tvm::ffi::Any& any);

  /* Construct a TIR call with gien TOPI handler.
   * Legalize handler can inject additional args over relax call args.
   */

  tvm::relax::Call Make(const tvm::ffi::Array<Expr>& topi_args, std::string topi_handler,
                        std::string fname);

 private:
  BlockBuilder bb_;
  Call call_;
  ffi::Map<tvm::tir::Var, tvm::PrimExpr> tir_var_map_;
  ffi::Array<tvm::ffi::Any> create_primfunc_args_;
  tvm::ffi::Array<Expr> call_tir_args_;
  ffi::Array<tvm::ffi::Any> extra_tir_args_list_;

  VDevice GetVDevice(void);
  ffi::Array<tvm::te::Tensor> CallTEHandler(ffi::Array<tvm::ffi::Any>& te_args,
                                            std::string& topi_handler);
};

#define JOIN(x, y) x##y
#define MAKE_NAME(fn, id) JOIN(fn, id)

#define TVM_LEGALIZE_BINARY_OP(OpName, TopiHandler)                                  \
  Expr MAKE_NAME(BinaryLegalize, OpName)(const BlockBuilder& bb, const Call& call) { \
    tvm::ffi::Array<Expr> topi_args;                                                 \
    for (auto arg : call->args) {                                                    \
      if (arg.as<ConstantNode>()) {                                                  \
        if (auto tinfo = GetStructInfo(arg).as<TensorStructInfoNode>()) {            \
          if (tinfo->ndim == 0) {                                                    \
            LOG(WARNING) << arg << " Is Const size 0";                               \
          }                                                                          \
        }                                                                            \
      } else {                                                                       \
        topi_args.push_back(arg);                                                    \
      }                                                                              \
    }                                                                                \
    auto m_te = MakeCallTE(bb, call);                                                \
    auto call_ret = m_te.Make(topi_args, std::string(#TopiHandler), #OpName);        \
    return call_ret;                                                                 \
  }                                                                                  \
  TVM_REGISTER_OP("relax." #OpName)                                                  \
      .set_attr<FLegalize>("FLegalize", MAKE_NAME(BinaryLegalize, OpName));

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_LEGALIZE_OPS_UTILS_H_
