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

#include <string>
#include <typeinfo>

#include "../../../te/operation/create_primfunc.h"
#include "../../ir/emit_te.h"

/*
 * Default level (python) is 10. Keep this lower to give priority to python legalizations.
 * One can override this setting with build options to enfore cpp legalization if needed.
 */
#ifndef TVM_LEGALIZE_CPP_LEVEL
#define TVM_LEGALIZE_CPP_LEVEL 20
#endif

namespace tvm {
namespace relax {

using FTOPIHandler = ffi::TypedFunction<ffi::Array<te::Tensor>(const ffi::Array<ffi::Any>)>;

class MakeCallTE {
 public:
  /* Constructor */
  MakeCallTE(const BlockBuilder& bb, const Call& call) : bb_(bb), call_(call) {}

  // Convert given args to TE
  ffi::Array<tvm::ffi::Any> CallArgsToTE(const tvm::ffi::Array<tvm::ffi::Any>& args);
  tvm::ffi::Any ConvertToTE(const tvm::ffi::Any& any);

  /* Construct a TIR call with gien TOPI handler.
   * Legalize handler can inject additional args over relax call args.
   */

  tvm::relax::Call Make(const tvm::ffi::Array<tvm::ffi::Any>& topi_args,
                        ffi::Variant<FTOPIHandler, ffi::String> topi_handler, std::string fname);

  std::tuple<tvm::tir::PrimFunc, tvm::ffi::Array<Expr>, ffi::Array<TensorStructInfo>,
             ffi::Array<PrimExpr>>
  GenCallTirInputs(const ffi::Array<ffi::Any>& args,
                   ffi::Variant<FTOPIHandler, ffi::String> topi_handler, std::string fname);

 private:
  BlockBuilder bb_;
  Call call_;
  ffi::Map<tvm::tir::Var, tvm::PrimExpr> tir_var_map_;
  ffi::Array<tvm::ffi::Any> create_primfunc_args_;
  tvm::ffi::Array<Expr> call_tir_args_;
  ffi::Array<tvm::ffi::Any> extra_tir_args_list_;

  VDevice GetVDevice(void);
  ffi::Array<tvm::te::Tensor> CallTEHandler(
      const ffi::Array<tvm::ffi::Any>& te_args,
      const ffi::Variant<FTOPIHandler, ffi::String>& topi_handler

  );
};

#define JOIN(x, y) x##y
#define MAKE_NAME(fn, id) JOIN(fn, id)

ffi::Any TryConvertToScalarConst(const relax::Expr& val);

ffi::Array<ffi::Any> GetConstTuple(const ffi::Array<PrimExpr>& tuple);

std::pair<ffi::Array<tvm::PrimExpr>, ffi::Array<tvm::PrimExpr>> GetPadTupleGeneric(
    const ffi::Array<tvm::PrimExpr>& padding, const ffi::Array<tvm::PrimExpr>& kernel_dims);

bool EqualsConstInt(const tvm::PrimExpr& expr, int64_t val);

te::Tensor Dilate(const te::Tensor& data, const ffi::Array<tvm::PrimExpr>& strides,
                  const PrimExpr& dilation_value, const ffi::String& name = "DilatedInput");

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_LEGALIZE_OPS_UTILS_H_
