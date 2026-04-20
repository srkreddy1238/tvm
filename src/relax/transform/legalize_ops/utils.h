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
#define TVM_LEGALIZE_CPP_LEVEL 1
#endif

namespace tvm {
namespace relax {

using FTOPIHandler = ffi::TypedFunction<ffi::Array<te::Tensor>(const ffi::Array<ffi::Any>)>;

class MakeCallTE {
 public:
  /*!
   * \brief Construct a MakeCallTE helper bound to the given block builder and call.
   *
   * \param bb The block builder used to emit new TIR functions.
   * \param call The Relax call expression being legalized.
   */
  MakeCallTE(const BlockBuilder& bb, const Call& call) : bb_(bb), call_(call) {}

  /*!
   * \brief Convert a flat list of Relax call arguments to their TE equivalents.
   *
   * \param args The Relax call arguments to convert.
   * \return The converted TE argument list.
   */
  ffi::Array<tvm::ffi::Any> CallArgsToTE(const tvm::ffi::Array<tvm::ffi::Any>& args);

  /*!
   * \brief Convert a single Relax value to its TE equivalent.
   *
   * Handles tensors, shapes, scalars, and nested arrays recursively.
   *
   * \param any The value to convert.
   * \return The TE-compatible representation of the value.
   */
  tvm::ffi::Any ConvertToTE(const tvm::ffi::Any& any);

  /*!
   * \brief Build a call_tir expression from a TOPI handler and argument list.
   *
   * The legalization handler may inject additional arguments beyond those
   * present in the original Relax call.
   *
   * \param topi_args Arguments to forward to the TOPI handler.
   * \param topi_handler The TOPI TE function or its registered global name.
   * \param fname Name hint for the generated TIR PrimFunc.
   * \param out_sinfo_override When set, replaces the output struct info derived
   *                           from the TE compute with the given StructInfo.
   *                           Mirrors the sinfo_args parameter of bb.call_te.
   * \return The resulting call_tir Relax expression.
   */
  tvm::relax::Call Make(const tvm::ffi::Array<tvm::ffi::Any>& topi_args,
                        ffi::Variant<FTOPIHandler, ffi::String> topi_handler, std::string fname,
                        ffi::Optional<StructInfo> out_sinfo_override = std::nullopt);

  /*!
   * \brief Generate the inputs needed to construct a call_tir node.
   *
   * Runs the TOPI handler to obtain output TE tensors, then builds the
   * corresponding TIR PrimFunc, call_tir argument list, output struct-info
   * array, and unbound TIR variable list.
   *
   * \param args Arguments to forward to the TOPI handler.
   * \param topi_handler The TOPI TE function or its registered global name.
   * \param fname Name hint for the generated TIR PrimFunc.
   * \return A tuple of (PrimFunc, call_tir_args, output_sinfo, tir_vars).
   */
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

/*!
 * \brief Try to convert a scalar-valued Relax constant to a PrimExpr immediate.
 *
 * If `val` is a 0-D relax::Constant with a supported dtype (float32, float16,
 * int32, uint8, bool), the function returns the corresponding IntImm or
 * FloatImm. Otherwise `val` itself is returned unchanged.
 *
 * \param val The Relax expression to inspect.
 * \return A scalar PrimExpr immediate, or `val` if conversion is not possible.
 */
ffi::Any TryConvertToScalarConst(const relax::Expr& val);

/*!
 * \brief Convert an array of PrimExprs to a flat ffi::Array<ffi::Any>.
 *
 * Each element is simplified and, if it reduces to an IntImm, stored as a
 * plain int64_t; otherwise the simplified PrimExpr is stored as-is.
 *
 * \param tuple The array of PrimExprs to convert.
 * \return The converted ffi::Any array.
 */
ffi::Array<ffi::Any> GetConstTuple(const ffi::Array<PrimExpr>& tuple);

/*!
 * \brief Compute symmetric or asymmetric pad-begin / pad-end arrays.
 *
 * Handles three padding conventions:
 *   - `padding.size() == kernel_dims.size()`: symmetric padding per dimension.
 *   - `padding.size() == kernel_dims.size() * 2`: explicit begin/end per dimension.
 *   - `padding.size() == 1`: the same scalar is applied to every dimension.
 *
 * \param padding The raw padding specification.
 * \param kernel_dims The kernel spatial dimensions (used to determine ndim).
 * \return A pair (pad_begin, pad_end), each of length `kernel_dims.size()`.
 */
std::pair<ffi::Array<tvm::PrimExpr>, ffi::Array<tvm::PrimExpr>> GetPadTupleGeneric(
    const ffi::Array<tvm::PrimExpr>& padding, const ffi::Array<tvm::PrimExpr>& kernel_dims);

/*!
 * \brief Check whether a PrimExpr simplifies to a given integer constant.
 *
 * \param expr The expression to test.
 * \param val The integer value to compare against.
 * \return True if `expr` simplifies to `val`.
 */
bool EqualsConstInt(const tvm::PrimExpr& expr, int64_t val);

/*!
 * \brief Insert zeros between elements of a tensor along every spatial axis.
 *
 * For each spatial axis `i`, the output size is
 * `(data->shape[i] - 1) * strides[i] + 1`. Positions that do not correspond
 * to an original element are filled with `dilation_value`.
 *
 * \param data The input tensor to dilate.
 * \param strides Dilation stride per axis (length must equal `data->shape.size()`).
 * \param dilation_value Fill value for the inserted zero positions.
 * \param name Optional name for the output tensor.
 * \return The dilated output tensor.
 */
te::Tensor Dilate(const te::Tensor& data, const ffi::Array<tvm::PrimExpr>& strides,
                  const PrimExpr& dilation_value, const ffi::String& name = "DilatedInput");

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_LEGALIZE_OPS_UTILS_H_
