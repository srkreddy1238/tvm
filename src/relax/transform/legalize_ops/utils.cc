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
 * \file tvm/relax/transform/legalize_ops/utils.cc
 * \brief Helpers for legalization
 */

#include "utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>

#include <numeric>
#include <string>
#include <vector>

namespace tvm {
namespace relax {

ffi::Any MakeCallTE::ConvertToTE(const ffi::Any& any) {
  std::function<void(const ffi::ObjectRef&)> _visit_expr = [&](const ffi::ObjectRef& expr) -> void {
    tvm::PrimExpr pexpr = Downcast<PrimExpr>(expr);
    if (pexpr.as<tvm::tir::Var>()) {
      auto p_var = pexpr.as<tvm::tir::Var>().value();
      if (tir_var_map_.find(p_var) == tir_var_map_.end()) {
        auto new_var = tvm::tir::Var(pexpr.as<tvm::tir::VarNode>()->name_hint,
                                     pexpr.as<tvm::tir::VarNode>()->dtype);
        tir_var_map_.Set(p_var, new_var.as<PrimExpr>().value());
      }
    }
  };

  std::function<std::string(const Expr&)> _get_name = [&](const Expr& arg) -> std::string {
    std::string te_name;
    auto n_args = create_primfunc_args_.size();
    if (arg.as<VarNode>()) {
      te_name = arg.as<VarNode>()->name_hint();
    } else if (n_args < 26) {
      te_name = 'A' + n_args;
    } else {
      te_name = "tensor_input_" + std::to_string(n_args);
    }
    return te_name;
  };
  if (any == nullptr) {
    return any;
  } else if (any.as<ffi::Array<ffi::Any>>()) {
    ffi::Array<ffi::Any> ret;
    for (auto val : any.cast<ffi::Array<ffi::Any>>()) {
      ret.push_back(ConvertToTE(val));
    }
    return ret;
  } else if (any.as<Expr>()) {
    auto arg = any.as<Expr>().value();
    auto sinfo = GetStructInfo(arg);
    if (sinfo.as<TensorStructInfoNode>()) {
      TVM_FFI_ICHECK_NOTNULL(sinfo.as<TensorStructInfoNode>()->shape.as<ShapeExprNode>());
      for (auto shape_val : sinfo.as<TensorStructInfoNode>()->shape.as<ShapeExprNode>()->values) {
        tvm::tir::PostOrderVisit(shape_val, _visit_expr);
      }
      auto te_name = _get_name(arg);
      auto te_param = TETensor(arg, tir_var_map_, te_name);

      call_tir_args_.push_back(arg);
      create_primfunc_args_.push_back(te_param);

      return te_param;
    } else if (sinfo.as<ShapeStructInfoNode>()) {
      TVM_FFI_ICHECK(arg.as<ShapeExpr>()) << "Only ShapeExpr supported now for ShapeStructInfo";
      ffi::Array<ffi::Any> ret;
      for (auto val : arg.as<ShapeExprNode>()->values) {
        ret.push_back(ConvertToTE(val));
      }
      return ret;
    } else if (sinfo.as<PrimStructInfoNode>()) {
      if (sinfo.as<PrimStructInfoNode>()->value.has_value()) {
        return ConvertToTE(sinfo.as<PrimStructInfoNode>()->value.value());
      } else {
        auto te_name = _get_name(arg);
        auto tir_param = tvm::tir::Var(te_name, sinfo.as<PrimStructInfoNode>()->dtype);

        call_tir_args_.push_back(arg);
        create_primfunc_args_.push_back(tir_param);

        return tir_param;
      }
    } else if (sinfo.as<TupleStructInfoNode>()) {
      ffi::Array<ffi::Any> ret;
      for (auto val : arg.as<TupleNode>()->fields) {
        ret.push_back(ConvertToTE(val));
      }
      return ret;
    } else {
      LOG(FATAL) << "Unexpeted sinfo: " << sinfo;
    }
  } else if (any.as<PrimExpr>()) {
    auto arg = any.as<PrimExpr>().value();
    tvm::tir::PostOrderVisit(arg, _visit_expr);
    ObjectRef p_val = tvm::tir::Substitute(arg, tir_var_map_);
    extra_tir_args_list_.push_back(p_val);
    return p_val;
  } else if (any.as<ShapeExpr>()) {
    ffi::Array<ffi::Any> ret;
    for (auto val : any.as<ShapeExprNode>()->values) {
      ret.push_back(ConvertToTE(val));
    }
    return ret;
  } else if (any.as<DataType>() || any.as<int>() || any.as<float>() || any.as<double>() ||
             any.as<ffi::String>() || any.as<bool>()) {
    return any;
  } else {
    LOG(FATAL) << "Conversion to TE arg not handled:" << any.GetTypeKey();
  }
}

ffi::Array<ffi::Any> MakeCallTE::CallArgsToTE(const ffi::Array<ffi::Any>& args) {
  ffi::Array<ffi::Any> ret;

  for (auto arg : args) {
    auto te_arg = ConvertToTE(arg);
    ret.push_back(te_arg);
  }
  return ret;
}

VDevice MakeCallTE::GetVDevice(void) {
  auto args = call_->args;
  VDevice vdev = VDevice();
  for (auto arg : args) {
    auto sinfo = GetStructInfo(arg);
    if (sinfo.as<TensorStructInfoNode>()) {
      if (sinfo.as<TensorStructInfoNode>()->vdevice.defined()) {
        return sinfo.as<TensorStructInfoNode>()->vdevice.value();
      }
    }
  }

  return vdev;
}

ffi::Array<tvm::te::Tensor> MakeCallTE::CallTEHandler(
    const ffi::Array<ffi::Any>& te_args,
    const ffi::Variant<FTOPIHandler, ffi::String>& topi_handler) {
  if (auto hndl = topi_handler.as<FTOPIHandler>()) {
    return hndl.value()(te_args);
  } else if (auto hndl = topi_handler.as<ffi::String>()) {
    std::vector<AnyView> packed_args;
    ffi::Array<tvm::te::Tensor> te_ret;
    ffi::Any ret;

    for (auto arg : te_args) {
      packed_args.push_back(arg);
    }

    auto topi_func = ffi::Function::GetGlobal(hndl.value());
    TVM_FFI_ICHECK(topi_func.has_value()) << "TOPI handler not found:" << hndl.value();
    topi_func.value().CallPacked(ffi::PackedArgs(packed_args.data(), packed_args.size()), &ret);
    if (ret.cast<tvm::te::Tensor>().defined()) {
      te_ret.push_back(ret.cast<tvm::te::Tensor>());
    } else if (ret.cast<ffi::Array<tvm::te::Tensor>>().defined()) {
      for (auto te_ret_val : ret.cast<ffi::Array<tvm::te::Tensor>>()) {
        te_ret.push_back(te_ret_val);
      }
    } else {
      LOG(FATAL) << "Not expecting any thing other than te.Tensor as return from topi handler";
    }
    return te_ret;
  } else {
    LOG(FATAL) << "Unexpected TOPI Handler:" << topi_handler;
  }
}

/*!
 * \brief Collect TIR variables that appear in `args` or `extras` but are not
 *        bound by any tensor shape in `args`.
 *
 * \param args Primary TE argument list (tensors and PrimExprs).
 * \param extras Additional PrimExpr arguments that may reference free vars.
 * \return The list of unbound TIR variables.
 */
ffi::Array<ObjectRef> GetUnboundTIRVars(const ffi::Array<ObjectRef>& args,
                                        const ffi::Array<ObjectRef>& extras) {
  ffi::Array<ObjectRef> bound_vars;
  ffi::Array<ObjectRef> used_vars;

  std::function<void(const ffi::ObjectRef&)> _populate_bound_vars =
      [&](const ffi::ObjectRef& expr) -> void {
    if (expr.as<tvm::te::Tensor>()) {
      for (auto dim : expr.as<tvm::te::Tensor>().value()->shape) {
        _populate_bound_vars(dim);
      }
    } else if (expr.as<tvm::tir::Var>()) {
      bound_vars.push_back(expr);
    }
  };

  std::function<void(const ffi::ObjectRef&)> _populate_used_vars =
      [&](const ffi::ObjectRef& expr) -> void {
    if (expr.as<tvm::te::Tensor>()) {
      for (auto dim : expr.as<tvm::te::Tensor>().value()->shape) {
        _populate_used_vars(dim);
      }
    } else if (expr.as<PrimExpr>()) {
      auto analysed_vars = tvm::tir::UndefinedVars(expr.as<PrimExpr>().value());
      for (auto var : analysed_vars) {
        if (std::find(used_vars.begin(), used_vars.end(), var.as<ObjectRef>().value()) ==
            used_vars.end()) {
          used_vars.push_back(var.as<ObjectRef>().value());
        }
      }
    }
  };

  for (auto arg : args) {
    _populate_used_vars(arg);
  }
  for (auto arg : extras) {
    _populate_used_vars(arg);
  }

  for (auto arg : args) {
    _populate_bound_vars(arg);
  }

  ffi::Array<ObjectRef> diff;
  for (auto arg : used_vars) {
    if (std::find(bound_vars.begin(), bound_vars.end(), arg) == bound_vars.end()) {
      diff.push_back(arg);
    }
  }

  return diff;
}

std::tuple<tvm::tir::PrimFunc, tvm::ffi::Array<Expr>, ffi::Array<TensorStructInfo>,
           ffi::Array<PrimExpr>>
MakeCallTE::GenCallTirInputs(const ffi::Array<ffi::Any>& args,
                             ffi::Variant<FTOPIHandler, ffi::String> topi_handler,
                             std::string fname) {
  auto te_args = CallArgsToTE(args);
  auto te_outs = CallTEHandler(te_args, topi_handler);
  ffi::Array<ObjectRef> prim_args;

  // Inputs
  for (auto te_in : create_primfunc_args_) {
    prim_args.push_back(te_in.as<ObjectRef>().value());
  }

  // Output
  for (auto te_out : te_outs) {
    prim_args.push_back(te_out.as<ObjectRef>().value());
  }

  // Extras
  ffi::Array<ObjectRef> extra_args;
  for (auto extra : extra_tir_args_list_) {
    extra_args.push_back(extra.as<ObjectRef>().value());
  }

  auto unbound_tir_vars = GetUnboundTIRVars(prim_args, extra_args);
  for (auto extra : unbound_tir_vars) {
    prim_args.push_back(extra);
  }

  auto prim_func =
      tvm::tir::CreatePrimFunc(prim_args, tvm::runtime::DataType(DLDataType({kDLInt, 64, 1})));

  ffi::Array<TensorStructInfo> output_sinfo;
  ffi::Map<tvm::tir::Var, tvm::PrimExpr> tir_var_inv_map_;

  for (auto it : tir_var_map_) {
    tir_var_inv_map_.Set(it.second.as<tvm::tir::Var>().value(),
                         it.first.as<tvm::PrimExpr>().value());
  }

  for (auto out : te_outs) {
    ffi::Array<PrimExpr> shape_arr;
    for (auto val : out->shape) {
      PrimExpr s_val = tvm::tir::Substitute(val, tir_var_inv_map_);
      shape_arr.push_back(s_val);
    }
    output_sinfo.push_back(TensorStructInfo(ShapeExpr(shape_arr), out->dtype, GetVDevice()));
  }

  ffi::Array<PrimExpr> tir_vars;
  if (unbound_tir_vars.size() > 0) {
    for (auto tir_var : unbound_tir_vars) {
      PrimExpr s_val = tvm::tir::Substitute(tir_var.as<PrimExpr>().value(), tir_var_inv_map_);
      tir_vars.push_back(s_val);
    }
  }

  return {prim_func, call_tir_args_, output_sinfo, tir_vars};
}

Call MakeCallTE::Make(const ffi::Array<ffi::Any>& args,
                      ffi::Variant<FTOPIHandler, ffi::String> topi_handler, std::string fname,
                      ffi::Optional<StructInfo> out_sinfo_override) {
  auto [tir_func, call_tir_args, output_sinfo, tir_vars] =
      GenCallTirInputs(args, topi_handler, fname);

  tir_func = tvm::WithoutAttr(tir_func, tvm::attr::kGlobalSymbol);
  auto gvar = bb_->AddFunction(tir_func, fname);
  static const Op& call_tir_op_ = Op::Get("relax.call_tir");
  ffi::Array<Expr> call_args = {gvar, Tuple(call_tir_args)};

  if (tir_vars.size() > 0) {
    call_args.push_back(ShapeExpr(tir_vars));
  }

  // If the caller supplied an explicit output StructInfo (mirrors Python's
  // sinfo_args parameter in bb.call_te), use it instead of the one derived
  // from the TE compute output shape.
  if (out_sinfo_override.defined()) {
    return Call(call_tir_op_, call_args, tvm::Attrs(), {out_sinfo_override.value()});
  }

  if (output_sinfo.size() == 1) {
    return Call(call_tir_op_, call_args, tvm::Attrs(), {output_sinfo[0]});
  } else {
    TupleStructInfo t_sinfo = TupleStructInfo(output_sinfo);
    return Call(call_tir_op_, call_args, tvm::Attrs(), {t_sinfo});
  }
}

/*!
 * \brief Convert a float16 bit-pattern stored as uint16 to a float32 value.
 *
 * Implements the IEEE 754 half-precision to single-precision conversion
 * using only integer arithmetic, with denormals mapped to zero.
 *
 * \param out Pointer to the float32 destination.
 * \param in The float16 bit-pattern to convert.
 */
void float32(float* __restrict out, const uint16_t in) {
  uint32_t t1;
  uint32_t t2;
  uint32_t t3;

  t1 = in & 0x7fffu;  // Non-sign bits
  t2 = in & 0x8000u;  // Sign bit
  t3 = in & 0x7c00u;  // Exponent

  t1 <<= 13u;  // Align mantissa on MSB
  t2 <<= 16u;  // Shift sign bit into position

  t1 += 0x38000000;  // Adjust bias

  t1 = (t3 == 0 ? 0 : t1);  // Denormals-as-zero

  t1 |= t2;  // Re-insert sign bit

  *((uint32_t*)out) = t1;
};

ffi::Any TryConvertToScalarConst(const relax::Expr& val) {
  if (auto c_val = val.as<relax::ConstantNode>()) {
    auto sinfo = GetStructInfo(val).as<TensorStructInfoNode>();
    if (sinfo->ndim == 0) {
      if (sinfo->dtype.is_float()) {
        if (sinfo->dtype.bits() == 32) {
          float* data_ptr = (float*)(c_val->data.operator->()->data);
          return tvm::FloatImm(sinfo->dtype, *data_ptr);
        } else if (sinfo->dtype.bits() == 16) {
          uint16_t* data_ptr = (uint16_t*)(c_val->data.operator->()->data);
          float fval;
          float32(&fval, *data_ptr);
          return tvm::FloatImm(sinfo->dtype, fval);
        }
      } else if (sinfo->dtype.is_int()) {
        if (sinfo->dtype.bits() == 32) {
          int* data_ptr = (int*)(c_val->data.operator->()->data);
          return tvm::IntImm(sinfo->dtype, *data_ptr);
        }
      } else if (sinfo->dtype.is_uint()) {
        if (sinfo->dtype.bits() == 8) {
          char* data_ptr = (char*)(c_val->data.operator->()->data);
          return tvm::IntImm(sinfo->dtype, (int64_t)*data_ptr);
        }
      } else if (sinfo->dtype.is_bool()) {
        char* data_ptr = (char*)(c_val->data.operator->()->data);
        return tvm::IntImm(sinfo->dtype, (int64_t)*data_ptr);
      }
    }
  }
  return val;
}

ffi::Array<ffi::Any> GetConstTuple(const ffi::Array<PrimExpr>& tuple) {
  ffi::Array<ffi::Any> ret;
  for (auto val : tuple) {
    if (auto ival = val.as<tvm::IntImmNode>()) {
      ret.push_back(ival->value);
    } else {
      arith::Analyzer analyzer;
      auto a_val = analyzer.Simplify(val);
      if (auto ival = a_val.as<IntImmNode>()) {
        ret.push_back(ival->value);
      } else {
        ret.push_back(a_val);
      }
    }
  }
  return ret;
}

std::pair<ffi::Array<tvm::PrimExpr>, ffi::Array<tvm::PrimExpr>> GetPadTupleGeneric(
    const ffi::Array<tvm::PrimExpr>& padding, const ffi::Array<tvm::PrimExpr>& kernel_dims) {
  ffi::Array<tvm::PrimExpr> pad_dimensions;
  ffi::Array<tvm::PrimExpr> pad_begin;
  ffi::Array<tvm::PrimExpr> pad_end;

  if (padding.size() == kernel_dims.size()) {
    arith::Analyzer analyzer;
    for (auto pval : padding) {
      pad_dimensions.push_back(pval * tvm::tir::make_const(pval->dtype, 2));
    }
    for (size_t i = 0; i < pad_dimensions.size(); ++i) {
      pad_begin.push_back(analyzer.Simplify(tvm::tir::FloorDiv(
          (pad_dimensions[i] + tvm::tir::make_const(pad_dimensions[i]->dtype, 1)),
          tvm::tir::make_const(pad_dimensions[i]->dtype, 2))));

      pad_end.push_back(analyzer.Simplify(pad_dimensions[i] - pad_begin[i]));
    }
  } else if (padding.size() == kernel_dims.size() * 2) {
    for (size_t i = 0; i < kernel_dims.size(); ++i) {
      pad_begin.push_back(padding[i]);
      pad_end.push_back(padding[kernel_dims.size() + i]);
    }
  } else if (padding.size() == 1) {
    for (size_t i = 0; i < kernel_dims.size(); ++i) {
      pad_begin.push_back(padding[0]);
      pad_end.push_back(padding[0]);
    }
  } else {
    LOG(FATAL) << "Padidng size " << padding.size() << " with kernel size " << kernel_dims.size()
               << " not supported";
  }
  return std::make_pair(pad_begin, pad_end);
}

bool EqualsConstInt(const tvm::PrimExpr& expr, int64_t val) {
  arith::Analyzer analyzer;
  if (auto int_node = analyzer.Simplify(expr).as<tvm::IntImmNode>()) {
    return (int_node->value == val);
  }
  return false;
}

te::Tensor Dilate(const te::Tensor& data, const ffi::Array<tvm::PrimExpr>& strides,
                  const PrimExpr& dilation_value, const ffi::String& name) {
  auto n = data->shape.size();
  TVM_FFI_ICHECK(strides.size() == n) << "Data and strides dimension mismatch";
  arith::Analyzer analyzer;
  ffi::Map<ffi::String, ffi::Any> attrs({});

  ffi::Array<tvm::PrimExpr> out_shape;
  for (size_t i = 0; i < n; ++i) {
    out_shape.push_back(
        analyzer.Simplify((data->shape[i] - tvm::IntImm(data->shape[0]->dtype, 1)) * strides[i] +
                          tvm::IntImm(data->shape[0]->dtype, 1)));
  }

  return tvm::te::compute(
      out_shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        ffi::Array<tvm::PrimExpr> not_zero;
        ffi::Array<tvm::PrimExpr> index_tuple;

        for (size_t i = 0; i < n; ++i) {
          if (!EqualsConstInt(strides[i], 1)) {
            index_tuple.push_back(tvm::tir::FloorDiv(indices[i], strides[i]));
            not_zero.push_back(tvm::tir::FloorMod(indices[i], strides[i]) ==
                               tvm::IntImm(indices[i]->dtype, 0));
          } else {
            index_tuple.push_back(indices[i]);
          }
        }
        if (not_zero.size()) {
          tvm::PrimExpr not_zero_all = not_zero[0];
          for (size_t i = 1; i < not_zero.size(); ++i) {
            not_zero_all = tvm::logical_and(analyzer.Simplify(not_zero_all), not_zero[i]);
          }
          return tvm::if_then_else(not_zero_all, data(index_tuple), dilation_value);
        }
        return data(index_tuple);
      },
      name, kInjective + ",dilate", attrs);
}

}  // namespace relax
}  // namespace tvm
