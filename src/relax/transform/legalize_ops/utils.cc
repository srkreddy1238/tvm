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

namespace tvm {
namespace relax {

tvm::ffi::Any MakeCallTE::ConvertToTE(const tvm::ffi::Any& any) {
  std::function<void(const tvm::ffi::ObjectRef&)> _visit_expr =
      [&](const tvm::ffi::ObjectRef& expr) -> void {
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
    if (arg.as<tvm::relax::VarNode>()) {
      te_name = arg.as<tvm::relax::VarNode>()->name_hint();
    } else if (n_args < 26) {
      te_name = 'A' + n_args;
    } else {
      te_name = "tensor_input_" + std::to_string(n_args);
    }
    return te_name;
  };

  if (any.as<Expr>()) {
    auto arg = any.as<Expr>().value();
    auto sinfo = GetStructInfo(arg);
    if (sinfo.as<TensorStructInfoNode>()) {
      ICHECK_NOTNULL(sinfo.as<TensorStructInfoNode>()->shape.as<ShapeExprNode>());
      for (auto shape_val : sinfo.as<TensorStructInfoNode>()->shape.as<ShapeExprNode>()->values) {
        tvm::tir::PostOrderVisit(shape_val, _visit_expr);
      }
      auto te_name = _get_name(arg);
      auto te_param = tvm::relax::TETensor(arg, tir_var_map_, te_name);

      call_tir_args_.push_back(arg);
      create_primfunc_args_.push_back(te_param);

      return te_param;
    } else if (sinfo.as<ShapeStructInfoNode>()) {
      ICHECK(arg.as<ShapeExpr>()) << "Only ShapeExpr supported now for ShapeStructInfo";
      tvm::ffi::Array<tvm::ffi::Any> ret;
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
    } else {
      LOG(FATAL) << "Unexpeted sinfo: " << sinfo;
    }
  } else if (any.as<PrimExpr>()) {
    auto arg = any.as<PrimExpr>().value();
    tvm::tir::PostOrderVisit(arg, _visit_expr);
    ObjectRef p_val = tvm::tir::Substitute(arg, tir_var_map_);
    extra_tir_args_list_.push_back(p_val);
    return p_val;
  } else {
    LOG(FATAL) << "Expected an Expr to convert to TE arg";
  }
}

ffi::Array<tvm::ffi::Any> MakeCallTE::CallArgsToTE(const tvm::ffi::Array<Expr>& args) {
  ffi::Array<tvm::ffi::Any> ret;

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

ffi::Array<tvm::te::Tensor> MakeCallTE::CallTEHandler(ffi::Array<tvm::ffi::Any>& te_args,
                                                      std::string& topi_handler) {
  std::vector<AnyView> packed_args;
  ffi::Any ret;
  ffi::Array<tvm::te::Tensor> te_ret;

  for (auto arg : te_args) {
    packed_args.push_back(arg);
  }

  auto topi_func = tvm::ffi::Function::GetGlobal(topi_handler);
  ICHECK(topi_func.has_value());
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
}

ffi::Array<ObjectRef> GetUnboundTIRVars(const ffi::Array<ObjectRef>& args,
                                        const ffi::Array<ObjectRef>& extras) {
  ffi::Array<ObjectRef> bound_vars;
  ffi::Array<ObjectRef> used_vars;

  std::function<void(const tvm::ffi::ObjectRef&)> _populate_bound_vars =
      [&](const tvm::ffi::ObjectRef& expr) -> void {
    if (expr.as<tvm::te::Tensor>()) {
      for (auto dim : expr.as<tvm::te::Tensor>().value()->shape) {
        _populate_bound_vars(dim);
      }
    } else if (expr.as<tvm::tir::Var>()) {
      bound_vars.push_back(expr);
    }
  };

  std::function<void(const tvm::ffi::ObjectRef&)> _populate_used_vars =
      [&](const tvm::ffi::ObjectRef& expr) -> void {
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

tvm::relax::Call MakeCallTE::Make(const tvm::ffi::Array<Expr>& args, std::string topi_handler,
                                  std::string fname) {
  LOG(WARNING) << "**** Make Start ****";

  auto te_args = CallArgsToTE(args);
  auto te_outs = CallTEHandler(te_args, topi_handler);
  ffi::Array<ObjectRef> prim_args;

  // Inputs
  for (auto te_in : te_args) {
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
  LOG(WARNING) << "Unbound:" << unbound_tir_vars;
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

  prim_func = tvm::WithoutAttr(prim_func, tvm::attr::kGlobalSymbol);
  auto gvar = bb_->AddFunction(prim_func, fname);
  static const Op& call_tir_op_ = Op::Get("relax.call_tir");

  ffi::Array<Expr> call_args = {gvar, Tuple(call_tir_args_)};

  if (unbound_tir_vars.size() > 0) {
    ffi::Array<PrimExpr> tir_vars;
    for (auto tir_var : unbound_tir_vars) {
      PrimExpr s_val = tvm::tir::Substitute(tir_var.as<PrimExpr>().value(), tir_var_inv_map_);
      tir_vars.push_back(s_val);
    }
    call_args.push_back(ShapeExpr(tir_vars));
  }

  LOG(WARNING) << "***** Make End ****";
  if (output_sinfo.size() == 1) {
    return Call(call_tir_op_, call_args, call_->attrs, {output_sinfo[0]});
  } else {
    TupleStructInfo t_sinfo = TupleStructInfo(output_sinfo);
    return Call(call_tir_op_, call_args, call_->attrs, {t_sinfo});
  }
}

}  // namespace relax
}  // namespace tvm
