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
 * \file src/relax/frontend/nn/exporter.cc
 * \brief C++ implementation of ExportToIRModule.
 */

#include "exporter.h"
#include "core.h"
#include "spec.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/expr.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ===========================================================================
// Internal helpers
// ===========================================================================

/*!
 * \brief Build a ShapeExpr from a SpecTensor's shape array.
 *
 * Each element is either:
 *   int64   → IntImm(int64, v)
 *   String  → tir::Var(name, int64)  [symbolic, deduplicated via str2var]
 */
static ShapeExpr BuildSpecShape(const ffi::Array<ffi::Any>& shape,
                                std::unordered_map<std::string, tir::Var>& str2var) {
  ffi::Array<PrimExpr> dims;
  for (const ffi::Any& elem : shape) {
    if (auto opt = elem.TryAs<int64_t>()) {
      dims.push_back(tir::IntImm(DataType::Int(64), opt.value()));
    } else if (auto opt = elem.TryAs<ffi::String>()) {
      std::string name = std::string(opt.value());
      auto it = str2var.find(name);
      if (it == str2var.end()) {
        tir::Var v(name, DataType::Int(64));
        str2var[name] = v;
        dims.push_back(v);
      } else {
        dims.push_back(it->second);
      }
    } else {
      TVM_FFI_THROW(TypeError) << "BuildSpecShape: invalid shape element: " << elem.GetTypeKey();
    }
  }
  return ShapeExpr(dims);
}

/*!
 * \brief Unwrap a forward() return value to a relax::Expr.
 *
 * Handles:
 *   TensorNode  → its .expr Var
 *   Array<Any>  → relax::Tuple of recursively unwrapped elements
 */
static Expr UnwrapReturn(const ffi::Any& val) {
  // Case 1: single Tensor
  if (auto opt = val.TryAs<NNTensor>()) {
    return opt.value()->expr;
  }
  // Case 2: NNParameter (subclass of TensorNode)
  if (auto opt = val.TryAs<NNParameter>()) {
    return opt.value()->expr;
  }
  // Case 3: Array (tuple of outputs)
  if (auto opt = val.TryAs<ffi::Array<ffi::Any>>()) {
    ffi::Array<Expr> fields;
    for (const ffi::Any& elem : opt.value()) {
      fields.push_back(UnwrapReturn(elem));
    }
    return relax::Tuple(fields);
  }
  // Case 4: already a Var (e.g. from WrapNested)
  if (auto opt = val.TryAs<Var>()) {
    return opt.value();
  }
  TVM_FFI_THROW(TypeError) << "UnwrapReturn: unsupported return type: " << val.GetTypeKey();
}

// ===========================================================================
// _initialize_effect  (debug=true only)
// ===========================================================================

static void EmitInitializeEffect(BlockBuilder& bb) {
  // @R.function
  // def _initialize_effect() -> R.Tuple(R.Object):
  //     with R.dataflow():
  //         _io: R.Object = R.null_value()
  //         lv: R.Tuple(R.Object) = (_io,)
  //         gv: R.Tuple(R.Object) = lv
  //         R.output(gv)
  //     return gv
  with(bb->function("_initialize_effect", {}, /*attrs=*/{}), [&]() {
    with(bb->dataflow(), [&]() {
      Var io = bb->Emit(relax::op::null_value(), "_io");
      Var lv = bb->Emit(relax::Tuple({io}), "lv");
      Var gv = bb->EmitOutput(lv, "gv");
      bb->EmitFuncOutput(gv, {});
    });
  });
}

// ===========================================================================
// EmitMethod
// ===========================================================================

static void EmitMethod(BlockBuilder& bb, const ffi::String& method_name,
                       const MethodSpecNode* ms,
                       const ffi::Map<ffi::String, NNParameter>& named_params,
                       bool debug) {
  // ---- 1. Build symbolic-shape deduplication map -------------------------
  std::unordered_map<std::string, tir::Var> str2var;

  // ---- 2. Build input Vars from arg_specs --------------------------------
  // explicit_inputs[i] = the nn.Tensor (or tir.Var for Int) for arg i
  std::vector<ffi::Any> explicit_inputs;
  std::vector<Var> input_vars;  // relax Vars for the function signature

  for (size_t i = 0; i < ms->arg_specs.size(); ++i) {
    const ffi::String& arg_name = ms->arg_names[i];
    const ffi::Any& spec = ms->arg_specs[i];

    if (auto opt = spec.TryAs<runtime::ObjectRef>()) {
      runtime::ObjectRef spec_obj = opt.value();

      if (spec_obj->IsInstance<SpecTensorNode>()) {
        const auto* st = spec_obj.as<SpecTensorNode>();
        ShapeExpr shape = BuildSpecShape(st->shape, str2var);
        DataType dtype = DataType(runtime::String2DLDataType(st->dtype));
        Var v(std::string(arg_name), TensorStructInfo(shape, dtype));
        input_vars.push_back(v);
        // Wrap as nn.Tensor so forward() receives TensorNode
        NNTensor t(v);
        explicit_inputs.push_back(ffi::Any(t));

      } else if (spec_obj->IsInstance<SpecIntNode>()) {
        // Int spec → ShapeVar (R.Shape([name]))
        tir::Var tvar(std::string(arg_name), DataType::Int(64));
        str2var[std::string(arg_name)] = tvar;
        Var v(std::string(arg_name), ShapeStructInfo({tvar}));
        input_vars.push_back(v);
        explicit_inputs.push_back(ffi::Any(tvar));

      } else {
        TVM_FFI_THROW(TypeError) << "EmitMethod: unsupported arg_spec type: "
                                 << spec_obj->GetTypeKey();
      }
    } else {
      TVM_FFI_THROW(TypeError) << "EmitMethod: arg_spec is not an ObjectRef";
    }
  }

  // ---- 3. Build param Vars -----------------------------------------------
  // Re-use symbolic shape vars so the same name maps to the same tir::Var.
  std::vector<Var> param_vars;
  // We need a mutable copy of named_params to update .expr in-place.
  // Collect as vector to preserve order.
  std::vector<std::pair<ffi::String, NNParameter>> params_vec;
  for (const auto& [name, param] : named_params) {
    params_vec.emplace_back(name, param);
  }

  for (auto& [name, param] : params_vec) {
    ffi::Array<PrimExpr> old_shape = param->GetShape();
    ffi::Array<PrimExpr> new_shape_dims;
    for (const PrimExpr& dim : old_shape) {
      if (const auto* var = dim.as<tir::VarNode>()) {
        auto it = str2var.find(var->name_hint);
        if (it != str2var.end()) {
          new_shape_dims.push_back(it->second);
          continue;
        }
        tir::Var v(var->name_hint, DataType::Int(64));
        str2var[var->name_hint] = v;
        new_shape_dims.push_back(v);
      } else {
        new_shape_dims.push_back(dim);
      }
    }
    DataType dtype = DataType(runtime::String2DLDataType(param->GetDtype()));
    Var v(std::string(name), TensorStructInfo(ShapeExpr(new_shape_dims), dtype));
    param_vars.push_back(v);
    // Update the parameter's expr so forward() sees the new Var
    param->expr = v;
  }

  // ---- 4. Build effect Var (debug only) ----------------------------------
  ffi::Optional<Var> io_var;
  if (debug) {
    io_var = Var("_io", ObjectStructInfo());
  }

  // ---- 5. Assemble full function input list ------------------------------
  // Signature: [explicit_inputs..., _io (if debug), params...]
  // num_input attr = len(explicit_inputs) + (1 if debug else 0)
  ffi::Array<Var> func_params;
  for (const Var& v : input_vars) func_params.push_back(v);
  if (debug) func_params.push_back(io_var.value());
  for (const Var& v : param_vars) func_params.push_back(v);

  int64_t num_input = static_cast<int64_t>(input_vars.size()) + (debug ? 1 : 0);

  // ---- 6. Emit the function ----------------------------------------------
  ffi::Map<ffi::String, ffi::Any> func_attrs;
  func_attrs.Set("num_input", ffi::Any(num_input));

  with(bb->function(std::string(method_name), func_params, func_attrs), [&]() {
    with(bb->dataflow(), [&]() {
      // Build named_args map for forward()
      ffi::Map<ffi::String, ffi::Any> named_args;
      for (size_t i = 0; i < ms->arg_names.size(); ++i) {
        named_args.Set(ms->arg_names[i], explicit_inputs[i]);
      }

      // Call the forward function
      ffi::Any raw_out = ms->forward(named_args);

      // Unwrap return to relax::Expr
      Expr out_expr = UnwrapReturn(raw_out);

      // Wrap effect output if debug
      Expr final_out;
      if (debug) {
        // outputs = (out_expr, (_io,))
        Expr effect_tuple = relax::Tuple({io_var.value()});
        final_out = relax::Tuple({out_expr, effect_tuple});
      } else {
        final_out = out_expr;
      }

      Var gv = bb->EmitOutput(final_out, "gv1");
      bb->EmitFuncOutput(gv, func_params);
    });
  });
}

// ===========================================================================
// ExportToIRModule
// ===========================================================================

IRModule ExportToIRModule(ModuleSpec spec, bool debug) {
  const ModuleSpecNode* ms_node = spec.get();
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);

  // Emit _initialize_effect if debug
  if (debug) {
    EmitInitializeEffect(bb);
  }

  // Emit each method
  for (size_t i = 0; i < ms_node->method_names.size(); ++i) {
    const ffi::String& method_name = ms_node->method_names[i];
    const ffi::Any& method_spec_any = ms_node->method_specs[i];

    auto opt = method_spec_any.TryAs<MethodSpec>();
    TVM_FFI_ICHECK(opt.has_value())
        << "ExportToIRModule: method_spec[" << i << "] is not a MethodSpec, got "
        << method_spec_any.GetTypeKey();

    EmitMethod(bb, method_name, opt.value().get(), ms_node->named_params, debug);
  }

  IRModule mod = bb->Finalize();
  TVM_FFI_ICHECK(relax::analysis::well_formed(mod))
      << "ExportToIRModule: produced ill-formed IRModule";
  return mod;
}

// ===========================================================================
// FFI registration
// ===========================================================================

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.frontend.nn.ExportToIRModule",
           [](ModuleSpec spec, bool debug) -> IRModule {
             return ExportToIRModule(std::move(spec), debug);
           });
}

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
