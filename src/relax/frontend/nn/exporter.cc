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
 *
 * Uses the low-level BlockBuilder API (BeginScope/EndScope,
 * BeginDataflowBlock/EndBlock, EmitOutput, relax::Function) that is
 * already present in the codebase, matching the pattern in
 * tests/cpp-compiler/compiler_base.cc.
 */

#include "exporter.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/expr.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "core.h"
#include "spec.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ===========================================================================
// BBScope: RAII helper that installs/restores the thread-local BlockBuilder
// so that WrapNested / Emit helpers in core.cc and op.cc can find it.
// ===========================================================================

struct BBScope {
  BlockBuilder* prev;
  explicit BBScope(BlockBuilder& bb) : prev(nullptr) {
    // Save whatever was current and install the new one
    BlockBuilder cur = BlockBuilder_Current();
    prev = cur.defined() ? new BlockBuilder(cur) : nullptr;
    BlockBuilder_SetCurrent(&bb);
  }
  ~BBScope() {
    BlockBuilder_SetCurrent(prev);
    delete prev;
  }
};

// Thread-local storage for the current debug _io Var, set by EmitMethod
// so that debug_func (called from Python forward()) can retrieve it.
static thread_local ffi::Optional<Var> g_current_io_var;  // NOLINT(*)

ffi::Optional<Var> GetCurrentIOVar() { return g_current_io_var; }
void SetCurrentIOVar(ffi::Optional<Var> v) { g_current_io_var = v; }

// RAII scope that installs/restores the current _io Var
struct IOVarScope {
  ffi::Optional<Var> prev;
  explicit IOVarScope(ffi::Optional<Var> v) : prev(g_current_io_var) { g_current_io_var = v; }
  ~IOVarScope() { g_current_io_var = prev; }
};

/*!
 * \brief Build a ShapeExpr from a SpecTensor shape array.
 *
 * Each element is either:
 *   int64   -> IntImm(int64, v)
 *   String  -> tir::Var(name, int64)  [symbolic, deduplicated via str2var]
 */
static ShapeExpr BuildSpecShape(const ffi::Array<ffi::Any>& shape,
                                std::unordered_map<std::string, tir::Var>& str2var) {
  ffi::Array<PrimExpr> dims;
  for (const ffi::Any& elem : shape) {
    if (auto opt = elem.try_cast<int64_t>()) {
      dims.push_back(IntImm(DataType::Int(64), opt.value()));
    } else if (auto opt = elem.try_cast<ffi::String>()) {
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
      TVM_FFI_UNREACHABLE();
    }
  }
  return ShapeExpr(dims);
}

/*!
 * \brief Unwrap a forward() return value to a relax::Expr.
 *
 * Handles:
 *   NNTensor / NNParameter -> their .expr Var
 *   Array<Any>             -> relax::Tuple of recursively unwrapped elements
 *   Var                    -> returned as-is
 */
static Expr UnwrapReturn(const ffi::Any& val) {
  if (auto opt = val.try_cast<NNTensor>()) return opt.value()->expr;
  if (auto opt = val.try_cast<NNParameter>()) return opt.value()->expr;
  if (auto opt = val.try_cast<ffi::Array<ffi::Any>>()) {
    ffi::Array<Expr> fields;
    for (const ffi::Any& elem : opt.value()) fields.push_back(UnwrapReturn(elem));
    return relax::Tuple(fields);
  }
  if (auto opt = val.try_cast<Var>()) return opt.value();
  TVM_FFI_THROW(TypeError) << "UnwrapReturn: unsupported return type: " << val.GetTypeKey();
  TVM_FFI_UNREACHABLE();
}

// ===========================================================================
// EmitInitializeEffect  (debug=true only)
//
// Builds and adds to bb the function:
//   @R.function
//   def _initialize_effect() -> R.Tuple(R.Object):
//       with R.dataflow():
//           _io  = R.null_value()
//           lv   = (_io,)
//           gv   = lv          # output
//       return gv
// ===========================================================================

static void EmitInitializeEffect(BlockBuilder& bb) {
  ffi::Array<Var> params;  // no parameters

  bb->BeginScope(params);
  bb->BeginDataflowBlock();

  // Set current BB so that relax ops called from here can use BlockBuilder::Current()
  BBScope bb_scope(bb);

  static const Op& null_value_op = Op::Get("relax.null_value");
  Var io = bb->Emit(Call(null_value_op, {}, {}, {}), "_io");
  Var lv = bb->Emit(relax::Tuple({io}), "lv");
  // Expected: gv = lv  (Tuple(Object), not Tuple(Tuple(Object)))
  Var gv = bb->EmitOutput(lv, "gv");

  BindingBlock df_block = bb->EndBlock();

  // Build SeqExpr body: the dataflow block + return gv
  Expr body = bb->Normalize(SeqExpr({df_block}, gv));

  bb->EndScope();

  // Build the function with global_symbol so it is public
  ffi::Map<ffi::String, ffi::Any> attrs_map;
  attrs_map.Set("global_symbol", ffi::Any(ffi::String("_initialize_effect")));
  Function func(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs_map));
  bb->AddFunction(func, "_initialize_effect");
}

// ===========================================================================
// EmitMethod
// ===========================================================================

static void EmitMethod(BlockBuilder& bb, const ffi::String& method_name, const MethodSpecNode* ms,
                       const ffi::Map<ffi::String, NNParameter>& named_params, bool debug) {
  // ---- 1. Build symbolic-shape deduplication map -------------------------
  std::unordered_map<std::string, tir::Var> str2var;

  // ---- 2. Build input Vars from arg_specs --------------------------------
  std::vector<ffi::Any> explicit_inputs;
  std::vector<Var> input_vars;

  for (size_t i = 0; i < ms->arg_specs.size(); ++i) {
    const ffi::String& arg_name = ms->arg_names[i];
    const ffi::Any& spec = ms->arg_specs[i];

    if (auto opt = spec.try_cast<runtime::ObjectRef>()) {
      runtime::ObjectRef spec_obj = opt.value();

      if (spec_obj->IsInstance<SpecTensorNode>()) {
        const auto* st = spec_obj.as<SpecTensorNode>();
        ShapeExpr shape = BuildSpecShape(st->shape, str2var);
        DataType dtype = DataType(ffi::StringToDLDataType(st->dtype));
        Var v(std::string(arg_name), TensorStructInfo(shape, dtype));
        input_vars.push_back(v);
        explicit_inputs.push_back(ffi::Any(NNTensor(v)));

      } else if (spec_obj->IsInstance<SpecIntNode>()) {
        tir::Var tvar(std::string(arg_name), DataType::Int(64));
        str2var[std::string(arg_name)] = tvar;
        Var v(std::string(arg_name), ShapeStructInfo(ffi::Array<PrimExpr>{tvar}));
        input_vars.push_back(v);
        explicit_inputs.push_back(ffi::Any(v));

      } else {
        TVM_FFI_THROW(TypeError) << "EmitMethod: unsupported arg_spec type: "
                                 << spec_obj->GetTypeKey();
        TVM_FFI_UNREACHABLE();
      }
    } else {
      TVM_FFI_THROW(TypeError) << "EmitMethod: arg_spec is not an ObjectRef";
      TVM_FFI_UNREACHABLE();
    }
  }

  // ---- 3. Build param Vars -----------------------------------------------
  // param_mode: "plain" = individual params, "packed" = single R.Tuple param,
  //             "none"  = no params appended
  std::string param_mode_str = std::string(ms->param_mode);
  std::string effect_mode_str = std::string(ms->effect_mode);

  std::vector<std::pair<ffi::String, NNParameter>> params_vec;
  for (const auto& [name, param] : named_params) params_vec.emplace_back(name, param);

  std::vector<Var> param_vars;
  ffi::Optional<Var> packed_params_var;

  if (param_mode_str == "plain") {
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
      DataType dtype = DataType(ffi::StringToDLDataType(param->GetDtype()));
      Var v(std::string(name), TensorStructInfo(ShapeExpr(new_shape_dims), dtype));
      param_vars.push_back(v);
      param->expr = v;
    }
  } else if (param_mode_str == "packed") {
    // Single packed_params: R.Tuple parameter; individual params extracted via TupleGetItem
    ffi::Array<StructInfo> field_sinfos;
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
      DataType dtype = DataType(ffi::StringToDLDataType(param->GetDtype()));
      field_sinfos.push_back(TensorStructInfo(ShapeExpr(new_shape_dims), dtype));
    }
    // Use empty TupleStructInfo when no params (matches R.Tuple with no fields)
    StructInfo packed_sinfo = TupleStructInfo(field_sinfos);
    packed_params_var = Var("packed_params", packed_sinfo);
    // Bind each param's expr to TupleGetItem(packed_params_var, i)
    // (actual emission into BB happens after BeginDataflowBlock in step 6)
    for (size_t pi = 0; pi < params_vec.size(); ++pi) {
      auto& [name, param] = params_vec[pi];
      // Store index; will emit TupleGetItem in the dataflow block
      (void)param;  // binding deferred to step 6
    }
  }
  // param_mode == "none": no params appended, param->expr stays as placeholder

  // ---- 4. Build effect Var -----------------------------------------------
  // effect_mode: "plain" = _io param added when debug=true
  //              "none"  = no _io param regardless of debug
  bool use_effect = (effect_mode_str != "none") && debug;
  ffi::Optional<Var> io_var;
  if (use_effect) io_var = Var("_io", ObjectStructInfo());

  // ---- 5. Assemble full function parameter list --------------------------
  ffi::Array<Var> func_params;
  for (const Var& v : input_vars) func_params.push_back(v);
  if (use_effect) func_params.push_back(io_var.value());
  if (param_mode_str == "plain") {
    for (const Var& v : param_vars) func_params.push_back(v);
  } else if (param_mode_str == "packed") {
    func_params.push_back(packed_params_var.value());
  }

  int64_t num_input = static_cast<int64_t>(input_vars.size()) + (use_effect ? 1 : 0);

  // ---- 6. Build the function body ----------------------------------------
  bb->BeginScope(func_params);
  bb->BeginDataflowBlock();

  // Install current BB so WrapNested / Emit helpers work
  BBScope bb_scope(bb);
  // Install current _io var so debug_func can retrieve it
  IOVarScope io_scope(io_var);

  // For packed params: emit TupleGetItem bindings now that we are in the dataflow block
  if (param_mode_str == "packed" && packed_params_var.defined()) {
    for (size_t pi = 0; pi < params_vec.size(); ++pi) {
      auto& [name, param] = params_vec[pi];
      Expr get_item = TupleGetItem(packed_params_var.value(), static_cast<int>(pi));
      Var bound = bb->Emit(get_item, std::string(name));
      param->expr = bound;
    }
  }

  // Build named_args map for forward()
  ffi::Map<ffi::String, ffi::Any> named_args;
  for (size_t i = 0; i < ms->arg_names.size(); ++i) {
    named_args.Set(ms->arg_names[i], explicit_inputs[i]);
  }

  // Call the forward function
  ffi::Any raw_out = ms->forward(named_args);
  Expr out_expr = UnwrapReturn(raw_out);

  // Wrap effect output if use_effect -- use the (possibly updated) _io var
  Expr final_out;
  if (use_effect) {
    // debug_func may have updated g_current_io_var; use the latest value
    Var final_io = g_current_io_var.defined() ? g_current_io_var.value() : io_var.value();
    Expr effect_tuple = relax::Tuple({final_io});
    final_out = relax::Tuple({out_expr, effect_tuple});
  } else {
    final_out = out_expr;
  }

  Var gv = bb->EmitOutput(final_out, "gv");
  BindingBlock df_block = bb->EndBlock();

  Expr body = bb->Normalize(SeqExpr({df_block}, gv));
  bb->EndScope();

  // ---- 7. Build function attrs and register ------------------------------
  ffi::Map<ffi::String, ffi::Any> func_attrs_map;
  func_attrs_map.Set("num_input", ffi::Any(num_input));
  func_attrs_map.Set("global_symbol", ffi::Any(ffi::String(std::string(method_name))));
  Function func(func_params, body, std::nullopt, /*is_pure=*/true, DictAttrs(func_attrs_map));
  bb->AddFunction(func, std::string(method_name));
}

// ===========================================================================
// ExportToIRModule
// ===========================================================================

IRModule ExportToIRModule(ModuleSpec spec, bool debug) {
  const ModuleSpecNode* ms_node = spec.get();
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);

  if (debug) EmitInitializeEffect(bb);

  for (size_t i = 0; i < ms_node->method_names.size(); ++i) {
    const ffi::String& method_name = ms_node->method_names[i];
    const ffi::Any& method_spec_any = ms_node->method_specs[i];

    auto opt = method_spec_any.try_cast<MethodSpec>();
    TVM_FFI_ICHECK(opt.has_value())
        << "ExportToIRModule: method_spec[" << i << "] is not a MethodSpec, got "
        << method_spec_any.GetTypeKey();

    EmitMethod(bb, method_name, opt.value().get(), ms_node->named_params, debug);
  }

  IRModule mod = bb->Finalize();
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
           })
      .def("relax.frontend.nn.GetCurrentIOVar", GetCurrentIOVar)
      .def("relax.frontend.nn.SetCurrentIOVar", SetCurrentIOVar);
}

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
