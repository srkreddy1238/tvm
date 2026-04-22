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
// Thread-local Exporter tracker  (mirrors BlockBuilder_Current / _SetCurrent)
// ExporterScope installs the current Exporter before calling forward() so
// that nn.add_extern() called from Python forward() can find it via
// Exporter_Current().
// ===========================================================================

static thread_local ExporterNode* g_current_exporter = nullptr;  // NOLINT(*)

ffi::Optional<Exporter> Exporter_Current() {
  if (!g_current_exporter) return std::nullopt;
  return ffi::details::ObjectUnsafe::ObjectRefFromObjectPtr<Exporter>(
      ffi::details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(g_current_exporter));
}

void Exporter_SetCurrent(ExporterNode* exporter) { g_current_exporter = exporter; }

// RAII scope that installs/restores the current Exporter
struct ExporterScope {
  ExporterNode* prev;
  explicit ExporterScope(ExporterNode* e) : prev(g_current_exporter) { g_current_exporter = e; }
  ~ExporterScope() { g_current_exporter = prev; }
};

// ===========================================================================
// BBScope: RAII helper that installs/restores the thread-local BlockBuilder
// so that WrapNested / Emit helpers in core.cc and op.cc can find it.
// ===========================================================================

struct BBScope {
  ffi::Optional<BlockBuilder> prev;
  explicit BBScope(BlockBuilder& bb) {
    // Save whatever was current and install the new one
    BlockBuilder cur = BlockBuilder_Current();
    prev = cur.defined() ? ffi::Optional<BlockBuilder>(cur) : std::nullopt;
    BlockBuilder_SetCurrent(&bb);
  }
  ~BBScope() {
    if (prev.has_value()) {
      // Restore previous BB: we need a stable address for the duration of
      // BlockBuilder_SetCurrent, so store it in a local and point to that.
      static thread_local BlockBuilder t_prev;  // NOLINT(*)
      t_prev = prev.value();
      BlockBuilder_SetCurrent(&t_prev);
    } else {
      BlockBuilder_SetCurrent(nullptr);
    }
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
//   def _initialize_effect() -> R.Tuple(R.Object, ...):
//       with R.dataflow():
//           effect1 = effect1.emit_init("effect1", bb)
//           effect2 = effect2.emit_init("effect2", bb)
//           ...
//           lv   = (effect1, effect2, ...)
//           gv   = lv          # output
//       return gv
//
// Calls emit_init() on each Effect in named_effects.
// ===========================================================================

static void EmitInitializeEffect(BlockBuilder& bb,
                                 const ffi::Map<ffi::String, runtime::ObjectRef>& named_effects,
                                 bool debug) {
  ffi::Array<Var> params;  // no parameters

  bb->BeginScope(params);
  bb->BeginDataflowBlock();

  // Set current BB so that relax ops called from here can use BlockBuilder::Current()
  BBScope bb_scope(bb);

  ffi::Array<Expr> effect_vars;

  // If debug=true, always emit _io = null_value() first
  if (debug) {
    static const Op& null_value_op = Op::Get("relax.null_value");
    Var io = bb->Emit(Call(null_value_op, {}, {}, {}), "_io");
    effect_vars.push_back(io);
  }

  // Then call emit_init() on each Effect in named_effects
  for (const auto& [name, effect_obj] : named_effects) {
    // Call effect._cpp_emit_init(name, bb) via the registered method.
    // Methods registered with .def("_cpp_emit_init", &Class::Method) are
    // callable via ffi::Function::GetMethod.
    std::string type_key = effect_obj->GetTypeKey();

    // Get the method function from the type's method table
    auto type_index = effect_obj->type_index();
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
    if (!type_info || !type_info->methods) {
      TVM_FFI_THROW(TypeError) << "Effect object has no type info: " << type_key;
      TVM_FFI_UNREACHABLE();
    }

    // Find the "_cpp_emit_init" method in the methods array
    const TVMFFIMethodInfo* method_info = nullptr;
    for (int32_t i = 0; i < type_info->num_methods; ++i) {
      const TVMFFIMethodInfo* m = &type_info->methods[i];
      if (std::string(m->name.data, m->name.size) == "_cpp_emit_init") {
        method_info = m;
        break;
      }
    }

    if (!method_info) {
      TVM_FFI_THROW(TypeError) << "Effect object has no _cpp_emit_init method: " << type_key;
      TVM_FFI_UNREACHABLE();
    }

    // Call the method: _cpp_emit_init(effect_obj, name, bb) -> Array<Var>
    // method_info->method is a TVMFFIAny containing the ffi::Function
    ffi::AnyView method_any = ffi::AnyView::CopyFromTVMFFIAny(method_info->method);
    auto method_func_opt = method_any.try_cast<ffi::Function>();
    TVM_FFI_ICHECK(method_func_opt.has_value()) << "Method is not an ffi::Function";
    ffi::Function method_func = method_func_opt.value();
    ffi::Any result = method_func(effect_obj, ffi::Any(name), ffi::Any(bb));
    auto vars_opt = result.try_cast<ffi::Array<Var>>();
    TVM_FFI_ICHECK(vars_opt.has_value()) << "Effect._cpp_emit_init must return Array<Var>";
    for (const Var& v : vars_opt.value()) effect_vars.push_back(v);
  }

  Var lv = bb->Emit(relax::Tuple(effect_vars), "lv");
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

/*!
 * \brief Recursively build StructInfo for a SpecTuple.
 */
static StructInfo BuildSpecTupleStructInfo(const SpecTupleNode* st,
                                           std::unordered_map<std::string, tir::Var>& str2var) {
  ffi::Array<StructInfo> fields;
  for (const ffi::Any& elem : st->elements) {
    auto opt = elem.try_cast<runtime::ObjectRef>();
    TVM_FFI_ICHECK(opt.has_value()) << "SpecTuple element is not an ObjectRef";
    runtime::ObjectRef e = opt.value();
    if (e->IsInstance<SpecTensorNode>()) {
      const auto* s = e.as<SpecTensorNode>();
      ShapeExpr shape = BuildSpecShape(s->shape, str2var);
      fields.push_back(TensorStructInfo(shape, DataType(ffi::StringToDLDataType(s->dtype))));
    } else if (e->IsInstance<SpecTupleNode>()) {
      fields.push_back(BuildSpecTupleStructInfo(e.as<SpecTupleNode>(), str2var));
    } else {
      TVM_FFI_THROW(TypeError) << "SpecTuple element has unsupported type: " << e->GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }
  }
  return TupleStructInfo(fields);
}

/*!
 * \brief Recursively build the ffi::Any input value for a SpecTuple.
 *
 * Returns an ffi::Array<ffi::Any> (a Python list/tuple of NNTensors or
 * nested arrays) that forward() receives as its tuple argument.
 * Also emits TupleGetItem bindings into bb for each leaf tensor.
 */
static ffi::Any BuildSpecTupleInput(const SpecTupleNode* st, Var tuple_var, BlockBuilder& bb) {
  ffi::Array<ffi::Any> result;
  for (size_t i = 0; i < st->elements.size(); ++i) {
    auto opt = st->elements[i].try_cast<runtime::ObjectRef>();
    TVM_FFI_ICHECK(opt.has_value());
    runtime::ObjectRef e = opt.value();
    // Emit TupleGetItem to extract element i from the tuple var
    Var elem_var = bb->Emit(TupleGetItem(tuple_var, static_cast<int>(i)),
                            std::string(st->name) + "_" + std::to_string(i));
    if (e->IsInstance<SpecTensorNode>()) {
      result.push_back(ffi::Any(NNTensor(elem_var)));
    } else if (e->IsInstance<SpecTupleNode>()) {
      result.push_back(BuildSpecTupleInput(e.as<SpecTupleNode>(), elem_var, bb));
    } else {
      TVM_FFI_THROW(TypeError) << "SpecTuple element has unsupported type: " << e->GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }
  }
  return ffi::Any(result);
}

static void EmitMethod(BlockBuilder& bb, const ffi::String& method_name, const MethodSpecNode* ms,
                       const ffi::Map<ffi::String, NNParameter>& named_params,
                       const ffi::Map<ffi::String, runtime::ObjectRef>& named_effects, bool debug) {
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
        // Pass the tir::Var to forward() (matches Python exporter behavior)
        explicit_inputs.push_back(ffi::Any(tvar));

      } else if (spec_obj->IsInstance<SpecTupleNode>()) {
        const auto* st = spec_obj.as<SpecTupleNode>();
        StructInfo sinfo = BuildSpecTupleStructInfo(st, str2var);
        Var v(std::string(arg_name), sinfo);
        input_vars.push_back(v);
        // Defer TupleGetItem extraction to after BeginDataflowBlock.
        // Store a sentinel: the SpecTuple ObjectRef paired with the Var.
        // We use a 2-element Array<Any>: [SpecTuple, Var].
        ffi::Array<ffi::Any> deferred{ffi::Any(spec_obj), ffi::Any(v)};
        explicit_inputs.push_back(ffi::Any(deferred));

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

  // ---- 4. Build effect Vars and call create/set_state -------------------
  // For each Effect in named_effects:
  //   1. Call effect._cpp_create(name) -> Array<Var> (effect state vars)
  //   2. Add those Vars to func_params (if effect_mode != "none")
  //   3. Call effect._cpp_set_state(state_vars) to initialize the Effect
  //
  // Special case: if debug=true and effect_mode != "none", ALWAYS add a
  // legacy _io effect var as the FIRST effect (for backward compatibility),
  // followed by any Effects from named_effects.
  std::string effect_mode_str = std::string(ms->effect_mode);
  std::vector<std::pair<ffi::String, runtime::ObjectRef>> effects_vec;
  std::vector<ffi::Array<Var>> effect_state_vars;  // per-effect state vars
  ffi::Array<Var> all_effect_vars;                 // flattened list for func params
  bool use_legacy_io = (effect_mode_str != "none") && debug;

  if (use_legacy_io) {
    // Always add legacy _io as first effect when debug=true
    Var io_var("_io", ObjectStructInfo());
    all_effect_vars.push_back(io_var);
  }

  // Then add Effects from named_effects (if any)
  for (const auto& [name, effect_obj] : named_effects) {
    effects_vec.emplace_back(name, effect_obj);

    // Call effect._cpp_create(name) -> Array<Var>
    auto type_index = effect_obj->type_index();
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
    if (!type_info || !type_info->methods) {
      TVM_FFI_THROW(TypeError) << "Effect has no type info: " << effect_obj->GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }

    // Find _cpp_create method
    const TVMFFIMethodInfo* create_method = nullptr;
    for (int32_t i = 0; i < type_info->num_methods; ++i) {
      const TVMFFIMethodInfo* m = &type_info->methods[i];
      if (std::string(m->name.data, m->name.size) == "_cpp_create") {
        create_method = m;
        break;
      }
    }
    if (!create_method) {
      TVM_FFI_THROW(TypeError) << "Effect has no _cpp_create method: " << effect_obj->GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }

    ffi::AnyView create_any = ffi::AnyView::CopyFromTVMFFIAny(create_method->method);
    auto create_func_opt = create_any.try_cast<ffi::Function>();
    TVM_FFI_ICHECK(create_func_opt.has_value());
    ffi::Function create_func = create_func_opt.value();
    ffi::Any create_result = create_func(effect_obj, ffi::Any(name));
    auto state_vars_opt = create_result.try_cast<ffi::Array<Var>>();
    TVM_FFI_ICHECK(state_vars_opt.has_value()) << "Effect._cpp_create must return Array<Var>";
    ffi::Array<Var> state_vars = state_vars_opt.value();
    effect_state_vars.push_back(state_vars);
    for (const Var& v : state_vars) all_effect_vars.push_back(v);
  }

  // ---- 5. Build param Vars -----------------------------------------------
  // param_mode: "plain" = individual params, "packed" = single R.Tuple param,
  //             "none"  = no params appended
  std::string param_mode_str = std::string(ms->param_mode);

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

  // ---- 7. Assemble full function parameter list --------------------------
  ffi::Array<Var> func_params;
  for (const Var& v : input_vars) func_params.push_back(v);
  // Add effect state vars if effect_mode != "none"
  if (effect_mode_str != "none") {
    for (const Var& v : all_effect_vars) func_params.push_back(v);
  }
  if (param_mode_str == "plain") {
    for (const Var& v : param_vars) func_params.push_back(v);
  } else if (param_mode_str == "packed") {
    func_params.push_back(packed_params_var.value());
  }

  int64_t num_input =
      static_cast<int64_t>(input_vars.size()) +
      (effect_mode_str != "none" ? static_cast<int64_t>(all_effect_vars.size()) : 0);

  // ---- 8. Build the function body ----------------------------------------
  bb->BeginScope(func_params);
  bb->BeginDataflowBlock();

  // Install current BB so WrapNested / Emit helpers work
  BBScope bb_scope(bb);
  // Install current _io var (first effect var if any) so debug_func can retrieve it
  ffi::Optional<Var> io_var_for_debug;
  if (!all_effect_vars.empty()) io_var_for_debug = all_effect_vars[0];
  IOVarScope io_scope(io_var_for_debug);

  // Call effect._cpp_set_state(state_vars) for each Effect
  for (size_t ei = 0; ei < effects_vec.size(); ++ei) {
    const auto& [name, effect_obj] = effects_vec[ei];
    const ffi::Array<Var>& state_vars = effect_state_vars[ei];

    auto type_index = effect_obj->type_index();
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
    const TVMFFIMethodInfo* set_state_method = nullptr;
    for (int32_t i = 0; i < type_info->num_methods; ++i) {
      const TVMFFIMethodInfo* m = &type_info->methods[i];
      if (std::string(m->name.data, m->name.size) == "_cpp_set_state") {
        set_state_method = m;
        break;
      }
    }
    if (!set_state_method) {
      TVM_FFI_THROW(TypeError) << "Effect has no _cpp_set_state method: "
                               << effect_obj->GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }

    ffi::AnyView set_state_any = ffi::AnyView::CopyFromTVMFFIAny(set_state_method->method);
    auto set_state_func_opt = set_state_any.try_cast<ffi::Function>();
    TVM_FFI_ICHECK(set_state_func_opt.has_value());
    ffi::Function set_state_func = set_state_func_opt.value();
    set_state_func(effect_obj, ffi::Any(state_vars));
  }

  // For packed params: emit TupleGetItem bindings now that we are in the dataflow block
  if (param_mode_str == "packed" && packed_params_var.defined()) {
    for (size_t pi = 0; pi < params_vec.size(); ++pi) {
      auto& [name, param] = params_vec[pi];
      Expr get_item = TupleGetItem(packed_params_var.value(), static_cast<int>(pi));
      Var bound = bb->Emit(get_item, std::string(name));
      param->expr = bound;
    }
  }

  // Resolve deferred SpecTuple inputs: emit TupleGetItem extractions now
  // that we are inside the dataflow block.
  for (size_t i = 0; i < explicit_inputs.size(); ++i) {
    if (auto arr_opt = explicit_inputs[i].try_cast<ffi::Array<ffi::Any>>()) {
      ffi::Array<ffi::Any> deferred = arr_opt.value();
      // Sentinel: [SpecTuple ObjectRef, tuple Var]
      if (deferred.size() == 2) {
        if (auto st_opt = deferred[0].try_cast<SpecTuple>()) {
          if (auto v_opt = deferred[1].try_cast<Var>()) {
            explicit_inputs[i] = BuildSpecTupleInput(st_opt.value().get(), v_opt.value(), bb);
          }
        }
      }
    }
  }

  // Build named_args map for forward()
  ffi::Map<ffi::String, ffi::Any> named_args;
  for (size_t i = 0; i < ms->arg_names.size(); ++i) {
    named_args.Set(ms->arg_names[i], explicit_inputs[i]);
  }

  // Call the forward function — if it throws, close the open BB blocks first
  // to avoid the "BlockBuilder destroyed with remaining blocks" warning and
  // the dangling-pointer segfault in the next test.
  ffi::Any raw_out;
  try {
    raw_out = ms->forward(named_args);
  } catch (...) {
    // Close the dataflow block and function scope before propagating.
    try {
      bb->EndBlock();
    } catch (...) {
    }
    try {
      bb->EndScope();
    } catch (...) {
    }
    throw;
  }
  Expr out_expr = UnwrapReturn(raw_out);

  // Build effect output vars:
  // 1. If use_legacy_io: return the (possibly updated) _io var
  // 2. Then call _cpp_finalize() for each Effect
  ffi::Array<Expr> effect_output_vars;
  if (use_legacy_io) {
    // Legacy: return the _io var (use updated value from g_current_io_var if available)
    Var final_io = g_current_io_var.defined() ? g_current_io_var.value() : all_effect_vars[0];
    effect_output_vars.push_back(final_io);
  }

  // Call finalize() for each Effect
  for (const auto& [name, effect_obj] : effects_vec) {
    auto type_index = effect_obj->type_index();
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
    const TVMFFIMethodInfo* finalize_method = nullptr;
    for (int32_t i = 0; i < type_info->num_methods; ++i) {
      const TVMFFIMethodInfo* m = &type_info->methods[i];
      if (std::string(m->name.data, m->name.size) == "_cpp_finalize") {
        finalize_method = m;
        break;
      }
    }
    if (!finalize_method) {
      TVM_FFI_THROW(TypeError) << "Effect has no _cpp_finalize method: "
                               << effect_obj->GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }

    ffi::AnyView finalize_any = ffi::AnyView::CopyFromTVMFFIAny(finalize_method->method);
    auto finalize_func_opt = finalize_any.try_cast<ffi::Function>();
    TVM_FFI_ICHECK(finalize_func_opt.has_value());
    ffi::Function finalize_func = finalize_func_opt.value();
    ffi::Any finalize_result = finalize_func(effect_obj);
    auto finalized_vars_opt = finalize_result.try_cast<ffi::Array<Var>>();
    TVM_FFI_ICHECK(finalized_vars_opt.has_value()) << "Effect._cpp_finalize must return Array<Var>";
    for (const Var& v : finalized_vars_opt.value()) effect_output_vars.push_back(v);
  }

  // Wrap effect outputs if effect_mode != "none" and we have effects
  Expr final_out;
  if (effect_mode_str != "none" && !effect_output_vars.empty()) {
    Expr effect_tuple = relax::Tuple(effect_output_vars);
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

// ===========================================================================
// Exporter class - implementation
// ===========================================================================

ExporterNode::ExporterNode(bool debug)
    : builder(BlockBuilder::Create(std::nullopt)), debug(debug) {}

void ExporterNode::AddExternalModule(runtime::ObjectRef extern_mod) {
  // Check for duplicate symbols (Python-side validation)
  extern_mods.push_back(extern_mod);
}

ffi::Array<ffi::Any> ExporterNode::Build(ModuleSpec spec) {
  // Install this Exporter as the thread-local current so that
  // nn.add_extern() called from Python forward() can find it.
  ExporterScope exporter_scope(this);

  // Delegate to ExportToIRModule
  IRModule mod = ExportToIRModule(spec, debug);

  // Return [mod, named_params, extern_mods] as Array<Any>
  ffi::Array<ffi::Any> result;
  result.push_back(ffi::Any(mod));
  result.push_back(ffi::Any(spec.get()->named_params));
  result.push_back(ffi::Any(extern_mods));
  return result;
}

void ExporterNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ExporterNode>()
      .def(refl::init<bool>())
      .def_rw("builder", &ExporterNode::builder)
      .def_rw("debug", &ExporterNode::debug)
      .def_rw("extern_mods", &ExporterNode::extern_mods)
      .def("add_external_module", &ExporterNode::AddExternalModule)
      .def("_build_cpp", &ExporterNode::Build);  // Use _build_cpp to avoid name collision
}

Exporter::Exporter(bool debug) { data_ = ffi::make_object<ExporterNode>(debug); }

// ===========================================================================
// ExportToIRModule
// ===========================================================================

IRModule ExportToIRModule(ModuleSpec spec, bool debug) {
  const ModuleSpecNode* ms_node = spec.get();
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);

  if (debug) {
    EmitInitializeEffect(bb, ms_node->named_effects, debug);
  }

  for (size_t i = 0; i < ms_node->method_names.size(); ++i) {
    const ffi::String& method_name = ms_node->method_names[i];
    const ffi::Any& method_spec_any = ms_node->method_specs[i];

    auto opt = method_spec_any.try_cast<MethodSpec>();
    TVM_FFI_ICHECK(opt.has_value())
        << "ExportToIRModule: method_spec[" << i << "] is not a MethodSpec, got "
        << method_spec_any.GetTypeKey();

    EmitMethod(bb, method_name, opt.value().get(), ms_node->named_params, ms_node->named_effects,
               debug);
  }

  IRModule mod = bb->Finalize();
  return mod;
}

// ===========================================================================
// FFI registration
// ===========================================================================

TVM_FFI_STATIC_INIT_BLOCK() {
  ExporterNode::RegisterReflection();

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.frontend.nn.ExportToIRModule",
           [](ModuleSpec spec, bool debug) -> IRModule {
             return ExportToIRModule(std::move(spec), debug);
           })
      .def("relax.frontend.nn.GetCurrentIOVar", GetCurrentIOVar)

      .def("relax.frontend.nn.SetCurrentIOVar", SetCurrentIOVar)
      .def("relax.frontend.nn.Exporter", [](bool debug) { return Exporter(debug); })

      // Thread-local Exporter accessor (mirrors BlockBuilder_Current/_SetCurrent)
      .def("relax.frontend.nn.GetCurrentExporter",
           []() -> ffi::Optional<Exporter> { return Exporter_Current(); })
      .def("relax.frontend.nn.SetCurrentExporter",
           [](ffi::Optional<Exporter> e) {
             static thread_local ExporterNode* t_exporter = nullptr;
             if (e.has_value() && e.value().defined()) {
               t_exporter = e.value().get();
               Exporter_SetCurrent(t_exporter);
             } else {
               Exporter_SetCurrent(nullptr);
             }
           })
      // nn.add_extern: register an ExternModule with the current Exporter
      .def("relax.frontend.nn.AddExtern", [](runtime::ObjectRef extern_mod) {
        auto e = Exporter_Current();
        TVM_FFI_ICHECK(e.has_value())
            << "`nn.add_extern` must be called during export_tvm (no active Exporter).";
        e.value()->AddExternalModule(extern_mod);
      });
}

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
