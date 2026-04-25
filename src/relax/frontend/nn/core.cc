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
 * \file src/relax/frontend/nn/core.cc
 * \brief Native C++ implementations for nn frontend core types (Tensor,
 *        Parameter, NNObject) and global helpers (default dtype, WrapNested,
 *        MakePlaceholder, etc.).
 */

#include "core.h"

#include <tvm/ffi/reflection/accessor.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/tensor.h>

#include <sstream>
#include <string>

// ExportTVM and Jit delegate to ExportToIRModule and nn::Jit respectively.
// Include their headers here (after core.h to avoid circular dependency).
#include "cpp_module.h"
#include "exporter.h"
#include "modules.h"
#include "spec.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ===========================================================================
// Thread-local BlockBuilder tracker
// Provides BlockBuilder::Current() semantics for WrapNested / Emit helpers.
// exporter.cc installs the current BB via BBScope before calling forward().
// ===========================================================================

static thread_local BlockBuilder* g_current_bb = nullptr;  // NOLINT(*)

BlockBuilder BlockBuilder_Current() { return g_current_bb ? *g_current_bb : BlockBuilder(); }

void BlockBuilder_SetCurrent(BlockBuilder* bb) { g_current_bb = bb; }

// ===========================================================================
// Default dtype (thread-local so nested scopes can override independently)
// ===========================================================================

static thread_local std::string g_default_dtype = "float32";  // NOLINT(*)

ffi::String GetDefaultDtype() { return g_default_dtype; }
void SetDefaultDtype(ffi::String dtype) { g_default_dtype = std::string(dtype); }

// ===========================================================================
// Helper: build a ShapeExpr from a mixed Array<Any>
// ===========================================================================

static ShapeExpr BuildShapeExpr(const ffi::Array<ffi::Any>& shape) {
  ffi::Array<PrimExpr> dims;
  for (const ffi::Any& elem : shape) {
    if (auto opt = elem.try_cast<int64_t>()) {
      int64_t v = opt.value();
      TVM_FFI_ICHECK(v >= 0) << "Shape dimension must be non-negative, got " << v;
      dims.push_back(IntImm(DataType::Int(64), v));
    } else if (auto opt = elem.try_cast<ffi::String>()) {
      dims.push_back(tir::Var(opt.value(), DataType::Int(64)));
    } else if (auto opt = elem.try_cast<tir::Var>()) {
      TVM_FFI_ICHECK(opt.value()->dtype == DataType::Int(64))
          << "Symbolic shape var must have dtype int64";
      dims.push_back(opt.value());
    } else if (auto opt = elem.try_cast<PrimExpr>()) {
      TVM_FFI_ICHECK(opt.value()->dtype == DataType::Int(64))
          << "PrimExpr shape must have dtype int64";
      dims.push_back(opt.value());
    } else {
      TVM_FFI_THROW(TypeError) << "Invalid shape element type: " << elem.GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }
  }
  return ShapeExpr(dims);
}

// ===========================================================================
// TensorNode
// ===========================================================================

TensorNode::TensorNode(Var expr) : expr(std::move(expr)) {
  TVM_FFI_ICHECK(this->expr->struct_info_.defined()) << "TensorNode: Var must have struct_info set";
  TVM_FFI_ICHECK(this->expr->struct_info_->IsInstance<TensorStructInfoNode>())
      << "TensorNode: Var struct_info must be TensorStructInfo, got "
      << this->expr->struct_info_->GetTypeKey();
}

ffi::Array<PrimExpr> TensorNode::GetShape() const {
  const auto* sinfo = expr->struct_info_.as<TensorStructInfoNode>();
  TVM_FFI_ICHECK(sinfo && sinfo->shape.defined()) << "TensorNode::GetShape: shape is not available";
  const auto* shape_sinfo = sinfo->shape.value()->struct_info_.as<ShapeStructInfoNode>();
  TVM_FFI_ICHECK(shape_sinfo && shape_sinfo->values.defined())
      << "TensorNode::GetShape: shape values are not available";
  return shape_sinfo->values.value();
}

int64_t TensorNode::GetNdim() const {
  const auto* sinfo = expr->struct_info_.as<TensorStructInfoNode>();
  TVM_FFI_ICHECK(sinfo) << "TensorNode::GetNdim: missing TensorStructInfo";
  return static_cast<int64_t>(sinfo->ndim);
}

ffi::String TensorNode::GetDtype() const {
  const auto* sinfo = expr->struct_info_.as<TensorStructInfoNode>();
  TVM_FFI_ICHECK(sinfo) << "TensorNode::GetDtype: missing TensorStructInfo";
  return ffi::String(ffi::DLDataTypeToString(sinfo->dtype));
}

TensorNode* TensorNode::MakePlaceholder(ffi::Array<ffi::Any> shape, ffi::String dtype,
                                        ffi::String name) {
  ShapeExpr shape_expr = BuildShapeExpr(shape);
  Var v(name, TensorStructInfo(shape_expr, DataType(ffi::StringToDLDataType(dtype))));
  return new TensorNode(std::move(v));
}

TensorNode* TensorNode::MakeFromStructInfo(TensorStructInfo sinfo, ffi::String name) {
  return new TensorNode(Var(name, sinfo));
}

NNTensor::NNTensor(Var expr) { data_ = ffi::make_object<TensorNode>(std::move(expr)); }

// ===========================================================================
// ParameterNode
// ===========================================================================

ParameterNode::ParameterNode(Var expr, ffi::Optional<runtime::Tensor> data,
                             ffi::Map<ffi::String, ffi::Any> attrs)
    : TensorNode(std::move(expr)), data(std::move(data)), attrs(std::move(attrs)) {}

void ParameterNode::To(ffi::String new_dtype) {
  TVM_FFI_ICHECK(!data.has_value())
      << "ParameterNode::To: cannot change dtype of a bound parameter";
  // Re-create the placeholder Var with the new dtype
  ffi::Array<PrimExpr> old_shape = GetShape();
  ffi::Array<ffi::Any> shape_any;
  for (const PrimExpr& dim : old_shape) shape_any.push_back(ffi::Any(dim));
  ShapeExpr shape_expr = BuildShapeExpr(shape_any);
  expr = Var(expr->name_hint(),
             TensorStructInfo(shape_expr, DataType(ffi::StringToDLDataType(new_dtype))));
}

NNParameter::NNParameter(Var expr, ffi::Optional<runtime::Tensor> data,
                         ffi::Map<ffi::String, ffi::Any> attrs) {
  data_ = ffi::make_object<ParameterNode>(std::move(expr), std::move(data), std::move(attrs));
}

// ===========================================================================
// NNObjectNode
// ===========================================================================

NNObjectNode::NNObjectNode(Var expr) : expr(std::move(expr)) {
  TVM_FFI_ICHECK(this->expr->struct_info_.defined())
      << "NNObjectNode: Var must have struct_info set";
  TVM_FFI_ICHECK(this->expr->struct_info_->IsInstance<ObjectStructInfoNode>())
      << "NNObjectNode: Var struct_info must be ObjectStructInfo, got "
      << this->expr->struct_info_->GetTypeKey();
}

NNObject::NNObject(Var expr) { data_ = ffi::make_object<NNObjectNode>(std::move(expr)); }

// ===========================================================================
// ModuleListNode / ModuleDictNode
// ===========================================================================

ModuleList::ModuleList(ffi::Array<ffi::Any> modules) {
  data_ = ffi::make_object<ModuleListNode>(std::move(modules));
}

ModuleDict::ModuleDict(ffi::Map<ffi::String, ffi::Any> modules) {
  data_ = ffi::make_object<ModuleDictNode>(std::move(modules));
}

// ===========================================================================
// WrapNested
// ===========================================================================

ffi::Any WrapNested(Expr expr, ffi::String name) {
  BlockBuilder bb = BlockBuilder_Current();
  TVM_FFI_ICHECK(bb.defined()) << "WrapNested must be called inside a BlockBuilder scope";

  if (!expr->IsInstance<DataflowVarNode>()) {
    expr = bb->Emit(expr, name);
  }

  StructInfo sinfo = GetStructInfo(expr);

  if (sinfo->IsInstance<TensorStructInfoNode>()) {
    return ffi::Any(Downcast<Var>(expr));
  }

  if (const auto* ts = sinfo.as<TupleStructInfoNode>()) {
    ffi::Array<ffi::Any> results;
    for (int i = 0; i < static_cast<int>(ts->fields.size()); ++i) {
      results.push_back(
          WrapNested(TupleGetItem(expr, i), std::string(name) + "." + std::to_string(i)));
    }
    return ffi::Any(results);
  }

  TVM_FFI_THROW(TypeError) << "WrapNested: unsupported struct_info: " << sinfo->GetTypeKey();
  TVM_FFI_UNREACHABLE();
}

// ===========================================================================
// Global FFI helpers (MakePlaceholder, MakeTensorFromStructInfo, etc.)
// ===========================================================================

static Var FFIMakePlaceholder(ffi::Array<ffi::Any> shape, ffi::String dtype,
                              ffi::String name = "tensor") {
  ShapeExpr shape_expr = BuildShapeExpr(shape);
  return Var(name, TensorStructInfo(shape_expr, DataType(ffi::StringToDLDataType(dtype))));
}

static Var FFIMakeTensorFromStructInfo(TensorStructInfo sinfo, ffi::String name = "tensor") {
  return Var(name, sinfo);
}

static ffi::Optional<BlockBuilder> GetCurrentBlockBuilder() {
  BlockBuilder bb = BlockBuilder_Current();
  if (!bb.defined()) return std::nullopt;
  return bb;
}

// ---------------------------------------------------------------------------
// Internal helper: recursively collect NNParameters from an ffi::Any value.
// Mirrors Python's _attribute_finder logic:
//   Case 1: ModuleList / ModuleDict  -> GetContainerParameters
//   Case 2: native C++ Object        -> GetNativeParameters
//   Case 3: NNModuleNode             -> recurse into its attrs map
//   Case 4: NNParameter              -> yield directly
// ---------------------------------------------------------------------------
static void CollectParameters(const ffi::Any& val, const std::string& prefix,
                              ffi::Map<ffi::String, NNParameter>& out) {
  // Case 4: direct NNParameter
  if (auto opt = val.try_cast<NNParameter>()) {
    if (!prefix.empty()) out.Set(ffi::String(prefix), opt.value());
    return;
  }

  auto opt_ref = val.try_cast<runtime::ObjectRef>();
  if (!opt_ref.has_value() || !opt_ref.value().defined()) return;
  runtime::ObjectRef obj = opt_ref.value();

  // Case 1: ModuleList — iterate by index, recurse into each element
  if (const auto* list = obj.as<ModuleListNode>()) {
    for (int64_t i = 0; i < static_cast<int64_t>(list->modules.size()); ++i) {
      std::string child = prefix.empty() ? std::to_string(i) : prefix + "." + std::to_string(i);
      CollectParameters(list->modules[i], child, out);
    }
    return;
  }

  // Case 1b: ModuleDict — iterate by key, recurse into each element
  if (const auto* dict = obj.as<ModuleDictNode>()) {
    for (const auto& [k, v] : dict->modules) {
      std::string child = prefix.empty() ? std::string(k) : prefix + "." + std::string(k);
      CollectParameters(v, child, out);
    }
    return;
  }

  // Case 3: NNModuleNode (or any subclass) — recurse into its attrs map.
  // Skip EffectNode subclasses: effects have no trainable parameters.
  if (const auto* mod = obj.as<NNModuleNode>()) {
    if (obj.as<EffectNode>()) return;  // Effects carry no parameters
    for (const auto& [fname, fval] : mod->attrs) {
      std::string child = prefix.empty() ? std::string(fname) : prefix + "." + std::string(fname);
      CollectParameters(fval, child, out);
    }
    return;
  }

  // Case 2: any other native C++ Object — inspect its registered fields
  auto params = GetNativeParameters(obj);
  for (const auto& [fname, param] : params) {
    std::string full = prefix.empty() ? std::string(fname) : prefix + "." + std::string(fname);
    out.Set(ffi::String(full), param);
  }
}

// ===========================================================================
// GetNativeParameters
// ===========================================================================

ffi::Map<ffi::String, NNParameter> GetNativeParameters(runtime::ObjectRef obj) {
  ffi::Map<ffi::String, NNParameter> result;
  if (!obj.defined()) return result;

  // If the object is an NNModuleNode (or subclass), use NamedParameters
  // which walks attrs — populated by Make* factories via PopulateAttrs.
  if (const auto* mod = obj.as<NNModuleNode>()) {
    return mod->NamedParameters(ffi::String(""));
  }

  int32_t type_index = obj->type_index();
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
  if (!type_info) return result;

  ffi::reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* field_info) {
    ffi::String field_name(field_info->name.data, field_info->name.size);
    ffi::reflection::FieldGetter getter(field_info);
    ffi::Any val = getter(obj);

    // Case 1: field is directly a ParameterNode
    if (auto opt = val.try_cast<NNParameter>()) {
      result.Set(field_name, opt.value());
      return;
    }
    // Case 2: field is Optional<NNParameter> stored as ObjectRef
    if (auto opt_ref = val.try_cast<runtime::ObjectRef>()) {
      if (opt_ref.value().defined() && opt_ref.value()->IsInstance<ParameterNode>()) {
        result.Set(field_name, Downcast<NNParameter>(opt_ref.value()));
      }
    }
  });
  return result;
}

// ===========================================================================
// GetContainerParameters / ContainerApplyTo
// ===========================================================================

/*!
 * \brief Recursively collect NNParameters from a ModuleList or ModuleDict.
 * Each element is an ffi::Any that may be:
 *   - A native C++ module object  -> call GetNativeParameters
 *   - A ModuleListNode            -> recurse
 *   - A ModuleDictNode            -> recurse
 *   - Anything else               -> skip (Python-side Module handled in Python)
 */
ffi::Map<ffi::String, NNParameter> GetContainerParameters(runtime::ObjectRef container,
                                                          ffi::String prefix) {
  ffi::Map<ffi::String, NNParameter> result;
  if (!container.defined()) return result;
  // Strip trailing '.' from prefix if present (Python _attribute_finder adds it)
  std::string pfx = std::string(prefix);
  if (!pfx.empty() && pfx.back() == '.') pfx.pop_back();
  // Delegate entirely to CollectParameters which handles ModuleListNode,
  // ModuleDictNode, NNModuleNode subclasses, and native C++ objects uniformly.
  CollectParameters(ffi::Any(container), pfx, result);
  return result;
}

void ContainerApplyTo(runtime::ObjectRef container, ffi::String dtype) {
  if (!container.defined()) return;
  // Walk via NNModuleNode::To if it is one, otherwise fall back to
  // GetNativeParameters for plain native objects.
  if (const auto* mod = container.as<NNModuleNode>()) {
    mod->To(dtype);
    return;
  }
  auto params = GetNativeParameters(container);
  for (const auto& [_, param] : params) param->To(dtype);
}

/*!
 * \brief Apply dtype conversion to all Parameters in a Python module's __dict__.
 * \param py_dict  Python dict (from module.__dict__) as ffi::Map<String, Any>.
 * \param dtype    Target dtype string.
 *
 * Recursively walks the dict, calling To() on Parameters, ModuleLists, ModuleDicts,
 * and any native C++ modules.
 */
void PythonModuleApplyTo(ffi::Map<ffi::String, ffi::Any> py_dict, ffi::String dtype) {
  for (const auto& [name, val] : py_dict) {
    // Case 1: Parameter -> call To()
    if (auto opt = val.try_cast<NNParameter>()) {
      opt.value()->To(dtype);
      continue;
    }
    // Case 2: ModuleList / ModuleDict -> delegate to ContainerApplyTo
    auto opt_ref = val.try_cast<runtime::ObjectRef>();
    if (!opt_ref.has_value() || !opt_ref.value().defined()) continue;
    runtime::ObjectRef obj = opt_ref.value();
    if (obj->IsInstance<ModuleListNode>() || obj->IsInstance<ModuleDictNode>()) {
      ContainerApplyTo(obj, dtype);
      continue;
    }
    // Case 3: NNModuleNode subclass -> call To()
    if (const auto* mod = obj.as<NNModuleNode>()) {
      mod->To(dtype);
      continue;
    }
    // Case 4: nested dict (pure-Python sub-module) -> recurse
    if (auto dict_opt = val.try_cast<ffi::Map<ffi::String, ffi::Any>>()) {
      PythonModuleApplyTo(dict_opt.value(), dtype);
      continue;
    }
  }
}

// ===========================================================================
// NNModuleNode
// ===========================================================================

ffi::Map<ffi::String, NNParameter> NNModuleNode::NamedParameters(ffi::String prefix) const {
  ffi::Map<ffi::String, NNParameter> result;
  for (const auto& [name, val] : attrs) {
    std::string child =
        prefix.empty() ? std::string(name) : std::string(prefix) + "." + std::string(name);
    CollectParameters(val, child, result);
  }
  return result;
}

ffi::Map<ffi::String, NNParameter> NNModuleNode::StateDict(ffi::String prefix) const {
  return NamedParameters(prefix);
}

ffi::Array<ffi::Array<ffi::String>> NNModuleNode::LoadStateDict(
    ffi::Map<ffi::String, NNParameter> state_dict, bool strict) const {
  // Build current state dict
  ffi::Map<ffi::String, NNParameter> self_sd = StateDict(ffi::String(""));

  ffi::Array<ffi::String> missing;
  ffi::Array<ffi::String> unexpected;

  for (const auto& [key, value] : state_dict) {
    if (!self_sd.count(key)) {
      unexpected.push_back(key);
      continue;
    }
    TVM_FFI_ICHECK(value->data.has_value())
        << "LoadStateDict: parameter '" << key << "' has no concrete data";
    self_sd.at(key)->data = value->data;
    self_sd.erase(key);
  }

  for (const auto& [key, _] : self_sd) missing.push_back(key);

  if (strict && (missing.size() > 0 || unexpected.size() > 0)) {
    std::ostringstream oss;
    oss << "LoadStateDict: Missing keys: [";
    for (size_t i = 0; i < missing.size(); ++i) {
      if (i) oss << ", ";
      oss << missing[i];
    }
    oss << "]  Unexpected keys: [";
    for (size_t i = 0; i < unexpected.size(); ++i) {
      if (i) oss << ", ";
      oss << unexpected[i];
    }
    oss << "]";
    TVM_FFI_THROW(KeyError) << oss.str();
  }

  return {missing, unexpected};
}

void NNModuleNode::To(ffi::String dtype) const {
  for (const auto& [name, val] : attrs) {
    // NNParameter: call To() directly
    if (auto opt = val.try_cast<NNParameter>()) {
      opt.value()->To(dtype);
      continue;
    }
    auto opt_ref = val.try_cast<runtime::ObjectRef>();
    if (!opt_ref.has_value() || !opt_ref.value().defined()) continue;
    runtime::ObjectRef obj = opt_ref.value();

    // ModuleList / ModuleDict: delegate to ContainerApplyTo
    if (obj->IsInstance<ModuleListNode>() || obj->IsInstance<ModuleDictNode>()) {
      ContainerApplyTo(obj, dtype);
      continue;
    }
    // Nested NNModuleNode: recurse
    if (const auto* mod = obj.as<NNModuleNode>()) {
      mod->To(dtype);
      continue;
    }
    // Other native C++ object: apply to() on its parameters
    auto params = GetNativeParameters(obj);
    for (const auto& [_, param] : params) param->To(dtype);
  }
}

// ===========================================================================
// NNModuleNode::ExportTVM / Jit
// ===========================================================================

ffi::Array<ffi::Any> NNModuleNode::ExportTVM(ModuleSpec spec, bool debug, bool allow_extern) const {
  // Use the Exporter class to build the IRModule and collect extern_mods.
  // This matches the Python implementation exactly.
  Exporter exporter(debug);
  ffi::Array<ffi::Any> result = exporter->Build(std::move(spec));
  // result = [mod, named_params, extern_mods]
  // If allow_extern=false, return [mod, named_params, empty_array]
  if (!allow_extern) {
    result.Set(2, ffi::Any(ffi::Array<runtime::ObjectRef>{}));
  }
  return result;
}

runtime::ObjectRef NNModuleNode::Jit(ModuleSpec spec, tvm::Device device, ffi::String pipeline,
                                     bool debug) const {
  return nn::Jit(std::move(spec), device, std::move(pipeline), debug);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  // Register Object types
  TensorNode::RegisterReflection();
  ParameterNode::RegisterReflection();
  NNObjectNode::RegisterReflection();
  ModuleListNode::RegisterReflection();
  ModuleDictNode::RegisterReflection();
  NNModuleNode::RegisterReflection();

  // Register global helpers
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      // Default dtype
      .def("relax.frontend.nn.GetDefaultDtype", GetDefaultDtype)
      .def("relax.frontend.nn.SetDefaultDtype", SetDefaultDtype)
      // Tensor construction
      .def("relax.frontend.nn.MakePlaceholder", FFIMakePlaceholder)
      .def("relax.frontend.nn.MakeTensorFromStructInfo", FFIMakeTensorFromStructInfo)
      // WrapNested
      .def("relax.frontend.nn.WrapNested", WrapNested)
      // BlockBuilder accessor
      .def("relax.frontend.nn.GetCurrentBlockBuilder", GetCurrentBlockBuilder)
      // BlockBuilder install/restore for Python exporter
      .def("relax.frontend.nn.SetCurrentBlockBuilder",
           [](ffi::Optional<BlockBuilder> bb) {
             static thread_local BlockBuilder t_bb;
             if (bb.defined()) {
               t_bb = bb.value();
               BlockBuilder_SetCurrent(&t_bb);
             } else {
               BlockBuilder_SetCurrent(nullptr);
             }
           })
      // Parameter discovery for named_parameters() traversal
      .def("relax.frontend.nn.GetNativeParameters", GetNativeParameters)
      // Container traversal helpers
      .def("relax.frontend.nn.GetContainerParameters", GetContainerParameters)
      .def("relax.frontend.nn.ContainerApplyTo", ContainerApplyTo)
      .def("relax.frontend.nn.PythonModuleApplyTo", PythonModuleApplyTo)
      // NNModule construction
      .def("relax.frontend.nn.MakeModule",
           [](ffi::Map<ffi::String, ffi::Any> attrs) { return NNModule(std::move(attrs)); });
}

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
