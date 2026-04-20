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

#include <string>

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

// ===========================================================================
// GetNativeParameters
// ===========================================================================

ffi::Map<ffi::String, NNParameter> GetNativeParameters(runtime::ObjectRef obj) {
  ffi::Map<ffi::String, NNParameter> result;
  if (!obj.defined()) return result;

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

  auto collect = [&](const ffi::String& key, const ffi::Any& elem) {
    std::string child_prefix =
        prefix.empty() ? std::string(key) : std::string(prefix) + "." + std::string(key);

    if (auto opt = elem.try_cast<runtime::ObjectRef>()) {
      runtime::ObjectRef child = opt.value();
      if (!child.defined()) return;

      if (child->IsInstance<ModuleListNode>()) {
        auto sub = GetContainerParameters(child, ffi::String(child_prefix));
        for (const auto& [k, v] : sub) result.Set(k, v);
      } else if (child->IsInstance<ModuleDictNode>()) {
        auto sub = GetContainerParameters(child, ffi::String(child_prefix));
        for (const auto& [k, v] : sub) result.Set(k, v);
      } else {
        // Native C++ module: inspect its fields
        auto params = GetNativeParameters(child);
        for (const auto& [fname, param] : params) {
          std::string full = child_prefix + "." + std::string(fname);
          result.Set(ffi::String(full), param);
        }
      }
    }
  };

  if (container->IsInstance<ModuleListNode>()) {
    const auto* node = container.as<ModuleListNode>();
    for (int64_t i = 0; i < static_cast<int64_t>(node->modules.size()); ++i) {
      collect(ffi::String(std::to_string(i)), node->modules[i]);
    }
  } else if (container->IsInstance<ModuleDictNode>()) {
    const auto* node = container.as<ModuleDictNode>();
    for (const auto& [k, v] : node->modules) {
      collect(k, v);
    }
  }
  return result;
}

void ContainerApplyTo(runtime::ObjectRef container, ffi::String dtype) {
  if (!container.defined()) return;

  auto apply_elem = [&](const ffi::Any& elem) {
    if (auto opt = elem.try_cast<runtime::ObjectRef>()) {
      runtime::ObjectRef child = opt.value();
      if (!child.defined()) return;
      if (child->IsInstance<ModuleListNode>() || child->IsInstance<ModuleDictNode>()) {
        ContainerApplyTo(child, dtype);
      } else {
        // Native C++ module: call to() on each parameter
        auto params = GetNativeParameters(child);
        for (const auto& [_, param] : params) {
          param->To(dtype);
        }
      }
    }
  };

  if (container->IsInstance<ModuleListNode>()) {
    const auto* node = container.as<ModuleListNode>();
    for (const auto& elem : node->modules) apply_elem(elem);
  } else if (container->IsInstance<ModuleDictNode>()) {
    const auto* node = container.as<ModuleDictNode>();
    for (const auto& [_, v] : node->modules) apply_elem(v);
  }
}

// ===========================================================================
// Registration
// ===========================================================================

TVM_FFI_STATIC_INIT_BLOCK() {
  // Register Object types
  TensorNode::RegisterReflection();
  ParameterNode::RegisterReflection();
  NNObjectNode::RegisterReflection();
  ModuleListNode::RegisterReflection();
  ModuleDictNode::RegisterReflection();

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
      .def("relax.frontend.nn.ContainerApplyTo", ContainerApplyTo);
}

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
