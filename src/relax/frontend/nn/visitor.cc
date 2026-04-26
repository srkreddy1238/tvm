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
 * \file src/relax/frontend/nn/visitor.cc
 * \brief C++ port of python/tvm/relax/frontend/nn/visitor.py
 *
 * Implements the Mutator class: a visitor/mutator for nn::Module trees.
 *
 * Design notes
 * ------------
 * The Python Mutator walks a Module's __dict__ to find children.  In C++
 * the equivalent is the NNModuleNode::attrs map, which is populated by
 * every Make* factory function via PopulateAttrs.  ModuleList and
 * ModuleDict are handled as special cases (they store children in their
 * own `modules` field rather than `attrs`).
 *
 * Dispatch priority (matches Python visitor.py):
 *   ModuleDictNode  -> visit_moduledict
 *   ModuleListNode  -> visit_modulelist
 *   EffectNode      -> visit_effect
 *   ParameterNode   -> visit_param
 *   NNModuleNode    -> visit_module
 *
 * All visit_* defaults delegate to visit(), which recurses into children.
 * visit() returns the (possibly replaced) node as ffi::Any so that
 * subclasses can substitute a different object at any level.
 *
 * Python–C++ correspondence
 * -------------------------
 *   Python                     C++
 *   ──────────────────────     ──────────────────────────────────────
 *   isinstance(n, ModuleDict)  n.as<ModuleDictNode>()
 *   isinstance(n, ModuleList)  n.as<ModuleListNode>()
 *   isinstance(n, Effect)      n.as<EffectNode>()
 *   isinstance(n, Parameter)   n.as<ParameterNode>()
 *   isinstance(n, Module)      n.as<NNModuleNode>()
 *   node.__dict__.items()      mod->attrs (ffi::Map<String,Any>)
 *   node[i] = ...              list->modules.Set(i, ...)
 *   node[k] = ...              dict->modules.Set(k, ...)
 *   setattr(node, k, v)        mod->attrs.Set(k, v)
 */

#include "visitor.h"

#include <string>

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/*static*/ std::string Mutator::ChildName(const std::string& parent, const std::string& child) {
  // Mirrors _get_child_name() in visitor.py:
  //   if parent == "": return child
  //   else:            return f"{parent}.{child}"
  if (parent.empty()) return child;
  return parent + "." + child;
}

ffi::Any Mutator::DispatchChild(const std::string& name, ffi::Any value) {
  // Try to cast to ObjectRef; non-object values (scalars, strings, etc.)
  // are returned unchanged — they are not module children.
  auto opt_ref = value.try_cast<runtime::ObjectRef>();
  if (!opt_ref.has_value() || !opt_ref.value().defined()) return value;
  runtime::ObjectRef obj = opt_ref.value();

  // Dispatch in the same priority order as visitor.py.
  if (obj->IsInstance<ModuleDictNode>()) {
    return visit_moduledict(name, Downcast<ModuleDict>(obj));
  }
  if (obj->IsInstance<ModuleListNode>()) {
    return visit_modulelist(name, Downcast<ModuleList>(obj));
  }
  if (obj->IsInstance<EffectNode>()) {
    return visit_effect(name, obj);
  }
  if (obj->IsInstance<ParameterNode>()) {
    return visit_param(name, Downcast<NNParameter>(obj));
  }
  if (obj->IsInstance<NNModuleNode>()) {
    return visit_module(name, obj);
  }
  // Not a recognised module child — return unchanged.
  return value;
}

// ---------------------------------------------------------------------------
// visit() — the main traversal driver
// ---------------------------------------------------------------------------

ffi::Any Mutator::visit(const std::string& name, ffi::Any node) {
  auto opt_ref = node.try_cast<runtime::ObjectRef>();
  if (!opt_ref.has_value() || !opt_ref.value().defined()) return node;
  runtime::ObjectRef obj = opt_ref.value();

  // ── ModuleList ────────────────────────────────────────────────────────────
  // Python:
  //   for i in range(len(node)):
  //       node[i] = dispatch(f"{name}.{i}", node[i])
  if (const auto* list_node = obj.as<ModuleListNode>()) {
    ModuleList ml = Downcast<ModuleList>(obj);
    ffi::Array<ffi::Any> new_modules = list_node->modules;
    for (int64_t i = 0; i < static_cast<int64_t>(new_modules.size()); ++i) {
      std::string child_name = ChildName(name, std::to_string(i));
      ffi::Any old_val = new_modules[i];
      ffi::Any new_val = DispatchChild(child_name, old_val);
      if (!new_val.same_as(old_val)) {
        new_modules.Set(i, new_val);
      }
    }
    ml->modules = new_modules;
    return ffi::Any(ml);
  }

  // ── ModuleDict ────────────────────────────────────────────────────────────
  // Python:
  //   for k, v in node.items():
  //       node[k] = dispatch(_get_child_name(name, k), v)
  if (const auto* dict_node = obj.as<ModuleDictNode>()) {
    ModuleDict md = Downcast<ModuleDict>(obj);
    ffi::Map<ffi::String, ffi::Any> new_modules = dict_node->modules;
    for (const auto& [k, v] : dict_node->modules) {
      std::string child_name = ChildName(name, std::string(k));
      ffi::Any new_val = DispatchChild(child_name, v);
      if (!new_val.same_as(v)) {
        new_modules.Set(k, new_val);
      }
    }
    md->modules = new_modules;
    return ffi::Any(md);
  }

  // ── Module (generic NNModuleNode) ─────────────────────────────────────────
  // Python:
  //   for key, value in node.__dict__.items():
  //       setattr(node, key, dispatch(_get_child_name(name, key), value))
  if (const auto* mod_node = obj.as<NNModuleNode>()) {
    // Work on a copy of the attrs map so we can mutate it safely.
    ffi::Map<ffi::String, ffi::Any> new_attrs = mod_node->attrs;
    for (const auto& [k, v] : mod_node->attrs) {
      std::string child_name = ChildName(name, std::string(k));
      ffi::Any new_val = DispatchChild(child_name, v);
      if (!new_val.same_as(v)) {
        new_attrs.Set(k, new_val);
      }
    }
    // NNModuleNode::_type_mutable = true, so operator->() returns NNModuleNode*.
    const_cast<NNModuleNode*>(mod_node)->attrs = new_attrs;
    return node;
  }

  // Not a module node — return unchanged.
  return node;
}

// ---------------------------------------------------------------------------
// Default visit_* implementations — all delegate back to visit()
// ---------------------------------------------------------------------------

ffi::Any Mutator::visit_module(const std::string& name, runtime::ObjectRef node) {
  return visit(name, ffi::Any(node));
}

ffi::Any Mutator::visit_effect(const std::string& name, runtime::ObjectRef node) {
  return visit(name, ffi::Any(node));
}

ffi::Any Mutator::visit_param(const std::string& name, NNParameter node) {
  return visit(name, ffi::Any(node));
}

ffi::Any Mutator::visit_moduledict(const std::string& name, ModuleDict node) {
  return visit(name, ffi::Any(node));
}

ffi::Any Mutator::visit_modulelist(const std::string& name, ModuleList node) {
  return visit(name, ffi::Any(node));
}

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
