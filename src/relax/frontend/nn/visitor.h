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
 * \file src/relax/frontend/nn/visitor.h
 * \brief Visitor / mutator infrastructure for nn::Module trees.
 *
 * Provides a single class, Mutator, whose virtual visit_* methods are
 * dispatched by the non-virtual visit() driver.  Users subclass Mutator
 * and override only the visit_* methods they care about; the default
 * implementations all delegate back to visit(), which recurses into the
 * module tree.
 *
 * The traversal logic mirrors visitor.py exactly:
 *
 *   visit(name, node)
 *     ModuleList  -> visit each element by index, dispatch on element type
 *     ModuleDict  -> visit each value by key,   dispatch on value type
 *     Module      -> visit each __dict__ entry,  dispatch on value type
 *
 * Dispatch priority (highest first):
 *   ModuleDict  -> visit_moduledict
 *   ModuleList  -> visit_modulelist
 *   EffectNode  -> visit_effect
 *   NNParameter -> visit_param
 *   NNModuleNode (other) -> visit_module
 *
 * All visit_* methods receive the dotted path name of the node within its
 * parent and return the (possibly replaced) node as ffi::Any.
 */

#ifndef TVM_RELAX_FRONTEND_NN_VISITOR_H_
#define TVM_RELAX_FRONTEND_NN_VISITOR_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/object.h>

#include <string>

#include "core.h"
#include "modules.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

/*!
 * \brief Mutator for nn::Module trees.
 *
 * Subclass and override visit_module / visit_effect / visit_param /
 * visit_moduledict / visit_modulelist to intercept specific node kinds.
 * Override visit() to change the traversal strategy entirely.
 *
 * All virtual methods are non-const so that subclasses can accumulate
 * state (e.g. a list of visited parameter names) during traversal.
 */
class Mutator {
 public:
  virtual ~Mutator() = default;

  /*!
   * \brief Visit an NNModuleNode (non-container, non-effect) node.
   *
   * Default: delegates to visit(name, node).
   *
   * \param name  Dotted path of this node within its parent.
   * \param node  The module node to visit.
   * \return      The (possibly replaced) node.
   */
  virtual ffi::Any visit_module(const std::string& name, runtime::ObjectRef node);

  /*!
   * \brief Visit an EffectNode.
   *
   * Default: delegates to visit(name, node).
   *
   * \param name  Dotted path of this node within its parent.
   * \param node  The effect node to visit.
   * \return      The (possibly replaced) node.
   */
  virtual ffi::Any visit_effect(const std::string& name, runtime::ObjectRef node);

  /*!
   * \brief Visit an NNParameter node.
   *
   * Default: delegates to visit(name, node).
   *
   * \param name  Dotted path of this node within its parent.
   * \param node  The parameter to visit.
   * \return      The (possibly replaced) parameter.
   */
  virtual ffi::Any visit_param(const std::string& name, NNParameter node);

  /*!
   * \brief Visit a ModuleDict node.
   *
   * Default: delegates to visit(name, node).
   *
   * \param name  Dotted path of this node within its parent.
   * \param node  The ModuleDict to visit.
   * \return      The (possibly replaced) node.
   */
  virtual ffi::Any visit_moduledict(const std::string& name, ModuleDict node);

  /*!
   * \brief Visit a ModuleList node.
   *
   * Default: delegates to visit(name, node).
   *
   * \param name  Dotted path of this node within its parent.
   * \param node  The ModuleList to visit.
   * \return      The (possibly replaced) node.
   */
  virtual ffi::Any visit_modulelist(const std::string& name, ModuleList node);

  /*!
   * \brief Dispatch driver: recurse into the module tree and call the
   *        appropriate visit_* method for each child.
   *
   * \param name  Dotted path of \p node within its parent (pass "" at root).
   * \param node  The node to visit; may be any ffi::Any value.
   * \return      The (possibly mutated) node.
   */
  virtual ffi::Any visit(const std::string& name, ffi::Any node);

 private:
  /*!
   * \brief Build the dotted child name from a parent path and a child key.
   *
   * When \p parent is empty the child name is returned as-is (top-level).
   * Otherwise the result is "parent.child".
   */
  static std::string ChildName(const std::string& parent, const std::string& child);

  /*!
   * \brief Dispatch a single child value to the correct visit_* method.
   *
   * \param name   Dotted path of the child.
   * \param value  The child value (ffi::Any).
   * \return       The (possibly replaced) child value.
   */
  ffi::Any DispatchChild(const std::string& name, ffi::Any value);
};

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_VISITOR_H_
