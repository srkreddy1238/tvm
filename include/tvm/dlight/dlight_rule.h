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

#ifndef TVM_DLIGHT_DLIGHT_RULE_H_
#define TVM_DLIGHT_DLIGHT_RULE_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/schedule/schedule.h>

namespace tvm {
namespace dlight {

/*! \brief Rules to schedule a TIR PrimFunc. */
class DlightRuleNode : public runtime::Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~DlightRuleNode() = default;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DlightRuleNode>();
  }

  /*!
   * \brief Schedule a TIR PrimFunc.
   * \param func The PrimFunc to be scheduled.
   * \param target The The target device for schedule.
   * \return The schedule if generated.
   */
  virtual ffi::Optional<s_tir::Schedule> Apply(const tir::PrimFunc& func,
                                               const tvm::Target& target) = 0;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("dlight.DlightRule", DlightRuleNode, Object);
};

/*!
 * \brief Managed reference to DlightRuleNode
 * \sa DlightRuleNode
 */
class DlightRule : public runtime::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DlightRule, ObjectRef, DlightRuleNode);
};

}  // namespace dlight
}  // namespace tvm

#endif  // TVM_DLIGHT_DLIGHT_RULE_H_
