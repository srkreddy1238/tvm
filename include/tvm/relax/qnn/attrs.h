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
 * \file tvm/relax/qnn/attrs.h
 * \brief Attributes for neural network operators.
 */
#ifndef TVM_RELAX_QNN_ATTRS_H_
#define TVM_RELAX_QNN_ATTRS_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/attrs/nn.h>

namespace tvm {
namespace relax {
namespace qnn {

struct BroadcastAttrs : public AttrsNodeReflAdapter<BroadcastAttrs> {
int lhs_axis;
int rhs_axis;
static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BroadcastAttrs>()
        .def_ro("lhs_axis", &BroadcastAttrs::lhs_axis,
                "The channel quantization axis of the lhs tensor. Use -1 for per-tensor.")
        .def_ro("rhs_axis", &BroadcastAttrs::rhs_axis,
                "The channel quantization axis of the rhs tensor. Use -1 for per-tensor.");
}
// Identifies the attribute for the TVM reflection and FFI system
TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.qnn.attrs.BroadcastAttrs",
    BroadcastAttrs, BaseAttrsNode);
};

}  // namespace qnn
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_QNN_ATTRS_H_
