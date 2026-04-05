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
 * \file tvm/relax/transform/legalize_ops/image.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */
#include <tvm/relax/attrs/image.h>
#include <tvm/relax/op_attr_types.h>

#include "utils.h"

namespace tvm {
namespace relax {

// relax.image.resize2d
Expr LegalizeImageResize2D(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<Resize2DAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);                          // data
  args.push_back(attrs->roi);                             // roi
  args.push_back(call->args[1]);                          // size
  args.push_back(attrs->layout);                          // layout
  args.push_back(attrs->method);                          // method
  args.push_back(attrs->coordinate_transformation_mode);  // coordinate_transformation_mode
  args.push_back(attrs->rounding_method);                 // rounding_method
  args.push_back(attrs->cubic_alpha);                     // bicubic_alpha
  args.push_back(attrs->cubic_exclude);                   // bicubic_exclude
  args.push_back(attrs->extrapolation_value);             // extrapolation_value
  return m_te.Make(args, ffi::String("topi.image.resize2d"), std::string("resize2d"));
}
TVM_REGISTER_OP("relax.image.resize2d")
    .set_attr<FLegalize>("FLegalize", LegalizeImageResize2D, TVM_LEGALIZE_CPP_LEVEL);

// relax.image.grid_sample
Expr LegalizeImageGridSample(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<GridSampleAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);         // data
  args.push_back(call->args[1]);         // grid
  args.push_back(attrs->method);         // method
  args.push_back(attrs->layout);         // layout
  args.push_back(attrs->padding_mode);   // padding_mode
  args.push_back(attrs->align_corners);  // align_corners
  return m_te.Make(args, ffi::String("topi.image.grid_sample"), std::string("grid_sample"));
}
TVM_REGISTER_OP("relax.image.grid_sample")
    .set_attr<FLegalize>("FLegalize", LegalizeImageGridSample, TVM_LEGALIZE_CPP_LEVEL);

}  // namespace relax
}  // namespace tvm
