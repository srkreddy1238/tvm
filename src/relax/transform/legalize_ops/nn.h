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

#ifndef TVM_RELAX_TRANSFORM_LEGALIZE_OPS_NN_H_
#define TVM_RELAX_TRANSFORM_LEGALIZE_OPS_NN_H_

#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/distributed/struct_info.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/utils.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <typeinfo>

#include "../../../te/operation/create_primfunc.h"
#include "../../ir/emit_te.h"
#include "utils.h"

namespace tvm {
namespace relax {

ffi::Array<te::Tensor> ConvTE(const ffi::Array<ffi::Any> args);

ffi::Array<te::Tensor> PadTE(const ffi::Array<ffi::Any> args);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_LEGALIZE_OPS_NN_H_
