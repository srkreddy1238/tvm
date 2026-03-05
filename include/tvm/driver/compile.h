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
 * \file tvm/driver/compile.h
 * \brief TVM compiler API
 */

#ifndef TVM_DRIVER_COMPILE_H_
#define TVM_DRIVER_COMPILE_H_

#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/transform.h>
#include <tvm/node/serialization.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/exec_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/tir/function.h>
#include <tvm/tir/index_map.h>

namespace ffi = tvm::ffi;
namespace relax = tvm::relax;
namespace tir = tvm::tir;
namespace runtime = tvm::runtime;

namespace tvm {
namespace driver {

/*!
 * \brief tvm IRModule compile function to VM Module
 * \param mod The IRModule to compile
 * \param target The Target as string or Target object
 * \param relax_pipeline the relax pipeline to be used like
 *        "cpu_generic", "gpu_generic" or target specific like "adreno"
 * \param tir_pipeline the TIR lowering pipeline to be used.
 *        can be "generic" for default or target specific as registered.
 * \return The compiled VM module.
 */
TVM_DLL ffi::Module Compile(IRModule mod, ffi::Any target,
                            ffi::Optional<ffi::String> relax_pipeline = std::nullopt,
                            ffi::Optional<ffi::String> tir_pipeline = std::nullopt);

}  // namespace driver
}  // namespace tvm

#endif  // TVM_DRIVER_COMPILE_H_
