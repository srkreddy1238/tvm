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
 * \file src/dlight/base/utils.h
 * \brief Utils
 */

#ifndef TVM_DLIGHT_BASE_UTILS_H_
#define TVM_DLIGHT_BASE_UTILS_H_

#include <tvm/dlight/dlight_rule.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace dlight {

int GetMaxThreadsPerBlock(const tvm::Target& target);

}  // namespace dlight
}  // namespace tvm

#endif  // TVM_DLIGHT_BASE_UTILS_H_
