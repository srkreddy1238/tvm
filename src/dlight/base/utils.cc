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
#include <tvm/dlight/dlight_rule.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace dlight {

int GetMaxThreadsPerBlock(const tvm::Target& target) {
  auto attrs = target->attrs;
  if (attrs.find("max_threads_per_block") != attrs.end()) {
    return attrs["max_threads_per_block"].cast<int>();
  }
  if (attrs.find("max_num_threads") != attrs.end()) {
    return attrs["max_num_threads"].cast<int>();
  }
  if (target->kind->name == "cuda") {
    return 1024;
  }
  return 256;
}

}  // namespace dlight
}  // namespace tvm
