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

#include "../analysis/common_analysis.h"

namespace tvm {
namespace dlight {

void DlightTryInline(tir::Schedule& sch, ffi::Array<DlightSBlockInfo>& blk_info) {
  while (true) {
    bool inlined = false;
    for (auto it = blk_info.begin(); it != blk_info.end(); ++it) {
      try {
        sch->ComputeInline((*it)->block_rv_);
        inlined = true;
        LOG(WARNING) << "Inlined successfully";
        blk_info.erase(it);
        break;
      } catch (const tvm::ffi::Error& e) {
        continue;
      }
    }
    if (!inlined) {
      for (auto it = blk_info.begin(); it != blk_info.end(); ++it) {
        try {
          sch->ReverseComputeInline((*it)->block_rv_);
          inlined = true;
          LOG(WARNING) << "ReverseInlined successfully";
          blk_info.erase(it);
          break;
        } catch (const tvm::ffi::Error& e) {
          continue;
        }
      }
    }
    if (!inlined) {
      break;
    }
  }
}

}  // namespace dlight
}  // namespace tvm
