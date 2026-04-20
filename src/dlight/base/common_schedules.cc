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

void DlightTryInline(const s_tir::Schedule& sch, ffi::Array<DlightSBlockInfo>* blk_info) {
  while (true) {
    bool inlined = false;
    for (auto it = blk_info->begin(); it != blk_info->end(); ++it) {
      try {
        sch->ComputeInline((*it)->block_rv_);
        inlined = true;
        blk_info->erase(it);
        break;
      } catch (const tvm::ffi::Error& e) {
        continue;
      }
    }
    if (!inlined) {
      for (auto it = blk_info->begin(); it != blk_info->end(); ++it) {
        try {
          sch->ReverseComputeInline((*it)->block_rv_);
          inlined = true;
          blk_info->erase(it);
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

void DlightTryInlineContiguousSpatial(const s_tir::Schedule& sch,
                                      ffi::Array<DlightSBlockInfo>* blk_info) {
  ffi::Array<DlightSBlockInfo> results;
  ffi::Array<DlightSBlockInfo> spatial_blocks;

  for (const DlightSBlockInfo& block : *blk_info) {
    const bool is_injective = std::string(block->DomKind()).find('R') == std::string::npos;

    if (is_injective) {
      spatial_blocks.push_back(block);
    } else {
      if (!spatial_blocks.empty()) {
        DlightTryInline(sch, &spatial_blocks);
        for (const DlightSBlockInfo& b : spatial_blocks) results.push_back(b);
        spatial_blocks.clear();
      }
      results.push_back(block);
    }
  }

  // Flush trailing spatial blocks.
  if (!spatial_blocks.empty()) {
    DlightTryInline(sch, &spatial_blocks);
    for (const DlightSBlockInfo& b : spatial_blocks) results.push_back(b);
  }

  *blk_info = results;
}

}  // namespace dlight
}  // namespace tvm
