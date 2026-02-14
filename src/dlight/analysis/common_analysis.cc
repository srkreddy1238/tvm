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
#include "common_analysis.h"

#include <tvm/dlight/dlight_rule.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/var.h>

#include <ranges>

namespace tvm {
namespace dlight {

DlightIterInfo::DlightIterInfo(tvm::tir::IterVar iter, tir::LoopRV loop_rv) {
  auto n = ffi::make_object<DlightIterInfoNode>();
  n->iter_ = std::move(iter);
  n->loop_rv_ = std::move(loop_rv);
  data_ = std::move(n);
}

DlightSBlockInfo::DlightSBlockInfo(ffi::String name, ffi::Array<DlightIterInfo> iters,
                                   tir::SBlockRV block_rv, bool reduction_block) {
  auto n = ffi::make_object<DlightSBlockInfoNode>();
  n->name_ = std::move(name);
  n->iters_ = std::move(iters);
  n->block_rv_ = std::move(block_rv);
  n->reduction_block_ = reduction_block;
  data_ = std::move(n);
}

ffi::String DlightSBlockInfoNode::DomKind() const {
  std::string d_kind;
  for (auto diter : iters_) {
    if (tir::kDataPar == diter->iter_->iter_type) {
      d_kind += "S";
    } else if (tir::kCommReduce == diter->iter_->iter_type) {
      d_kind += "R";
    } else {
      d_kind += "O";
    }
  }
  return d_kind;
}

ffi::Array<DlightSBlockInfo> DlightNormalizePrimFunc(tir::Schedule& sch) {
  ffi::Array<DlightSBlockInfo> blkinfo;

  auto gfunc = tvm::ffi::Function::GetGlobal("tir.schedule.NormalizePrimFunc");
  auto nor_info = (*gfunc)(sch).cast<ffi::Array<ObjectRef>>();

  LOG(WARNING) << "Normalized:" << nor_info;

  ICHECK(nor_info.size() == 4);

  auto blocks = Downcast<ffi::Array<tir::SBlockRV>>(nor_info[0]);
  auto loops = Downcast<ffi::Array<ffi::Array<tir::LoopRV>>>(nor_info[1]);
  auto iters = Downcast<ffi::Array<ffi::Array<tir::IterVar>>>(nor_info[2]);
  auto is_reduction = Downcast<ffi::Array<IntImm>>(nor_info[3]);
  ICHECK((blocks.size() == loops.size()) && (loops.size() == iters.size()) &&
         (loops.size() == is_reduction.size()));

  for (size_t i = 0; i < blocks.size(); ++i) {
    ffi::Array<DlightIterInfo> dl_iter_info;
    for (size_t j = 0; j < loops[i].size(); ++j) {
      auto diter = DlightIterInfo(iters[i][j], loops[i][j]);
      dl_iter_info.push_back(diter);
    }
    blkinfo.push_back(DlightSBlockInfo(sch->Get(blocks[i])->name_hint, dl_iter_info, blocks[i],
                                       is_reduction[i]->value ? true : false));
  }
  return blkinfo;
}

}  // namespace dlight
}  // namespace tvm
