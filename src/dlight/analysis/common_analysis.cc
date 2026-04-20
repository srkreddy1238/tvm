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
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/var.h>

#include <ranges>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace dlight {

DlightIterInfo::DlightIterInfo(char kind, tir::Var var, PrimExpr dom, s_tir::LoopRV loop_rv) {
  auto n = ffi::make_object<DlightIterInfoNode>();
  n->kind_ = kind;
  n->var_ = std::move(var);
  n->dom_ = std::move(dom);
  n->loop_rv_ = std::move(loop_rv);
  data_ = std::move(n);
}

/*!
 * \brief Extract the primary Var from a Range::min expression.
 *
 * Returns the single Var that drives the dimension, or nullptr for
 * FloorDiv expressions (which have no single primary variable).
 *
 * \param expr  The min expression of a buffer access range.
 * \return      Pointer to the primary VarNode, or nullptr.
 */
static const tir::VarNode* ExtractPrimaryVar(const PrimExpr& expr) {
  // Plain variable: c
  if (const auto* v = expr.as<tir::VarNode>()) return v;
  // FloorMod: c % len
  if (const auto* fm = expr.as<tir::FloorModNode>()) {
    if (const auto* v = fm->a.as<tir::VarNode>()) return v;
  }
  // Mul+Add merge: co*len + cb
  if (const auto* add = expr.as<tir::AddNode>()) {
    if (const auto* mul = add->a.as<tir::MulNode>()) {
      if (const auto* v = mul->a.as<tir::VarNode>()) return v;
    }
    if (const auto* v = add->b.as<tir::VarNode>()) return v;
  }
  // FloorDiv: c // len — no single primary variable
  return nullptr;
}

DlightBufferInfo::DlightBufferInfo(const s_tir::Schedule& sch, const s_tir::SBlockRV& block_rv,
                                   const tir::BufferRegion& buf_region,
                                   const ffi::Array<s_tir::LoopRV>& loops) {
  auto n = ffi::make_object<DlightBufferInfoNode>();
  n->buf_region_ = buf_region;

  // Build loop_var → LoopRV map
  std::unordered_map<const tir::VarNode*, s_tir::LoopRV> lpvar_to_rv;
  for (const s_tir::LoopRV& lrv : loops) {
    tir::For loop = sch->Get(lrv);
    lpvar_to_rv[loop->loop_var.get()] = lrv;
  }

  // Build block iter-var → LoopRV map via iter_values of the block realize.
  tir::SBlock blk_stmt = sch->Get(block_rv);
  std::unordered_map<const tir::VarNode*, s_tir::LoopRV> itervar_to_rv;
  {
    auto get_realize_fn = tvm::ffi::Function::GetGlobal("s_tir.schedule.GetSBlockRealize");
    if (get_realize_fn.has_value()) {
      try {
        auto realize_any = (*get_realize_fn)(sch, block_rv);
        if (auto* realize = realize_any.cast<ObjectRef>().as<tir::SBlockRealizeNode>()) {
          const ffi::Array<PrimExpr>& iter_values = realize->iter_values;
          size_t n_iters = std::min(blk_stmt->iter_vars.size(), iter_values.size());
          for (size_t i = 0; i < n_iters; ++i) {
            const tir::VarNode* iter_var = blk_stmt->iter_vars[i]->var.get();
            if (const auto* lv = iter_values[i].as<tir::VarNode>()) {
              auto it = lpvar_to_rv.find(lv);
              if (it != lpvar_to_rv.end()) {
                itervar_to_rv[iter_var] = it->second;
              }
            }
          }
        }
      } catch (...) {
        // Fall back to positional matching below
      }
    }
    // Fallback: positional matching
    if (itervar_to_rv.empty()) {
      size_t n_match = std::min(blk_stmt->iter_vars.size(), loops.size());
      for (size_t i = 0; i < n_match; ++i) {
        itervar_to_rv[blk_stmt->iter_vars[i]->var.get()] = loops[i];
      }
    }
  }

  // For each dimension of the access region, find the associated LoopRV.
  for (const Range& r : buf_region->region) {
    const tir::VarNode* primary = ExtractPrimaryVar(r->min);
    if (primary == nullptr) {
      n->assoc_lps_.push_back(ffi::Optional<s_tir::LoopRV>());
      continue;
    }
    auto it_iv = itervar_to_rv.find(primary);
    if (it_iv != itervar_to_rv.end()) {
      n->assoc_lps_.push_back(it_iv->second);
      continue;
    }
    auto it_lp = lpvar_to_rv.find(primary);
    if (it_lp != lpvar_to_rv.end()) {
      n->assoc_lps_.push_back(it_lp->second);
      continue;
    }
    n->assoc_lps_.push_back(ffi::Optional<s_tir::LoopRV>());
  }

  data_ = std::move(n);
}

std::string DlightBufferInfoNode::GetScope() const {
  return std::string(buf_region_->buffer.scope());
}

int DlightBufferInfoNode::GetVecSize(int vbits) const {
  if (assoc_lps_.empty() || !assoc_lps_.back().defined()) return 1;

  const tir::Buffer& buf = buf_region_->buffer;
  if (buf->shape.empty()) return 1;

  const PrimExpr& last_shape_expr = buf->shape[buf->shape.size() - 1];
  const auto* shape_imm = last_shape_expr.as<IntImmNode>();
  if (!shape_imm || shape_imm->value <= 0) return 1;

  int64_t vbuf_extent = shape_imm->value & (-shape_imm->value);  // largest pow2 divisor
  int dtype_bits = buf->dtype.bits();
  int max_vec = (dtype_bits > 0) ? (vbits / dtype_bits) : 1;
  return static_cast<int>(std::min(vbuf_extent, static_cast<int64_t>(max_vec)));
}

ffi::String DlightSBlockInfoNode::DomKind() const {
  std::string d_kind;
  for (const DlightIterInfo& di : iters_) {
    d_kind += di->kind_;
  }
  return d_kind;
}

bool DlightSBlockInfoNode::IsInjective() const {
  for (const DlightIterInfo& di : iters_) {
    if (di->kind_ != 'S') return false;
  }
  return true;
}

bool DlightSBlockInfoNode::IsReduction() const {
  bool has_r = false;
  for (const DlightIterInfo& di : iters_) {
    if (di->kind_ == 'O') return false;
    if (di->kind_ == 'R') has_r = true;
  }
  return has_r;
}

bool DlightSBlockInfoNode::IsElementwise() const {
  if (!IsInjective() || write_bufs_.size() != 1) return false;
  const ffi::Array<Range>& w_region = write_bufs_[0]->buf_region_->region;
  for (const DlightBufferInfo& rbuf : read_bufs_) {
    const ffi::Array<Range>& r_region = rbuf->buf_region_->region;
    if (r_region.size() != w_region.size()) return false;
    for (size_t i = 0; i < r_region.size(); ++i) {
      if (!ffi::StructuralEqual()(r_region[i], w_region[i])) return false;
    }
  }
  return true;
}

bool DlightSBlockInfoNode::IsDataPad(const s_tir::Schedule& sch) const {
  if (!IsInjective()) return false;
  if (read_bufs_.size() != 1 || write_bufs_.size() != 1) return false;
  if (read_bufs_[0]->buf_region_->region.size() != write_bufs_[0]->buf_region_->region.size())
    return false;
  if (IsElementwise()) return false;
  tir::SBlock stmt = sch->Get(block_rv_);
  bool has_if = false;
  tir::PostOrderVisit(stmt->body, [&](const ObjectRef& node) {
    if (node.as<tir::IfThenElseNode>()) has_if = true;
  });
  return has_if;
}

bool DlightSBlockInfoNode::IsLayoutTransform(const s_tir::Schedule& sch) const {
  if (!IsInjective()) return false;
  if (read_bufs_.size() != 1 || write_bufs_.size() != 1) return false;
  if (IsElementwise()) return false;
  tir::SBlock stmt = sch->Get(block_rv_);
  bool has_if = false;
  tir::PostOrderVisit(stmt->body, [&](const ObjectRef& node) {
    if (node.as<tir::IfThenElseNode>()) has_if = true;
  });
  return !has_if;
}

DlightSBlockInfo GetSBlockInfo(const s_tir::Schedule& sch, const s_tir::SBlockRV& block_rv) {
  auto n = ffi::make_object<DlightSBlockInfoNode>();

  tir::SBlock blk_stmt = sch->Get(block_rv);
  n->name_ = blk_stmt->name_hint;
  n->block_rv_ = block_rv;

  ffi::Array<s_tir::LoopRV> lps = sch->GetLoops(block_rv);
  size_t n_iters = std::min(lps.size(), blk_stmt->iter_vars.size());
  for (size_t i = 0; i < n_iters; ++i) {
    const tir::IterVar& iv = blk_stmt->iter_vars[i];
    char kind;
    if (iv->iter_type == tir::kDataPar) {
      kind = 'S';
    } else if (iv->iter_type == tir::kCommReduce) {
      kind = 'R';
    } else {
      kind = 'O';
    }
    n->iters_.push_back(DlightIterInfo(kind, iv->var, iv->dom->extent, lps[i]));
  }

  // read_bufs / write_bufs
  for (const tir::BufferRegion& br : blk_stmt->reads) {
    n->read_bufs_.push_back(DlightBufferInfo(sch, block_rv, br, lps));
  }
  for (const tir::BufferRegion& bw : blk_stmt->writes) {
    n->write_bufs_.push_back(DlightBufferInfo(sch, block_rv, bw, lps));
  }

  // producers / consumers
  n->producers_ = sch->GetProducers(block_rv);
  n->consumers_ = sch->GetConsumers(block_rv);

  return DlightSBlockInfo(n);
}

ffi::Array<DlightBufferInfo> GetReadBufferInfos(const s_tir::Schedule& sch,
                                                const s_tir::SBlockRV& block_rv) {
  ffi::Array<DlightBufferInfo> result;
  ffi::Array<s_tir::LoopRV> loops = sch->GetLoops(block_rv);
  tir::SBlock stmt = sch->Get(block_rv);
  for (const tir::BufferRegion& br : stmt->reads) {
    result.push_back(DlightBufferInfo(sch, block_rv, br, loops));
  }
  return result;
}

ffi::Array<DlightBufferInfo> GetWriteBufferInfos(const s_tir::Schedule& sch,
                                                 const s_tir::SBlockRV& block_rv) {
  ffi::Array<DlightBufferInfo> result;
  ffi::Array<s_tir::LoopRV> loops = sch->GetLoops(block_rv);
  tir::SBlock stmt = sch->Get(block_rv);
  for (const tir::BufferRegion& bw : stmt->writes) {
    result.push_back(DlightBufferInfo(sch, block_rv, bw, loops));
  }
  return result;
}

std::pair<std::vector<std::string>, std::vector<std::string>> GetInOutDtypes(
    const tir::SBlock& block) {
  std::vector<std::string> in_dtypes, out_dtypes;
  for (const tir::BufferRegion& br : block->reads) {
    in_dtypes.push_back(std::string(ffi::DLDataTypeToString(br->buffer->dtype)));
  }
  for (const tir::BufferRegion& bw : block->writes) {
    out_dtypes.push_back(std::string(ffi::DLDataTypeToString(bw->buffer->dtype)));
  }
  return {in_dtypes, out_dtypes};
}

ffi::Array<DlightSBlockInfo> DlightNormalizePrimFunc(const s_tir::Schedule& sch) {
  ffi::Array<DlightSBlockInfo> blkinfo;

  auto gfunc = tvm::ffi::Function::GetGlobal("s_tir.schedule.NormalizePrimFunc");
  ffi::Any result;
  try {
    result = (*gfunc)(sch);
  } catch (...) {
    return blkinfo;
  }
  if (result == nullptr) return blkinfo;

  ffi::Array<ObjectRef> nor_info;
  try {
    nor_info = result.cast<ffi::Array<ObjectRef>>();
  } catch (...) {
    return blkinfo;
  }

  if (nor_info.size() != 4) return blkinfo;

  auto blocks = Downcast<ffi::Array<s_tir::SBlockRV>>(nor_info[0]);
  for (size_t i = 0; i < blocks.size(); ++i) {
    try {
      blkinfo.push_back(GetSBlockInfo(sch, blocks[i]));
    } catch (...) {
    }
  }
  return blkinfo;
}

}  // namespace dlight
}  // namespace tvm
