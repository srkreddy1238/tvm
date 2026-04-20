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
 * \file src/dlight/gpu/general_reduction.cc
 * \brief General reduction schedule rule for GPU operators.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/dlight/dlight_rule.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "../analysis/common_analysis.h"
#include "../base/common_schedules.h"
#include "../base/utils.h"

namespace tvm {
namespace dlight {

class ApplyGeneralReductionRuleNode : public DlightRuleNode {
 public:
  ffi::Optional<s_tir::Schedule> Apply(const tir::PrimFunc& func, const tvm::Target& target) final {
    if (!target->HasKey("gpu")) {
      return std::nullopt;
    }

    int len_tx = 64;
    int unroll_depth = 64;
    if (target->kind->name == "cuda") {
      len_tx = 256;
      unroll_depth = 256;
    } else if (target->kind->name == "opencl") {
      len_tx = 256;
      unroll_depth = 64;
    }

    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> func_map;
    func_map.Set(GlobalVar("main"), func);
    auto sch = s_tir::Schedule::Traced(IRModule(func_map), -1, 0,
                                       s_tir::ScheduleErrorRenderLevel::kDetail, true);

    auto block_infos = DlightNormalizePrimFunc(sch);
    DlightTryInlineContiguousSpatial(sch, &block_infos);
    if (block_infos.empty()) {
      return std::nullopt;
    }

    ffi::String dom_kind = block_infos[0]->DomKind();
    size_t dom_len = dom_kind.size();

    size_t num_leading_s = 0;
    for (size_t i = 0; i < dom_len; ++i) {
      if (dom_kind.at(i) == 'S') {
        ++num_leading_s;
      } else {
        break;
      }
    }

    size_t num_trailing_r = 0;
    for (int i = static_cast<int>(dom_len) - 1; i >= 0; --i) {
      if (dom_kind.at(i) == 'R') {
        ++num_trailing_r;
      } else {
        break;
      }
    }

    size_t num_last_block_iter = block_infos.back()->iters_.size();

    if (num_last_block_iter < dom_len) {
      if (num_last_block_iter == 0) {
        for (size_t i = 0; i < block_infos.size(); ++i) {
          auto loop_rv = sch->AddUnitLoop(block_infos[i]->block_rv_);
          if (i == 0) {
            sch->Bind(loop_rv, "blockIdx.x");
          } else {
            sch->Bind(loop_rv, "threadIdx.x");
          }
        }
        return sch;
      }

      ffi::Array<PrimExpr> first_block_extents;
      for (auto& iter_info : block_infos[0]->iters_) {
        first_block_extents.push_back(iter_info->dom_);
      }
      ffi::Array<PrimExpr> last_block_extents;
      for (auto& iter_info : block_infos.back()->iters_) {
        last_block_extents.push_back(iter_info->dom_);
      }
      size_t n_first = first_block_extents.size();
      size_t n_last = last_block_extents.size();  // == num_last_block_iter

      auto index_map = tir::IndexMap::FromFunc(
          static_cast<int>(n_last),
          [n_first, n_last, dom_len, first_block_extents,
           last_block_extents](ffi::Array<tir::Var> iters) -> ffi::Array<PrimExpr> {
            arith::Analyzer analyzer;

            size_t num_matched = 0;
            ffi::Array<PrimExpr> target_layout_iters;

            for (size_t fi = 0; fi < n_first; ++fi) {
              if (num_matched < n_last && analyzer.CanProveEqual(first_block_extents[fi],
                                                                 last_block_extents[num_matched])) {
                target_layout_iters.push_back(iters[num_matched]);
                ++num_matched;
              } else {
                target_layout_iters.push_back(tvm::tir::make_const(DataType::Int(64), 0));
              }
            }

            if (num_matched == n_last) {
              return target_layout_iters;
            }

            ffi::Array<PrimExpr> fallback;
            size_t n_zeros = dom_len - n_last;
            for (size_t i = 0; i < n_zeros; ++i) {
              fallback.push_back(tvm::tir::make_const(DataType::Int(64), 0));
            }
            for (auto& v : iters) {
              fallback.push_back(v);
            }
            return fallback;
          });

      sch->TransformBlockLayout(block_infos.back()->block_rv_, index_map);
    }

    // Must have at least one trailing reduction dim.
    if (num_trailing_r == 0) return std::nullopt;
    // All intermediate blocks must share the same dom_kind as the first block.
    for (size_t i = 1; i + 1 < block_infos.size(); ++i) {
      if (block_infos[i]->DomKind() != dom_kind) return std::nullopt;
    }
    // The last block must be injective (no 'R' in its dom_kind).
    ffi::String last_dom_kind = block_infos.back()->DomKind();
    bool last_is_injective = (last_dom_kind.find("R") == std::string::npos);
    if (!last_is_injective) return std::nullopt;
    // The last block's iter count must be <= the first block's.
    if (block_infos.back()->iters_.size() > dom_len) return std::nullopt;

    if (last_dom_kind.find("R") == std::string::npos) {
      std::vector<tir::Buffer> reduced_buffers;
      for (size_t i = 0; i + 1 < block_infos.size(); ++i) {
        auto sblock = sch->Get(block_infos[i]->block_rv_);
        for (auto& buf_region : sblock->writes) {
          reduced_buffers.push_back(buf_region->buffer);
        }
      }

      auto spatial_block = sch->Get(block_infos.back()->block_rv_);
      auto loops = sch->GetLoops(block_infos.back()->block_rv_);

      std::unordered_map<const tir::VarNode*, tir::Var> block_var_to_loop_var;
      for (size_t i = 0; i < spatial_block->iter_vars.size() && i < loops.size(); ++i) {
        auto loop_for = sch->Get(loops[i]);
        block_var_to_loop_var[spatial_block->iter_vars[i]->var.get()] = loop_for->loop_var;
      }

      std::unordered_set<const tir::VarNode*> spatial_loop_vars;
      auto visit_expr = [&](const PrimExpr& e) {
        tir::PostOrderVisit(e, [&](const ObjectRef& node) {
          if (auto var = node.as<tir::VarNode>()) {
            auto it = block_var_to_loop_var.find(var);
            if (it != block_var_to_loop_var.end()) {
              spatial_loop_vars.insert(it->second.get());
            }
          }
        });
      };

      for (auto& buf_region : spatial_block->reads) {
        bool is_reduced = false;
        for (auto& rb : reduced_buffers) {
          if (buf_region->buffer.same_as(rb)) {
            is_reduced = true;
            break;
          }
        }
        if (!is_reduced) continue;
        for (auto& range : buf_region->region) {
          visit_expr(range->min);
          visit_expr(range->extent);
        }
      }

      ffi::Array<s_tir::LoopRV> s_loops, other_loops;
      for (auto& loop_rv : loops) {
        auto loop_for = sch->Get(loop_rv);
        bool is_spatial = (spatial_loop_vars.count(loop_for->loop_var.get()) > 0) ||
                          tir::is_one(loop_for->extent);
        if (is_spatial) {
          s_loops.push_back(loop_rv);
        } else {
          other_loops.push_back(loop_rv);
        }
      }

      ffi::Array<s_tir::LoopRV> reordered;
      for (auto& l : s_loops) reordered.push_back(l);
      for (auto& l : other_loops) reordered.push_back(l);
      sch->Reorder(reordered);
    }

    {
      auto loops = sch->GetLoops(block_infos.back()->block_rv_);

      ffi::Array<s_tir::LoopRV> leading_loops;
      for (size_t i = 0; i < num_leading_s && i < loops.size(); ++i) {
        leading_loops.push_back(loops[i]);
      }
      auto bx = sch->Fuse(leading_loops);

      auto last_loop = loops[loops.size() - 1];
      ffi::Array<ffi::Optional<s_tir::ExprRV>> split_factors = {
          ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), len_tx))};
      auto splits = sch->Split(last_loop, split_factors);
      auto r_loop = splits[0];
      auto tx = splits[1];

      sch->Reorder({tx, r_loop});

      sch->Bind(bx, "blockIdx.x");
      sch->Bind(tx, "threadIdx.x");
      sch->Annotate(r_loop, "pragma_auto_unroll_max_step", IntImm(DataType::Int(32), unroll_depth));
      sch->Annotate(r_loop, "pragma_unroll_explicit", IntImm(DataType::Int(32), 1));
    }

    auto last_block_loops_after = sch->GetLoops(block_infos.back()->block_rv_);
    auto bx_rv = last_block_loops_after[0];

    for (int bi = static_cast<int>(block_infos.size()) - 2; bi >= 0; --bi) {
      auto blk = block_infos[bi]->block_rv_;

      auto sblock = sch->Get(blk);
      for (size_t wi = 0; wi < sblock->writes.size(); ++wi) {
        sch->SetScope(blk, static_cast<int>(wi), "shared");
      }

      sch->ComputeAt(blk, bx_rv, /*preserve_unit_loops=*/true);

      auto blk_loops = sch->GetLoops(blk);
      ffi::Array<s_tir::LoopRV> trailing_r_loops;
      size_t start = blk_loops.size() >= num_trailing_r ? blk_loops.size() - num_trailing_r : 0;
      for (size_t li = start; li < blk_loops.size(); ++li) {
        trailing_r_loops.push_back(blk_loops[li]);
      }
      auto r_loop = sch->Fuse(trailing_r_loops);

      ffi::Array<ffi::Optional<s_tir::ExprRV>> split_factors = {
          ffi::Optional<s_tir::ExprRV>(), s_tir::ExprRV(IntImm(DataType::Int(32), len_tx))};
      auto splits = sch->Split(r_loop, split_factors);
      r_loop = splits[0];
      auto tx = splits[1];

      sch->Reorder({tx, r_loop});

      sch->Bind(tx, "threadIdx.x");
      sch->Annotate(r_loop, "pragma_auto_unroll_max_step", IntImm(DataType::Int(32), unroll_depth));
      sch->Annotate(r_loop, "pragma_unroll_explicit", IntImm(DataType::Int(32), 1));
    }

    return sch;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dlight.ApplyGeneralReductionRule",
                                    ApplyGeneralReductionRuleNode, DlightRuleNode);
};

ffi::Optional<s_tir::Schedule> ApplyGeneralReductionRule(tir::PrimFunc func, tvm::Target target) {
  return ApplyGeneralReductionRuleNode().Apply(func, target);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("dl.gpu.GeneralReduction", ApplyGeneralReductionRule);
}

}  // namespace dlight
}  // namespace tvm
