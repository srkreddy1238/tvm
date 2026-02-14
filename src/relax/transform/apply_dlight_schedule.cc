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
 * \file src/relax/backend/adreno/apply_adreno_schedule.cc
 * \brief Dlight Schedules
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace relax {

class ApplySchedule {
 public:
  explicit ApplySchedule(const ffi::Array<ffi::String>& dlight_rules)
      : dlight_rules_(dlight_rules) {}

  IRModule Run(IRModule& mod) {
    IRModule updates_;
    Target target_;
    for (const auto& [gv, func] : mod->functions) {
      if (func->IsInstance<tvm::tir::PrimFuncNode>()) {
        const auto& base_func = mod->Lookup(gv);
        if ((!base_func->HasNonzeroAttr("tir.is_scheduled"))) {
          auto func_target = base_func->GetAttr<tvm::Target>("target");
          if (func_target.has_value()) {
            target_ = func_target.value();
          } else {
            target_ = Target::Current(false);
          }
          /* Apply Schedules */
          for (auto rule : dlight_rules_) {
            auto rule_h = tvm::ffi::Function::GetGlobal(rule);
            if (!rule_h.has_value()) continue;
            auto sch = (*rule_h)(base_func, target_).cast<ffi::Optional<tir::Schedule>>();
            if (sch.has_value()) {
              auto sch_func = Downcast<tir::PrimFunc>(sch.value()->mod()->Lookup("main"));
              auto sch_func_scheduled = WithAttr(sch_func, "tir.is_scheduled", Bool(true));
              updates_->Add(gv, sch_func_scheduled);
              break;
            }
          }
        }
      }
    }
    mod.CopyOnWrite()->Update(updates_);

    return mod;
  }

 private:
  ffi::Array<ffi::String> dlight_rules_;
};

namespace transform {

Pass ApplyDlightSchedule(ffi::Array<ffi::String> dlight_rules) {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    return tvm::relax::ApplySchedule(dlight_rules).Run(mod);
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"ApplyDlightSchedule",
                          /*required=*/{});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.ApplyDlightSchedule", ApplyDlightSchedule);
}
}  // namespace transform
}  // namespace relax
}  // namespace tvm
