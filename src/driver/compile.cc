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

#include <tvm/driver/compile.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace driver {

std::pair<tvm::IRModule, tvm::ffi::Map<tvm::Target, tvm::IRModule>> SplitHostDeviceMods(
    tvm::IRModule& mod) {
  tvm::Target target;
  tvm::IRModule host_mod;
  tvm::ffi::Map<tvm::Target, tvm::IRModule> device_mod_dict;
  bool is_host = true;

  std::function<bool(const tvm::tir::PrimFunc&)> is_host_func =
      [&](const tvm::tir::PrimFunc& func) -> bool {
    tvm::Target tgt = func->GetAttr<tvm::Target>("target", tvm::Target("llvm")).value();
    if (tgt->kind->name == "llvm" || tgt->kind->name == "c") {
      return is_host ? true : false;
    } else {
      return is_host ? false : true;
    }
  };

  auto pass = ffi::Function::GetGlobal("tir.transform.Filter");
  host_mod = (*pass)(tvm::ffi::TypedFunction<bool(tvm::tir::PrimFunc)>(is_host_func))
                 .cast<tvm::transform::Pass>()(mod);
  is_host = false;
  auto device_mod = (*pass)(tvm::ffi::TypedFunction<bool(tvm::tir::PrimFunc)>(is_host_func))
                        .cast<tvm::transform::Pass>()(mod);

  tvm::ffi::Map<tvm::ffi::String, tvm::Target> target_str2target;
  tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc>> device_func_dict;
  for (auto it : device_mod->functions) {
    tvm::Target tgt = it.second->GetAttr<tvm::Target>("target", tvm::Target()).value();
    target_str2target.Set(tgt->str(), tgt);
    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> dev_funcs;
    if (device_func_dict.find(tgt->str()) != device_func_dict.end()) {
      dev_funcs = device_func_dict[tgt->str()];
    }
    dev_funcs.Set(it.first, it.second);
    device_func_dict.Set(tgt->str(), dev_funcs);
  }

  for (auto it : device_func_dict) {
    auto tgt = tvm::Target(it.first);

    auto d_mod =
        tvm::IRModule(it.second, tvm::SourceMap({}), device_mod->attrs,
                      tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Array<tvm::GlobalInfo>>({}));
    device_mod_dict.Set(tgt, d_mod);
  }

  return std::pair<tvm::IRModule, tvm::ffi::Map<tvm::Target, tvm::IRModule>>(
      {host_mod, device_mod_dict});
}

ffi::Module Compile(IRModule mod, ffi::Any target, ffi::Optional<ffi::String> relax_pipeline,
                    ffi::Optional<ffi::String> tir_pipeline) {
  ObjectPtr<tvm::runtime::vm::VMExecutable> executable;
  Target target_;
  ffi::String relax_pipeline_;
  ffi::String tir_pipeline_;

  // Target
  if (target.as<Target>()) {
    target_ = target.as<Target>().value();
  } else if (target.as<ffi::String>()) {
    target_ = Target(target.as<ffi::String>().value());
  } else {
    LOG(FATAL) << "Only Target or string are allow as target:" << target;
  }

  // Target Host
  Target target_host_ = tvm::runtime::RuntimeEnabled("llvm") ? Target("llvm") : Target("c");
  if (target_->GetHost()) {
    target_host_ = target_->GetHost().value();
  } else if (target_->GetAttr<ffi::String>("target_host")) {
    target_host_ = Target(target_->GetAttr<ffi::String>("target_host").value());
  }
  target_ = Target::WithHost(target_, target_host_);

  // relax_pipeline
  if (relax_pipeline.has_value()) {
    relax_pipeline_ = relax_pipeline.value();
  } else if (target_->GetAttr<ffi::String>("relax_pipeline")) {
    relax_pipeline_ = target_->GetAttr<ffi::String>("relax_pipeline").value();
  } else {
    LOG(FATAL) << "Can't get relax pipeline for : " << target;
  }

  // TIR pipeline
  if (tir_pipeline.has_value()) {
    tir_pipeline_ = tir_pipeline.value();
  } else if (target_->GetAttr<ffi::String>("tir_pipeline")) {
    tir_pipeline_ = target_->GetAttr<ffi::String>("tir_pipeline").value();
  } else {
    // Use generic TIR Pipeline
    tir_pipeline_ = "generic";
  }

  // Generic passes
  mod = relax::transform::Normalize()(mod);
  mod = relax::transform::CanonicalizeBindings()(mod);

  std::function<IRModule(IRModule & mod_, const ffi::String& pass)> apply_module_pass_ =
      [&target_](IRModule& mod_, const tvm::ffi::String& pass) -> IRModule {
    auto gf = ffi::Function::GetGlobal(pass);
    ICHECK(gf) << "Global function not found : " << pass;
    tvm::transform::Pass pass_h = (*gf)(target_).cast<tvm::transform::Pass>();
    return pass_h(mod_);
  };

  // Relax Pipeline
  mod = apply_module_pass_(mod, ffi::String("relax.pipeline.") + relax_pipeline_);

  // VM Codegen
  tvm::relax::ExecBuilder ex_builder =
      (*ffi::Function::GetGlobal("relax.ExecBuilderCreate"))().cast<tvm::relax::ExecBuilder>();
  mod = (*ffi::Function::GetGlobal("relax.VMCodeGen"))(ex_builder, mod).cast<tvm::IRModule>();

  // TOR Mod extraction
  std::function<tvm::IRModule(tvm::IRModule&)> _filter_tir =
      [&](tvm::IRModule& mod) -> tvm::IRModule {
    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> tir_funcs;
    for (auto func : mod->functions) {
      if (func.second.as<tvm::tir::PrimFunc>()) {
        tir_funcs.Set(func.first, func.second);
      }
    }
    return tvm::IRModule(tir_funcs, tvm::SourceMap({}), mod->attrs,
                         tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Array<tvm::GlobalInfo>>({}));
  };
  auto tir_mod = _filter_tir(mod);

  // TIR Pipeline
  tir_mod = tvm::tir::transform::BindTarget(target_)(tir_mod);

  tir_mod = apply_module_pass_(tir_mod,
                               ffi::String("tir.pipeline.") + tir_pipeline_ + ffi::String(".Base"));

  // Split host and device
  auto [host_mod, device_mod_dict] = SplitHostDeviceMods(tir_mod);

  // TIR Host
  host_mod = apply_module_pass_(
      host_mod, ffi::String("tir.pipeline.") + tir_pipeline_ + ffi::String(".Host"));

  // TIR Device
  for (auto [tgt, dmod] : device_mod_dict) {
    dmod = apply_module_pass_(
        dmod, ffi::String("tir.pipeline.") + tir_pipeline_ + ffi::String(".Device"));
    device_mod_dict.Set(tgt, dmod);
  }

  // TIR To Runtime
  auto mhost_all =
      tvm::IRModule({}, tvm::SourceMap({}), host_mod->attrs,
                    tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Array<tvm::GlobalInfo>>({}));
  mhost_all->Update(host_mod);
  tvm::ffi::Array<tvm::ffi::Module> runtime_mods;

  /*
  auto gf = ffi::Function::GetGlobal("target.build." + target_host_->kind->name);
  ICHECK(gf) << "Global function not found : target.build." << target_host_->kind->name;
  auto m_host = (*gf)(mhost_all, target_host_).cast<tvm::ffi::Module>();
  */

  for (auto [tgt, d_mod] : device_mod_dict) {
    if (d_mod->functions.size() != 0) {
      LOG(WARNING) << "Device Compile:" << target_->kind->name;

      auto gf = ffi::Function::GetGlobal("target.build.opencl");
      ICHECK(gf) << "Global function not found : target.build.opencl";
      runtime_mods.push_back((*gf)(d_mod, target_).cast<tvm::ffi::Module>());
      /*
      runtime_mods.push_back((*ffi::Function::GetGlobal(ffi::String("target.build.") +
                                                        target_->kind->name))(d_mod, target_)
                                 .cast<tvm::ffi::Module>()); */
    }
  }

  auto m_host = (*ffi::Function::GetGlobal(ffi::String("target.build.") +
                                           target_host_->kind->name))(mhost_all, target_host_)
                    .cast<tvm::ffi::Module>();

  for (auto r_mod : runtime_mods) {
    m_host->ImportModule(r_mod);
  }

  // VM Link
  auto vm_mod = (*ffi::Function::GetGlobal("relax.VMLink"))(
                    ex_builder, target_, m_host, tvm::ffi::Array<tvm::ffi::Module>({}),
                    tvm::ffi::Map<tvm::ffi::String, tvm::runtime::Tensor>({}))
                    .cast<tvm::ffi::Module>();
  return vm_mod;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tvm.driver.compile", Compile);
}

}  // namespace driver
}  // namespace tvm
