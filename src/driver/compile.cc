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

std::pair<tvm::IRModule, ffi::Map<tvm::Target, tvm::IRModule>> SplitHostDeviceMods(
    const tvm::IRModule& mod) {
  tvm::Target target;
  tvm::IRModule host_mod;
  ffi::Map<tvm::Target, tvm::IRModule> device_mod_dict;
  bool is_host = true;

  std::function<bool(const tir::PrimFunc&)> is_host_func = [&](const tir::PrimFunc& func) -> bool {
    tvm::Target tgt = func->GetAttr<tvm::Target>("target", tvm::Target("llvm")).value();
    if (tgt->kind->name == "llvm" || tgt->kind->name == "c") {
      return is_host ? true : false;
    } else {
      return is_host ? false : true;
    }
  };

  auto pass = ffi::Function::GetGlobal("tir.transform.Filter");
  host_mod = (*pass)(ffi::TypedFunction<bool(tir::PrimFunc)>(is_host_func))
                 .cast<tvm::transform::Pass>()(mod);
  is_host = false;
  auto device_mod = (*pass)(ffi::TypedFunction<bool(tir::PrimFunc)>(is_host_func))
                        .cast<tvm::transform::Pass>()(mod);

  ffi::Map<ffi::String, tvm::Target> target_str2target;
  ffi::Map<ffi::String, ffi::Map<tvm::GlobalVar, tvm::BaseFunc>> device_func_dict;
  for (auto it : device_mod->functions) {
    tvm::Target tgt = it.second->GetAttr<tvm::Target>("target", tvm::Target()).value();
    target_str2target.Set(tgt->str(), tgt);
    ffi::Map<tvm::GlobalVar, tvm::BaseFunc> dev_funcs;
    if (device_func_dict.find(tgt->str()) != device_func_dict.end()) {
      dev_funcs = device_func_dict[tgt->str()];
    }
    dev_funcs.Set(it.first, it.second);
    device_func_dict.Set(tgt->str(), dev_funcs);
  }

  for (auto it : device_func_dict) {
    auto tgt = tvm::Target(it.first);

    auto d_mod = tvm::IRModule(it.second, tvm::SourceMap({}), device_mod->attrs,
                               ffi::Map<ffi::String, ffi::Array<tvm::GlobalInfo>>({}));
    device_mod_dict.Set(tgt, d_mod);
  }

  return std::pair<tvm::IRModule, ffi::Map<tvm::Target, tvm::IRModule>>(
      {host_mod, device_mod_dict});
}

ffi::Module Compile(IRModule mod, ffi::Any target, ffi::Optional<ffi::String> relax_pipeline,
                    ffi::Optional<ffi::String> tir_pipeline) {
  ObjectPtr<runtime::vm::VMExecutable> executable;
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
  Target target_host_ = runtime::RuntimeEnabled("llvm") ? Target("llvm") : Target("c");
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
      [&target_](IRModule& mod_, const ffi::String& pass) -> IRModule {
    auto gf = ffi::Function::GetGlobal(pass);
    TVM_FFI_ICHECK(gf) << "Global function not found : " << pass;
    tvm::transform::Pass pass_h = (*gf)(target_).cast<tvm::transform::Pass>();
    return pass_h(mod_);
  };

  // Relax Pipeline
  mod = apply_module_pass_(mod, ffi::String("relax.pipeline.") + relax_pipeline_);

  // VM Codegen
  relax::ExecBuilder ex_builder =
      (*ffi::Function::GetGlobal("relax.ExecBuilderCreate"))().cast<relax::ExecBuilder>();
  mod = (*ffi::Function::GetGlobal("relax.VMCodeGen"))(ex_builder, mod).cast<tvm::IRModule>();

  // TOR Mod extraction
  std::function<tvm::IRModule(tvm::IRModule&)> _filter_tir =
      [&](tvm::IRModule& mod) -> tvm::IRModule {
    ffi::Map<tvm::GlobalVar, tvm::BaseFunc> tir_funcs;
    for (auto func : mod->functions) {
      if (func.second.as<tir::PrimFunc>()) {
        tir_funcs.Set(func.first, func.second);
      }
    }
    return tvm::IRModule(tir_funcs, tvm::SourceMap({}), mod->attrs,
                         ffi::Map<ffi::String, ffi::Array<tvm::GlobalInfo>>({}));
  };
  auto tir_mod = _filter_tir(mod);

  // TIR Pipeline
  tir_mod = tir::transform::BindTarget(target_)(tir_mod);

  tir_mod = apply_module_pass_(
      tir_mod, ffi::String("s_tir.pipeline.") + tir_pipeline_ + ffi::String(".Base"));

  // Split host and device
  auto [host_mod, device_mod_dict] = SplitHostDeviceMods(tir_mod);

  // TIR Host
  host_mod = apply_module_pass_(
      host_mod, ffi::String("s_tir.pipeline.") + tir_pipeline_ + ffi::String(".Host"));

  // TIR Device
  for (auto [tgt, dmod] : device_mod_dict) {
    dmod = apply_module_pass_(
        dmod, ffi::String("s_tir.pipeline.") + tir_pipeline_ + ffi::String(".Device"));
    device_mod_dict.Set(tgt, dmod);
  }

  // TIR To Runtime
  auto mhost_all = tvm::IRModule({}, tvm::SourceMap({}), host_mod->attrs,
                                 ffi::Map<ffi::String, ffi::Array<tvm::GlobalInfo>>({}));
  mhost_all->Update(host_mod);
  ffi::Array<ffi::Module> runtime_mods;

  for (auto [tgt, d_mod] : device_mod_dict) {
    if (d_mod->functions.size() != 0) {
      ffi::String target_codegen = tgt->GetAttr<ffi::String>("codegen")
                                       ? tgt->GetAttr<ffi::String>("codegen").value()
                                       : ffi::String("target.build.") + tgt->kind->name;
      runtime_mods.push_back(
          (*ffi::Function::GetGlobal(target_codegen))(d_mod, tgt).cast<ffi::Module>());
    }
  }

  auto m_host = (*ffi::Function::GetGlobal(ffi::String("target.build.") +
                                           target_host_->kind->name))(mhost_all, target_host_)
                    .cast<ffi::Module>();

  for (auto r_mod : runtime_mods) {
    m_host->ImportModule(r_mod);
  }

  // VM Link
  auto vm_mod = (*ffi::Function::GetGlobal("relax.VMLink"))(
                    ex_builder, target_, m_host, ffi::Array<ffi::Module>({}),
                    ffi::Map<ffi::String, runtime::Tensor>({}))
                    .cast<ffi::Module>();
  return vm_mod;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::GlobalDef().def("tvm.driver.compile", Compile);
}

}  // namespace driver
}  // namespace tvm
