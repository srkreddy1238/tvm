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

#include "compiler_base.h"

#include <tvm/relax/struct_info.h>
#include <tvm/tir/function.h>

#include <algorithm>
#include <random>
#include <type_traits>

bool CPPCompilerBase::TargetSetup(void) {
  if (!runtime::DeviceAPI::Get(device, true)) {
    return false;
  }
  tgt = tvm::Target(dev_name);
  tgt_host = tvm::Target("llvm");
  tgt = tvm::Target::WithHost(tgt, tgt_host);
  return true;
}

relax::BlockBuilder CPPCompilerBase::BuilderSetup(const ffi::Array<relax::Var>& args) {
  relax::BlockBuilder ctx_ = relax::BlockBuilder::Create(std::nullopt);
  ctx_->BeginScope(args);
  ctx_->BeginDataflowBlock();
  return ctx_;
}

tvm::IRModule CPPCompilerBase::BuilderFinalize(const relax::BlockBuilder& ctx_,
                                               const ffi::Array<relax::Var>& args,
                                               const relax::Var& out) {
  auto bind_block = ctx_->EndBlock();
  auto body = ctx_->Normalize(out);
  body = ctx_->Normalize(relax::SeqExpr({bind_block}, body));

  ffi::Map<ffi::String, ffi::Any> fattrs;
  fattrs.Set(tvm::attr::kGlobalSymbol, ffi::String("main"));
  auto func = relax::Function(args, body, std::nullopt, true, tvm::DictAttrs(fattrs));
  ctx_->EndScope();
  ctx_->AddFunction(func, "main");
  return ctx_->Finalize();
}

ffi::Module CPPCompilerBase::Compile(tvm::IRModule mod_, tvm::Target& tgt, DLDeviceType dl_dev_type,
                                     ffi::String relax_pipeline, ffi::String tir_pipeline,
                                     ffi::Map<ffi::Any, ffi::ObjectRef> params) {
  auto vm_mod = tvm::driver::Compile(mod_, tgt, params, relax_pipeline, tir_pipeline);
  auto vm_ex = vm_mod.as<runtime::vm::VMExecutable>();
  auto vm = vm_ex->VMLoadExecutable();
  std::vector<ffi::AnyView> vm_args;
  ffi::Any rv;
  if (mod_->global_infos.find("vdevice") != mod_->global_infos.end()) {
    for (const auto& ginfo : mod_->global_infos["vdevice"]) {
      // Assuming there only one type of target device and is same as dl_device_type
      // May not work for heterogenious
      vm_args.push_back(dl_dev_type);
      vm_args.push_back(0);
      vm_args.push_back(runtime::memory::AllocatorType::kPooled);
    }
  } else {
    vm_args.push_back(dl_dev_type);
    vm_args.push_back(0);
    vm_args.push_back(runtime::memory::AllocatorType::kPooled);
  }
  // Fallback for shape functions
  vm_args.push_back(kDLCPU);
  vm_args.push_back(0);
  vm_args.push_back(runtime::memory::AllocatorType::kPooled);

  vm->GetFunction("vm_initialization")
      .value()
      .CallPacked(ffi::PackedArgs(vm_args.data(), vm_args.size()), &rv);
  return vm;
}

template <typename T>
void FillRandomUniform(T* arr, size_t size, T min, T max) {
  std::random_device rd;
  std::mt19937 gen(rd());

  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> dist(min, max);
    for (size_t i = 0; i < size; ++i) arr[i] = dist(gen);
  } else if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> dist(min, max);
    for (size_t i = 0; i < size; ++i) arr[i] = dist(gen);
  }
}

std::vector<runtime::Tensor> CPPCompilerBase::InitRandomInputs(const tvm::IRModule& mod,
                                                               const tvm::Device& dev) {
  std::vector<runtime::Tensor> args;
  auto main_func = tvm::Downcast<relax::Function>(mod->Lookup("main"));

  for (const auto& param : main_func->params) {
    std::vector<int64_t> r_shape;
    DLDataType r_dtype;
    auto tsinfo = tvm::Downcast<relax::TensorStructInfo>(relax::GetStructInfo(param));
    ffi::Array<tvm::PrimExpr> shape = tsinfo->GetShape().value();
    for (const auto& sval : shape) {
      r_shape.push_back(sval.as<tvm::IntImmNode>()->value);
    }
    r_dtype = DLDataType(tsinfo->dtype);
    auto tensor = runtime::Tensor::Empty(ffi::Shape(r_shape), r_dtype, dev, std::nullopt);
    args.push_back(tensor);

    /*
    size_t data_size = GetDataSize(tensor);
    TVM_FFI_ICHECK(tsinfo->dtype.is_float()) << "Random init supported for float32 only";

    float* data = new float[data_size / sizeof(float)];
    TVM_FFI_ICHECK(data != nullptr) << "Malloc failed";
    FillRandomUniform<float>(data, data_size / sizeof(float), 0.0, 0.1);
    tensor.CopyFromBytes(data, data_size);

    delete[] data;
    */
  }

  return args;
}

runtime::Tensor CPPCompilerBase::VMRun(const ffi::Module& vm,
                                       const std::vector<ffi::AnyView>& args) {
  ffi::Any ret;

  vm->GetFunction("set_input").value().CallPacked(ffi::PackedArgs(args.data(), args.size()), &ret);
  vm->GetFunction("invoke_stateful").value()("main");
  return vm->GetFunction("get_output").value()("main").cast<runtime::Tensor>();
}
