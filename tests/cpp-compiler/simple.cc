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

// CPP Compiler tests

#include <gtest/gtest.h>
#include <tvm/driver/compile.h>

#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_set>

class CPPCompiler : public ::testing::TestWithParam<std::tuple<DLDeviceType, std::string>> {
  void SetUp() override {
    std::cout << "Setting up for: " << std::get<0>(GetParam()) << " : " << std::get<1>(GetParam())
              << std::endl;
    dtype = runtime::DataType(dl_type);
    dl_dev_type = std::get<0>(GetParam());
    dev_name = std::get<1>(GetParam());
    device = tvm::Device({dl_dev_type, 0});
  }

 public:
  bool TargetSetup(void) {
    if (!runtime::DeviceAPI::Get(device, true)) {
      return false;
    }
    tgt = tvm::Target(dev_name);
    tgt_host = tvm::Target("llvm");
    tgt = tvm::Target::WithHost(tgt, tgt_host);
    return true;
  }

  relax::BlockBuilder BuilderSetup(const ffi::Array<relax::Var>& args) {
    relax::BlockBuilder ctx_ = relax::BlockBuilder::Create(std::nullopt);
    ctx_->BeginScope(args);
    ctx_->BeginDataflowBlock();
    return ctx_;
  }

  tvm::IRModule BuilderFinalize(const relax::BlockBuilder& ctx_, const ffi::Array<relax::Var>& args,
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

  ffi::Module Compile(const tvm::IRModule& mod_) {
    auto vm_mod =
        tvm::driver::Compile(mod_, tgt, ffi::String("gpu_generic"), ffi::String("generic"));
    auto vm_ex = vm_mod.as<runtime::vm::VMExecutable>();
    LOG(INFO) << "About to Load Executable";
    auto vm = vm_ex->VMLoadExecutable();
    vm->GetFunction("vm_initialization")
        .value()(dl_dev_type, 0, runtime::memory::AllocatorType::kPooled, kDLCPU, 0,
                 runtime::memory::AllocatorType::kPooled);
    return vm;
  }

  runtime::Tensor Run(const ffi::Module& vm, const std::vector<ffi::AnyView>& args) {
    ffi::Any ret;

    LOG(INFO) << "About to set_input";
    vm->GetFunction("set_input")
        .value()
        .CallPacked(ffi::PackedArgs(args.data(), args.size()), &ret);

    LOG(INFO) << "About to Invoke";
    vm->GetFunction("invoke_stateful").value()("main");

    LOG(INFO) << "About to Get Output";
    return vm->GetFunction("get_output").value()("main").cast<runtime::Tensor>();
  }

  DLDataType dl_type = {kDLFloat, 32, 1};
  runtime::DataType dtype;
  DLDeviceType dl_dev_type;
  std::string dev_name;
  tvm::Device device;
  tvm::Target tgt;
  tvm::Target tgt_host;
};

TEST_P(CPPCompiler, Binary) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  auto N = tir::Var("N", runtime::DataType::Int(64));
  auto M = tir::Var("M", runtime::DataType::Int(64));
  ffi::Array<tvm::PrimExpr> tvm_shape = {N.as<tvm::PrimExpr>().value(),
                                         M.as<tvm::PrimExpr>().value(), 4, 5};
  ffi::Array<tvm::PrimExpr> tvm_shape_c = {1, 1, 4, 5};

  relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);
  relax::TensorStructInfo ts_info_c = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape_c), dtype);
  auto A = relax::Var("A", ts_info);
  auto B = relax::Var("B", ts_info);
  auto input_tensor_c =
      runtime::Tensor::Empty(ffi::Shape({1, 1, 4, 5}), dl_type, device, std::nullopt);

  auto C = relax::Var("C", ts_info_c);
  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(A);
  tvm_args.push_back(B);
  tvm_args.push_back(C);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& add_op_ = tvm::Op::Get("relax.add");
  const tvm::Op& subtract_op_ = tvm::Op::Get("relax.subtract");

  auto result = ctx_->Emit(relax::Call(add_op_, {A, B}, tvm::Attrs(), {}));
  result = ctx_->Emit(relax::Call(subtract_op_, {result, C}, tvm::Attrs(), {}));
  auto result_out = ctx_->EmitOutput(result, "add_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);
  LOG(INFO) << "Mod:" << mod_;

  // Compile
  auto vm = Compile(mod_);

  // Inputs
  auto input_tensor_a =
      runtime::Tensor::Empty(ffi::Shape({10, 30, 4, 5}), dl_type, device, std::nullopt);
  auto input_tensor_b =
      runtime::Tensor::Empty(ffi::Shape({10, 30, 4, 5}), dl_type, device, std::nullopt);
  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor_a, input_tensor_b,
                                           input_tensor_c};

  // Run
  auto output = Run(vm, packed_args);
  LOG(INFO) << "Result:" << output.shape();
}

TEST_P(CPPCompiler, BinaryScalar) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  auto N = tir::Var("N", runtime::DataType::Int(64));
  auto M = tir::Var("M", runtime::DataType::Int(64));
  ffi::Array<tvm::PrimExpr> tvm_shape = {N.as<tvm::PrimExpr>().value(),
                                         M.as<tvm::PrimExpr>().value(), 4, 5};
  relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);
  auto A = relax::Var("A", ts_info);
  auto B = relax::Var("B", ts_info);

  float c_scalar = 2.5f;
  auto c_tensor =
      runtime::Tensor::Empty(ffi::Shape({}), dl_type, tvm::Device({kDLCPU, 0}), std::nullopt);
  c_tensor.CopyFromBytes(static_cast<void*>(&c_scalar), 4);
  auto C = relax::Constant(c_tensor, std::nullopt);

  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(A);
  tvm_args.push_back(B);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& add_op_ = tvm::Op::Get("relax.add");
  const tvm::Op& subtract_op_ = tvm::Op::Get("relax.subtract");

  auto result = ctx_->Emit(relax::Call(add_op_, {A, B}, tvm::Attrs(), {}));
  result = ctx_->Emit(relax::Call(subtract_op_, {result, C}, tvm::Attrs(), {}));
  auto result_out = ctx_->EmitOutput(result, "add_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);
  LOG(INFO) << "Mod:" << mod_;

  // Compile
  auto vm = Compile(mod_);

  // Inputs
  auto input_tensor_a =
      runtime::Tensor::Empty(ffi::Shape({10, 30, 4, 5}), dl_type, device, std::nullopt);
  auto input_tensor_b =
      runtime::Tensor::Empty(ffi::Shape({10, 30, 4, 5}), dl_type, device, std::nullopt);
  std::vector<ffi::AnyView> packed_args = {
      ffi::String("main"),
      input_tensor_a,
      input_tensor_b,
  };

  // Run
  auto output = Run(vm, packed_args);
  LOG(INFO) << "Result:" << output.shape();
}

INSTANTIATE_TEST_SUITE_P(CPPCompierDevices, CPPCompiler,
                         ::testing::Values(std::make_tuple(kDLCPU, "llvm"),
                                           std::make_tuple(kDLOpenCL, "opencl")));
