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

#include "compiler_base.h"

class Simple : public ::testing::TestWithParam<
                   std::tuple<DLDeviceType, std::string, std::string, std::string>>,
               public CPPCompilerBase {
  void SetUp() override {
    dl_type = {kDLFloat, 32, 1};
    dtype = runtime::DataType(dl_type);
    dl_dev_type = std::get<0>(GetParam());
    dev_name = std::get<1>(GetParam());
    relax_pipeline = std::get<2>(GetParam());
    tir_pipeline = std::get<3>(GetParam());
    device = tvm::Device({dl_dev_type, 0});
  }
};

TEST_P(Simple, Binary) {
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

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor_a =
      runtime::Tensor::Empty(ffi::Shape({10, 30, 4, 5}), dl_type, device, std::nullopt);
  auto input_tensor_b =
      runtime::Tensor::Empty(ffi::Shape({10, 30, 4, 5}), dl_type, device, std::nullopt);
  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor_a, input_tensor_b,
                                           input_tensor_c};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output.shape();
}

TEST_P(Simple, BinaryScalar) {
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

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

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

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output.shape();
}

INSTANTIATE_TEST_SUITE_P(
    CPPCompier, Simple,
    ::testing::Values(std::make_tuple(kDLCPU, "llvm", "cpu_generic", "generic"),
                      std::make_tuple(kDLOpenCL, "opencl", "gpu_generic", "generic"),
                      std::make_tuple(kDLOpenCL, "qcom/adreno-opencl", "adreno", "adreno")));
