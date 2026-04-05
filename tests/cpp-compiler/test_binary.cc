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

// CPP Compiler tests - Binary

#include "compiler_base.h"

class Binary : public ::testing::TestWithParam<
                   std::tuple<std::tuple<DLDeviceType, std::string, std::string, std::string>,
                              std::tuple<std::string, DLDataType>>>,
               public CPPCompilerBase {
  void SetUp() override {
    auto env = std::get<0>(GetParam());
    dl_dev_type = std::get<0>(env);
    dev_name = std::get<1>(env);
    relax_pipeline = std::get<2>(env);
    tir_pipeline = std::get<3>(env);
    device = tvm::Device({dl_dev_type, 0});

    auto op_spec = std::get<1>(GetParam());
    relax_op = std::get<0>(op_spec);
    dl_type = std::get<1>(op_spec);
    dtype = runtime::DataType(dl_type);
  }

 public:
  std::string relax_op;
};

TEST_P(Binary, Ops) {
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
  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(A);
  tvm_args.push_back(B);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& add_op_ = tvm::Op::Get("relax." + relax_op);

  auto result = ctx_->Emit(relax::Call(add_op_, {A, B}, tvm::Attrs(), {}));
  auto result_out = ctx_->EmitOutput(result, relax_op + "_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor_a =
      runtime::Tensor::Empty(ffi::Shape({10, 30, 4, 5}), dl_type, device, std::nullopt);
  auto input_tensor_b =
      runtime::Tensor::Empty(ffi::Shape({10, 30, 4, 5}), dl_type, device, std::nullopt);
  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor_a, input_tensor_b};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

INSTANTIATE_TEST_SUITE_P(
    CPPCompierBinary, Binary,
    ::testing::Combine(
        ::testing::ValuesIn(
            std::vector<std::tuple<DLDeviceType, std::string, std::string, std::string>>{
                std::make_tuple(kDLCPU, "llvm", "cpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "opencl", "gpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "qcom/adreno-opencl", "adreno", "adreno")}),
        ::testing::ValuesIn(std::vector<std::tuple<std::string, DLDataType>>{
            std::make_tuple("add", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("subtract", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("divide", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("floor_divide", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("log_add_exp", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("multiply", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("power", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("equal", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("mod", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("floor_mod", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("greater", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("greater_equal", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("less", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("less_equal", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("not_equal", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("maximum", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("minimum", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("bitwise_and", DLDataType({kDLInt, 32, 1})),
            std::make_tuple("bitwise_or", DLDataType({kDLInt, 32, 1})),
            std::make_tuple("bitwise_xor", DLDataType({kDLInt, 32, 1})),
            std::make_tuple("left_shift", DLDataType({kDLInt, 32, 1})),
            std::make_tuple("right_shift", DLDataType({kDLInt, 32, 1})),
            std::make_tuple("logical_and", DLDataType({kDLBool, 8, 1})),
            std::make_tuple("logical_or", DLDataType({kDLBool, 8, 1})),
            std::make_tuple("logical_xor", DLDataType({kDLBool, 8, 1}))})));
