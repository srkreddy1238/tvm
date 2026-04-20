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

// CPP Compiler tests - Search

#include <tvm/relax/attrs/search.h>

#include "compiler_base.h"

class Search : public ::testing::TestWithParam<
                   std::tuple<std::tuple<DLDeviceType, std::string, std::string, std::string>,
                              DLDataType, std::string, bool>>,
               public CPPCompilerBase {
  void SetUp() override {
    auto env = std::get<0>(GetParam());
    dl_dev_type = std::get<0>(env);
    dev_name = std::get<1>(env);
    relax_pipeline = std::get<2>(env);
    tir_pipeline = std::get<3>(env);
    device = tvm::Device({dl_dev_type, 0});

    dl_type = std::get<1>(GetParam());
    dtype = runtime::DataType(dl_type);
    relax_op = std::get<2>(GetParam());
    is_symbolic = std::get<3>(GetParam());
  }

 public:
  std::string relax_op;
  bool is_symbolic;
};

TEST_P(Search, ArgMinArgMax) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<tvm::PrimExpr> tvm_shape;
  if (is_symbolic) {
    auto N = tir::Var("N", runtime::DataType::Int(64));
    tvm_shape = ffi::Array<tvm::PrimExpr>({N.as<tvm::PrimExpr>().value(), 3, 4, 5});
  } else {
    tvm_shape = ffi::Array<tvm::PrimExpr>({2, 3, 4, 5});
  }
  relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);

  auto input = relax::Var("input", ts_info);
  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(input);
  ffi::ObjectPtr<relax::ArgmaxArgminAttrs> attrs = ffi::make_object<relax::ArgmaxArgminAttrs>();

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax." + relax_op);

  auto result = ctx_->Emit(relax::Call(relax_op_, {input}, tvm::Attrs(attrs), {}));
  auto result_out = ctx_->EmitOutput(result, relax_op + "_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor =
      runtime::Tensor::Empty(ffi::Shape({2, 3, 4, 5}), dl_type, device, std::nullopt);

  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

INSTANTIATE_TEST_SUITE_P(
    CPPCompierSearch, Search,
    ::testing::Combine(
        ::testing::ValuesIn(
            std::vector<std::tuple<DLDeviceType, std::string, std::string, std::string>>{
                std::make_tuple(kDLCPU, "llvm", "cpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "opencl", "gpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "qcom/adreno-opencl", "adreno", "adreno")}),
        ::testing::ValuesIn(std::vector<DLDataType>{DLDataType({kDLFloat, 32, 1}),
                                                    DLDataType({kDLInt, 32, 1})}),
        ::testing::ValuesIn(std::vector<std::string>{"argmin", "argmax"}),
        ::testing::ValuesIn(std::vector<bool>{true, false})));
