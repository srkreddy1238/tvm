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

// CPP Compiler tests - Unary

#include <tvm/relax/attrs/create.h>

#include "compiler_base.h"

class UnaryBase : public CPPCompilerBase {
 public:
  void GraphInit(void) {
    // Inputs
    N = tir::Var("N", runtime::DataType::Int(64));
    M = tir::Var("M", runtime::DataType::Int(64));
    ffi::Array<tvm::PrimExpr> tvm_shape = {N.as<tvm::PrimExpr>().value(),
                                           M.as<tvm::PrimExpr>().value(), 4, 5};
    relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);

    A = relax::Var("A", ts_info);
    tvm_args.push_back(A);

    // BlockBuilder
    ctx_ = BuilderSetup(tvm_args);
  }

  void GraphRun(const relax::Var& result_out) {
    auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

    // Compile
    auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

    // Inputs
    auto input_tensor_a =
        runtime::Tensor::Empty(ffi::Shape({10, 30, 4, 5}), dl_type, device, std::nullopt);
    std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor_a};

    // VMRun
    auto output = VMRun(vm, packed_args);
    LOG(INFO) << "Result:" << output.shape();
  }

  tir::Var N, M;
  std::string relax_op;
  relax::Var A;
  ffi::Array<relax::Var> tvm_args;
  relax::BlockBuilder ctx_;
};

class Unary : public ::testing::TestWithParam<
                  std::tuple<std::tuple<DLDeviceType, std::string, std::string, std::string>,
                             std::tuple<std::string, DLDataType>>>,
              public UnaryBase {
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
};

TEST_P(Unary, Ops) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  GraphInit();
  const tvm::Op& relax_op_ = tvm::Op::Get("relax." + relax_op);

  auto result = ctx_->Emit(relax::Call(relax_op_, {A}, tvm::Attrs(), {}));
  auto result_out = ctx_->EmitOutput(result, relax_op + "_out");

  GraphRun(result_out);
}

INSTANTIATE_TEST_SUITE_P(
    CPPCompierUnary, Unary,
    ::testing::Combine(
        ::testing::ValuesIn(
            std::vector<std::tuple<DLDeviceType, std::string, std::string, std::string>>{
                std::make_tuple(kDLCPU, "llvm", "cpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "opencl", "gpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "qcom/adreno-opencl", "adreno", "adreno")}),
        ::testing::ValuesIn(std::vector<std::tuple<std::string, DLDataType>>{
            std::make_tuple("acos", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("acosh", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("asin", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("asinh", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("atan", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("atanh", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("bitwise_not", DLDataType({kDLInt, 32, 1})),
            std::make_tuple("cos", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("erf", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("exp", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("fast_erf", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("fast_exp", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("fast_tanh", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("identity", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("log", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("log10", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("log2", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("logical_not", DLDataType({kDLBool, 8, 1})),
            std::make_tuple("negative", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("rsqrt", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("sigmoid", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("sin", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("sinh", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("sqrt", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("tan", DLDataType({kDLFloat, 32, 1})),
            std::make_tuple("tanh", DLDataType({kDLFloat, 32, 1}))})));

class UnaryClip : public ::testing::TestWithParam<
                      std::tuple<std::tuple<DLDeviceType, std::string, std::string, std::string>,
                                 std::tuple<DLDataType, float, float>>>,
                  public UnaryBase {
  void SetUp() override {
    auto env = std::get<0>(GetParam());
    dl_dev_type = std::get<0>(env);
    dev_name = std::get<1>(env);
    relax_pipeline = std::get<2>(env);
    tir_pipeline = std::get<3>(env);
    device = tvm::Device({dl_dev_type, 0});

    auto op_spec = std::get<1>(GetParam());
    dl_type = std::get<0>(op_spec);
    dtype = runtime::DataType(dl_type);

    min_val = std::get<1>(op_spec);
    max_val = std::get<2>(op_spec);
  }

 public:
  float min_val;
  float max_val;
};

TEST_P(UnaryClip, Clip) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  GraphInit();
  const tvm::Op& relax_op_ = tvm::Op::Get("relax.clip");
  relax::Expr min, max;
  if (dtype.code() == kDLInt) {
    min = relax::PrimValue(tvm::PrimExpr((int)min_val));
    max = relax::PrimValue(tvm::PrimExpr((int)max_val));
  } else {
    min = relax::PrimValue(tvm::PrimExpr(min_val));
    max = relax::PrimValue(tvm::PrimExpr(max_val));
  }

  auto result = ctx_->Emit(relax::Call(relax_op_, {A, min, max}, tvm::Attrs(), {}));
  auto result_out = ctx_->EmitOutput(result, "clip_out");

  GraphRun(result_out);
}

INSTANTIATE_TEST_SUITE_P(
    CPPCompierUnaryClip, UnaryClip,
    ::testing::Combine(
        ::testing::ValuesIn(
            std::vector<std::tuple<DLDeviceType, std::string, std::string, std::string>>{
                std::make_tuple(kDLCPU, "llvm", "cpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "opencl", "gpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "qcom/adreno-opencl", "adreno", "adreno")}),
        ::testing::ValuesIn(std::vector<std::tuple<DLDataType, float, float>>{
            std::make_tuple(DLDataType({kDLFloat, 32, 1}), 0.0f, 0.6f),
            std::make_tuple(DLDataType({kDLFloat, 32, 1}), -0.3f, 0.3f),
            std::make_tuple(DLDataType({kDLInt, 32, 1}), 1, 5),
            std::make_tuple(DLDataType({kDLInt, 32, 1}), -3, 3)})));
