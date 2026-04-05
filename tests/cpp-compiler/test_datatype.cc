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

// CPP Compiler tests - DataType

#include <tvm/relax/attrs/datatype.h>

#include "compiler_base.h"

class AsTypeBase : public CPPCompilerBase {
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
    LOG(WARNING) << "Mod:" << mod_;

    // Compile
    auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

    // Inputs
    auto input_tensor_a =
        runtime::Tensor::Empty(ffi::Shape({10, 30, 4, 5}), dl_type, device, std::nullopt);
    std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor_a};

    // VMRun
    auto output = VMRun(vm, packed_args);
    LOG(INFO) << "Result:" << output[0].shape();
  }

  tir::Var N, M;
  relax::Var A;
  ffi::Array<relax::Var> tvm_args;
  relax::BlockBuilder ctx_;
};

class AsTypeTest : public ::testing::TestWithParam<
                       std::tuple<std::tuple<DLDeviceType, std::string, std::string, std::string>,
                                  std::tuple<DLDataType, DLDataType>>>,
                   public AsTypeBase {
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
    to_dtype = runtime::DataType(std::get<1>(op_spec));
  }

 public:
  runtime::DataType to_dtype;
};

TEST_P(AsTypeTest, Test) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  GraphInit();

  ffi::ObjectPtr<relax::AstypeAttrs> attrs = ffi::make_object<relax::AstypeAttrs>();
  attrs->dtype = to_dtype;

  auto result = ctx_->Emit(relax::Call(tvm::Op::Get("relax.astype"), {A}, tvm::Attrs(attrs), {}));

  auto result_out = ctx_->EmitOutput(result, "astype_out");

  GraphRun(result_out);
}

INSTANTIATE_TEST_SUITE_P(
    CPPCompierAsTypeTest, AsTypeTest,
    ::testing::Combine(
        ::testing::ValuesIn(
            std::vector<std::tuple<DLDeviceType, std::string, std::string, std::string>>{
                std::make_tuple(kDLCPU, "llvm", "cpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "opencl", "gpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "qcom/adreno-opencl", "adreno", "adreno")}),
        ::testing::ValuesIn(std::vector<std::tuple<DLDataType, DLDataType>>{
            std::make_tuple(DLDataType({kDLFloat, 32, 1}), DLDataType({kDLInt, 32, 1})),
            std::make_tuple(DLDataType({kDLInt, 32, 1}), DLDataType({kDLFloat, 32, 1})),
        })));
