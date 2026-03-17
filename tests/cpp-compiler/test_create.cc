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

// CPP Compiler tests - Create

#include <tvm/relax/attrs/create.h>

#include "compiler_base.h"

class CreateBase : public CPPCompilerBase {
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

class CreateFull : public ::testing::TestWithParam<
                       std::tuple<std::tuple<DLDeviceType, std::string, std::string, std::string>,
                                  std::tuple<std::string, bool, bool, DLDataType, float>>>,
                   public CreateBase {
  void SetUp() override {
    auto env = std::get<0>(GetParam());
    dl_dev_type = std::get<0>(env);
    dev_name = std::get<1>(env);
    relax_pipeline = std::get<2>(env);
    tir_pipeline = std::get<3>(env);
    device = tvm::Device({dl_dev_type, 0});

    auto op_spec = std::get<1>(GetParam());
    relax_op = std::get<0>(op_spec);
    is_like = std::get<1>(op_spec);
    by_value = std::get<2>(op_spec);
    dl_type = std::get<3>(op_spec);
    dtype = runtime::DataType(dl_type);
    value = std::get<4>(op_spec);
  }

 public:
  float value;
  bool is_like;
  bool by_value;
};

TEST_P(CreateFull, Full) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  const tvm::Op& relax_op_ = tvm::Op::Get("relax." + relax_op);

  GraphInit();

  ffi::ObjectPtr<relax::InitAttrs> attrs = ffi::make_object<relax::InitAttrs>();
  attrs->dtype = dtype;

  relax::Expr out;
  tvm::ffi::Array<relax::Expr> call_args;

  if (is_like) {
    call_args.push_back(A);
  } else {
    auto shape = ffi::Array<tvm::PrimExpr>(
        {N.as<tvm::PrimExpr>().value(), M.as<tvm::PrimExpr>().value(), 4, 5});
    call_args.push_back(relax::ShapeExpr(shape));
  }

  if (!by_value) {
    auto value_tensor =
        runtime::Tensor::Empty(ffi::Shape({}), dl_type, tvm::Device({kDLCPU, 0}), std::nullopt);
    if (dtype.code() == kDLInt) {
      int i_val = value;
      value_tensor.CopyFromBytes(static_cast<void*>(&i_val), 4);
    } else {
      float f_val = value;
      value_tensor.CopyFromBytes(static_cast<void*>(&f_val), 4);
    }
    auto V = relax::Constant(
        value_tensor,
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({})), dtype));

    call_args.push_back(V);
  }

  out = relax::Call(relax_op_, call_args, tvm::Attrs(attrs), {});

  auto result = ctx_->Emit(relax::Call(tvm::Op::Get("relax.add"), {A, out}, tvm::Attrs(), {}));

  auto result_out = ctx_->EmitOutput(result, relax_op + "_out");

  GraphRun(result_out);
}

INSTANTIATE_TEST_SUITE_P(
    CPPCompierCreateFull, CreateFull,
    ::testing::Combine(
        ::testing::ValuesIn(
            std::vector<std::tuple<DLDeviceType, std::string, std::string, std::string>>{
                std::make_tuple(kDLCPU, "llvm", "cpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "opencl", "gpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "qcom/adreno-opencl", "adreno", "adreno")}),
        ::testing::ValuesIn(std::vector<std::tuple<std::string, bool, bool, DLDataType, float>>{
            std::make_tuple("full", false, false, DLDataType({kDLFloat, 32, 1}), 3.5f),
            std::make_tuple("full", false, false, DLDataType({kDLInt, 32, 1}), 6.0f),
            std::make_tuple("full_like", true, false, DLDataType({kDLFloat, 32, 1}), 3.5f),
            std::make_tuple("full_like", true, false, DLDataType({kDLInt, 32, 1}), 6.0f),
            std::make_tuple("ones", false, true, DLDataType({kDLFloat, 32, 1}),
                            0.0f), /* value ignored */
            std::make_tuple("ones", false, true, DLDataType({kDLInt, 32, 1}),
                            0.0f), /* value ignored */
            std::make_tuple("ones_like", true, true, DLDataType({kDLFloat, 32, 1}),
                            0.0f), /* value ignored */
            std::make_tuple("ones_like", true, true, DLDataType({kDLInt, 32, 1}),
                            0.0f), /* value ignored */
            std::make_tuple("zeros", false, true, DLDataType({kDLFloat, 32, 1}),
                            0.0f), /* value ignored */
            std::make_tuple("zeros", false, true, DLDataType({kDLInt, 32, 1}),
                            0.0f), /* value ignored */
            std::make_tuple("zeros_like", true, true, DLDataType({kDLFloat, 32, 1}),
                            0.0f), /* value ignored */
            std::make_tuple("zeros_like", true, true, DLDataType({kDLInt, 32, 1}),
                            0.0f), /* value ignored */

        })));
