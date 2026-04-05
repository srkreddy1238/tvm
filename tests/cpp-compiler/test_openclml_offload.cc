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

// CPP Compiler tests : OpenCLML Offloading

#include "compiler_base.h"

class CLMLOffLoadConv2D
    : public ::testing::TestWithParam<
          std::tuple<DLDeviceType, std::string, std::string, std::string,
                     std::tuple<ffi::Array<int64_t>, ffi::Array<int64_t>, ffi::Array<int64_t>,
                                ffi::Array<int64_t>, ffi::Array<int64_t>, int>,
                     bool, bool, bool, bool, bool>>,
      public CPPCompilerBase {
 public:
  void SetUp() override {
    if (!std::getenv("ADRENO_TARGET")) {
      GTEST_SKIP() << "OpenCLML need Adreno target Env: ADRENO_TARGET";
      return;
    }
    // Parent
    dl_type = {kDLFloat, 32, 1};
    dtype = runtime::DataType(dl_type);
    dl_dev_type = std::get<0>(GetParam());
    dev_name = std::get<1>(GetParam());
    relax_pipeline = std::get<2>(GetParam());
    tir_pipeline = std::get<3>(GetParam());
    device = tvm::Device({dl_dev_type, 0});

    // Local
    auto conv2d_tuple = std::get<4>(GetParam());

    data_shape_i = std::get<0>(conv2d_tuple);
    kernel_shape_i = std::get<1>(conv2d_tuple);
    strides = std::get<2>(conv2d_tuple);
    pad = std::get<3>(conv2d_tuple);
    dilation = std::get<4>(conv2d_tuple);
    groups = std::get<5>(conv2d_tuple);

    has_bias = std::get<5>(GetParam());
    has_bn = std::get<6>(GetParam());
    has_activation = std::get<7>(GetParam());
    has_pad = std::get<8>(GetParam());
    use_consts = std::get<9>(GetParam());

    for (auto val : data_shape_i) data_shape_prim.push_back((int32_t)val);
    for (auto val : kernel_shape_i) kernel_shape_prim.push_back((int32_t)val);
    data_shape = relax::ShapeExpr(data_shape_prim);
    kernel_shape = relax::ShapeExpr(kernel_shape_prim);

    // Ref device settings
    ref_target = tvm::Target("llvm");
    ref_dl_dev_type = kDLCPU;
    ref_relax_pipe = "gpu_generic";
    ref_tir_pipe = "generic";

    if (!TargetSetup()) {
      GTEST_SKIP() << "Target setup failed for: " << dev_name;
      return;
    }
  }

  // Conv2D Op
  ffi::Array<tvm::PrimExpr> data_shape_prim;
  ffi::Array<tvm::PrimExpr> kernel_shape_prim;
  ffi::Array<int64_t> data_shape_i;
  ffi::Array<int64_t> kernel_shape_i;
  relax::ShapeExpr data_shape;
  relax::ShapeExpr kernel_shape;
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> pad;
  ffi::Array<int64_t> dilation;
  int groups;
  bool has_bias;
  bool has_bn;
  bool has_activation;
  bool has_pad;
  bool use_consts;

  // Ref dev
  tvm::Target ref_target;
  DLDeviceType ref_dl_dev_type;
  ffi::String ref_relax_pipe;
  ffi::String ref_tir_pipe;
};

TEST_P(CLMLOffLoadConv2D, Conv2D) {
  relax::TensorStructInfo data_tsinfo = relax::TensorStructInfo(data_shape, dtype);
  relax::TensorStructInfo kernel_tsinfo = relax::TensorStructInfo(kernel_shape, dtype);
  relax::TensorStructInfo bias_tsinfo =
      relax::TensorStructInfo(relax::ShapeExpr({1, kernel_shape_prim[0], 1, 1}), dtype);
  relax::TensorStructInfo bn_tsinfo =
      relax::TensorStructInfo(relax::ShapeExpr({kernel_shape_prim[0]}), dtype);

  ffi::Map<ffi::Any, runtime::Tensor> params = {};
  ffi::Array<relax::Var> tvm_args;

  auto data = relax::Var("data", data_tsinfo);
  tvm_args.push_back(data);

  relax::Expr weight, bias, gamma, beta, mean, variance;

  if (use_consts) {
    params.Set("weight", runtime::Tensor::Empty(ffi::Shape(kernel_shape_i), dl_type,
                                                tvm::Device({kDLCPU, 0}), std::nullopt));
    weight = relax::Constant(params["weight"], std::nullopt);

    if (has_bias) {
      params.Set("bias", runtime::Tensor::Empty(ffi::Shape({1, kernel_shape_i[0], 1, 1}), dl_type,
                                                tvm::Device({kDLCPU, 0}), std::nullopt));
      bias = relax::Constant(params["bias"], std::nullopt);
    }

    if (has_bn) {
      params.Set("gamma", runtime::Tensor::Empty(ffi::Shape({kernel_shape_i[0]}), dl_type,
                                                 tvm::Device({kDLCPU, 0}), std::nullopt));
      params.Set("beta", runtime::Tensor::Empty(ffi::Shape({kernel_shape_i[0]}), dl_type,
                                                tvm::Device({kDLCPU, 0}), std::nullopt));
      params.Set("mean", runtime::Tensor::Empty(ffi::Shape({kernel_shape_i[0]}), dl_type,
                                                tvm::Device({kDLCPU, 0}), std::nullopt));
      params.Set("variance", runtime::Tensor::Empty(ffi::Shape({kernel_shape_i[0]}), dl_type,
                                                    tvm::Device({kDLCPU, 0}), std::nullopt));

      gamma = relax::Constant(params["gamma"], std::nullopt);
      beta = relax::Constant(params["beta"], std::nullopt);
      mean = relax::Constant(params["mean"], std::nullopt);
      variance = relax::Constant(params["variance"], std::nullopt);
    }
  } else {
    weight = relax::Var("weight", kernel_tsinfo);
    tvm_args.push_back(tvm::Downcast<relax::Var>(weight));
    if (has_bias) {
      bias = relax::Var("bias", bias_tsinfo);
      tvm_args.push_back(tvm::Downcast<relax::Var>(bias));
    }
    if (has_bn) {
      gamma = relax::Var("gamma", bn_tsinfo);
      beta = relax::Var("beta", bn_tsinfo);
      mean = relax::Var("mean", bn_tsinfo);
      variance = relax::Var("variance", bn_tsinfo);
      tvm_args.push_back(tvm::Downcast<relax::Var>(gamma));
      tvm_args.push_back(tvm::Downcast<relax::Var>(beta));
      tvm_args.push_back(tvm::Downcast<relax::Var>(mean));
      tvm_args.push_back(tvm::Downcast<relax::Var>(variance));
    }
  }

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);
  ffi::Array<int64_t> padding = {pad[0], pad[1], pad[0], pad[1]};
  relax::Expr conv_in = data.as<relax::Expr>().value();
  if (has_pad) {
    ffi::Array<tvm::Integer> pad_width = {0,
                                          0,
                                          0,
                                          0,
                                          (int32_t)padding[0],
                                          (int32_t)padding[0],
                                          (int32_t)padding[1],
                                          (int32_t)padding[1]};
    conv_in = (*ffi::Function::GetGlobal("relax.op.nn.pad"))(data, pad_width, "constant", 0.0)
                  .cast<relax::Call>();
    padding = {0, 0, 0, 0};
  }
  relax::Expr out = (*ffi::Function::GetGlobal("relax.op.nn.conv2d"))(
                        conv_in, weight, strides, padding, dilation, groups, ffi::String("NCHW"),
                        ffi::String("OIHW"), ffi::String("NCHW"), dtype)
                        .cast<relax::Call>();
  if (has_bias) {
    out = (*ffi::Function::GetGlobal("relax.op.add"))(out, bias).cast<relax::Call>();
  }
  if (has_bn) {
    out = (*ffi::Function::GetGlobal("relax.op.nn.batch_norm"))(out, gamma, beta, mean, variance, 1,
                                                                0.00001, true, true, 0.1, false)
              .cast<relax::Call>();
    out = relax::TupleGetItem(out, 0);
  }
  if (has_activation) {
    out = (*ffi::Function::GetGlobal("relax.op.nn.relu"))(out).cast<relax::Call>();
  }
  auto result_out = ctx_->EmitOutput(ctx_->Emit(out), "out");
  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  auto ref_mod = mod_;
  ref_mod.CopyOnWrite();

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);
  // auto ref_vm = Compile(ref_mod, ref_target, ref_dl_dev_type, ref_relax_pipe, ref_tir_pipe)

  // Inputs
  std::vector<runtime::Tensor> inputs = InitRandomInputs(ref_mod, device);
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  for (const auto& tensor : inputs) {
    packed_args.push_back(tensor);
  }

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

INSTANTIATE_TEST_SUITE_P(
    CPPOpenCLML, CLMLOffLoadConv2D,
    ::testing::Combine(
        ::testing::ValuesIn(std::vector<DLDeviceType>({kDLOpenCL})),
        ::testing::ValuesIn(std::vector<std::string>({"qcom/adreno-opencl-clml"})),
        ::testing::ValuesIn(std::vector<std::string>({"adreno"})),
        ::testing::ValuesIn(std::vector<std::string>({"adreno"})),
        ::testing::ValuesIn(
            std::vector<std::tuple<ffi::Array<int64_t>, ffi::Array<int64_t>, ffi::Array<int64_t>,
                                   ffi::Array<int64_t>, ffi::Array<int64_t>, int>>{
                std::make_tuple(ffi::Array<int64_t>({1, 4, 224, 224}),
                                ffi::Array<int64_t>({64, 4, 3, 3}), ffi::Array<int64_t>({2, 2}),
                                ffi::Array<int64_t>({1, 1}), ffi::Array<int64_t>({1, 1}), 1),
                std::make_tuple(ffi::Array<int64_t>({1, 256, 14, 14}),
                                ffi::Array<int64_t>({512, 256, 3, 3}), ffi::Array<int64_t>({1, 1}),
                                ffi::Array<int64_t>({0, 0}), ffi::Array<int64_t>({1, 1}), 1),
                std::make_tuple(ffi::Array<int64_t>({1, 512, 7, 7}),
                                ffi::Array<int64_t>({1024, 512, 1, 1}), ffi::Array<int64_t>({1, 1}),
                                ffi::Array<int64_t>({0, 0}), ffi::Array<int64_t>({1, 1}), 1),
                std::make_tuple(ffi::Array<int64_t>({1, 64, 7, 7}),
                                ffi::Array<int64_t>({64, 64, 1, 3}), ffi::Array<int64_t>({1, 1}),
                                ffi::Array<int64_t>({0, 0}), ffi::Array<int64_t>({1, 1}), 1),
                std::make_tuple(ffi::Array<int64_t>({1, 64, 3, 1}),
                                ffi::Array<int64_t>({64, 64, 1, 1}), ffi::Array<int64_t>({1, 1}),
                                ffi::Array<int64_t>({0, 0}), ffi::Array<int64_t>({1, 1}), 1),
            }),
        ::testing::ValuesIn(std::vector<bool>({true, false})),
        ::testing::ValuesIn(std::vector<bool>({true, false})),
        ::testing::ValuesIn(std::vector<bool>({true, false})),
        ::testing::ValuesIn(std::vector<bool>({true, false})),
        ::testing::ValuesIn(std::vector<bool>({true, false}))));
