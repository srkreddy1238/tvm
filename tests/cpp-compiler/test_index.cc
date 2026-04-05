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

// CPP Compiler tests - IndexTake

#include <tvm/relax/attrs/index.h>

#include "compiler_base.h"

class IndexTake
    : public ::testing::TestWithParam<std::tuple<
          std::tuple<DLDeviceType, std::string, std::string, std::string>, std::tuple<DLDataType>>>,
      public CPPCompilerBase {
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
  }
};

TEST_P(IndexTake, Take) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<tvm::PrimExpr> tvm_shape = {2, 3, 4};
  relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);

  ffi::Array<tvm::PrimExpr> ind_shape = {4};
  auto ind_dl_type = DLDataType({kDLInt, 64, 1});
  auto ind_dtype = runtime::DataType(ind_dl_type);
  relax::TensorStructInfo ind_ts_info =
      relax::TensorStructInfo(relax::ShapeExpr(ind_shape), ind_dtype);

  auto input = relax::Var("input", ts_info);
  auto indices = relax::Var("indices", ind_ts_info);
  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(input);
  tvm_args.push_back(indices);

  ffi::ObjectPtr<relax::TakeAttrs> attrs = ffi::make_object<relax::TakeAttrs>();
  attrs->axis = 1;
  attrs->mode = "fast";

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.take");

  auto result = ctx_->Emit(relax::Call(relax_op_, {input, indices}, tvm::Attrs(attrs), {}));
  auto result_out = ctx_->EmitOutput(result, "take_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor = runtime::Tensor::Empty(ffi::Shape({2, 3, 4}), dl_type, device, std::nullopt);
  auto indices_tensor = runtime::Tensor::Empty(ffi::Shape({4}), ind_dl_type, device, std::nullopt);

  std::vector<int64_t> indices_data = {1, 2, 2, 1};
  indices_tensor.CopyFromBytes(static_cast<void*>(indices_data.data()), 4 * sizeof(int64_t));

  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor, indices_tensor};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(IndexTake, TakePrim) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<tvm::PrimExpr> tvm_shape = {2, 3, 4};
  relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);

  auto input = relax::Var("input", ts_info);
  auto indices = relax::PrimValue(tvm::PrimExpr(3));
  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(input);

  ffi::ObjectPtr<relax::TakeAttrs> attrs = ffi::make_object<relax::TakeAttrs>();
  attrs->axis = 1;
  attrs->mode = "fast";

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.take");

  auto result = ctx_->Emit(relax::Call(relax_op_, {input, indices}, tvm::Attrs(attrs), {}));
  auto result_out = ctx_->EmitOutput(result, "take_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor = runtime::Tensor::Empty(ffi::Shape({2, 3, 4}), dl_type, device, std::nullopt);

  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(IndexTake, TakeSymbolic) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  auto N = tir::Var("N", runtime::DataType::Int(64));
  auto M = tir::Var("M", runtime::DataType::Int(64));
  ffi::Array<tvm::PrimExpr> tvm_shape = {N.as<tvm::PrimExpr>().value(),
                                         M.as<tvm::PrimExpr>().value(), 4};

  relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);

  ffi::Array<tvm::PrimExpr> ind_shape = {4};
  auto ind_dl_type = DLDataType({kDLInt, 64, 1});
  auto ind_dtype = runtime::DataType(ind_dl_type);
  relax::TensorStructInfo ind_ts_info =
      relax::TensorStructInfo(relax::ShapeExpr(ind_shape), ind_dtype);

  auto input = relax::Var("input", ts_info);
  auto indices = relax::Var("indices", ind_ts_info);

  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(input);
  tvm_args.push_back(indices);

  ffi::ObjectPtr<relax::TakeAttrs> attrs = ffi::make_object<relax::TakeAttrs>();
  attrs->axis = 2;
  attrs->mode = "fast";

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.take");

  auto result = ctx_->Emit(relax::Call(relax_op_, {input, indices}, tvm::Attrs(attrs), {}));
  auto result_out = ctx_->EmitOutput(result, "take_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor =
      runtime::Tensor::Empty(ffi::Shape({20, 30, 4}), dl_type, device, std::nullopt);
  auto indices_tensor = runtime::Tensor::Empty(ffi::Shape({4}), ind_dl_type, device, std::nullopt);

  std::vector<int64_t> indices_data = {1, 2, 2, 1};
  indices_tensor.CopyFromBytes(static_cast<void*>(indices_data.data()), 4 * sizeof(int64_t));

  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor, indices_tensor};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(IndexTake, TakeSymbolicPrim) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  auto N = tir::Var("N", runtime::DataType::Int(64));
  auto M = tir::Var("M", runtime::DataType::Int(64));
  ffi::Array<tvm::PrimExpr> tvm_shape = {N.as<tvm::PrimExpr>().value(),
                                         M.as<tvm::PrimExpr>().value(), 4};

  relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);

  auto input = relax::Var("input", ts_info);
  auto indices = relax::PrimValue(N.as<tvm::PrimExpr>().value() - tvm::PrimExpr(1));
  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(input);

  ffi::ObjectPtr<relax::TakeAttrs> attrs = ffi::make_object<relax::TakeAttrs>();
  attrs->axis = 1;
  attrs->mode = "fast";

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.take");

  auto result = ctx_->Emit(relax::Call(relax_op_, {input, indices}, tvm::Attrs(attrs), {}));
  auto result_out = ctx_->EmitOutput(result, "take_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor =
      runtime::Tensor::Empty(ffi::Shape({20, 30, 4}), dl_type, device, std::nullopt);

  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

INSTANTIATE_TEST_SUITE_P(
    CPPCompierIndexTake, IndexTake,
    ::testing::Combine(
        ::testing::ValuesIn(
            std::vector<std::tuple<DLDeviceType, std::string, std::string, std::string>>{
                std::make_tuple(kDLCPU, "llvm", "cpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "opencl", "gpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "qcom/adreno-opencl", "adreno", "adreno")}),
        ::testing::ValuesIn(std::vector<std::tuple<DLDataType>>{
            std::make_tuple(DLDataType({kDLFloat, 32, 1})),
            std::make_tuple(DLDataType({kDLInt, 32, 1})),
        })));

class IndexStridedSlice
    : public ::testing::TestWithParam<std::tuple<
          std::tuple<DLDeviceType, std::string, std::string, std::string>, std::tuple<DLDataType>>>,
      public CPPCompilerBase {
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
  }
};

TEST_P(IndexStridedSlice, Simple) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs

  ffi::Array<tvm::PrimExpr> tvm_shape = {8, 9, 10, 10};
  relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);

  auto input = relax::Var("input", ts_info);
  auto axes = relax::Tuple(ffi::Array<relax::Expr>(
      {relax::PrimValue::Int64(0), relax::PrimValue::Int64(1), relax::PrimValue::Int64(3)}));
  auto begin = relax::Tuple(ffi::Array<relax::Expr>(
      {relax::PrimValue::Int64(1), relax::PrimValue::Int64(0), relax::PrimValue::Int64(8)}));
  auto end = relax::Tuple(ffi::Array<relax::Expr>(
      {relax::PrimValue::Int64(8), relax::PrimValue::Int64(9), relax::PrimValue::Int64(0)}));
  auto strides = relax::Tuple(ffi::Array<relax::Expr>(
      {relax::PrimValue::Int64(2), relax::PrimValue::Int64(1), relax::PrimValue::Int64(-3)}));

  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(input);

  ffi::ObjectPtr<relax::StridedSliceAttrs> attrs = ffi::make_object<relax::StridedSliceAttrs>();
  attrs->assume_inbound = true;

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.strided_slice");

  auto result =
      ctx_->Emit(relax::Call(relax_op_, {input, axes, begin, end, strides}, tvm::Attrs(attrs), {}));
  auto result_out = ctx_->EmitOutput(result, "strided_slice_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor =
      runtime::Tensor::Empty(ffi::Shape({8, 9, 10, 10}), dl_type, device, std::nullopt);

  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(IndexStridedSlice, NoStrides) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs

  ffi::Array<tvm::PrimExpr> tvm_shape = {8, 9, 10, 10};
  relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);

  auto input = relax::Var("input", ts_info);
  auto axes = relax::Tuple(ffi::Array<relax::Expr>(
      {relax::PrimValue::Int64(0), relax::PrimValue::Int64(1), relax::PrimValue::Int64(3)}));
  auto begin = relax::Tuple(ffi::Array<relax::Expr>(
      {relax::PrimValue::Int64(1), relax::PrimValue::Int64(0), relax::PrimValue::Int64(2)}));
  auto end = relax::Tuple(ffi::Array<relax::Expr>(
      {relax::PrimValue::Int64(8), relax::PrimValue::Int64(9), relax::PrimValue::Int64(4)}));

  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(input);

  ffi::ObjectPtr<relax::StridedSliceAttrs> attrs = ffi::make_object<relax::StridedSliceAttrs>();
  attrs->assume_inbound = true;

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.strided_slice");

  auto result =
      ctx_->Emit(relax::Call(relax_op_, {input, axes, begin, end}, tvm::Attrs(attrs), {}));
  auto result_out = ctx_->EmitOutput(result, "strided_slice_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor =
      runtime::Tensor::Empty(ffi::Shape({8, 9, 10, 10}), dl_type, device, std::nullopt);

  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(IndexStridedSlice, SymbolicSlicedAxis) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs

  auto M = tir::Var("M", runtime::DataType::Int(64));
  auto N = tir::Var("N", runtime::DataType::Int(64));
  ffi::Array<tvm::PrimExpr> tvm_shape = {M.as<tvm::PrimExpr>().value(),
                                         N.as<tvm::PrimExpr>().value()};
  relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);

  auto input = relax::Var("input", ts_info);
  auto axes = relax::Tuple(ffi::Array<relax::Expr>({relax::PrimValue::Int64(0)}));
  auto begin = relax::Tuple(ffi::Array<relax::Expr>({relax::PrimValue::Int64(1)}));
  auto end = relax::Tuple(ffi::Array<relax::Expr>({relax::PrimValue::Int64(8)}));
  auto strides = relax::Tuple(ffi::Array<relax::Expr>({relax::PrimValue::Int64(3)}));

  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(input);

  ffi::ObjectPtr<relax::StridedSliceAttrs> attrs = ffi::make_object<relax::StridedSliceAttrs>();
  attrs->assume_inbound = true;

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.strided_slice");

  auto result =
      ctx_->Emit(relax::Call(relax_op_, {input, axes, begin, end, strides}, tvm::Attrs(attrs), {}));
  auto result_out = ctx_->EmitOutput(result, "strided_slice_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor = runtime::Tensor::Empty(ffi::Shape({8, 9}), dl_type, device, std::nullopt);

  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(IndexStridedSlice, SymbolicSlice) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs

  auto N = tir::Var("N", runtime::DataType::Int(64));
  ffi::Array<tvm::PrimExpr> tvm_shape = {10, N.as<tvm::PrimExpr>().value()};
  relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);

  auto input = relax::Var("input", ts_info);
  auto axes = relax::Tuple(ffi::Array<relax::Expr>({relax::PrimValue::Int64(0)}));
  auto begin = relax::Tuple(ffi::Array<relax::Expr>({relax::PrimValue::Int64(1)}));
  auto end = relax::Tuple(ffi::Array<relax::Expr>({relax::PrimValue::Int64(8)}));
  auto strides = relax::Tuple(ffi::Array<relax::Expr>({relax::PrimValue::Int64(3)}));

  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(input);

  ffi::ObjectPtr<relax::StridedSliceAttrs> attrs = ffi::make_object<relax::StridedSliceAttrs>();
  attrs->assume_inbound = true;

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.strided_slice");

  auto result =
      ctx_->Emit(relax::Call(relax_op_, {input, axes, begin, end, strides}, tvm::Attrs(attrs), {}));
  auto result_out = ctx_->EmitOutput(result, "strided_slice_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor = runtime::Tensor::Empty(ffi::Shape({10, 9}), dl_type, device, std::nullopt);

  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(IndexStridedSlice, SymbolicSliceBound) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs

  auto N = tir::Var("N", runtime::DataType::Int(64));
  ffi::Array<tvm::PrimExpr> tvm_shape = {10, N.as<tvm::PrimExpr>().value()};
  relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);

  auto input = relax::Var("input", ts_info);
  auto axes = relax::Tuple(
      ffi::Array<relax::Expr>({relax::PrimValue::Int64(0), relax::PrimValue::Int64(1)}));
  auto begin = relax::Tuple(
      ffi::Array<relax::Expr>({relax::PrimValue::Int64(1), relax::PrimValue::Int64(0)}));
  auto end = relax::Tuple(ffi::Array<relax::Expr>(
      {relax::PrimValue::Int64(8), relax::PrimValue(N.as<tvm::PrimExpr>().value())}));
  auto strides = relax::Tuple(
      ffi::Array<relax::Expr>({relax::PrimValue::Int64(3), relax::PrimValue::Int64(1)}));

  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(input);

  ffi::ObjectPtr<relax::StridedSliceAttrs> attrs = ffi::make_object<relax::StridedSliceAttrs>();
  attrs->assume_inbound = true;

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.strided_slice");

  auto result =
      ctx_->Emit(relax::Call(relax_op_, {input, axes, begin, end, strides}, tvm::Attrs(attrs), {}));
  auto result_out = ctx_->EmitOutput(result, "strided_slice_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor = runtime::Tensor::Empty(ffi::Shape({10, 20}), dl_type, device, std::nullopt);

  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

INSTANTIATE_TEST_SUITE_P(
    CPPCompierIndexStridedSlice, IndexStridedSlice,
    ::testing::Combine(
        ::testing::ValuesIn(
            std::vector<std::tuple<DLDeviceType, std::string, std::string, std::string>>{
                std::make_tuple(kDLCPU, "llvm", "cpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "opencl", "gpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "qcom/adreno-opencl", "adreno", "adreno")}),
        ::testing::ValuesIn(std::vector<std::tuple<DLDataType>>{
            std::make_tuple(DLDataType({kDLFloat, 32, 1})),
            std::make_tuple(DLDataType({kDLInt, 32, 1})),
        })));

// Dynamic Strided Slice
class IndexDynamicStridedSlice
    : public ::testing::TestWithParam<std::tuple<
          std::tuple<DLDeviceType, std::string, std::string, std::string>, std::tuple<DLDataType>>>,
      public CPPCompilerBase {
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
  }
};

TEST_P(IndexDynamicStridedSlice, Simple) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs

  ffi::Array<tvm::PrimExpr> tvm_shape = {8, 9, 10, 10};
  ffi::Array<tvm::PrimExpr> arg_shape = {4};
  relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);
  relax::TensorStructInfo arg_ts_info = relax::TensorStructInfo(
      relax::ShapeExpr(arg_shape), runtime::DataType(DLDataType({kDLInt, 64, 1})));

  auto input = relax::Var("input", ts_info);
  auto begin = relax::Var("input", arg_ts_info);
  auto end = relax::Var("input", arg_ts_info);
  auto strides = relax::Var("input", arg_ts_info);

  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(input);
  tvm_args.push_back(begin);
  tvm_args.push_back(end);
  tvm_args.push_back(strides);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.dynamic_strided_slice");

  auto result = ctx_->Emit(relax::Call(relax_op_, {input, begin, end, strides}, tvm::Attrs(), {}));
  auto result_out = ctx_->EmitOutput(result, "dynamic_strided_slice_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);
}

TEST_P(IndexDynamicStridedSlice, Symbolic) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs

  auto N = tir::Var("N", runtime::DataType::Int(64));
  ffi::Array<tvm::PrimExpr> tvm_shape = {10, N.as<tvm::PrimExpr>().value()};
  ffi::Array<tvm::PrimExpr> arg_shape = {2};
  relax::TensorStructInfo ts_info = relax::TensorStructInfo(relax::ShapeExpr(tvm_shape), dtype);
  relax::TensorStructInfo arg_ts_info = relax::TensorStructInfo(
      relax::ShapeExpr(arg_shape), runtime::DataType(DLDataType({kDLInt, 64, 1})));

  auto input = relax::Var("input", ts_info);
  auto begin = relax::Var("input", arg_ts_info);
  auto end = relax::Var("input", arg_ts_info);
  auto strides = relax::Var("input", arg_ts_info);

  ffi::Array<relax::Var> tvm_args;
  tvm_args.push_back(input);
  tvm_args.push_back(begin);
  tvm_args.push_back(end);
  tvm_args.push_back(strides);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.dynamic_strided_slice");

  auto result = ctx_->Emit(relax::Call(relax_op_, {input, begin, end, strides}, tvm::Attrs(), {}));
  auto result_out = ctx_->EmitOutput(result, "dynamic_strided_slice_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);
}

INSTANTIATE_TEST_SUITE_P(
    CPPCompierIndexDynamicStridedSlice, IndexDynamicStridedSlice,
    ::testing::Combine(
        ::testing::ValuesIn(
            std::vector<std::tuple<DLDeviceType, std::string, std::string, std::string>>{
                std::make_tuple(kDLCPU, "llvm", "cpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "opencl", "gpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "qcom/adreno-opencl", "adreno", "adreno")}),
        ::testing::ValuesIn(std::vector<std::tuple<DLDataType>>{
            std::make_tuple(DLDataType({kDLFloat, 32, 1})),
            std::make_tuple(DLDataType({kDLInt, 32, 1})),
        })));
