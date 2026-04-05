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

// CPP Compiler tests - Manipulate

#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/transform.h>

#include "compiler_base.h"

class Manipulate
    : public ::testing::TestWithParam<
          std::tuple<std::tuple<DLDeviceType, std::string, std::string, std::string>, bool>>,
      public CPPCompilerBase {
  void SetUp() override {
    auto env = std::get<0>(GetParam());
    dl_dev_type = std::get<0>(env);
    dev_name = std::get<1>(env);
    relax_pipeline = std::get<2>(env);
    tir_pipeline = std::get<3>(env);
    device = tvm::Device({dl_dev_type, 0});

    dl_type = DLDataType({kDLFloat, 32, 1});
    dtype = runtime::DataType(dl_type);

    is_symbolic = std::get<1>(GetParam());
  }

 public:
  bool is_symbolic;
};

TEST_P(Manipulate, Concat) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  ffi::ObjectPtr<relax::ConcatAttrs> attrs = ffi::make_object<relax::ConcatAttrs>();
  ffi::Array<relax::Expr> input_arr;
  if (is_symbolic) {
    auto N = tir::Var("N", runtime::DataType::Int(64));
    auto n_prim = N.as<tvm::PrimExpr>().value();
    auto input1 = relax::Var(
        "input1", relax::TensorStructInfo(
                      relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({1, n_prim, 4, 5})), dtype));
    auto input2 = relax::Var(
        "input2", relax::TensorStructInfo(
                      relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({1, n_prim, 3, 5})), dtype));
    auto input3 = relax::Var(
        "input3", relax::TensorStructInfo(
                      relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({1, n_prim, 2, 5})), dtype));
    auto input4 = relax::Var(
        "input4", relax::TensorStructInfo(
                      relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({1, n_prim, 1, 5})), dtype));
    input_arr.push_back(input1);
    input_arr.push_back(input2);
    input_arr.push_back(input3);
    input_arr.push_back(input4);
    attrs->axis = 2;

    tvm_args.push_back(input1);
    tvm_args.push_back(input2);
    tvm_args.push_back(input3);
    tvm_args.push_back(input4);
  } else {
    auto input1 = relax::Var(
        "input1",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({1, 4, 4, 5})), dtype));
    auto input2 = relax::Var(
        "input2",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({1, 4, 3, 5})), dtype));
    auto input3 = relax::Var(
        "input3",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({1, 4, 2, 5})), dtype));
    auto input4 = relax::Var(
        "input4",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({1, 4, 1, 5})), dtype));
    input_arr.push_back(input1);
    input_arr.push_back(input2);
    input_arr.push_back(input3);
    input_arr.push_back(input4);
    attrs->axis = 2;

    tvm_args.push_back(input1);
    tvm_args.push_back(input2);
    tvm_args.push_back(input3);
    tvm_args.push_back(input4);
  }

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.concat");

  auto result =
      ctx_->Emit(relax::Call(relax_op_, {relax::Tuple(input_arr)}, tvm::Attrs(attrs), {}));
  auto result_out = ctx_->EmitOutput(result, "concat_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor_1 =
      runtime::Tensor::Empty(ffi::Shape({1, 4, 4, 5}), dl_type, device, std::nullopt);
  auto input_tensor_2 =
      runtime::Tensor::Empty(ffi::Shape({1, 4, 3, 5}), dl_type, device, std::nullopt);
  auto input_tensor_3 =
      runtime::Tensor::Empty(ffi::Shape({1, 4, 2, 5}), dl_type, device, std::nullopt);
  auto input_tensor_4 =
      runtime::Tensor::Empty(ffi::Shape({1, 4, 1, 5}), dl_type, device, std::nullopt);

  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor_1, input_tensor_2,
                                           input_tensor_3, input_tensor_4};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, ConcatSymbolic) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  ffi::ObjectPtr<relax::ConcatAttrs> attrs = ffi::make_object<relax::ConcatAttrs>();
  ffi::Array<relax::Expr> input_arr;

  auto N = tir::Var("N", runtime::DataType::Int(64));
  auto n_prim = N.as<tvm::PrimExpr>().value();
  auto A = tir::Var("A", runtime::DataType::Int(64));
  auto a_prim = A.as<tvm::PrimExpr>().value();
  auto B1 = tir::Var("B1", runtime::DataType::Int(64));
  auto b1_prim = B1.as<tvm::PrimExpr>().value();
  auto B2 = tir::Var("B2", runtime::DataType::Int(64));
  auto b2_prim = B2.as<tvm::PrimExpr>().value();
  auto B3 = tir::Var("B3", runtime::DataType::Int(64));
  auto b3_prim = B3.as<tvm::PrimExpr>().value();
  auto B4 = tir::Var("B4", runtime::DataType::Int(64));
  auto b4_prim = B4.as<tvm::PrimExpr>().value();
  auto C = tir::Var("C", runtime::DataType::Int(64));
  auto c_prim = C.as<tvm::PrimExpr>().value();

  auto input1 = relax::Var(
      "input1",
      relax::TensorStructInfo(
          relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, n_prim, b1_prim, c_prim})), dtype));
  auto input2 = relax::Var(
      "input2",
      relax::TensorStructInfo(
          relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, n_prim, b2_prim, c_prim})), dtype));
  auto input3 = relax::Var(
      "input3",
      relax::TensorStructInfo(
          relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, n_prim, b3_prim, c_prim})), dtype));
  auto input4 = relax::Var(
      "input4",
      relax::TensorStructInfo(
          relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, n_prim, b4_prim, c_prim})), dtype));
  input_arr.push_back(input1);
  input_arr.push_back(input2);
  input_arr.push_back(input3);
  input_arr.push_back(input4);
  attrs->axis = 2;

  tvm_args.push_back(input1);
  tvm_args.push_back(input2);
  tvm_args.push_back(input3);
  tvm_args.push_back(input4);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.concat");

  auto result =
      ctx_->Emit(relax::Call(relax_op_, {relax::Tuple(input_arr)}, tvm::Attrs(attrs), {}));
  auto result_out = ctx_->EmitOutput(result, "concat_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor_1 =
      runtime::Tensor::Empty(ffi::Shape({1, 4, 4, 5}), dl_type, device, std::nullopt);
  auto input_tensor_2 =
      runtime::Tensor::Empty(ffi::Shape({1, 4, 3, 5}), dl_type, device, std::nullopt);
  auto input_tensor_3 =
      runtime::Tensor::Empty(ffi::Shape({1, 4, 2, 5}), dl_type, device, std::nullopt);
  auto input_tensor_4 =
      runtime::Tensor::Empty(ffi::Shape({1, 4, 1, 5}), dl_type, device, std::nullopt);

  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor_1, input_tensor_2,
                                           input_tensor_3, input_tensor_4};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, BroadCastTo) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  auto N = tir::Var("N", runtime::DataType::Int(64));
  auto n_prim = N.as<tvm::PrimExpr>().value();
  relax::ShapeExpr out_shape;

  if (is_symbolic) {
    auto input1 = relax::Var(
        "input1", relax::TensorStructInfo(
                      relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({1, n_prim, 4, 5})), dtype));
    out_shape = relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({3, 4, n_prim, 4, 5}));
    tvm_args.push_back(input1);
  } else {
    auto input1 = relax::Var(
        "input1",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({1, 4, 4, 5})), dtype));
    out_shape = relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({3, 4, 4, 4, 5}));
    tvm_args.push_back(input1);
  }

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.broadcast_to");

  auto result = ctx_->Emit(relax::Call(relax_op_, {tvm_args[0], out_shape}, tvm::Attrs(), {}));
  auto result_out = ctx_->EmitOutput(result, "broadcast_to_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  auto input_tensor =
      runtime::Tensor::Empty(ffi::Shape({1, 4, 4, 5}), dl_type, device, std::nullopt);

  std::vector<ffi::AnyView> packed_args = {ffi::String("main"), input_tensor};

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, ExpandDims) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  ffi::ObjectPtr<relax::ExpandDimsAttrs> attrs = ffi::make_object<relax::ExpandDimsAttrs>();
  auto N = tir::Var("N", runtime::DataType::Int(64));
  auto n_prim = N.as<tvm::PrimExpr>().value();
  ffi::Array<tvm::Integer> axis;

  if (is_symbolic) {
    auto input1 = relax::Var(
        "input1", relax::TensorStructInfo(
                      relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2, n_prim, 4})), dtype));
    axis = ffi::Array<tvm::Integer>({1, 3, 5});
    tvm_args.push_back(input1);
  } else {
    auto input1 = relax::Var(
        "input1",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2, 3, 4})), dtype));
    axis = ffi::Array<tvm::Integer>({-1, 1, -6, 3, 5});
    tvm_args.push_back(input1);
  }

  attrs->axis = axis;

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.expand_dims");

  auto result = ctx_->Emit(relax::Call(relax_op_, {tvm_args[0]}, tvm::Attrs(attrs), {}));
  auto result_out = ctx_->EmitOutput(result, "expand_dims_out");

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

TEST_P(Manipulate, Flatten) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();
    auto B = tir::Var("B", runtime::DataType::Int(64));
    auto b_prim = B.as<tvm::PrimExpr>().value();
    auto C = tir::Var("C", runtime::DataType::Int(64));
    auto c_prim = C.as<tvm::PrimExpr>().value();

    auto input1 = relax::Var(
        "input1",
        relax::TensorStructInfo(
            relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, b_prim, c_prim})), dtype));
    tvm_args.push_back(input1);
  } else {
    auto input1 = relax::Var(
        "input1",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2, 3, 4})), dtype));
    tvm_args.push_back(input1);
  }

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.flatten");

  auto result = ctx_->Emit(relax::Call(relax_op_, {tvm_args[0]}, tvm::Attrs(), {}));
  auto result_out = ctx_->EmitOutput(result, "flatten_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto input_tensor = runtime::Tensor::Empty(ffi::Shape({2, 3, 4}), dl_type, device, std::nullopt);
  packed_args.push_back(input_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, PermuteDims) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::ObjectPtr<relax::PermuteDimsAttrs> attrs = ffi::make_object<relax::PermuteDimsAttrs>();
  ffi::Array<tvm::Integer> axis;
  ffi::Array<relax::Var> tvm_args;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();
    auto B = tir::Var("B", runtime::DataType::Int(64));
    auto b_prim = B.as<tvm::PrimExpr>().value();
    auto C = tir::Var("C", runtime::DataType::Int(64));
    auto c_prim = C.as<tvm::PrimExpr>().value();
    auto D = tir::Var("D", runtime::DataType::Int(64));
    auto d_prim = D.as<tvm::PrimExpr>().value();

    auto input1 = relax::Var(
        "input1",
        relax::TensorStructInfo(
            relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, b_prim, c_prim, d_prim})), dtype));
    tvm_args.push_back(input1);
    axis = ffi::Array<tvm::Integer>({1, -1, 2, -4});
  } else {
    auto input1 = relax::Var(
        "input1",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2, 4, 3, 1})), dtype));
    tvm_args.push_back(input1);
    axis = ffi::Array<tvm::Integer>({1, -1, 2, -4});
  }
  attrs->axes = axis;

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  const tvm::Op& relax_op_ = tvm::Op::Get("relax.permute_dims");

  auto result = ctx_->Emit(relax::Call(relax_op_, {tvm_args[0]}, tvm::Attrs(attrs), {}));
  auto result_out = ctx_->EmitOutput(result, "permute_dims_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto input_tensor =
      runtime::Tensor::Empty(ffi::Shape({2, 4, 3, 1}), dl_type, device, std::nullopt);
  packed_args.push_back(input_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, Reshape) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;

  auto input1 = relax::Var(
      "input1",
      relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({1, 2, 3, 4})), dtype));
  tvm_args.push_back(input1);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.reshape");
  relax::Call call;
  // is_symbolic flag here is used to use shape computation instead of list/array
  if (is_symbolic) {
    call = (*relax_op_)(input1, relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({8, 3})))
               .cast<relax::Call>();
  } else {
    call = (*relax_op_)(input1, ffi::Array<tvm::PrimExpr>({8, 3})).cast<relax::Call>();
  }

  auto result = ctx_->Emit(call);
  auto result_out = ctx_->EmitOutput(result, "reshape_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto input_tensor =
      runtime::Tensor::Empty(ffi::Shape({1, 2, 3, 4}), dl_type, device, std::nullopt);
  packed_args.push_back(input_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, ReshapeSymbolic) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  auto A = tir::Var("A", runtime::DataType::Int(64));
  auto a_prim = A.as<tvm::PrimExpr>().value();
  auto B = tir::Var("B", runtime::DataType::Int(64));
  auto b_prim = B.as<tvm::PrimExpr>().value();

  auto input1 = relax::Var(
      "input1", relax::TensorStructInfo(
                    relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, b_prim})), dtype));
  tvm_args.push_back(input1);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.reshape");
  relax::Call call;
  // is_symbolic flag here is used to use shape computation instead of list/array
  if (is_symbolic) {
    call =
        (*relax_op_)(input1, relax::ShapeExpr(ffi::Array<tvm::PrimExpr>(
                                 {tvm::tir::Div(a_prim, tvm::IntImm(runtime::DataType::Int(64), 2)),
                                  b_prim * tvm::IntImm(runtime::DataType::Int(64), 2)})))
            .cast<relax::Call>();
  } else {
    call =
        (*relax_op_)(input1, ffi::Array<tvm::PrimExpr>(
                                 {tvm::tir::Div(a_prim, tvm::IntImm(runtime::DataType::Int(64), 2)),
                                  b_prim * tvm::IntImm(runtime::DataType::Int(64), 2)}))
            .cast<relax::Call>();
  }

  auto result = ctx_->Emit(call);
  auto result_out = ctx_->EmitOutput(result, "reshape_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto input_tensor = runtime::Tensor::Empty(ffi::Shape({10, 20}), dl_type, device, std::nullopt);
  packed_args.push_back(input_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, ReshapeDataDependent) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  relax::Var shape_in, data_in;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();

    shape_in = relax::Var("shape_in",
                          relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2})),
                                                  runtime::DataType::Int(64)));
    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim})), dtype));
  } else {
    shape_in = relax::Var("shape_in",
                          relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2})),
                                                  runtime::DataType::Int(64)));
    data_in = relax::Var("data_in", relax::TensorStructInfo(
                                        relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({16})), dtype));
  }
  tvm_args.push_back(shape_in);
  tvm_args.push_back(data_in);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.reshape");
  auto shape_relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.tensor_to_shape");

  auto shape_arg = ctx_->Emit((*shape_relax_op_)(shape_in).cast<relax::Call>());

  auto result = ctx_->Emit((*relax_op_)(data_in, shape_arg).cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "reshape_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);
  mod_ = relax::transform::DecomposeOpsForInference(std::nullopt)(mod_);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};

  std::vector<int64_t> shape_arr = {2, 8};
  auto shape_tensor = runtime::Tensor::Empty(ffi::Shape({2}), runtime::DataType::Int(64),
                                             tvm::Device({kDLCPU, 0}), std::nullopt);
  shape_tensor.CopyFromBytes(static_cast<void*>(shape_arr.data()), 2 * sizeof(int64_t));

  auto data_tensor = runtime::Tensor::Empty(ffi::Shape({16}), dl_type, device, std::nullopt);

  packed_args.push_back(shape_tensor);
  packed_args.push_back(data_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, Split) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();

    data_in = relax::Var("data_in",
                         relax::TensorStructInfo(
                             relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, 10, 4})), dtype));
  } else {
    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2, 10, 4})), dtype));
  }
  tvm_args.push_back(data_in);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.split");

  auto result =
      ctx_->Emit((*relax_op_)(data_in,
                              ffi::Array<tvm::IntImm>({tvm::IntImm(runtime::DataType::Int(64), 3),
                                                       tvm::IntImm(runtime::DataType::Int(64), 7)}),
                              1)
                     .cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "split_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor = runtime::Tensor::Empty(ffi::Shape({2, 10, 4}), dl_type, device, std::nullopt);
  packed_args.push_back(data_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args, 3);
  LOG(INFO) << "Result:" << output[0].shape() << ":" << output[1].shape() << ":"
            << output[2].shape();
}

TEST_P(Manipulate, SplitByIndicesNSectionIndivisible) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();

    data_in = relax::Var("data_in",
                         relax::TensorStructInfo(
                             relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, 10, 4})), dtype));
  } else {
    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2, 10, 4})), dtype));
  }
  tvm_args.push_back(data_in);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.split");

  auto result = ctx_->Emit(
      (*relax_op_)(data_in, tvm::IntImm(runtime::DataType::Int(64), 3), 1).cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "split_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor = runtime::Tensor::Empty(ffi::Shape({2, 10, 4}), dl_type, device, std::nullopt);
  packed_args.push_back(data_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args, 3);
  LOG(INFO) << "Result:" << output[0].shape() << ":" << output[1].shape() << ":"
            << output[2].shape();
}

TEST_P(Manipulate, SplitByIndicesNSectionDivisible) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();
    auto B = tir::Var("B", runtime::DataType::Int(64));
    auto b_prim = B.as<tvm::PrimExpr>().value();

    data_in = relax::Var(
        "data_in", relax::TensorStructInfo(
                       relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, 10, b_prim})), dtype));
  } else {
    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2, 10, 4})), dtype));
  }
  tvm_args.push_back(data_in);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.split");

  auto result = ctx_->Emit(
      (*relax_op_)(data_in, tvm::IntImm(runtime::DataType::Int(64), 2), 1).cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "split_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor = runtime::Tensor::Empty(ffi::Shape({2, 10, 4}), dl_type, device, std::nullopt);
  packed_args.push_back(data_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args, 2);
  LOG(INFO) << "Result:" << output[0].shape() << ":" << output[1].shape();
}

TEST_P(Manipulate, Squeeze) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();
    auto B = tir::Var("B", runtime::DataType::Int(64));
    auto b_prim = B.as<tvm::PrimExpr>().value();

    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(
            relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, 1, 3, 1, 1, b_prim})), dtype));
  } else {
    data_in = relax::Var(
        "data_in", relax::TensorStructInfo(
                       relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2, 1, 3, 1, 1, 4})), dtype));
  }
  tvm_args.push_back(data_in);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.squeeze");

  auto result =
      ctx_->Emit((*relax_op_)(data_in, ffi::Array<tvm::Integer>({1, 4})).cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "squeeze_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor =
      runtime::Tensor::Empty(ffi::Shape({2, 1, 3, 1, 1, 4}), dl_type, device, std::nullopt);
  packed_args.push_back(data_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, SqueezeNoAxis) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in;

  // TODO(Siva): Not handling symbolic: Legalization pass is not invoking the legalization with
  // dynamic shape and no axis.
  data_in = relax::Var("data_in",
                       relax::TensorStructInfo(
                           relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2, 1, 3, 1, 1, 4})), dtype));
  tvm_args.push_back(data_in);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.squeeze");

  auto result = ctx_->Emit((*relax_op_)(data_in, nullptr).cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "squeeze_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor =
      runtime::Tensor::Empty(ffi::Shape({2, 1, 3, 1, 1, 4}), dl_type, device, std::nullopt);
  packed_args.push_back(data_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, CollapseSumLike) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in, like;

  data_in = relax::Var("data_in", relax::TensorStructInfo(
                                      relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2, 3})), dtype));
  like = relax::Var(
      "like", relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({1, 3})), dtype));
  tvm_args.push_back(data_in);
  tvm_args.push_back(like);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.collapse_sum_like");

  auto result = ctx_->Emit((*relax_op_)(data_in, like).cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "collapse_sum_like_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor = runtime::Tensor::Empty(ffi::Shape({2, 3}), dl_type, device, std::nullopt);
  auto like_tensor = runtime::Tensor::Empty(ffi::Shape({1, 3}), dl_type, device, std::nullopt);

  packed_args.push_back(data_tensor);
  packed_args.push_back(like_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, CollapseSumTo) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in, like;

  data_in = relax::Var(
      "data_in",
      relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({3, 2, 3})), dtype));
  tvm_args.push_back(data_in);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.collapse_sum_to");

  auto result =
      ctx_->Emit((*relax_op_)(data_in, relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2, 1})))
                     .cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "collapse_sum_to_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor = runtime::Tensor::Empty(ffi::Shape({3, 2, 3}), dl_type, device, std::nullopt);

  packed_args.push_back(data_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, Repeat) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in, like;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();
    auto B = tir::Var("B", runtime::DataType::Int(64));
    auto b_prim = B.as<tvm::PrimExpr>().value();
    auto C = tir::Var("C", runtime::DataType::Int(64));
    auto c_prim = C.as<tvm::PrimExpr>().value();

    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(
            relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, b_prim, c_prim})), dtype));
  } else {
    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({3, 2, 3})), dtype));
  }
  tvm_args.push_back(data_in);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.repeat");

  auto result = ctx_->Emit((*relax_op_)(data_in, 2, 0).cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "repeat_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor = runtime::Tensor::Empty(ffi::Shape({3, 2, 3}), dl_type, device, std::nullopt);

  packed_args.push_back(data_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, RepeatNoAxis) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in, like;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();
    auto B = tir::Var("B", runtime::DataType::Int(64));
    auto b_prim = B.as<tvm::PrimExpr>().value();
    auto C = tir::Var("C", runtime::DataType::Int(64));
    auto c_prim = C.as<tvm::PrimExpr>().value();

    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(
            relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, b_prim, c_prim})), dtype));
  } else {
    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({3, 2, 3})), dtype));
  }
  tvm_args.push_back(data_in);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.repeat");

  auto result = ctx_->Emit((*relax_op_)(data_in, 2, nullptr).cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "repeat_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor = runtime::Tensor::Empty(ffi::Shape({3, 2, 3}), dl_type, device, std::nullopt);

  packed_args.push_back(data_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, Tile) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in, like;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();
    auto B = tir::Var("B", runtime::DataType::Int(64));
    auto b_prim = B.as<tvm::PrimExpr>().value();
    auto C = tir::Var("C", runtime::DataType::Int(64));
    auto c_prim = C.as<tvm::PrimExpr>().value();

    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(
            relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, b_prim, c_prim})), dtype));
  } else {
    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({3, 2, 3})), dtype));
  }
  tvm_args.push_back(data_in);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.tile");

  auto result =
      ctx_->Emit((*relax_op_)(data_in, ffi::Array<tvm::Integer>({2, 1, 2, 3})).cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "tile_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor = runtime::Tensor::Empty(ffi::Shape({3, 2, 3}), dl_type, device, std::nullopt);

  packed_args.push_back(data_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, Flip) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in, like;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();
    auto B = tir::Var("B", runtime::DataType::Int(64));
    auto b_prim = B.as<tvm::PrimExpr>().value();

    data_in = relax::Var("data_in",
                         relax::TensorStructInfo(
                             relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, b_prim})), dtype));
  } else {
    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2, 3})), dtype));
  }
  tvm_args.push_back(data_in);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.flip");

  auto result = ctx_->Emit((*relax_op_)(data_in, 0).cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "flip_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor = runtime::Tensor::Empty(ffi::Shape({2, 3}), dl_type, device, std::nullopt);

  packed_args.push_back(data_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, GatherElements) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in, indices;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();
    auto B = tir::Var("B", runtime::DataType::Int(64));
    auto b_prim = B.as<tvm::PrimExpr>().value();
    auto C = tir::Var("C", runtime::DataType::Int(64));
    auto c_prim = C.as<tvm::PrimExpr>().value();

    data_in = relax::Var("data_in",
                         relax::TensorStructInfo(
                             relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, b_prim})), dtype));
    indices = relax::Var(
        "indices", relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({c_prim, 2})),
                                           runtime::DataType::Int(64)));
  } else {
    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({3, 3})), dtype));
    indices = relax::Var(
        "indices", relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({3, 2})),
                                           runtime::DataType::Int(64)));
  }
  tvm_args.push_back(data_in);
  tvm_args.push_back(indices);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.gather_elements");

  auto result = ctx_->Emit((*relax_op_)(data_in, indices, 1).cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "gather_elements_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor = runtime::Tensor::Empty(ffi::Shape({3, 3}), dl_type, device, std::nullopt);
  auto indices_tensor =
      runtime::Tensor::Empty(ffi::Shape({3, 2}), runtime::DataType::Int(64), device, std::nullopt);
  std::vector<int64_t> indices_arr = {0, 1, 1, 2, 2, 0};  // 3x2 represented as 1D
  indices_tensor.CopyFromBytes(static_cast<void*>(indices_arr.data()), 3 * 2 * sizeof(int64_t));

  packed_args.push_back(data_tensor);
  packed_args.push_back(indices_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, GatherND) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in, indices;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();
    auto B = tir::Var("B", runtime::DataType::Int(64));
    auto b_prim = B.as<tvm::PrimExpr>().value();
    auto C = tir::Var("C", runtime::DataType::Int(64));
    auto c_prim = C.as<tvm::PrimExpr>().value();
    auto D = tir::Var("D", runtime::DataType::Int(64));
    auto d_prim = D.as<tvm::PrimExpr>().value();

    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(
            relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, b_prim, c_prim, d_prim})), dtype));
    indices = relax::Var("indices", relax::TensorStructInfo(
                                        relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, 2, 2})),
                                        runtime::DataType::Int(64)));
  } else {
    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2, 3, 3, 3})), dtype));
    indices = relax::Var(
        "indices", relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({2, 2, 2})),
                                           runtime::DataType::Int(64)));
  }
  tvm_args.push_back(data_in);
  tvm_args.push_back(indices);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.gather_nd");

  auto result = ctx_->Emit((*relax_op_)(data_in, indices, 1).cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "gather_nd_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor =
      runtime::Tensor::Empty(ffi::Shape({2, 3, 3, 3}), dl_type, device, std::nullopt);
  auto indices_tensor = runtime::Tensor::Empty(ffi::Shape({2, 2, 2}), runtime::DataType::Int(64),
                                               device, std::nullopt);
  std::vector<int64_t> indices_arr = {0, 1, 1, 2, 2, 0, 0, 0};  // 2x2x2 represented as 1D
  indices_tensor.CopyFromBytes(static_cast<void*>(indices_arr.data()), 2 * 2 * 2 * sizeof(int64_t));

  packed_args.push_back(data_tensor);
  packed_args.push_back(indices_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, LayoutTransform) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();
    auto B = tir::Var("B", runtime::DataType::Int(64));
    auto b_prim = B.as<tvm::PrimExpr>().value();
    auto C = tir::Var("C", runtime::DataType::Int(64));
    auto c_prim = C.as<tvm::PrimExpr>().value();

    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(
            relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, b_prim, c_prim})), dtype));
  } else {
    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({10, 21, 30})), dtype));
  }
  tvm_args.push_back(data_in);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.layout_transform");

  std::function<ffi::Array<tvm::PrimExpr>(const ffi::Array<tir::Var>)> _index_func =
      [&](const ffi::Array<tir::Var> args) -> ffi::Array<tvm::PrimExpr> {
    return {args[0], args[2], tvm::tir::FloorDiv(args[1], 3), tvm::tir::FloorMod(args[1], 3)};
  };
  auto index_map = tvm::tir::IndexMap::FromFunc(3, _index_func);

  auto pad_value = relax::PrimValue(tvm::PrimExpr(2.0f));
  ffi::Array<tvm::IntImm> axis_separators;
  ffi::Array<tvm::IntImm> input_axis_separators;

  auto result =
      ctx_->Emit((*relax_op_)(data_in, index_map, pad_value, axis_separators, input_axis_separators)
                     .cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "layout_transform_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor =
      runtime::Tensor::Empty(ffi::Shape({10, 21, 30}), dl_type, device, std::nullopt);

  packed_args.push_back(data_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

TEST_P(Manipulate, LayoutTransformWithPad) {
  if (!TargetSetup()) {
    GTEST_SKIP() << "Device not available: " << dev_name;
    return;
  }

  // Inputs
  ffi::Array<relax::Var> tvm_args;
  tvm::relax::Var data_in;

  if (is_symbolic) {
    auto A = tir::Var("A", runtime::DataType::Int(64));
    auto a_prim = A.as<tvm::PrimExpr>().value();
    auto B = tir::Var("B", runtime::DataType::Int(64));
    auto b_prim = B.as<tvm::PrimExpr>().value();
    auto C = tir::Var("C", runtime::DataType::Int(64));
    auto c_prim = C.as<tvm::PrimExpr>().value();

    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(
            relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({a_prim, b_prim, c_prim})), dtype));
  } else {
    data_in = relax::Var(
        "data_in",
        relax::TensorStructInfo(relax::ShapeExpr(ffi::Array<tvm::PrimExpr>({10, 20, 30})), dtype));
  }
  tvm_args.push_back(data_in);

  // Graph
  auto ctx_ = BuilderSetup(tvm_args);

  auto relax_op_ = tvm::ffi::Function::GetGlobal("relax.op.layout_transform");

  std::function<ffi::Array<tvm::PrimExpr>(const ffi::Array<tir::Var>)> _index_func =
      [&](const ffi::Array<tir::Var> args) -> ffi::Array<tvm::PrimExpr> {
    return {args[0], args[2], tvm::tir::FloorDiv(args[1], 3), tvm::tir::FloorMod(args[1], 3)};
  };
  auto index_map = tvm::tir::IndexMap::FromFunc(3, _index_func);

  auto pad_value = relax::PrimValue(tvm::PrimExpr(2.0f));
  ffi::Array<tvm::IntImm> axis_separators;
  ffi::Array<tvm::IntImm> input_axis_separators;

  auto result =
      ctx_->Emit((*relax_op_)(data_in, index_map, pad_value, axis_separators, input_axis_separators)
                     .cast<relax::Call>());
  auto result_out = ctx_->EmitOutput(result, "layout_transform_out");

  auto mod_ = BuilderFinalize(ctx_, tvm_args, result_out);

  // Compile
  auto vm = Compile(mod_, tgt, dl_dev_type, relax_pipeline, tir_pipeline);

  // Inputs
  std::vector<ffi::AnyView> packed_args = {ffi::String("main")};
  auto data_tensor =
      runtime::Tensor::Empty(ffi::Shape({10, 20, 30}), dl_type, device, std::nullopt);

  packed_args.push_back(data_tensor);

  // VMRun
  auto output = VMRun(vm, packed_args);
  LOG(INFO) << "Result:" << output[0].shape();
}

INSTANTIATE_TEST_SUITE_P(
    CPPCompierManipulate, Manipulate,
    ::testing::Combine(
        ::testing::ValuesIn(
            std::vector<std::tuple<DLDeviceType, std::string, std::string, std::string>>{
                std::make_tuple(kDLCPU, "llvm", "cpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "opencl", "gpu_generic", "generic"),
                std::make_tuple(kDLOpenCL, "qcom/adreno-opencl", "adreno", "adreno")}),
        ::testing::ValuesIn(std::vector<bool>{true, false})));
