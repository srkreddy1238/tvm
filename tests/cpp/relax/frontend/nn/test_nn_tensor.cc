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

/*!
 * \file tests/cpp/relax/frontend/nn/test_nn_tensor.cc
 * \brief C++ port of tests/python/relax/test_frontend_nn_tensor.py
 *
 * Ported tests:
 *   - TestTensorFromNumpy      (test_tensor_from_numpy)
 *   - TestTensorFromScalar     (test_tensor_from_scalar)
 *   - TestTensorOpBinaryTensorTensor  (test_tensor_op_binary_tensor_tensor)
 *   - TestTensorOpBinaryTensorScalar  (test_tensor_op_binary_tensor_scalar)
 *   - TestTensorOpDatatype     (test_tensor_op_datatype)
 *   - TestTensorOpManipulate   (test_tensor_op_manipulate)
 *
 * Design notes:
 *   - test_tensor_from_numpy / test_tensor_from_scalar test Python-level
 *     Tensor.from_const / Tensor.from_scalar which are Python-only wrappers
 *     around rx.const().  In C++ we test the equivalent: creating a
 *     TensorNode from a relax.Constant expression and verifying shape/dtype.
 *   - The IR-equality tests (binary ops, datatype, manipulate) follow the
 *     same pattern as test_nn_ops.cc: build a ModuleSpec with a forward
 *     lambda, export to IRModule, and assert structural equality against a
 *     manually-built expected IRModule.
 *   - All tests use debug=True to match the Python test's export_tvm calls.
 */

#include <gtest/gtest.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/op.h>

#include <string>
#include <vector>

#include "../../../../../src/relax/frontend/nn/core.h"
#include "../../../../../src/relax/frontend/nn/exporter.h"
#include "../../../../../src/relax/frontend/nn/spec.h"
#include "../../../../../src/relax/op/tensor/binary.h"
#include "../../../../../src/relax/op/tensor/manipulate.h"
#include "../../../../../src/relax/op/tensor/datatype.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace testing {

// ===========================================================================
// Shared helpers
// ===========================================================================

static TensorStructInfo TSInfo(std::initializer_list<int64_t> dims, DataType dtype) {
  ffi::Array<PrimExpr> shape_dims;
  for (int64_t d : dims) shape_dims.push_back(IntImm(DataType::Int(64), d));
  return TensorStructInfo(ShapeExpr(shape_dims), dtype);
}

static SpecTensor MakeSpecTensor(std::initializer_list<int64_t> dims, const std::string& dtype) {
  ffi::Array<ffi::Any> shape;
  for (int64_t d : dims) shape.push_back(ffi::Any(d));
  return SpecTensor(shape, dtype);
}

static void AssertStructEqual(const IRModule& actual, const IRModule& expected) {
  EXPECT_TRUE(ffi::StructuralEqual()(actual, expected))
      << "\n=== Actual ===\n" << actual << "\n=== Expected ===\n" << expected;
}

// Retrieve the registered nn.op FFI function by short name.
static ffi::Function NNOp(const std::string& name) {
  auto f = ffi::Function::GetGlobal("relax.frontend.nn.op." + name);
  TVM_FFI_ICHECK(f.has_value()) << "nn.op not found: " << name;
  return f.value();
}

// Export a single method (debug=True) via NNModule->ExportTVM.
static IRModule ExportDebug(ffi::Function forward_fn,
                            const std::string& method_name,
                            ffi::Array<ffi::String> arg_names,
                            ffi::Array<ffi::Any> arg_specs) {
  // Create a minimal NNModule wrapper
  NNModule mod;
  
  // Build MethodSpec and ModuleSpec
  MethodSpec ms(forward_fn, arg_names, arg_specs, "plain", "plain");
  ModuleSpec mod_spec(ffi::Array<ffi::String>{ffi::String(method_name)},
                      ffi::Array<ffi::Any>{ffi::Any(ms)}, {}, {});
  
  // Use NNModule->ExportTVM
  ffi::Array<ffi::Any> result = mod->ExportTVM(mod_spec, /*debug=*/true, /*allow_extern=*/false);
  return result[0].cast<IRModule>();
}

// Emit the debug output tuple (result, (_io,)) and return the binding Var.
static Var EmitDebugOutput(BlockBuilder& bb, Expr result, Var io,
                           const std::string& hint = "gv1") {
  return bb->EmitOutput(relax::Tuple({result, relax::Tuple({io})}), hint);
}

// Emit the _initialize_effect function into bb.
static void EmitInitEffect(BlockBuilder& bb) {
  ffi::Array<Var> params;
  bb->BeginScope(params);
  bb->BeginDataflowBlock();
  static const Op& null_value_op = Op::Get("relax.null_value");
  Var io = bb->Emit(Call(null_value_op, {}, {}, {}), "_io");
  Var lv = bb->Emit(relax::Tuple({io}), "lv");
  Var gv = bb->EmitOutput(lv, "gv");
  BindingBlock df = bb->EndBlock();
  Expr body = bb->Normalize(SeqExpr({df}, gv));
  bb->EndScope();
  ffi::Map<ffi::String, ffi::Any> attrs;
  attrs.Set("global_symbol", ffi::Any(ffi::String("_initialize_effect")));
  bb->AddFunction(Function(params, body, std::nullopt, true, DictAttrs(attrs)),
                  "_initialize_effect");
}

// ===========================================================================
// TestTensorFromNumpy
//
// Python equivalent:
//   x = np.random.rand(1, 10)
//   tensor_x = Tensor.from_const(x)
//   assert tensor_x.shape == [1, 10]
//   assert tensor_x.ndim == 2
//   assert tensor_x.dtype == "float32"
//   assert repr(tensor_x) == 'Tensor([1, 10], "float32")'
//
// In C++ Tensor.from_const is Python-only.  We test the equivalent: creating
// a TensorNode from a relax.Constant and verifying shape/dtype/ndim.
// ===========================================================================
TEST(NNTensor, TestTensorFromNumpy) {
  // Create a (1, 10) float32 tensor and wrap it in a relax.Constant.
  std::vector<float> data(10, 0.0f);
  runtime::Tensor nd = runtime::Tensor::Empty(
      ffi::Shape{1, 10}, DLDataType{kDLFloat, 32, 1},
      DLDevice{kDLCPU, 0}, std::nullopt);
  nd.CopyFromBytes(data.data(), data.size() * sizeof(float));
  relax::Constant c(nd, std::nullopt);

  const auto* ts = c->struct_info_.as<TensorStructInfoNode>();
  ASSERT_NE(ts, nullptr);

  ffi::Array<PrimExpr> shape = ts->GetShape().value();
  ASSERT_EQ(shape.size(), 2u);
  EXPECT_EQ(Downcast<IntImm>(shape[0])->value, 1);
  EXPECT_EQ(Downcast<IntImm>(shape[1])->value, 10);
  EXPECT_EQ(ts->ndim, 2);
  EXPECT_EQ(runtime::DataType(ts->dtype), DataType::Float(32));
}

// ===========================================================================
// TestTensorFromScalar
//
// Python equivalent:
//   x = 123.321
//   tensor_x = Tensor.from_scalar(x, dtype="float16")
//   assert tensor_x.shape == []
//   assert tensor_x.ndim == 0
//   assert tensor_x.dtype == "float16"
//   assert repr(tensor_x) == 'Tensor([], "float16")'
//
// In C++ we create a scalar relax.Constant and verify its struct_info.
// ===========================================================================
TEST(NNTensor, TestTensorFromScalar) {
  // Create a scalar float16 constant.
  runtime::Tensor nd = runtime::Tensor::Empty(
      ffi::Shape{}, DLDataType{kDLFloat, 16, 1},
      DLDevice{kDLCPU, 0}, std::nullopt);
  relax::Constant c(nd, std::nullopt);
  const auto* ts = c->struct_info_.as<TensorStructInfoNode>();
  ASSERT_NE(ts, nullptr);
  EXPECT_EQ(ts->ndim, 0);
  EXPECT_EQ(runtime::DataType(ts->dtype), DataType::Float(16));
}

// ===========================================================================
// TestTensorOpBinaryTensorTensor
//
// Python equivalent:
//   class Model(Module):
//       def test(self, x: Tensor, y: Tensor):
//           z0 = x + y
//           z1 = x * y
//           z2 = x / y
//           z3 = x.maximum(y)
//           z4 = x.minimum(y)
//           return (z0, z1, z2, z3, z4)
//
//   irmodule, _ = m.export_tvm(
//       spec={"test": {"x": spec.Tensor([1,10],"float32"),
//                      "y": spec.Tensor([2,1],"float32")}},
//       debug=True)
//
// Expected (debug=True):
//   def test(x: (1,10)f32, y: (2,1)f32, _io: Object):
//     num_input = 3
//     add, mul, divide, maximum, minimum = ...
//     gv1 = (add,mul,divide,maximum,minimum), (_io,)
// ===========================================================================
TEST(NNTensor, TestTensorOpBinaryTensorTensor) {
  const ffi::Function op_add     = NNOp("add");
  const ffi::Function op_mul     = NNOp("multiply");
  const ffi::Function op_div     = NNOp("divide");
  const ffi::Function op_maximum = NNOp("maximum");
  const ffi::Function op_minimum = NNOp("minimum");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> test_fn =
      [op_add, op_mul, op_div, op_maximum, op_minimum](
          ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    NNTensor y = args.at("y").cast<NNTensor>();
    ffi::Any z0 = op_add(x->expr, y->expr, ffi::String("add"));
    ffi::Any z1 = op_mul(x->expr, y->expr, ffi::String("mul"));
    ffi::Any z2 = op_div(x->expr, y->expr, ffi::String("divide"));
    ffi::Any z3 = op_maximum(x->expr, y->expr, ffi::String("maximum"));
    ffi::Any z4 = op_minimum(x->expr, y->expr, ffi::String("minimum"));
    ffi::Array<ffi::Any> out{z0, z1, z2, z3, z4};
    return ffi::Any(out);
  };

  IRModule actual = ExportDebug(test_fn, "test", {"x", "y"},
                                {ffi::Any(MakeSpecTensor({1, 10}, "float32")),
                                 ffi::Any(MakeSpecTensor({2, 1}, "float32"))});

  // Build expected IR.
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    DataType f32 = DataType::Float(32);
    Var x("x", TSInfo({1, 10}, f32));
    Var y("y", TSInfo({2, 1}, f32));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, y, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var add     = bb->Emit(relax::add(x, y), "add");
    Var mul     = bb->Emit(relax::multiply(x, y), "mul");
    Var divide  = bb->Emit(relax::divide(x, y), "divide");
    Var maximum = bb->Emit(relax::maximum(x, y), "maximum");
    Var minimum = bb->Emit(relax::minimum(x, y), "minimum");
    Var gv1 = EmitDebugOutput(bb,
        relax::Tuple({add, mul, divide, maximum, minimum}), io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(3)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, true, DictAttrs(attrs)), "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestTensorOpBinaryTensorScalar
//
// Python equivalent:
//   class Model(Module):
//       def test(self, x: Tensor):
//           y = 10
//           z0 = x + y;  z1 = y + x;  z2 = x * y
//           z3 = x / y;  z4 = x.maximum(y);  z5 = x.minimum(y)
//           return (z0, z1, z2, z3, z4, z5)
//
//   irmodule, _ = m.export_tvm(
//       spec={"test": {"x": spec.Tensor([1,10],"float32")}}, debug=True)
//
// The scalar 10 is folded into R.const(10, "float32") by the nn.op binary ops.
// ===========================================================================
TEST(NNTensor, TestTensorOpBinaryTensorScalar) {
  const ffi::Function op_add     = NNOp("add");
  const ffi::Function op_mul     = NNOp("multiply");
  const ffi::Function op_div     = NNOp("divide");
  const ffi::Function op_maximum = NNOp("maximum");
  const ffi::Function op_minimum = NNOp("minimum");

  // Helper: build a scalar float32 relax.Constant.
  auto MakeF32Const = [](float v) -> relax::Constant {
    runtime::Tensor t = runtime::Tensor::Empty(
        ffi::Shape{}, DLDataType{kDLFloat, 32, 1},
        DLDevice{kDLCPU, 0}, std::nullopt);
    t.CopyFromBytes(&v, sizeof(float));
    return relax::Constant(t, std::nullopt);
  };

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> test_fn =
      [op_add, op_mul, op_div, op_maximum, op_minimum, MakeF32Const](
          ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    relax::Constant scalar10 = MakeF32Const(10.0f);
    ffi::Any z0 = op_add(x->expr, scalar10, ffi::String("add"));
    ffi::Any z1 = op_add(x->expr, scalar10, ffi::String("add1"));
    ffi::Any z2 = op_mul(x->expr, scalar10, ffi::String("mul"));
    ffi::Any z3 = op_div(x->expr, scalar10, ffi::String("divide"));
    ffi::Any z4 = op_maximum(x->expr, scalar10, ffi::String("maximum"));
    ffi::Any z5 = op_minimum(x->expr, scalar10, ffi::String("minimum"));
    ffi::Array<ffi::Any> out{z0, z1, z2, z3, z4, z5};
    return ffi::Any(out);
  };

  IRModule actual = ExportDebug(test_fn, "test", {"x"},
                                {ffi::Any(MakeSpecTensor({1, 10}, "float32"))});

  // Build expected IR.
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    DataType f32 = DataType::Float(32);
    Var x("x", TSInfo({1, 10}, f32));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    relax::Constant c10 = MakeF32Const(10.0f);
    Var add     = bb->Emit(relax::add(x, c10), "add");
    Var add1    = bb->Emit(relax::add(x, c10), "add1");
    Var mul     = bb->Emit(relax::multiply(x, c10), "mul");
    Var divide  = bb->Emit(relax::divide(x, c10), "divide");
    Var maximum = bb->Emit(relax::maximum(x, c10), "maximum");
    Var minimum = bb->Emit(relax::minimum(x, c10), "minimum");
    Var gv1 = EmitDebugOutput(bb,
        relax::Tuple({add, add1, mul, divide, maximum, minimum}), io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, true, DictAttrs(attrs)), "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestTensorOpDatatype
//
// Python equivalent:
//   class Model(Module):
//       def test(self, x: Tensor):
//           z0 = x.astype(dtype="float16")
//           return z0
//
//   irmodule, _ = m.export_tvm(
//       spec={"test": {"x": spec.Tensor([1,10],"float32")}}, debug=True)
//
// Expected:
//   def test(x: (1,10)f32, _io: Object):
//     num_input = 2
//     astype = R.astype(x, "float16")
//     gv1 = astype, (_io,)
// ===========================================================================
TEST(NNTensor, TestTensorOpDatatype) {
  const ffi::Function op_astype = NNOp("astype");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> test_fn =
      [op_astype](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    return op_astype(x->expr, ffi::String("float16"), ffi::String("astype"));
  };

  IRModule actual = ExportDebug(test_fn, "test", {"x"},
                                {ffi::Any(MakeSpecTensor({1, 10}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    DataType f32 = DataType::Float(32);
    DataType f16 = DataType::Float(16);
    Var x("x", TSInfo({1, 10}, f32));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var astype = bb->Emit(relax::astype(x, f16), "astype");
    Var gv1    = EmitDebugOutput(bb, astype, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, true, DictAttrs(attrs)), "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestTensorOpManipulate
//
// Python equivalent:
//   class Model(Module):
//       def test(self, x: Tensor):
//           z0 = x.reshape(2, 5, 2)
//           z1 = x.permute_dims(2, 1, 0)
//           z2 = x.repeat(2, axis=1)
//           return (z0, z1, z2)
//
//   irmodule, _ = m.export_tvm(
//       spec={"test": {"x": spec.Tensor([2,1,10],"float32")}}, debug=True)
//
// Expected:
//   def test(x: (2,1,10)f32, _io: Object):
//     num_input = 2
//     reshape   = R.reshape(x, [2,5,2])
//     permute_dims = R.permute_dims(x, axes=[2,1,0])
//     repeat    = R.repeat(x, repeats=2, axis=1)
//     gv1 = (reshape, permute_dims, repeat), (_io,)
// ===========================================================================
TEST(NNTensor, TestTensorOpManipulate) {
  const ffi::Function op_reshape = NNOp("reshape");
  const ffi::Function op_permute = NNOp("permute_dims");
  const ffi::Function op_repeat  = NNOp("repeat");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> test_fn =
      [op_reshape, op_permute, op_repeat](
          ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();

    // reshape(x, [2, 5, 2])
    ffi::Array<ffi::Any> new_shape{ffi::Any(int64_t(2)), ffi::Any(int64_t(5)),
                                   ffi::Any(int64_t(2))};
    ffi::Any z0 = op_reshape(x->expr, new_shape, ffi::String("reshape"));

    // permute_dims(x, axes=[2, 1, 0])
    ffi::Array<Integer> axes{Integer(2), Integer(1), Integer(0)};
    ffi::Any z1 = op_permute(x->expr, ffi::Optional<ffi::Array<Integer>>(axes),
                             ffi::String("permute_dims"));

    // repeat(x, repeats=2, axis=1)
    ffi::Any z2 = op_repeat(x->expr, int64_t(2), ffi::Optional<int64_t>(1),
                            ffi::String("repeat"));

    ffi::Array<ffi::Any> out{z0, z1, z2};
    return ffi::Any(out);
  };

  IRModule actual = ExportDebug(test_fn, "test", {"x"},
                                {ffi::Any(MakeSpecTensor({2, 1, 10}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    DataType f32 = DataType::Float(32);
    Var x("x", TSInfo({2, 1, 10}, f32));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // reshape(x, [2, 5, 2])
    Var reshape = bb->Emit(
        relax::reshape(x, ShapeExpr(ffi::Array<PrimExpr>{
            IntImm(DataType::Int(64), 2),
            IntImm(DataType::Int(64), 5),
            IntImm(DataType::Int(64), 2)})),
        "reshape");

    // permute_dims(x, axes=[2, 1, 0])
    Var permute = bb->Emit(
        relax::permute_dims(x, ffi::Optional<ffi::Array<Integer>>(
            ffi::Array<Integer>{Integer(2), Integer(1), Integer(0)})),
        "permute_dims");

    // repeat(x, repeats=2, axis=1)
    Var repeat = bb->Emit(relax::repeat(x, 2, ffi::Optional<int64_t>(1)), "repeat");

    Var gv1 = EmitDebugOutput(bb, relax::Tuple({reshape, permute, repeat}), io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, true, DictAttrs(attrs)), "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

}  // namespace testing
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
