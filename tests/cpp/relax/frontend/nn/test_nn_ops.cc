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
 * \file tests/cpp/relax/frontend/nn/test_nn_ops.cc
 * \brief C++ unit tests for the nn.op FFI handlers.
 *
 * Uses only APIs that actually exist in this codebase:
 *  - Internal src/ op headers (binary, unary, manipulate, etc.)
 *  - BlockBuilder low-level API (BeginScope/EndScope, BeginDataflowBlock/EndBlock,
 *    Emit, EmitOutput, Finalize)
 *  - ExportToIRModule from exporter.h
 *  - ffi::StructuralEqual for IR comparison
 */

#include <gtest/gtest.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/tensor.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

// Internal op headers (test lives in src tree, so relative includes are fine)
#include "../../../../../src/relax/op/nn/attention.h"
#include "../../../../../src/relax/op/nn/nn.h"
#include "../../../../../src/relax/op/tensor/binary.h"
#include "../../../../../src/relax/op/tensor/create.h"
#include "../../../../../src/relax/op/tensor/datatype.h"
#include "../../../../../src/relax/op/tensor/index.h"
#include "../../../../../src/relax/op/tensor/linear_algebra.h"
#include "../../../../../src/relax/op/tensor/manipulate.h"
#include "../../../../../src/relax/op/tensor/sampling.h"
#include "../../../../../src/relax/op/tensor/sorting.h"
#include "../../../../../src/relax/op/tensor/statistical.h"
#include "../../../../../src/relax/op/tensor/unary.h"

// nn frontend headers
#include "../../../../../src/relax/frontend/nn/core.h"
#include "../../../../../src/relax/frontend/nn/exporter.h"
#include "../../../../../src/relax/frontend/nn/spec.h"
#include "../../../../../src/relax/ir/emit_te.h"
#include "../../../../../src/te/operation/create_primfunc.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace testing {

// ===========================================================================
// Helpers
// ===========================================================================

static ffi::Function NNOp(const std::string& name) {
  const std::string key = "relax.frontend.nn.op." + name;
  auto f = ffi::Function::GetGlobal(key);
  TVM_FFI_ICHECK(f.has_value()) << "nn.op not found: " << key;
  return f.value();
}

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

// Build and export a single-method module (debug=true by default).
static IRModule ExportSingle(const std::string& method_name, ffi::Function forward_fn,
                             ffi::Array<ffi::String> arg_names, ffi::Array<ffi::Any> arg_specs,
                             ffi::Map<ffi::String, NNParameter> named_params = {},
                             bool debug = true) {
  MethodSpec ms(forward_fn, arg_names, arg_specs, "plain", "plain");
  ModuleSpec mod_spec(ffi::Array<ffi::String>{ffi::String(method_name)},
                      ffi::Array<ffi::Any>{ffi::Any(ms)}, named_params, {});  // named_effects
  return ExportToIRModule(mod_spec, debug);
}

// ---------------------------------------------------------------------------
// Build the _initialize_effect function and add it to bb.
//
//   def _initialize_effect() -> R.Tuple(R.Object):
//       with R.dataflow():
//           _io  = R.null_value()
//           lv   = (_io,)
//           gv   = lv
//       return gv
// ---------------------------------------------------------------------------
static void EmitInitEffect(BlockBuilder& bb) {
  ffi::Array<Var> params;
  bb->BeginScope(params);
  bb->BeginDataflowBlock();

  static const Op& null_value_op = Op::Get("relax.null_value");
  Var io = bb->Emit(Call(null_value_op, {}, {}, {}), "_io");
  Var lv = bb->Emit(relax::Tuple({io}), "lv");
  Var gv = bb->EmitOutput(lv, "gv");

  BindingBlock df_block = bb->EndBlock();
  Expr body = bb->Normalize(SeqExpr({df_block}, gv));
  bb->EndScope();

  ffi::Map<ffi::String, ffi::Any> init_attrs;
  init_attrs.Set("global_symbol", ffi::Any(ffi::String("_initialize_effect")));
  Function func(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(init_attrs));
  bb->AddFunction(func, "_initialize_effect");
}

// Emit the debug-mode output: ((outputs), (_io,)) and return the bound Var.
static Var EmitDebugOutput(BlockBuilder& bb, Expr outputs, Var io,
                           const std::string& hint = "gv1") {
  return bb->EmitOutput(relax::Tuple({outputs, relax::Tuple({io})}), hint);
}

// Assert structural equality.
static void AssertStructEqual(const IRModule& actual, const IRModule& expected) {
  EXPECT_TRUE(ffi::StructuralEqual()(actual, expected)) << "\n=== Actual ===\n"
                                                        << actual << "\n=== Expected ===\n"
                                                        << expected;
}

// Build a float32 scalar constant: mirrors R.const(v, "float32").
static Expr MakeF32Const(float v) {
  auto tensor = runtime::Tensor::Empty(ffi::Shape({}), DLDataType{kDLFloat, 32, 1},
                                       DLDevice{kDLCPU, 0}, std::nullopt);
  tensor.CopyFromBytes(&v, sizeof(float));
  return relax::Constant(tensor, std::nullopt);
}

// Build a [10,10] ShapeExpr.
static ShapeExpr Shape10x10() {
  return ShapeExpr(
      ffi::Array<PrimExpr>{IntImm(DataType::Int(64), 10), IntImm(DataType::Int(64), 10)});
}

// ===========================================================================
// TestUnary
// ===========================================================================
TEST(NNOps, TestUnary) {
  static const ffi::Function op_square = NNOp("square");
  static const ffi::Function op_sqrt = NNOp("sqrt");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    ffi::Array<ffi::Any> out;
    out.push_back(op_square(x->expr, ffi::String("square")));
    out.push_back(op_sqrt(x->expr, ffi::String("sqrt")));
    return ffi::Any(out);
  };

  IRModule actual =
      ExportSingle("test", forward, {"x"}, {ffi::Any(MakeSpecTensor({1, 10}, "float32"))});

  // Build expected
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({1, 10}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var sq = bb->Emit(relax::square(x), "square");
    Var sr = bb->Emit(relax::sqrt(x), "sqrt");
    Var gv1 = EmitDebugOutput(bb, relax::Tuple({sq, sr}), io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    Function func(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs));
    bb->AddFunction(func, "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestBinary
// ===========================================================================
TEST(NNOps, TestBinary) {
  static const ffi::Function op_add = NNOp("add");
  static const ffi::Function op_mul = NNOp("multiply");
  static const ffi::Function op_div = NNOp("divide");
  static const ffi::Function op_matmul = NNOp("matmul");
  static const ffi::Function op_max = NNOp("maximum");
  static const ffi::Function op_min = NNOp("minimum");
  static const ffi::Function op_sub = NNOp("subtract");
  static const ffi::Function op_gt = NNOp("greater");
  static const ffi::Function op_ge = NNOp("greater_equal");
  static const ffi::Function op_lt = NNOp("less");
  static const ffi::Function op_le = NNOp("less_equal");
  static const ffi::Function op_eq = NNOp("equal");
  static const ffi::Function op_ne = NNOp("not_equal");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    NNTensor y = args.at("y").cast<NNTensor>();
    ffi::Array<ffi::Any> out;
    out.push_back(op_add(x->expr, y->expr, ffi::String("add")));
    out.push_back(op_mul(x->expr, y->expr, ffi::String("mul")));
    out.push_back(op_div(x->expr, y->expr, ffi::String("divide")));
    out.push_back(op_matmul(x->expr, y->expr, ffi::Optional<ffi::String>(), ffi::String("matmul")));
    out.push_back(op_max(x->expr, y->expr, ffi::String("maximum")));
    out.push_back(op_min(x->expr, y->expr, ffi::String("minimum")));
    out.push_back(op_sub(x->expr, y->expr, ffi::String("subtract")));
    out.push_back(op_gt(x->expr, y->expr, ffi::String("greater")));
    out.push_back(op_ge(x->expr, y->expr, ffi::String("greater_equal")));
    out.push_back(op_lt(x->expr, y->expr, ffi::String("less")));
    out.push_back(op_le(x->expr, y->expr, ffi::String("less_equal")));
    out.push_back(op_eq(x->expr, y->expr, ffi::String("equal")));
    out.push_back(op_ne(x->expr, y->expr, ffi::String("not_equal")));
    return ffi::Any(out);
  };

  IRModule actual = ExportSingle(
      "test", forward, {"x", "y"},
      {ffi::Any(MakeSpecTensor({1, 10}, "float32")), ffi::Any(MakeSpecTensor({10, 1}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({1, 10}, DataType::Float(32)));
    Var y("y", TSInfo({10, 1}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, y, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var z0 = bb->Emit(relax::add(x, y), "add");
    Var z1 = bb->Emit(relax::multiply(x, y), "mul");
    Var z2 = bb->Emit(relax::divide(x, y), "divide");
    Var z3 = bb->Emit(relax::matmul(x, y, std::nullopt), "matmul");
    Var z4 = bb->Emit(relax::maximum(x, y), "maximum");
    Var z5 = bb->Emit(relax::minimum(x, y), "minimum");
    Var z6 = bb->Emit(relax::subtract(x, y), "subtract");
    Var z7 = bb->Emit(relax::greater(x, y), "greater");
    Var z8 = bb->Emit(relax::greater_equal(x, y), "greater_equal");
    Var z9 = bb->Emit(relax::less(x, y), "less");
    Var z10 = bb->Emit(relax::less_equal(x, y), "less_equal");
    Var z11 = bb->Emit(relax::equal(x, y), "equal");
    Var z12 = bb->Emit(relax::not_equal(x, y), "not_equal");
    Expr outputs = relax::Tuple({z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12});
    Var gv1 = EmitDebugOutput(bb, outputs, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(3)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    Function func(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs));
    bb->AddFunction(func, "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestSum
// ===========================================================================
TEST(NNOps, TestSum) {
  static const ffi::Function op_sum = NNOp("sum");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    return op_sum(x->expr, ffi::Array<Integer>{1, 2}, /*keepdims=*/true, ffi::String("sum"));
  };

  IRModule actual =
      ExportSingle("test", forward, {"x"}, {ffi::Any(MakeSpecTensor({3, 5, 2, 4}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({3, 5, 2, 4}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var s = bb->Emit(relax::sum(x, ffi::Array<Integer>{1, 2}, true), "sum");
    Var gv1 = EmitDebugOutput(bb, s, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestDatatype
// ===========================================================================
TEST(NNOps, TestDatatype) {
  static const ffi::Function op_astype = NNOp("astype");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    return op_astype(x->expr, ffi::String("float16"), ffi::String("astype"));
  };

  IRModule actual =
      ExportSingle("test", forward, {"x"}, {ffi::Any(MakeSpecTensor({2, 1, 10}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({2, 1, 10}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var a = bb->Emit(relax::astype(x, DataType::Float(16)), "astype");
    Var gv1 = EmitDebugOutput(bb, a, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestIndex
// ===========================================================================
TEST(NNOps, TestIndex) {
  static const ffi::Function op_take = NNOp("take");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    NNTensor y = args.at("y").cast<NNTensor>();
    return op_take(x->expr, y->expr, Integer(2), ffi::String("take"));
  };

  IRModule actual = ExportSingle(
      "test", forward, {"x", "y"},
      {ffi::Any(MakeSpecTensor({2, 1, 10}, "float32")), ffi::Any(MakeSpecTensor({5}, "int32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({2, 1, 10}, DataType::Float(32)));
    Var y("y", TSInfo({5}, DataType::Int(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, y, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var t = bb->Emit(relax::take(x, y, ffi::Optional<int64_t>(2)), "take");
    Var gv1 = EmitDebugOutput(bb, t, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(3)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestTensorExprOp
// ===========================================================================
TEST(NNOps, TestTensorExprOp) {
  static const ffi::Function op_te_op = NNOp("tensor_expr_op");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> named_args) -> ffi::Any {
    NNTensor x = named_args.at("x").cast<NNTensor>();

    ffi::TypedFunction<ffi::Array<te::Tensor>(ffi::Array<te::Tensor>)> te_func =
        [](ffi::Array<te::Tensor> inputs) -> ffi::Array<te::Tensor> {
      te::Tensor data = inputs[0];
      te::Tensor out = te::compute(
          data->shape,
          [&](const ffi::Array<tir::Var>& idx) -> PrimExpr {
            return data(idx) + tir::make_const(DataType::Float(32), 1.0f);
          },
          "T_add");
      return {out};
    };

    return op_te_op(te_func, ffi::String("add_one"), ffi::Array<Expr>{x->expr},
                    ffi::Optional<ffi::Map<ffi::String, ffi::Any>>());
  };

  IRModule actual =
      ExportSingle("test", forward, {"x"}, {ffi::Any(MakeSpecTensor({10, 10}, "float32"))});

  // Build expected: same call_tir directly
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({10, 10}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // Build the same TE compute.
    // The exporter's tensor_expr_op calls WithoutAttr(prim_func, "global_symbol") to make
    // the TIR func private. We must do the same so NormalizeGlobalVar keeps it as
    // a private func named "add_one" (not renamed to public "main").
    ffi::Map<tir::Var, PrimExpr> empty_map;
    te::Tensor te_x = TETensor(x, empty_map, "input_0");
    te::Tensor te_out = te::compute(
        te_x->shape,
        [&](const ffi::Array<tir::Var>& indices) -> PrimExpr {
          return te_x(indices) + tir::make_const(DataType::Float(32), 1.0f);
        },
        "T_add");
    tir::PrimFunc prim_func = tir::CreatePrimFunc({te_x, te_out});
    // Strip global_symbol so the TIR func is private (matches what tensor_expr_op does)
    prim_func = WithoutAttr(prim_func, "global_symbol");
    GlobalVar gv_func = bb->AddFunction(prim_func, "add_one");

    static const Op& call_tir_op = Op::Get("relax.call_tir");
    TensorStructInfo out_sinfo = TSInfo({10, 10}, DataType::Float(32));
    // hint "add_one" deduplicates to "add_one1" since GlobalVar "add_one" already exists
    Var lv1 = bb->Emit(Call(call_tir_op, {gv_func, relax::Tuple({x})}, tvm::Attrs(), {out_sinfo}),
                       "add_one");
    Var gv1 = EmitDebugOutput(bb, lv1, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestMax
// ===========================================================================
TEST(NNOps, TestMax) {
  static const ffi::Function op_max = NNOp("max");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    return op_max(x->expr, ffi::Array<Integer>{1, 2}, /*keepdims=*/true, ffi::String("max"));
  };

  IRModule actual =
      ExportSingle("test", forward, {"x"}, {ffi::Any(MakeSpecTensor({3, 5, 2, 4}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({3, 5, 2, 4}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var m = bb->Emit(relax::max(x, ffi::Array<Integer>{1, 2}, true), "max");
    Var gv1 = EmitDebugOutput(bb, m, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestMin
// ===========================================================================
TEST(NNOps, TestMin) {
  static const ffi::Function op_min = NNOp("min");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    return op_min(x->expr, ffi::Array<Integer>{1, 2}, /*keepdims=*/true, ffi::String("min"));
  };

  IRModule actual =
      ExportSingle("test", forward, {"x"}, {ffi::Any(MakeSpecTensor({3, 5, 2, 4}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({3, 5, 2, 4}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var m = bb->Emit(relax::min(x, ffi::Array<Integer>{1, 2}, true), "min");
    Var gv1 = EmitDebugOutput(bb, m, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestManipulate
// ===========================================================================
TEST(NNOps, TestManipulate) {
  static const ffi::Function op_broadcast_to = NNOp("broadcast_to");
  static const ffi::Function op_permute_dims = NNOp("permute_dims");
  static const ffi::Function op_reshape = NNOp("reshape");
  static const ffi::Function op_repeat = NNOp("repeat");
  static const ffi::Function op_squeeze = NNOp("squeeze");
  static const ffi::Function op_unsqueeze = NNOp("unsqueeze");
  static const ffi::Function op_concat = NNOp("concat");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();

    ffi::Array<ffi::Any> bcast_shape;
    bcast_shape.push_back(ffi::Any(int64_t(2)));
    bcast_shape.push_back(ffi::Any(int64_t(5)));
    bcast_shape.push_back(ffi::Any(int64_t(2)));
    ffi::Any z0 = op_broadcast_to(x->expr, bcast_shape, ffi::String("broadcast_to"));

    ffi::Any z1 = op_permute_dims(
        x->expr,
        ffi::Optional<ffi::Array<Integer>>(ffi::Array<Integer>{Integer(2), Integer(1), Integer(0)}),
        ffi::String("permute_dims"));

    ffi::Array<ffi::Any> new_shape;
    new_shape.push_back(ffi::Any(int64_t(1)));
    new_shape.push_back(ffi::Any(int64_t(10)));
    ffi::Any z2 = op_reshape(x->expr, new_shape, ffi::String("reshape"));

    ffi::Any z3 =
        op_repeat(x->expr, int(2), ffi::Optional<Integer>(Integer(1)), ffi::String("repeat"));

    ffi::Any z4 = op_squeeze(x->expr, int(0), ffi::String("squeeze"));

    ffi::Any z5 = op_unsqueeze(x->expr, int(0), ffi::String("unsqueeze"));

    ffi::Array<Var> tensors{x->expr, x->expr};
    ffi::Any z6 = op_concat(tensors, int(0), ffi::String("concat"));

    ffi::Array<ffi::Any> out;
    out.push_back(z0);
    out.push_back(z1);
    out.push_back(z2);
    out.push_back(z3);
    out.push_back(z4);
    out.push_back(z5);
    out.push_back(z6);
    return ffi::Any(out);
  };

  IRModule actual =
      ExportSingle("test", forward, {"x"}, {ffi::Any(MakeSpecTensor({1, 5, 2}, "float32"))});

  // Build expected IR
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({1, 5, 2}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    Var z0 = bb->Emit(
        relax::broadcast_to(x, ShapeExpr(ffi::Array<PrimExpr>{IntImm(DataType::Int(64), 2),
                                                              IntImm(DataType::Int(64), 5),
                                                              IntImm(DataType::Int(64), 2)})),
        "broadcast_to");
    Var z1 =
        bb->Emit(relax::permute_dims(x, ffi::Array<Integer>{Integer(2), Integer(1), Integer(0)}),
                 "permute_dims");
    Var z2 =
        bb->Emit(relax::reshape(x, ShapeExpr(ffi::Array<PrimExpr>{IntImm(DataType::Int(64), 1),
                                                                  IntImm(DataType::Int(64), 10)})),
                 "reshape");
    Var z3 = bb->Emit(relax::repeat(x, 2, ffi::Optional<int64_t>(1)), "repeat");
    Var z4 = bb->Emit(
        relax::squeeze(x, ffi::Optional<ffi::Array<Integer>>(ffi::Array<Integer>{Integer(0)})),
        "squeeze");
    Var z5 = bb->Emit(relax::expand_dims(x, ffi::Array<Integer>{Integer(0)}), "unsqueeze");
    Var z6 = bb->Emit(relax::concat(relax::Tuple({x, x}), ffi::Optional<int64_t>(0)), "concat");

    Expr outputs = relax::Tuple({z0, z1, z2, z3, z4, z5, z6});
    Var gv1 = EmitDebugOutput(bb, outputs, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestCreate
// ===========================================================================
TEST(NNOps, TestCreate) {
  static const ffi::Function op_triu = NNOp("triu");
  static const ffi::Function op_full = NNOp("full");
  static const ffi::Function op_zeros = NNOp("zeros");
  static const ffi::Function op_arange = NNOp("arange");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();

    // triu(x, k=0)
    op_triu(x->expr, int(0), ffi::String("triu"));

    // full([10,10], fill_value=const(10,"float32"), dtype="float32")  x3
    // Emit the scalar constant through the active BlockBuilder so it becomes
    // a proper dataflow binding (not a free Var) that NNFull can accept.
    BlockBuilder cur_bb = BlockBuilder_Current();
    TVM_FFI_ICHECK(cur_bb.defined()) << "must be called inside a BlockBuilder scope";
    float fval = 10.0f;
    auto fill_tensor = runtime::Tensor::Empty(ffi::Shape({}), DLDataType{kDLFloat, 32, 1},
                                              DLDevice{kDLCPU, 0}, std::nullopt);
    fill_tensor.CopyFromBytes(&fval, sizeof(float));
    Var fv = cur_bb->Emit(relax::Constant(fill_tensor, std::nullopt), "fill_value");

    ffi::Array<ffi::Any> shape10x10;
    shape10x10.push_back(ffi::Any(int64_t(10)));
    shape10x10.push_back(ffi::Any(int64_t(10)));
    op_full(shape10x10, fv, ffi::String("float32"), ffi::String("full"));
    op_full(shape10x10, fv, ffi::String("float32"), ffi::String("full1"));
    op_full(shape10x10, fv, ffi::String("float32"), ffi::String("full2"));

    // zeros([10,10], dtype="float32")
    ffi::Array<ffi::Any> shape10x10b;
    shape10x10b.push_back(ffi::Any(int64_t(10)));
    shape10x10b.push_back(ffi::Any(int64_t(10)));
    op_zeros(shape10x10b, ffi::String("float32"), ffi::String("zeros"));

    // zeros([10,10], dtype="float16")
    ffi::Array<ffi::Any> shape10x10c;
    shape10x10c.push_back(ffi::Any(int64_t(10)));
    shape10x10c.push_back(ffi::Any(int64_t(10)));
    op_zeros(shape10x10c, ffi::String("float16"), ffi::String("zeros1"));

    // arange(0, 10, 1, "float32")
    op_arange(ffi::Any(int64_t(0)), ffi::Any(int64_t(10)), ffi::Any(int64_t(1)),
              ffi::Optional<ffi::String>(ffi::String("float32")), ffi::String("arange"));

    return ffi::Any(x->expr);  // return x unchanged
  };

  IRModule actual =
      ExportSingle("test", forward, {"x"}, {ffi::Any(MakeSpecTensor({10, 10}, "float32"))});

  // Build expected IR
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({10, 10}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    bb->Emit(relax::triu(x, 0), "triu");

    // fill_value = R.const(10.0, "float32") emitted as a dataflow binding
    Var fill_val = bb->Emit(MakeF32Const(10.0f), "fill_value");
    bb->Emit(relax::full(Shape10x10(), fill_val, DataType::Float(32)), "full");
    bb->Emit(relax::full(Shape10x10(), fill_val, DataType::Float(32)), "full1");
    bb->Emit(relax::full(Shape10x10(), fill_val, DataType::Float(32)), "full2");

    bb->Emit(relax::zeros(Shape10x10(), DataType::Float(32)), "zeros");
    bb->Emit(relax::zeros(Shape10x10(), DataType::Float(16)), "zeros1");

    bb->Emit(relax::arange(PrimValue(IntImm(DataType::Int(64), 0)),
                           PrimValue(IntImm(DataType::Int(64), 10)),
                           PrimValue(IntImm(DataType::Int(64), 1)), DataType::Float(32)),
             "arange");

    Var gv1 = EmitDebugOutput(bb, x, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestChunk
// ===========================================================================
TEST(NNOps, TestChunk) {
  // NNChunk signature: (x: Var, chunks: int, dim: int, name: String) -> Any
  // It calls relax::split(x, IntImm(64, chunks), dim) which produces a
  // TupleStructInfo; WrapNested then emits TupleGetItem for each element.
  static const ffi::Function op_chunk = NNOp("chunk");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    // chunk(x, chunks=4, dim=0)
    return op_chunk(x->expr, int(4), int(0), ffi::String("chunk"));
  };

  IRModule actual =
      ExportSingle("test", forward, {"x"}, {ffi::Any(MakeSpecTensor({8}, "float32"))});

  // Build expected IR.
  // op.chunk returns a tuple of 4 tensors each of shape (2,).
  // The exporter emits:
  //   chunk   = split(x, 4, axis=0)   [TupleStructInfo]
  //   chunk.0 = chunk[0]
  //   chunk.1 = chunk[1]
  //   chunk.2 = chunk[2]
  //   chunk.3 = chunk[3]
  //   gv1     = (chunk.0, chunk.1, chunk.2, chunk.3), (_io,)
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({8}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // split(x, 4, axis=0) → TupleStructInfo of 4 × (2,) tensors
    Var chunk = bb->Emit(relax::split(x, IntImm(DataType::Int(64), 4), 0), "chunk");

    // TupleGetItem for each element
    Var c0 = bb->Emit(TupleGetItem(chunk, 0), "chunk.0");
    Var c1 = bb->Emit(TupleGetItem(chunk, 1), "chunk.1");
    Var c2 = bb->Emit(TupleGetItem(chunk, 2), "chunk.2");
    Var c3 = bb->Emit(TupleGetItem(chunk, 3), "chunk.3");

    Expr outputs = relax::Tuple({c0, c1, c2, c3});
    Var gv1 = EmitDebugOutput(bb, outputs, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ---------------------------------------------------------------------------
// PrimFunc builders and constants for TensorIr tests
// ---------------------------------------------------------------------------

// MakeTirFunc: simple (16,16) float32 PrimFunc that evaluates 0.
static tir::PrimFunc MakeTirFunc() {
  DataType f32 = DataType::Float(32);
  auto I64 = [](int64_t v) { return IntImm(DataType::Int(64), v); };
  ffi::Array<PrimExpr> shape{I64(16), I64(16)};
  tir::Var buf_a_var("A", DataType::Handle());
  tir::Var buf_b_var("B", DataType::Handle());
  tir::Buffer buf_a = tir::decl_buffer(shape, f32, "A");
  tir::Buffer buf_b = tir::decl_buffer(shape, f32, "B");
  ffi::Map<tir::Var, tir::Buffer> buffer_map;
  buffer_map.Set(buf_a_var, buf_a);
  buffer_map.Set(buf_b_var, buf_b);
  ffi::Map<ffi::String, ffi::Any> attrs_map;
  attrs_map.Set("tir.noalias", ffi::Any(Bool(true)));
  attrs_map.Set("global_symbol", ffi::Any(ffi::String("tir_func")));
  return tir::PrimFunc({buf_a_var, buf_b_var}, tir::Evaluate(IntImm(DataType::Int(32), 0)),
                       VoidType(), buffer_map, DictAttrs(attrs_map));
}

// Constants for TestTensorIrOp
static constexpr int kNumQHeads = 8;
static constexpr int kNumKVHeads = 8;
static constexpr int kHeadDim = 16;
static constexpr int kFusedHeads = kNumQHeads + kNumKVHeads * 2;  // 24

// MakeFusedRopePrimFunc: llama_fused_rope with dynamic batch/seq dims.
static tir::PrimFunc MakeFusedRopePrimFunc() {
  DataType f16 = DataType::Float(16);
  DataType i64 = DataType::Int(64);
  tir::Var batch_size("batch_size", i64);
  tir::Var seq_len("seq_len", i64);
  auto I64 = [&](int64_t v) { return IntImm(i64, v); };
  ffi::Array<PrimExpr> qkv_shape{batch_size, seq_len, I64(kFusedHeads), I64(kHeadDim)};
  ffi::Array<PrimExpr> q_shape{batch_size, seq_len, I64(kNumQHeads), I64(kHeadDim)};
  ffi::Array<PrimExpr> kv_shape{batch_size, seq_len, I64(kNumKVHeads), I64(kHeadDim)};
  tir::Var var_qkv("var_qkv", DataType::Handle());
  tir::Var var_q("var_q", DataType::Handle());
  tir::Var var_k("var_k", DataType::Handle());
  tir::Var var_v("var_v", DataType::Handle());
  tir::Var offset("offset", i64);
  tir::Buffer buf_qkv = tir::decl_buffer(qkv_shape, f16, "qkv");
  tir::Buffer buf_q = tir::decl_buffer(q_shape, f16, "q");
  tir::Buffer buf_k = tir::decl_buffer(kv_shape, f16, "k");
  tir::Buffer buf_v = tir::decl_buffer(kv_shape, f16, "v");
  ffi::Map<tir::Var, tir::Buffer> buffer_map;
  buffer_map.Set(var_qkv, buf_qkv);
  buffer_map.Set(var_q, buf_q);
  buffer_map.Set(var_k, buf_k);
  buffer_map.Set(var_v, buf_v);
  ffi::Map<ffi::String, ffi::Any> attrs_map;
  attrs_map.Set("global_symbol", ffi::Any(ffi::String("llama_fused_rope")));
  return tir::PrimFunc({var_qkv, var_q, var_k, var_v, offset}, tir::Evaluate(offset), VoidType(),
                       buffer_map, DictAttrs(attrs_map));
}

// Constants and builder for TestTensorIrInplaceOp
static constexpr int64_t kHiddenSize = 4096;
static const char* kInplaceDtype = "float16";

static tir::PrimFunc MakeInplaceTakePrimFunc() {
  DataType f16 = DataType::Float(16);
  DataType i32 = DataType::Int(32);
  DataType i64 = DataType::Int(64);
  tir::Var vocab_size("vocab_size", i64);
  tir::Var seq_len("seq_len", i64);
  tir::Var total_seq_len("total_seq_len", i64);
  tir::Var offset("offset", i64);
  auto I64 = [&](int64_t v) { return IntImm(i64, v); };
  tir::Var var_weight("var_weight", DataType::Handle());
  tir::Var var_pos("var_pos", DataType::Handle());
  tir::Var var_embeddings("var_embeddings", DataType::Handle());
  tir::Buffer buf_weight = tir::decl_buffer({vocab_size, I64(kHiddenSize)}, f16, "weight");
  tir::Buffer buf_pos = tir::decl_buffer({seq_len}, i32, "pos");
  tir::Buffer buf_embeddings =
      tir::decl_buffer({total_seq_len, I64(kHiddenSize)}, f16, "embeddings");
  ffi::Map<tir::Var, tir::Buffer> buffer_map;
  buffer_map.Set(var_weight, buf_weight);
  buffer_map.Set(var_pos, buf_pos);
  buffer_map.Set(var_embeddings, buf_embeddings);
  tir::Var ax0("ax0", i64), ax1("ax1", i64);
  PrimExpr pos_val = tir::BufferLoad(buf_pos, {ax0});
  PrimExpr w_val = tir::BufferLoad(buf_weight, {pos_val, ax1});
  tir::Stmt store = tir::BufferStore(buf_embeddings, w_val, {ax0 + offset, ax1});
  tir::Stmt inner = tir::SBlockRealize(
      {ax0, ax1}, tir::make_const(DataType::Bool(), true),
      tir::SBlock(
          {tir::IterVar(Range(I64(0), seq_len), tir::Var("v0", i64), tir::kDataPar),
           tir::IterVar(Range(I64(0), I64(kHiddenSize)), tir::Var("v1", i64), tir::kDataPar)},
          {tir::BufferRegion(buf_pos, {Range(ax0, ax0 + I64(1))}),
           tir::BufferRegion(buf_weight,
                             {Range(pos_val, pos_val + I64(1)), Range(ax1, ax1 + I64(1))})},
          {tir::BufferRegion(buf_embeddings, {Range(ax0 + offset, ax0 + offset + I64(1)),
                                              Range(ax1, ax1 + I64(1))})},
          "T_take", store));
  tir::Stmt loop = tir::For(ax0, I64(0), seq_len, tir::ForKind::kSerial,
                            tir::For(ax1, I64(0), I64(kHiddenSize), tir::ForKind::kSerial, inner));
  ffi::Map<ffi::String, ffi::Any> attrs_map;
  attrs_map.Set("tir.noalias", ffi::Any(Bool(true)));
  attrs_map.Set("global_symbol", ffi::Any(ffi::String("inplace_take")));
  return tir::PrimFunc({var_weight, var_pos, var_embeddings, offset}, loop, VoidType(), buffer_map,
                       DictAttrs(attrs_map));
}

// ===========================================================================
// TestNN
// ===========================================================================
TEST(NNOps, TestNN) {
  // Note: relax.nn.relu6 is not registered as a C++ Op; the nn.op.relu6 FFI
  // handler implements it as clip(x, 0, 6).  We use op.clip directly here.
  static const ffi::Function op_log = NNOp("log");
  static const ffi::Function op_floor = NNOp("floor");
  static const ffi::Function op_relu = NNOp("relu");
  static const ffi::Function op_clip = NNOp("clip");
  static const ffi::Function op_silu = NNOp("silu");
  static const ffi::Function op_gelu = NNOp("gelu");
  static const ffi::Function op_sigmoid = NNOp("sigmoid");
  static const ffi::Function op_tanh = NNOp("tanh");
  static const ffi::Function op_exp = NNOp("exp");
  static const ffi::Function op_negative = NNOp("negative");
  static const ffi::Function op_softplus = NNOp("softplus");
  static const ffi::Function op_softmax = NNOp("softmax");
  static const ffi::Function op_prelu = NNOp("prelu");
  static const ffi::Function op_rms_norm = NNOp("rms_norm");
  static const ffi::Function op_group_norm = NNOp("group_norm");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    NNTensor weight = args.at("weight").cast<NNTensor>();
    NNTensor bias = args.at("bias").cast<NNTensor>();
    op_log(x->expr, ffi::String("log"));
    op_floor(x->expr, ffi::String("floor"));
    op_relu(x->expr, ffi::String("relu"));
    op_clip(x->expr, double(0.0), double(6.0), ffi::String("relu6"));
    op_silu(x->expr, ffi::String("silu"));
    op_gelu(x->expr, ffi::Optional<ffi::String>(), ffi::String("gelu"));
    op_sigmoid(x->expr, ffi::String("sigmoid"));
    op_tanh(x->expr, ffi::String("tanh"));
    op_exp(x->expr, ffi::String("exp"));
    op_negative(x->expr, ffi::String("negative"));
    op_softplus(x->expr, double(1.0), double(20.0), ffi::String("softplus"));
    op_softmax(x->expr, int(2), ffi::String("softmax"));
    op_prelu(x->expr, bias->expr, ffi::String("prelu"));
    op_rms_norm(x->expr, weight->expr, ffi::Array<Integer>{Integer(-2), Integer(-1)}, double(1e-5),
                ffi::String("rms_norm"));
    op_rms_norm(x->expr, weight->expr, ffi::Array<Integer>{Integer(-2), Integer(-1)}, double(1e-5),
                ffi::String("rms_norm1"));
    op_group_norm(x->expr, ffi::Optional<Var>(bias->expr), ffi::Optional<Var>(bias->expr), int(1),
                  int(1), ffi::Array<Integer>{Integer(2), Integer(3)}, double(1e-5),
                  ffi::String("group_norm"));
    return ffi::Any(x->expr);
  };

  IRModule actual = ExportSingle(
      "test", forward, {"x", "weight", "bias"},
      {ffi::Any(MakeSpecTensor({2, 3, 4, 5}, "float32")),
       ffi::Any(MakeSpecTensor({4, 5}, "float32")), ffi::Any(MakeSpecTensor({3}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({2, 3, 4, 5}, DataType::Float(32)));
    Var weight("weight", TSInfo({4, 5}, DataType::Float(32)));
    Var bias("bias", TSInfo({3}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, weight, bias, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    bb->Emit(relax::log(x), "log");
    bb->Emit(relax::floor(x), "floor");
    bb->Emit(relax::relu(x), "relu");
    bb->Emit(relax::clip(x, PrimValue(FloatImm(DataType::Float(32), 0.0)),
                         PrimValue(FloatImm(DataType::Float(32), 6.0))),
             "relu6");
    bb->Emit(relax::silu(x), "silu");
    bb->Emit(relax::gelu(x), "gelu");
    bb->Emit(relax::sigmoid(x), "sigmoid");
    bb->Emit(relax::tanh(x), "tanh");
    bb->Emit(relax::exp(x), "exp");
    bb->Emit(relax::negative(x), "negative");
    bb->Emit(relax::softplus(x, 1.0, 20.0), "softplus");
    bb->Emit(relax::softmax(x, 2), "softmax");
    bb->Emit(relax::prelu(x, bias, 1), "prelu");
    bb->Emit(relax::rms_norm(x, weight, ffi::Array<Integer>{Integer(-2), Integer(-1)}, 1e-5),
             "rms_norm");
    bb->Emit(relax::rms_norm(x, weight, ffi::Array<Integer>{Integer(-2), Integer(-1)}, 1e-5),
             "rms_norm1");
    bb->Emit(relax::group_norm(x, bias, bias, 1, 1, ffi::Array<Integer>{Integer(2), Integer(3)},
                               1e-5, true, true),
             "group_norm");
    Var gv1 = EmitDebugOutput(bb, x, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(4)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestScaledDotProductAttention
// ===========================================================================
TEST(NNOps, TestScaledDotProductAttention) {
  static const ffi::Function op_sdpa = NNOp("scaled_dot_product_attention");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor query = args.at("query").cast<NNTensor>();
    NNTensor key = args.at("key").cast<NNTensor>();
    NNTensor value = args.at("value").cast<NNTensor>();
    return op_sdpa(query->expr, key->expr, value->expr, ffi::Optional<ffi::String>(),
                   ffi::Optional<double>(), ffi::String("scaled_dot_product_attention"));
  };

  IRModule actual = ExportSingle("test", forward, {"query", "key", "value"},
                                 {ffi::Any(MakeSpecTensor({1, 32, 32, 32}, "float32")),
                                  ffi::Any(MakeSpecTensor({1, 32, 32, 32}, "float32")),
                                  ffi::Any(MakeSpecTensor({1, 32, 32, 32}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var query("query", TSInfo({1, 32, 32, 32}, DataType::Float(32)));
    Var key("key", TSInfo({1, 32, 32, 32}, DataType::Float(32)));
    Var value("value", TSInfo({1, 32, 32, 32}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{query, key, value, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var sdpa = bb->Emit(
        relax::attention(query, key, value, std::nullopt, std::nullopt, std::nullopt, std::nullopt),
        "scaled_dot_product_attention");
    Var gv1 = EmitDebugOutput(bb, sdpa, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(4)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestSortArgsortTopk
// ===========================================================================
TEST(NNOps, TestSortArgsortTopk) {
  static const ffi::Function op_sort = NNOp("sort");
  static const ffi::Function op_argsort = NNOp("argsort");
  static const ffi::Function op_topk = NNOp("topk");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    ffi::Any z0 = op_sort(x->expr, int(-1), bool(true), ffi::String("sort"));
    ffi::Any z1 =
        op_argsort(x->expr, int(-1), bool(false), ffi::String("int32"), ffi::String("argsort"));
    ffi::Any z2 = op_topk(x->expr, int(2), int(-1), ffi::String("both"), bool(true),
                          ffi::String("int32"), ffi::String("topk"));
    ffi::Array<ffi::Any> out;
    out.push_back(z0);
    out.push_back(z1);
    out.push_back(z2);
    return ffi::Any(out);
  };

  // Symbolic first dim; effect_mode="none" → no _io, no _initialize_effect.
  ffi::Array<ffi::Any> spec_shape;
  spec_shape.push_back(ffi::Any(ffi::String("seq_len")));
  spec_shape.push_back(ffi::Any(int64_t(64)));
  MethodSpec ms_sort(forward, {"x"}, {ffi::Any(SpecTensor(spec_shape, "float16"))}, "plain",
                     "none");
  IRModule actual = ExportToIRModule(
      ModuleSpec(ffi::Array<ffi::String>{"foo"}, ffi::Array<ffi::Any>{ffi::Any(ms_sort)}, {}, {}),
      /*debug=*/false);

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  {
    tir::Var seq_len("seq_len", DataType::Int(64));
    TensorStructInfo x_sinfo(
        ShapeExpr(ffi::Array<PrimExpr>{seq_len, IntImm(DataType::Int(64), 64)}),
        DataType::Float(16));
    Var x("x", x_sinfo);
    ffi::Array<Var> params{x};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var sort_v = bb->Emit(relax::sort(x, -1, true), "sort");
    Var argsort_v = bb->Emit(relax::argsort(x, -1, false, DataType::Int(32)), "argsort");
    Var topk_v = bb->Emit(relax::topk(x, 2, -1, "both", true, DataType::Int(32)), "topk");
    Var topk_0 = bb->Emit(TupleGetItem(topk_v, 0), "topk.0");
    Var topk_1 = bb->Emit(TupleGetItem(topk_v, 1), "topk.1");
    Var gv =
        bb->EmitOutput(relax::Tuple({sort_v, argsort_v, relax::Tuple({topk_0, topk_1})}), "gv");
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(1)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("foo")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "foo");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestTensorIrOpNoTirVar
// ===========================================================================
TEST(NNOps, TestTensorIrOpNoTirVar) {
  static const ffi::Function op_tensor_ir_op = NNOp("tensor_ir_op");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor A = args.at("A").cast<NNTensor>();
    Var out_ph("B", TSInfo({16, 16}, DataType::Float(32)));
    return op_tensor_ir_op(MakeTirFunc(), ffi::String("tir_func"), ffi::Array<Expr>{A->expr},
                           ffi::Array<ffi::Any>{ffi::Any(out_ph)});
  };

  MethodSpec ms_notv(forward, {"A"}, {ffi::Any(MakeSpecTensor({16, 16}, "float32"))}, "plain",
                     "none");
  IRModule actual = ExportToIRModule(
      ModuleSpec(ffi::Array<ffi::String>{"test"}, ffi::Array<ffi::Any>{ffi::Any(ms_notv)}, {}, {}),
      /*debug=*/false);

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  {
    Var A("A", TSInfo({16, 16}, DataType::Float(32)));
    ffi::Array<Var> params{A};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    GlobalVar gv_func = bb->AddFunction(MakeTirFunc(), "tir_func");
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    Var lv = bb->Emit(Call(call_tir_op, {gv_func, relax::Tuple({A})}, tvm::Attrs(),
                           {TSInfo({16, 16}, DataType::Float(32))}),
                      "tir_func");
    Var gv = bb->EmitOutput(lv, "gv");
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(1)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestTensorIrOp
// ===========================================================================
TEST(NNOps, TestTensorIrOp) {
  static const ffi::Function op_tensor_ir_op = NNOp("tensor_ir_op");
  tir::PrimFunc fused_rope = MakeFusedRopePrimFunc();

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [fused_rope](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor qkv = args.at("qkv").cast<NNTensor>();
    Var offset_var = args.at("offset").cast<Var>();
    DataType f16 = DataType::Float(16);
    auto I64 = [](int64_t v) { return IntImm(DataType::Int(64), v); };
    auto make_out = [&](int heads) -> Var {
      return Var(
          "out",
          TensorStructInfo(
              ShapeExpr(ffi::Array<PrimExpr>{I64(1), I64(1), I64(heads), I64(kHeadDim)}), f16));
    };
    ffi::Array<Expr> call_args{qkv->expr, offset_var};
    ffi::Array<ffi::Any> out_list{ffi::Any(make_out(kNumQHeads)), ffi::Any(make_out(kNumKVHeads)),
                                  ffi::Any(make_out(kNumKVHeads))};
    return op_tensor_ir_op(fused_rope, ffi::String("llama_fused_rope"), call_args, out_list);
  };

  SpecInt spec_int;
  IRModule actual = ExportSingle(
      "test", forward, {"qkv", "offset"},
      {ffi::Any(MakeSpecTensor({1, 1, kFusedHeads, kHeadDim}, "float16")), ffi::Any(spec_int)});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    DataType f16 = DataType::Float(16);
    Var qkv("qkv", TSInfo({1, 1, kFusedHeads, kHeadDim}, f16));
    tir::Var offset_sym("offset", DataType::Int(64));
    Var offset("offset_1", ShapeStructInfo(ffi::Array<PrimExpr>{offset_sym}));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{qkv, offset, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    GlobalVar gv_func = bb->AddFunction(MakeFusedRopePrimFunc(), "llama_fused_rope");
    TensorStructInfo q_sinfo = TSInfo({1, 1, kNumQHeads, kHeadDim}, f16);
    TensorStructInfo kv_sinfo = TSInfo({1, 1, kNumKVHeads, kHeadDim}, f16);
    StructInfo out_sinfo{TupleStructInfo(ffi::Array<StructInfo>{q_sinfo, kv_sinfo, kv_sinfo})};
    ShapeExpr tir_vars(ffi::Array<PrimExpr>{offset_sym});
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    Var lv1 = bb->Emit(
        Call(call_tir_op, {gv_func, relax::Tuple({qkv}), tir_vars}, tvm::Attrs(), {out_sinfo}),
        "llama_fused_rope");
    Var r0 = bb->Emit(TupleGetItem(lv1, 0), "llama_fused_rope.0");
    Var r1 = bb->Emit(TupleGetItem(lv1, 1), "llama_fused_rope.1");
    Var r2 = bb->Emit(TupleGetItem(lv1, 2), "llama_fused_rope.2");
    Var gv1 = EmitDebugOutput(bb, relax::Tuple({r0, r1, r2}), io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(3)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestTensorIrInplaceOp
// ===========================================================================
TEST(NNOps, TestTensorIrInplaceOp) {
  static const ffi::Function op_inplace = NNOp("tensor_ir_inplace_op");
  tir::PrimFunc inplace_take = MakeInplaceTakePrimFunc();

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [inplace_take](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor embedding_table = args.at("embedding_table").cast<NNTensor>();
    NNTensor input_ids = args.at("input_ids").cast<NNTensor>();
    NNTensor embedding_dst = args.at("embedding_dst").cast<NNTensor>();
    Var offset_var = args.at("offset").cast<Var>();
    StructInfo dst_sinfo = GetStructInfo(embedding_dst->expr);
    const auto* ts = dst_sinfo.as<TensorStructInfoNode>();
    TVM_FFI_ICHECK(ts) << "embedding_dst must have TensorStructInfo";
    Var out_ph("out", ffi::GetRef<TensorStructInfo>(ts));
    ffi::Array<Expr> call_args{embedding_table->expr, input_ids->expr, embedding_dst->expr,
                               offset_var};
    ffi::Array<Integer> inplace_indices{Integer(2)};
    ffi::Array<ffi::Any> out_list{ffi::Any(out_ph)};
    return op_inplace(inplace_take, ffi::String("inplace_take"), call_args, inplace_indices,
                      out_list);
  };

  ffi::Array<ffi::Any> et_shape, ii_shape, ed_shape;
  et_shape.push_back(ffi::Any(ffi::String("vocab_size")));
  et_shape.push_back(ffi::Any(int64_t(kHiddenSize)));
  ii_shape.push_back(ffi::Any(ffi::String("seq_len")));
  ed_shape.push_back(ffi::Any(ffi::String("total_seq_len")));
  ed_shape.push_back(ffi::Any(int64_t(kHiddenSize)));
  MethodSpec ms_ip(
      forward, {"embedding_table", "input_ids", "embedding_dst", "offset"},
      {ffi::Any(SpecTensor(et_shape, kInplaceDtype)), ffi::Any(SpecTensor(ii_shape, "int32")),
       ffi::Any(SpecTensor(ed_shape, kInplaceDtype)), ffi::Any(SpecInt())},
      "packed", "none");
  IRModule actual = ExportToIRModule(
      ModuleSpec(ffi::Array<ffi::String>{"test"}, ffi::Array<ffi::Any>{ffi::Any(ms_ip)}, {}, {}),
      /*debug=*/false);

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  {
    DataType f16 = DataType::Float(16);
    DataType i32 = DataType::Int(32);
    DataType i64 = DataType::Int(64);
    auto I64 = [&](int64_t v) { return IntImm(i64, v); };
    tir::Var vocab_size("vocab_size", i64);
    tir::Var seq_len("seq_len", i64);
    tir::Var total_seq_len("total_seq_len", i64);
    tir::Var offset_tvar("offset", i64);
    TensorStructInfo et_sinfo(ShapeExpr(ffi::Array<PrimExpr>{vocab_size, I64(kHiddenSize)}), f16);
    TensorStructInfo ii_sinfo(ShapeExpr(ffi::Array<PrimExpr>{seq_len}), i32);
    TensorStructInfo ed_sinfo(ShapeExpr(ffi::Array<PrimExpr>{total_seq_len, I64(kHiddenSize)}),
                              f16);
    Var embedding_table("embedding_table", et_sinfo);
    Var input_ids("input_ids", ii_sinfo);
    Var embedding_dst("embedding_dst", ed_sinfo);
    Var offset("offset_1", ShapeStructInfo(ffi::Array<PrimExpr>{offset_tvar}));
    Var packed_params("packed_params", TupleStructInfo(ffi::Array<StructInfo>{}));
    ffi::Array<Var> params{embedding_table, input_ids, embedding_dst, offset, packed_params};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    GlobalVar gv_func = bb->AddFunction(MakeInplaceTakePrimFunc(), "inplace_take");
    ObjectPtr<CallTIRInplaceAttrs> inplace_attrs = ffi::make_object<CallTIRInplaceAttrs>();
    inplace_attrs->inplace_indices = ffi::Array<Integer>{Integer(2)};
    ShapeExpr tir_vars(ffi::Array<PrimExpr>{offset_tvar});
    static const Op& call_tir_inplace_op = Op::Get("relax.call_tir_inplace");
    Var lv1 = bb->Emit(
        Call(call_tir_inplace_op,
             {gv_func, relax::Tuple({embedding_table, input_ids, embedding_dst}), tir_vars},
             tvm::Attrs(inplace_attrs), {ed_sinfo}),
        "inplace_take");
    Var gv = bb->EmitOutput(lv1, "gv");
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(4)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "test");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestMultinomialFromUniform
// ===========================================================================
TEST(NNOps, TestMultinomialFromUniform) {
  static const ffi::Function op_multi = NNOp("multinomial_from_uniform");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor prob = args.at("prob").cast<NNTensor>();
    NNTensor usample = args.at("uniform_sample").cast<NNTensor>();
    NNTensor sindices = args.at("sample_indices").cast<NNTensor>();
    return op_multi(prob->expr, usample->expr, sindices->expr, ffi::String("int64"),
                    ffi::String("multinomial_from_uniform"));
  };

  IRModule actual = ExportSingle(
      "foo", forward, {"prob", "uniform_sample", "sample_indices"},
      {ffi::Any(MakeSpecTensor({3, 5}, "float32")), ffi::Any(MakeSpecTensor({6, 1}, "float32")),
       ffi::Any(MakeSpecTensor({6, 1}, "int64"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var prob("prob", TSInfo({3, 5}, DataType::Float(32)));
    Var usample("uniform_sample", TSInfo({6, 1}, DataType::Float(32)));
    Var sindices("sample_indices", TSInfo({6, 1}, DataType::Int(64)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{prob, usample, sindices, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var m = bb->Emit(relax::multinomial_from_uniform(prob, usample, sindices, DataType::Int(64)),
                     "multinomial_from_uniform");
    Var gv1 = EmitDebugOutput(bb, m, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(4)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("foo")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "foo");
  }
  AssertStructEqual(actual, bb->Finalize());
}

}  // namespace testing
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
