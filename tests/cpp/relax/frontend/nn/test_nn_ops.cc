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
 * \brief C++ ports of all tests in tests/python/relax/test_frontend_nn_op.py.
 *
 * Each test follows the same three-step pattern:
 *   1. Build the "actual" IRModule by calling ExportToIRModule with a forward
 *      function that invokes the nn.op FFI handlers.
 *   2. Build the "expected" IRModule directly with BlockBuilder using the
 *      corresponding relax C++ op functions.
 *   3. Assert ffi::StructuralEqual()(actual, expected).
 *
 * Tests that require GPU execution (multinomial, sample_top_p, renormalize)
 * port only the IR structural check; runtime execution is left to the Python
 * suite.  Tests whose expected IR involves call_tir / call_tir_inplace /
 * emit_te (tensor_expr_op, tensor_ir_op, tensor_ir_inplace_op, extern,
 * get_timestep_embedding) are ported as IR structural checks using the
 * equivalent relax ops where a direct C++ op exists, or are noted as
 * Python-only where the expected IR requires TE/TIR PrimFunc construction
 * that has no C++ BlockBuilder equivalent.
 *
 * Moved from test_nn_export.cc:
 *   TestUnary  (was the only test in that file)
 */

#include <gtest/gtest.h>

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/op/tensor/binary.h>
#include <tvm/relax/op/tensor/create.h>
#include <tvm/relax/op/tensor/index.h>
#include <tvm/relax/op/tensor/manipulate.h>
#include <tvm/relax/op/tensor/nn.h>
#include <tvm/relax/op/tensor/reduce.h>
#include <tvm/relax/op/tensor/search.h>
#include <tvm/relax/op/tensor/statistical.h>
#include <tvm/relax/op/tensor/unary.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include "../../../../src/relax/frontend/nn/core.h"
#include "../../../../src/relax/frontend/nn/exporter.h"
#include "../../../../src/relax/frontend/nn/spec.h"
#include "../../../../src/relax/ir/emit_te.h"
#include "../../../../src/te/operation/create_primfunc.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace testing {

// ===========================================================================
// Helpers shared across all tests
// ===========================================================================

// Retrieve a registered nn.op FFI function by its short name.
// Full registry key is "relax.frontend.nn.op.<name>".
static ffi::Function NNOp(const std::string& name) {
  const std::string key = "relax.frontend.nn.op." + name;
  const ffi::Function* f = tvm::runtime::Registry::Get(key);
  TVM_FFI_ICHECK(f != nullptr) << "nn.op not found: " << key;
  return *f;
}

// Convenience: make a static-shape TensorStructInfo.
static TensorStructInfo TSInfo(std::initializer_list<int64_t> dims, DataType dtype) {
  ffi::Array<PrimExpr> shape_dims;
  for (int64_t d : dims) shape_dims.push_back(tir::IntImm(DataType::Int(64), d));
  return TensorStructInfo(ShapeExpr(shape_dims), dtype);
}

// Convenience: make a symbolic-shape TensorStructInfo (one tir::Var per name).
static TensorStructInfo TSInfoSym(
    std::initializer_list<std::pair<std::string, int64_t>> dims_or_syms,
    DataType dtype,
    std::unordered_map<std::string, tir::Var>& sym_map) {
  ffi::Array<PrimExpr> shape_dims;
  for (const auto& [name, val] : dims_or_syms) {
    if (val < 0) {
      // symbolic
      if (!sym_map.count(name)) sym_map[name] = tir::Var(name, DataType::Int(64));
      shape_dims.push_back(sym_map.at(name));
    } else {
      shape_dims.push_back(tir::IntImm(DataType::Int(64), val));
    }
  }
  return TensorStructInfo(ShapeExpr(shape_dims), dtype);
}

// Emit the standard _initialize_effect function into bb.
static void EmitInitEffect(BlockBuilder& bb) {
  with(bb->function("_initialize_effect", {}, {}), [&]() {
    with(bb->dataflow(), [&]() {
      Var io  = bb->Emit(relax::op::null_value(), "_io");
      Var lv  = bb->Emit(relax::Tuple({io}), "lv");
      Var gv  = bb->EmitOutput(lv, "gv");
      bb->EmitFuncOutput(gv, {});
    });
  });
}

// Wrap a single output Expr + _io Var into the debug-mode output tuple
// ((outputs...), (_io,)) and emit it as the dataflow output.
static Var EmitDebugOutput(BlockBuilder& bb, Expr outputs, Var io,
                           const std::string& hint = "gv1") {
  Expr effects = relax::Tuple({io});
  return bb->EmitOutput(relax::Tuple({outputs, effects}), hint);
}

// Build a SpecTensor from a list of static int64 dims.
static SpecTensor MakeSpecTensor(std::initializer_list<int64_t> dims,
                                 const std::string& dtype) {
  ffi::Array<ffi::Any> shape;
  for (int64_t d : dims) shape.push_back(ffi::Any(d));
  return SpecTensor(shape, dtype);
}

// Build a SpecTensor with symbolic (string) or static (int64) dims.
// Pass dims as pairs: {"name", -1} for symbolic, {"", value} for static.
static SpecTensor MakeSpecTensorMixed(
    std::initializer_list<std::pair<std::string, int64_t>> dims,
    const std::string& dtype) {
  ffi::Array<ffi::Any> shape;
  for (const auto& [name, val] : dims) {
    if (val < 0)
      shape.push_back(ffi::Any(ffi::String(name)));
    else
      shape.push_back(ffi::Any(val));
  }
  return SpecTensor(shape, dtype);
}

// Build and export a single-method module.
// forward_fn receives Map<String,Any> and returns Any.
// arg_names / arg_specs describe the method inputs.
// named_params is empty for parameter-free models.
static IRModule ExportSingle(
    const std::string& method_name,
    ffi::Function forward_fn,
    ffi::Array<ffi::String> arg_names,
    ffi::Array<ffi::Any> arg_specs,
    ffi::Map<ffi::String, NNParameter> named_params = {},
    bool debug = true,
    const std::string& param_mode = "plain",
    const std::string& effect_mode = "plain") {
  MethodSpec ms(forward_fn, arg_names, arg_specs, param_mode, effect_mode);
  ModuleSpec mod_spec(
      ffi::Array<ffi::String>{ffi::String(method_name)},
      ffi::Array<ffi::Any>{ffi::Any(ms)},
      named_params);
  return ExportToIRModule(mod_spec, debug);
}

// ---------------------------------------------------------------------------
// Helper: emit a call_tir node into the current dataflow block.
//
// Registers prim_func in bb under func_name, then emits:
//   var = R.call_tir(gv, Tuple(args), out_sinfo=out_sinfo_list)
// or, if tir_vars is non-empty:
//   var = R.call_tir(gv, Tuple(args), out_sinfo=..., tir_vars=ShapeExpr(tir_vars))
// ---------------------------------------------------------------------------
static Var EmitCallTir(
    BlockBuilder& bb,
    tir::PrimFunc prim_func,
    const std::string& func_name,
    ffi::Array<Expr> args,
    ffi::Array<TensorStructInfo> out_sinfo_list,
    ffi::Array<PrimExpr> tir_vars = {},
    const std::string& binding_name = "lv1") {
  GlobalVar gv = bb->AddFunction(prim_func, func_name);
  static const Op& call_tir_op = Op::Get("relax.call_tir");

  StructInfo out_sinfo = (out_sinfo_list.size() == 1)
      ? StructInfo(out_sinfo_list[0])
      : StructInfo(TupleStructInfo(
            ffi::Array<StructInfo>(out_sinfo_list.begin(), out_sinfo_list.end())));

  ffi::Array<Expr> call_args{gv, relax::Tuple(args)};
  if (!tir_vars.empty()) call_args.push_back(ShapeExpr(tir_vars));

  return bb->Emit(Call(call_tir_op, call_args, tvm::Attrs(), {out_sinfo}), binding_name);
}

// ---------------------------------------------------------------------------
// Helper: emit a call_tir_inplace node into the current dataflow block.
// ---------------------------------------------------------------------------
static Var EmitCallTirInplace(
    BlockBuilder& bb,
    tir::PrimFunc prim_func,
    const std::string& func_name,
    ffi::Array<Expr> args,
    ffi::Array<Integer> inplace_indices,
    ffi::Array<TensorStructInfo> out_sinfo_list,
    ffi::Array<PrimExpr> tir_vars = {},
    const std::string& binding_name = "lv1") {
  GlobalVar gv = bb->AddFunction(prim_func, func_name);
  static const Op& call_tir_inplace_op = Op::Get("relax.call_tir_inplace");

  ObjectPtr<CallTIRInplaceAttrs> attrs = ffi::make_object<CallTIRInplaceAttrs>();
  attrs->inplace_indices = inplace_indices;

  StructInfo out_sinfo = (out_sinfo_list.size() == 1)
      ? StructInfo(out_sinfo_list[0])
      : StructInfo(TupleStructInfo(
            ffi::Array<StructInfo>(out_sinfo_list.begin(), out_sinfo_list.end())));

  ffi::Array<Expr> call_args{gv, relax::Tuple(args)};
  if (!tir_vars.empty()) call_args.push_back(ShapeExpr(tir_vars));

  return bb->Emit(
      Call(call_tir_inplace_op, call_args, tvm::Attrs(attrs), {out_sinfo}),
      binding_name);
}

// Assert structural equality and print both modules on failure.
static void AssertStructEqual(const IRModule& actual, const IRModule& expected) {
  EXPECT_TRUE(relax::analysis::well_formed(actual));
  EXPECT_TRUE(relax::analysis::well_formed(expected));
  EXPECT_TRUE(ffi::StructuralEqual()(actual, expected))
      << "\n=== Actual ===\n"   << actual
      << "\n=== Expected ===\n" << expected;
}

// ===========================================================================
// TestUnary  (moved from test_nn_export.cc)
//
// Python original:
//   class Model(Module):
//       def test(self, x: Tensor):
//           z0 = op.square(x)
//           z1 = op.sqrt(x)
//           return (z0, z1)
//
//   spec = {"test": {"x": spec.Tensor([1, 10], "float32")}}
// ===========================================================================
TEST(NNOps, TestUnary) {
  static const ffi::Function op_square = NNOp("square");
  static const ffi::Function op_sqrt   = NNOp("sqrt");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").operator NNTensor();
    ffi::Array<ffi::Any> out;
    out.push_back(op_square(x->expr, ffi::String("square")));
    out.push_back(op_sqrt(x->expr, ffi::String("sqrt")));
    return ffi::Any(out);
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"x"}, {ffi::Any(MakeSpecTensor({1, 10}, "float32"))});

  // Build expected
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({1, 10}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    with(bb->function("test", {x, io}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        Var sq = bb->Emit(relax::square(x), "square");
        Var sr = bb->Emit(relax::sqrt(x), "sqrt");
        Var gv1 = EmitDebugOutput(bb, relax::Tuple({sq, sr}), io);
        bb->EmitFuncOutput(gv1, {x, io});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestBinary
//
// Python original:
//   def test(self, x: Tensor, y: Tensor):
//       z0  = op.add(x, y)        z1  = op.multiply(x, y)
//       z2  = op.divide(x, y)     z3  = op.matmul(x, y)
//       z4  = op.maximum(x, y)    z5  = op.minimum(x, y)
//       z6  = op.subtract(x, y)   z7  = op.greater(x, y)
//       z8  = op.greater_equal(x, y)  z9  = op.less(x, y)
//       z10 = op.less_equal(x, y) z11 = op.equal(x, y)
//       z12 = op.not_equal(x, y)
//       return (z0..z12)
//
//   x: [1,10] f32,  y: [10,1] f32
// ===========================================================================
TEST(NNOps, TestBinary) {
  static const ffi::Function op_add      = NNOp("add");
  static const ffi::Function op_mul      = NNOp("multiply");
  static const ffi::Function op_div      = NNOp("divide");
  static const ffi::Function op_matmul   = NNOp("matmul");
  static const ffi::Function op_max      = NNOp("maximum");
  static const ffi::Function op_min      = NNOp("minimum");
  static const ffi::Function op_sub      = NNOp("subtract");
  static const ffi::Function op_gt       = NNOp("greater");
  static const ffi::Function op_ge       = NNOp("greater_equal");
  static const ffi::Function op_lt       = NNOp("less");
  static const ffi::Function op_le       = NNOp("less_equal");
  static const ffi::Function op_eq       = NNOp("equal");
  static const ffi::Function op_ne       = NNOp("not_equal");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").operator NNTensor();
    NNTensor y = args.at("y").operator NNTensor();
    ffi::Array<ffi::Any> out;
    out.push_back(op_add   (x->expr, y->expr, ffi::String("add")));
    out.push_back(op_mul   (x->expr, y->expr, ffi::String("mul")));
    out.push_back(op_div   (x->expr, y->expr, ffi::String("divide")));
    out.push_back(op_matmul(x->expr, y->expr, ffi::Optional<ffi::String>(), ffi::String("matmul")));
    out.push_back(op_max   (x->expr, y->expr, ffi::String("maximum")));
    out.push_back(op_min   (x->expr, y->expr, ffi::String("minimum")));
    out.push_back(op_sub   (x->expr, y->expr, ffi::String("subtract")));
    out.push_back(op_gt    (x->expr, y->expr, ffi::String("greater")));
    out.push_back(op_ge    (x->expr, y->expr, ffi::String("greater_equal")));
    out.push_back(op_lt    (x->expr, y->expr, ffi::String("less")));
    out.push_back(op_le    (x->expr, y->expr, ffi::String("less_equal")));
    out.push_back(op_eq    (x->expr, y->expr, ffi::String("equal")));
    out.push_back(op_ne    (x->expr, y->expr, ffi::String("not_equal")));
    return ffi::Any(out);
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"x", "y"},
      {ffi::Any(MakeSpecTensor({1, 10}, "float32")),
       ffi::Any(MakeSpecTensor({10, 1}, "float32"))});

  // Build expected
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({1, 10}, DataType::Float(32)));
    Var y("y", TSInfo({10, 1}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(3)));
    with(bb->function("test", {x, y, io}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        Var z0  = bb->Emit(relax::add(x, y),                          "add");
        Var z1  = bb->Emit(relax::multiply(x, y),                     "mul");
        Var z2  = bb->Emit(relax::divide(x, y),                       "divide");
        Var z3  = bb->Emit(relax::matmul(x, y, DataType::Void()),     "matmul");
        Var z4  = bb->Emit(relax::maximum(x, y),                      "maximum");
        Var z5  = bb->Emit(relax::minimum(x, y),                      "minimum");
        Var z6  = bb->Emit(relax::subtract(x, y),                     "subtract");
        Var z7  = bb->Emit(relax::greater(x, y),                      "greater");
        Var z8  = bb->Emit(relax::greater_equal(x, y),                "greater_equal");
        Var z9  = bb->Emit(relax::less(x, y),                         "less");
        Var z10 = bb->Emit(relax::less_equal(x, y),                   "less_equal");
        Var z11 = bb->Emit(relax::equal(x, y),                        "equal");
        Var z12 = bb->Emit(relax::not_equal(x, y),                    "not_equal");
        Expr outputs = relax::Tuple({z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12});
        Var gv1 = EmitDebugOutput(bb, outputs, io);
        bb->EmitFuncOutput(gv1, {x, y, io});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestSum / TestMax / TestMin  (reduce ops with axis + keepdims)
//
// Python original (sum):
//   def test(self, x: Tensor):
//       return op.sum(x, axis=[1, 2], keepdims=True)
//   x: [3,5,2,4] f32
// ===========================================================================
TEST(NNOps, TestSum) {
  static const ffi::Function op_sum = NNOp("sum");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").operator NNTensor();
    return op_sum(x->expr,
                  ffi::Array<int64_t>{1, 2},
                  /*keepdims=*/true,
                  ffi::String("sum"));
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"x"}, {ffi::Any(MakeSpecTensor({3, 5, 2, 4}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({3, 5, 2, 4}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    with(bb->function("test", {x, io}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        Var s   = bb->Emit(relax::sum(x, {1, 2}, /*keepdims=*/true), "sum");
        Var gv1 = EmitDebugOutput(bb, s, io);
        bb->EmitFuncOutput(gv1, {x, io});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

TEST(NNOps, TestMax) {
  static const ffi::Function op_max = NNOp("max");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").operator NNTensor();
    return op_max(x->expr,
                  ffi::Array<int64_t>{1, 2},
                  /*keepdims=*/true,
                  ffi::String("max"));
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"x"}, {ffi::Any(MakeSpecTensor({3, 5, 2, 4}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({3, 5, 2, 4}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    with(bb->function("test", {x, io}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        Var m   = bb->Emit(relax::max(x, {1, 2}, /*keepdims=*/true), "max");
        Var gv1 = EmitDebugOutput(bb, m, io);
        bb->EmitFuncOutput(gv1, {x, io});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

TEST(NNOps, TestMin) {
  static const ffi::Function op_min = NNOp("min");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").operator NNTensor();
    return op_min(x->expr,
                  ffi::Array<int64_t>{1, 2},
                  /*keepdims=*/true,
                  ffi::String("min"));
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"x"}, {ffi::Any(MakeSpecTensor({3, 5, 2, 4}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({3, 5, 2, 4}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    with(bb->function("test", {x, io}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        Var m   = bb->Emit(relax::min(x, {1, 2}, /*keepdims=*/true), "min");
        Var gv1 = EmitDebugOutput(bb, m, io);
        bb->EmitFuncOutput(gv1, {x, io});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestManipulate
//
// Python original:
//   def test(self, x: Tensor):   # x: [1,5,2] f32
//       z0 = op.broadcast_to(x, [2, 5, 2])
//       z1 = op.permute_dims(x, [2, 1, 0])
//       z2 = op.reshape(x, [1, 10])
//       z3 = op.repeat(x, repeats=2, axis=1)
//       z4 = op.squeeze(x, 0)
//       z5 = op.unsqueeze(x, 0)
//       z6 = op.concat([x, x], dim=0)
//       return (z0, z1, z2, z3, z4, z5, z6)
// ===========================================================================
TEST(NNOps, TestManipulate) {
  static const ffi::Function op_bcast  = NNOp("broadcast_to");
  static const ffi::Function op_perm   = NNOp("permute_dims");
  static const ffi::Function op_resh   = NNOp("reshape");
  static const ffi::Function op_rep    = NNOp("repeat");
  static const ffi::Function op_sq     = NNOp("squeeze");
  static const ffi::Function op_unsq   = NNOp("unsqueeze");
  static const ffi::Function op_concat = NNOp("concat");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").operator NNTensor();
    ffi::Array<ffi::Any> out;
    out.push_back(op_bcast (x->expr, ffi::Array<int64_t>{2,5,2},  ffi::String("broadcast_to")));
    out.push_back(op_perm  (x->expr, ffi::Array<int64_t>{2,1,0},  ffi::String("permute_dims")));
    out.push_back(op_resh  (x->expr, ffi::Array<int64_t>{1,10},   ffi::String("reshape")));
    out.push_back(op_rep   (x->expr, /*repeats=*/2, /*axis=*/1,   ffi::String("repeat")));
    out.push_back(op_sq    (x->expr, /*axis=*/0,                  ffi::String("squeeze")));
    out.push_back(op_unsq  (x->expr, /*dim=*/0,                   ffi::String("unsqueeze")));
    ffi::Array<Expr> xs{x->expr, x->expr};
    out.push_back(op_concat(xs, /*dim=*/0,                        ffi::String("concat")));
    return ffi::Any(out);
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"x"}, {ffi::Any(MakeSpecTensor({1, 5, 2}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({1, 5, 2}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    with(bb->function("test", {x, io}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        Var z0 = bb->Emit(relax::broadcast_to(x, ShapeExpr({2,5,2})),       "broadcast_to");
        Var z1 = bb->Emit(relax::permute_dims(x, {2,1,0}),                  "permute_dims");
        Var z2 = bb->Emit(relax::reshape(x, ShapeExpr({1,10})),             "reshape");
        Var z3 = bb->Emit(relax::repeat(x, 2, 1),                           "repeat");
        Var z4 = bb->Emit(relax::squeeze(x, {0}),                           "squeeze");
        Var z5 = bb->Emit(relax::expand_dims(x, {0}),                       "unsqueeze");
        Var z6 = bb->Emit(relax::concat({x, x}, 0),                         "concat");
        Expr outputs = relax::Tuple({z0,z1,z2,z3,z4,z5,z6});
        Var gv1 = EmitDebugOutput(bb, outputs, io);
        bb->EmitFuncOutput(gv1, {x, io});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestIndex
//
// Python original:
//   def test(self, x: Tensor, y: Tensor):
//       return op.take(x, y, axis=2)
//   x: [2,1,10] f32,  y: [5] i32
// ===========================================================================
TEST(NNOps, TestIndex) {
  static const ffi::Function op_take = NNOp("take");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").operator NNTensor();
    NNTensor y = args.at("y").operator NNTensor();
    return op_take(x->expr, y->expr, /*axis=*/2, ffi::String("take"));
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"x", "y"},
      {ffi::Any(MakeSpecTensor({2, 1, 10}, "float32")),
       ffi::Any(MakeSpecTensor({5}, "int32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({2, 1, 10}, DataType::Float(32)));
    Var y("y", TSInfo({5}, DataType::Int(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(3)));
    with(bb->function("test", {x, y, io}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        Var t   = bb->Emit(relax::take(x, y, 2), "take");
        Var gv1 = EmitDebugOutput(bb, t, io);
        bb->EmitFuncOutput(gv1, {x, y, io});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestDatatype
//
// Python original:
//   def test(self, x: Tensor):
//       return op.astype(x, "float16")
//   x: [2,1,10] f32
// ===========================================================================
TEST(NNOps, TestDatatype) {
  static const ffi::Function op_astype = NNOp("astype");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").operator NNTensor();
    return op_astype(x->expr, ffi::String("float16"), ffi::String("astype"));
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"x"}, {ffi::Any(MakeSpecTensor({2, 1, 10}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({2, 1, 10}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    with(bb->function("test", {x, io}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        Var a   = bb->Emit(relax::astype(x, DataType::Float(16)), "astype");
        Var gv1 = EmitDebugOutput(bb, a, io);
        bb->EmitFuncOutput(gv1, {x, io});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestNN
//
// Python original:
//   def test(self, x, weight, bias):   # x:[2,3,4,5] weight:[4,5] bias:[3]
//       log_out      = op.log(x)
//       floor_out    = op.floor(x)
//       relu_out     = op.relu(x)
//       relu6_out    = op.relu6(x)
//       silu_out     = op.silu(x)
//       gelu_out     = op.gelu(x)
//       sigmoid_out  = op.sigmoid(x)
//       tanh_out     = op.tanh(x)
//       exp_out      = op.exp(x)
//       negative_out = op.negative(x)
//       softplus_out = op.softplus(x, beta=1.0, threshold=20.0)
//       softmax_out  = op.softmax(x, axis=2)
//       prelu_out    = op.prelu(x, alpha=bias)
//       rms_norm_out = op.rms_norm(x, weight, axes=[-2,-1])
//       rms_norm_with_bias_out = op.rms_norm(x, weight, axes=[-2,-1])
//       group_norm_out = op.group_norm(x, num_groups=1, weight=bias, bias=bias)
//       return x
// ===========================================================================
TEST(NNOps, TestNN) {
  static const ffi::Function op_log      = NNOp("log");
  static const ffi::Function op_floor    = NNOp("floor");
  static const ffi::Function op_relu     = NNOp("relu");
  static const ffi::Function op_relu6    = NNOp("relu6");
  static const ffi::Function op_silu     = NNOp("silu");
  static const ffi::Function op_gelu     = NNOp("gelu");
  static const ffi::Function op_sigmoid  = NNOp("sigmoid");
  static const ffi::Function op_tanh     = NNOp("tanh");
  static const ffi::Function op_exp      = NNOp("exp");
  static const ffi::Function op_neg      = NNOp("negative");
  static const ffi::Function op_softplus = NNOp("softplus");
  static const ffi::Function op_softmax  = NNOp("softmax");
  static const ffi::Function op_prelu    = NNOp("prelu");
  static const ffi::Function op_rms_norm = NNOp("rms_norm");
  static const ffi::Function op_grp_norm = NNOp("group_norm");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x      = args.at("x").operator NNTensor();
    NNTensor weight = args.at("weight").operator NNTensor();
    NNTensor bias   = args.at("bias").operator NNTensor();
    // emit all ops but return x unchanged (matches Python test)
    op_log     (x->expr,                                          ffi::String("log"));
    op_floor   (x->expr,                                          ffi::String("floor"));
    op_relu    (x->expr,                                          ffi::String("relu"));
    op_relu6   (x->expr,                                          ffi::String("relu6"));
    op_silu    (x->expr,                                          ffi::String("silu"));
    op_gelu    (x->expr, ffi::Optional<ffi::String>(),            ffi::String("gelu"));
    op_sigmoid (x->expr,                                          ffi::String("sigmoid"));
    op_tanh    (x->expr,                                          ffi::String("tanh"));
    op_exp     (x->expr,                                          ffi::String("exp"));
    op_neg     (x->expr,                                          ffi::String("negative"));
    op_softplus(x->expr, 1.0, 20.0,                               ffi::String("softplus"));
    op_softmax (x->expr, 2,                                       ffi::String("softmax"));
    op_prelu   (x->expr, bias->expr,                              ffi::String("prelu"));
    op_rms_norm(x->expr, weight->expr, ffi::Array<int64_t>{-2,-1}, 1e-5, ffi::String("rms_norm"));
    op_rms_norm(x->expr, weight->expr, ffi::Array<int64_t>{-2,-1}, 1e-5, ffi::String("rms_norm"));
    op_grp_norm(x->expr, bias->expr, bias->expr, 1, 1,
                ffi::Array<int64_t>{2,3}, 1e-5,                   ffi::String("group_norm"));
    // return x itself
    return ffi::Any(x->expr);
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"x", "weight", "bias"},
      {ffi::Any(MakeSpecTensor({2,3,4,5}, "float32")),
       ffi::Any(MakeSpecTensor({4,5},     "float32")),
       ffi::Any(MakeSpecTensor({3},       "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x     ("x",      TSInfo({2,3,4,5}, DataType::Float(32)));
    Var weight("weight", TSInfo({4,5},     DataType::Float(32)));
    Var bias  ("bias",   TSInfo({3},       DataType::Float(32)));
    Var io    ("_io",    ObjectStructInfo());
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(4)));
    with(bb->function("test", {x, weight, bias, io}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        bb->Emit(relax::log(x),                                          "log");
        bb->Emit(relax::floor(x),                                        "floor");
        bb->Emit(relax::op::relu(x),                                     "relu");
        bb->Emit(relax::op::relu6(x),                                    "relu6");
        bb->Emit(relax::op::silu(x),                                     "silu");
        bb->Emit(relax::op::gelu(x),                                     "gelu");
        bb->Emit(relax::sigmoid(x),                                      "sigmoid");
        bb->Emit(relax::tanh(x),                                         "tanh");
        bb->Emit(relax::exp(x),                                          "exp");
        bb->Emit(relax::negative(x),                                     "negative");
        bb->Emit(relax::op::softplus(x, 1.0, 20.0),                     "softplus");
        bb->Emit(relax::op::softmax(x, 2),                               "softmax");
        bb->Emit(relax::op::prelu(x, bias),                              "prelu");
        bb->Emit(relax::op::rms_norm(x, weight, {-2,-1}, 1e-5),         "rms_norm");
        bb->Emit(relax::op::rms_norm(x, weight, {-2,-1}, 1e-5),         "rms_norm");
        bb->Emit(relax::op::group_norm(x, bias, bias, 1, 1, {2,3}, 1e-5), "group_norm");
        Var gv1 = EmitDebugOutput(bb, x, io);
        bb->EmitFuncOutput(gv1, {x, weight, bias, io});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestCreate
//
// Python original:
//   def test(self, x: Tensor):   # x: [10,10] f32
//       triu_out              = op.triu(x)
//       full_with_scalar_out  = op.full([10,10], fill_value=10)
//       full_with_FloatImm    = op.full([10,10], fill_value=tir.FloatImm("float32",10))
//       full_with_Tensor      = op.full([10,10], fill_value=Tensor.from_scalar(10,"float32"))
//       zeros_out             = op.zeros([10,10])
//       zeros_fp16_out        = op.zeros([10,10], dtype="float16")
//       arange_out            = op.arange(0, 10, 1, "float32")
//       return x
// ===========================================================================
TEST(NNOps, TestCreate) {
  static const ffi::Function op_triu   = NNOp("triu");
  static const ffi::Function op_full   = NNOp("full");
  static const ffi::Function op_zeros  = NNOp("zeros");
  static const ffi::Function op_arange = NNOp("arange");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").operator NNTensor();
    // triu(x, diagonal=0)
    op_triu(x->expr, 0, ffi::String("triu"));
    // full with scalar 10 (three variants all produce the same IR)
    Expr fill = relax::const_(10.0f);
    op_full(ffi::Array<int64_t>{10,10}, fill, ffi::String("float32"), ffi::String("full"));
    op_full(ffi::Array<int64_t>{10,10}, fill, ffi::String("float32"), ffi::String("full"));
    op_full(ffi::Array<int64_t>{10,10}, fill, ffi::String("float32"), ffi::String("full"));
    // zeros f32 and f16
    op_zeros(ffi::Array<int64_t>{10,10}, ffi::String("float32"), ffi::String("zeros"));
    op_zeros(ffi::Array<int64_t>{10,10}, ffi::String("float16"), ffi::String("zeros"));
    // arange(0,10,1,"float32")
    op_arange(int64_t(0), int64_t(10), int64_t(1), ffi::String("float32"), ffi::String("arange"));
    return ffi::Any(x->expr);
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"x"}, {ffi::Any(MakeSpecTensor({10, 10}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({10, 10}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    with(bb->function("test", {x, io}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        Expr fill = relax::const_(10.0f);
        bb->Emit(relax::triu(x, 0),                                          "triu");
        bb->Emit(relax::full(ShapeExpr({10,10}), fill, DataType::Float(32)), "full");
        bb->Emit(relax::full(ShapeExpr({10,10}), fill, DataType::Float(32)), "full");
        bb->Emit(relax::full(ShapeExpr({10,10}), fill, DataType::Float(32)), "full");
        bb->Emit(relax::zeros(ShapeExpr({10,10}), DataType::Float(32)),      "zeros");
        bb->Emit(relax::zeros(ShapeExpr({10,10}), DataType::Float(16)),      "zeros");
        bb->Emit(relax::arange(
                     tir::IntImm(DataType::Int(64), 0),
                     tir::IntImm(DataType::Int(64), 10),
                     tir::IntImm(DataType::Int(64), 1),
                     DataType::Float(32)),                                   "arange");
        Var gv1 = EmitDebugOutput(bb, x, io);
        bb->EmitFuncOutput(gv1, {x, io});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestChunk
//
// Python original:
//   def test(self, x: Tensor):   # x: [8] f32
//       return op.chunk(x, chunks=4)
//
// op.chunk(x, 4) lowers to R.split(x, 4, axis=0) then 4 TupleGetItem bindings.
// ===========================================================================
TEST(NNOps, TestChunk) {
  static const ffi::Function op_chunk = NNOp("chunk");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").operator NNTensor();
    return op_chunk(x->expr, /*chunks=*/4, /*dim=*/0, ffi::String("chunk"));
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"x"}, {ffi::Any(MakeSpecTensor({8}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({8}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    with(bb->function("test", {x, io}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        // R.split(x, 4, axis=0) → tuple of 4 tensors
        Var chunk = bb->Emit(relax::split(x, 4, 0), "chunk");
        Var c0 = bb->Emit(TupleGetItem(chunk, 0), "chunk_0");
        Var c1 = bb->Emit(TupleGetItem(chunk, 1), "chunk_1");
        Var c2 = bb->Emit(TupleGetItem(chunk, 2), "chunk_2");
        Var c3 = bb->Emit(TupleGetItem(chunk, 3), "chunk_3");
        Expr outputs = relax::Tuple({c0, c1, c2, c3});
        Var gv1 = EmitDebugOutput(bb, outputs, io);
        bb->EmitFuncOutput(gv1, {x, io});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestScaledDotProductAttention
//
// Python original:
//   def test(self, query, key, value):   # all [1,32,32,32] f32
//       return op.scaled_dot_product_attention(query, key, value)
// ===========================================================================
TEST(NNOps, TestScaledDotProductAttention) {
  static const ffi::Function op_sdpa = NNOp("scaled_dot_product_attention");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor q = args.at("query").operator NNTensor();
    NNTensor k = args.at("key").operator NNTensor();
    NNTensor v = args.at("value").operator NNTensor();
    // causal_mask=None, scale=None
    return op_sdpa(q->expr, k->expr, v->expr,
                   ffi::Optional<ffi::String>(),
                   ffi::Optional<double>(),
                   ffi::String("scaled_dot_product_attention"));
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"query", "key", "value"},
      {ffi::Any(MakeSpecTensor({1,32,32,32}, "float32")),
       ffi::Any(MakeSpecTensor({1,32,32,32}, "float32")),
       ffi::Any(MakeSpecTensor({1,32,32,32}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var q("query", TSInfo({1,32,32,32}, DataType::Float(32)));
    Var k("key",   TSInfo({1,32,32,32}, DataType::Float(32)));
    Var v("value", TSInfo({1,32,32,32}, DataType::Float(32)));
    Var io("_io",  ObjectStructInfo());
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(4)));
    with(bb->function("test", {q, k, v, io}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        // R.nn.attention(q, k, v, scale=None, causal_mask=None)
        Var sdpa = bb->Emit(
            relax::op::attention(q, k, v,
                                 /*bias=*/ffi::Optional<Expr>(),
                                 /*scale=*/ffi::Optional<FloatImm>(),
                                 /*causal_mask=*/ffi::Optional<ffi::String>(),
                                 /*window_size=*/ffi::Optional<int64_t>()),
            "scaled_dot_product_attention");
        Var gv1 = EmitDebugOutput(bb, sdpa, io);
        bb->EmitFuncOutput(gv1, {q, k, v, io});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestSortArgsortTopk
//
// Python original (no debug=True, no _io):
//   def foo(self, x: Tensor):   # x: ["seq_len", 64] f16
//       z0 = op.sort(x, axis=-1, descending=True)
//       z1 = op.argsort(x, axis=-1, descending=False)
//       z2 = op.topk(x, k=2, axis=-1)
//       return z0, z1, z2
// ===========================================================================
TEST(NNOps, TestSortArgsortTopk) {
  static const ffi::Function op_sort    = NNOp("sort");
  static const ffi::Function op_argsort = NNOp("argsort");
  static const ffi::Function op_topk    = NNOp("topk");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").operator NNTensor();
    ffi::Any z0 = op_sort   (x->expr, -1, true,  ffi::String("sort"));
    ffi::Any z1 = op_argsort(x->expr, -1, false, ffi::String("int32"), ffi::String("argsort"));
    ffi::Any z2 = op_topk   (x->expr, 2, -1, ffi::String("both"), true,
                              ffi::String("int32"), ffi::String("topk"));
    ffi::Array<ffi::Any> out;
    out.push_back(z0);
    out.push_back(z1);
    out.push_back(z2);
    return ffi::Any(out);
  };

  // debug=false → no _io, num_input=1
  IRModule actual = ExportSingle(
      "foo", ffi::Function(forward),
      {"x"},
      {ffi::Any(MakeSpecTensorMixed({{"seq_len", -1}, {"", 64}}, "float16"))},
      /*named_params=*/{},
      /*debug=*/false);

  // Build expected (no _initialize_effect, no _io)
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  {
    std::unordered_map<std::string, tir::Var> sym;
    sym["seq_len"] = tir::Var("seq_len", DataType::Int(64));
    TensorStructInfo x_sinfo(
        ShapeExpr({sym["seq_len"], tir::IntImm(DataType::Int(64), 64)}),
        DataType::Float(16));
    Var x("x", x_sinfo);
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(1)));
    with(bb->function("foo", {x}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        Var z0 = bb->Emit(relax::sort(x, -1, true),                    "sort");
        Var z1 = bb->Emit(relax::argsort(x, -1, false, DataType::Int(32)), "argsort");
        Var topk_raw = bb->Emit(
            relax::topk(x, 2, -1, "both", true, DataType::Int(32)),    "topk");
        Var t0 = bb->Emit(TupleGetItem(topk_raw, 0), "topk_0");
        Var t1 = bb->Emit(TupleGetItem(topk_raw, 1), "topk_1");
        Var gv = bb->EmitOutput(
            relax::Tuple({z0, z1, relax::Tuple({t0, t1})}), "gv");
        bb->EmitFuncOutput(gv, {x});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestMultinomialFromUniform  (IR structural check only; runtime needs CUDA)
//
// Python original:
//   def foo(self, prob, uniform_sample, sample_indices):
//       return op.multinomial_from_uniform(prob, uniform_sample, sample_indices)
//   prob: [3,5] f32,  uniform_sample: [6,1] f32,  sample_indices: [6,1] i64
// ===========================================================================
TEST(NNOps, TestMultinomialFromUniform) {
  static const ffi::Function op_multi = NNOp("multinomial_from_uniform");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor prob    = args.at("prob").operator NNTensor();
    NNTensor usample = args.at("uniform_sample").operator NNTensor();
    NNTensor sindices= args.at("sample_indices").operator NNTensor();
    return op_multi(prob->expr, usample->expr, sindices->expr,
                    ffi::String("int64"), ffi::String("multinomial_from_uniform"));
  };

  IRModule actual = ExportSingle(
      "foo", ffi::Function(forward),
      {"prob", "uniform_sample", "sample_indices"},
      {ffi::Any(MakeSpecTensor({3,5}, "float32")),
       ffi::Any(MakeSpecTensor({6,1}, "float32")),
       ffi::Any(MakeSpecTensor({6,1}, "int64"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var prob    ("prob",           TSInfo({3,5}, DataType::Float(32)));
    Var usample ("uniform_sample", TSInfo({6,1}, DataType::Float(32)));
    Var sindices("sample_indices", TSInfo({6,1}, DataType::Int(64)));
    Var io      ("_io",            ObjectStructInfo());
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(4)));
    with(bb->function("foo", {prob, usample, sindices, io}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        Var m   = bb->Emit(
            relax::multinomial_from_uniform(prob, usample, sindices, DataType::Int(64)),
            "multinomial_from_uniform");
        Var gv1 = EmitDebugOutput(bb, m, io);
        bb->EmitFuncOutput(gv1, {prob, usample, sindices, io});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestTensorExprOp
//
// Python original:
//   class Model(Module):
//       def test(self, x: Tensor):   # x: [10,10] f32
//           return op.tensor_expr_op(
//               tensor_expr_func=lambda x: x + 1,
//               name_hint="add_one",
//               args=[x]
//           )
//
// Expected IRModule:
//   @T.prim_func(private=True)
//   def add_one(A: T.Buffer((10,10),"float32"),
//               T_add: T.Buffer((10,10),"float32")):
//       for ax0, ax1 in T.grid(10, 10):
//           with T.sblock("T_add"):
//               T_add[ax0,ax1] = A[ax0,ax1] + T.float32(1)
//
//   @R.function
//   def test(x: R.Tensor((10,10),"float32"), _io: R.Object):
//       R.func_attr({"num_input": 2})
//       with R.dataflow():
//           lv1 = R.call_tir(cls.add_one, (x,),
//                            out_sinfo=R.Tensor((10,10),"float32"))
//           gv1 = lv1, (_io,)
//       return gv1
//
// C++ approach:
//   Both the actual and expected sides build the same call_tir node by:
//     1. Wrapping the relax Var in a TE placeholder via TETensor.
//     2. Defining the output via te::compute (data + 1.0f).
//     3. Lowering to a TIR PrimFunc via tir::CreatePrimFunc.
//     4. Adding the PrimFunc to the BlockBuilder and emitting call_tir.
//
//   The actual side drives this through op.tensor_expr_op's C++ FFI entry
//   ("relax.frontend.nn.op.tensor_expr_op"), which accepts an ffi::Function
//   that maps Array<te::Tensor> -> Array<te::Tensor> and a name hint.
//   The expected side builds the identical call_tir directly.
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: build the add_one TE compute and emit call_tir into bb.
//
// Given a relax Var `x` with shape [10,10] f32 already live in bb's current
// dataflow block, this function:
//   1. Creates a TE placeholder tensor from x via TETensor.
//   2. Defines T_add = te::compute(x.shape, [](indices){ return data(indices)+1 }).
//   3. Lowers {placeholder, T_add} to a TIR PrimFunc via CreatePrimFunc.
//   4. Adds the PrimFunc to bb under the name "add_one".
//   5. Emits and returns a call_tir Var bound to the result.
// ---------------------------------------------------------------------------
static Var EmitAddOneCallTir(BlockBuilder& bb, Var x) {
  // Step 1: wrap the relax Var in a TE placeholder
  ffi::Map<tir::Var, PrimExpr> empty_map;
  te::Tensor te_x = TETensor(x, empty_map, "A");

  // Step 2: te::compute that adds 1.0f element-wise
  te::Tensor te_out = te::compute(
      te_x->shape,
      [&](const ffi::Array<tir::Var>& indices) -> PrimExpr {
        return te_x(indices) + tir::make_const(DataType::Float(32), 1.0f);
      },
      "T_add");

  // Step 3: lower to TIR PrimFunc
  // CreatePrimFunc takes all tensors in data-flow order: inputs first, then outputs.
  tir::PrimFunc prim_func = tir::CreatePrimFunc({te_x, te_out});
  // Mark private (matches Python test's @T.prim_func(private=True))
  prim_func = WithAttr(prim_func, tvm::attr::kIsPrivateFunc, tvm::Bool(true));

  // Step 4: add to the module under the name "add_one"
  GlobalVar gv_func = bb->AddFunction(prim_func, "add_one");

  // Step 5: emit call_tir
  static const Op& call_tir_op = Op::Get("relax.call_tir");
  TensorStructInfo out_sinfo = TSInfo({10, 10}, DataType::Float(32));
  return bb->Emit(
      Call(call_tir_op,
           {gv_func, relax::Tuple({x})},
           tvm::Attrs(),
           {out_sinfo}),
      "lv1");
}

TEST(NNOps, TestTensorExprOp) {
  // ---- Actual: drive through op.tensor_expr_op FFI ----------------------
  //
  // op.tensor_expr_op(tensor_expr_func, name_hint, args) is registered as
  // "relax.frontend.nn.op.tensor_expr_op".  Its C++ signature accepts:
  //   - tensor_expr_func: ffi::Function(Array<te::Tensor>) -> Array<te::Tensor>
  //   - name_hint:        String
  //   - args:             Array<Expr>   (the relax Var expressions)
  //   - attrs:            Optional<Map<String,Any>>  (primfunc_attrs, default null)
  static const ffi::Function op_te_op = NNOp("tensor_expr_op");

  auto forward = [](ffi::Map<ffi::String, ffi::Any> named_args) -> ffi::Any {
    NNTensor x = named_args.at("x").operator NNTensor();

    // The tensor_expr_func: Array<te::Tensor> -> Array<te::Tensor>
    // Mirrors Python's `lambda x: x + 1`.
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

    return op_te_op(
        ffi::Function(te_func),
        ffi::String("add_one"),
        ffi::Array<Expr>{x->expr},
        ffi::Optional<ffi::Map<ffi::String, ffi::Any>>());
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"x"}, {ffi::Any(MakeSpecTensor({10, 10}, "float32"))});

  // ---- Expected: build the same IRModule directly -----------------------
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({10, 10}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    with(bb->function("test", {x, io}, attrs), [&]() {
      with(bb->dataflow(), [&]() {
        Var lv1 = EmitAddOneCallTir(bb, x);
        Var gv1 = EmitDebugOutput(bb, lv1, io);
        bb->EmitFuncOutput(gv1, {x, io});
      });
    });
  }

  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestTensorIrOpNoTirVar
//
// Python original (no debug, no _io, no tir_vars):
//   @T.prim_func(private=True)
//   def tir_func(A: T.Buffer((16,16),"float32"),
//                B: T.Buffer((16,16),"float32")):
//       T.evaluate(0)
//
//   class Model(Module):
//       def test(self, A: Tensor):
//           return op.tensor_ir_op(
//               tir_func, "tir_func",
//               args=[A],
//               out=[Tensor.placeholder((16,16),"float32")])
//
// Expected:
//   @R.function
//   def test(A: R.Tensor((16,16),"float32")) -> R.Tensor((16,16),"float32"):
//       R.func_attr({"num_input": 1})
//       with R.dataflow():
//           lv = R.call_tir(cls.tir_func, (A,),
//                           out_sinfo=R.Tensor((16,16),"float32"))
//           gv = lv
//       return gv
// ===========================================================================

// Build the trivial tir_func: two 16x16 f32 buffers, body = T.evaluate(0).
static tir::PrimFunc MakeTirFuncNoTirVar() {
  // Buffers
  tir::Var a_data("A", DataType::Handle());
  tir::Var b_data("B", DataType::Handle());
  tir::Buffer buf_a = tir::decl_buffer({16, 16}, DataType::Float(32), "A");
  tir::Buffer buf_b = tir::decl_buffer({16, 16}, DataType::Float(32), "B");

  // Body: T.evaluate(0)
  tir::Stmt body = tir::Evaluate(tir::make_const(DataType::Int(32), 0));

  // Params and buffer_map
  ffi::Array<tir::Var> params{a_data, b_data};
  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(a_data, buf_a);
  buf_map.Set(b_data, buf_b);

  tir::PrimFunc func(params, body, tvm::Type(), buf_map);
  func = WithAttr(func, tvm::attr::kIsPrivateFunc, tvm::Bool(true));
  return func;
}

TEST(NNOps, TestTensorIrOpNoTirVar) {
  static const ffi::Function op_tir_op = NNOp("tensor_ir_op");

  tir::PrimFunc tir_func = MakeTirFuncNoTirVar();

  // ---- Actual -------------------------------------------------------
  auto forward = [&tir_func](ffi::Map<ffi::String, ffi::Any> named_args) -> ffi::Any {
    NNTensor A = named_args.at("A").operator NNTensor();
    // out placeholder: shape [16,16] f32
    NNTensor out_ph = NNTensor(MakePlaceholder(
        ffi::Array<ffi::Any>{ffi::Any(int64_t(16)), ffi::Any(int64_t(16))},
        ffi::String("float32"), ffi::String("tensor")));
    return op_tir_op(
        tir_func,
        ffi::String("tir_func"),
        ffi::Array<Expr>{A->expr},
        ffi::Array<ffi::Any>{ffi::Any(out_ph)});
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"A"}, {ffi::Any(MakeSpecTensor({16, 16}, "float32"))},
      /*named_params=*/{}, /*debug=*/false);

  // ---- Expected -----------------------------------------------------
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  {
    Var A("A", TSInfo({16, 16}, DataType::Float(32)));
    ffi::Map<ffi::String, ffi::Any> fn_attrs;
    fn_attrs.Set("num_input", ffi::Any(int64_t(1)));
    with(bb->function("test", {A}, fn_attrs), [&]() {
      with(bb->dataflow(), [&]() {
        Var lv = EmitCallTir(bb, tir_func, "tir_func", {A},
                             {TSInfo({16, 16}, DataType::Float(32))},
                             /*tir_vars=*/{}, "lv");
        Var gv = bb->EmitOutput(lv, "gv");
        bb->EmitFuncOutput(gv, {A});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestTensorIrOp
//
// Python original (debug=True, tir_vars=[offset]):
//   num_q_heads, num_kv_heads, head_dim = 8, 8, 16
//   fused_heads = 24   # 8 + 8*2
//   dtype = "float16"
//
//   @T.prim_func(private=True)
//   def fused_rope(var_qkv, var_q, var_k, var_v, offset: T.int64):
//       batch_size = T.int64()
//       seq_len    = T.int64()
//       qkv = T.match_buffer(var_qkv, (batch_size, seq_len, 24, 16), "float16")
//       q   = T.match_buffer(var_q,   (batch_size, seq_len,  8, 16), "float16")
//       k   = T.match_buffer(var_k,   (batch_size, seq_len,  8, 16), "float16")
//       v   = T.match_buffer(var_v,   (batch_size, seq_len,  8, 16), "float16")
//       T.evaluate(offset)
//
//   class Model(Module):
//       def test(self, qkv: Tensor, offset: tir.Var):
//           return op.tensor_ir_op(
//               fused_rope, "llama_fused_rope",
//               args=[qkv, offset],
//               out=[
//                   Tensor.placeholder((1,1,8,16),"float16"),
//                   Tensor.placeholder((1,1,8,16),"float16"),
//                   Tensor.placeholder((1,1,8,16),"float16"),
//               ])
//
// Expected:
//   @R.function
//   def test(qkv: R.Tensor((1,1,24,16),"float16"),
//            offset: R.Shape(["offset_1"]), _io: R.Object):
//       offset_1 = T.int64()
//       R.func_attr({"num_input": 3})
//       with R.dataflow():
//           lv1 = R.call_tir(cls.llama_fused_rope, (qkv,),
//                            out_sinfo=[R.Tensor((1,1,8,16),"float16") x3],
//                            tir_vars=R.shape([offset_1]))
//           lv0 = lv1[0]; lv1_ = lv1[1]; lv2 = lv1[2]
//           gv1 = (lv0, lv1_, lv2), (_io,)
//       return gv1
// ===========================================================================

// Build the fused_rope PrimFunc with symbolic batch_size, seq_len.
static tir::PrimFunc MakeFusedRopePrimFunc() {
  const int fused_heads = 24;  // 8 + 8*2
  const int num_q_heads = 8;
  const int num_kv_heads = 8;
  const int head_dim = 16;
  DataType dtype = DataType::Float(16);
  DataType idx = DataType::Int(64);

  // Symbolic shape vars
  tir::Var batch_size("batch_size", idx);
  tir::Var seq_len("seq_len", idx);

  // Handle params
  tir::Var h_qkv("var_qkv", DataType::Handle());
  tir::Var h_q  ("var_q",   DataType::Handle());
  tir::Var h_k  ("var_k",   DataType::Handle());
  tir::Var h_v  ("var_v",   DataType::Handle());
  tir::Var offset("offset", idx);

  auto I = [&](int64_t v) { return tir::IntImm(idx, v); };

  // Buffers
  tir::Buffer buf_qkv = tir::decl_buffer(
      {batch_size, seq_len, I(fused_heads), I(head_dim)}, dtype, "qkv");
  tir::Buffer buf_q = tir::decl_buffer(
      {batch_size, seq_len, I(num_q_heads), I(head_dim)}, dtype, "q");
  tir::Buffer buf_k = tir::decl_buffer(
      {batch_size, seq_len, I(num_kv_heads), I(head_dim)}, dtype, "k");
  tir::Buffer buf_v = tir::decl_buffer(
      {batch_size, seq_len, I(num_kv_heads), I(head_dim)}, dtype, "v");

  // Body: T.evaluate(offset)
  tir::Stmt body = tir::Evaluate(offset);

  ffi::Array<tir::Var> params{h_qkv, h_q, h_k, h_v, offset};
  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_qkv, buf_qkv);
  buf_map.Set(h_q,   buf_q);
  buf_map.Set(h_k,   buf_k);
  buf_map.Set(h_v,   buf_v);

  tir::PrimFunc func(params, body, tvm::Type(), buf_map);
  func = WithAttr(func, tvm::attr::kIsPrivateFunc, tvm::Bool(true));
  return func;
}

TEST(NNOps, TestTensorIrOp) {
  static const ffi::Function op_tir_op = NNOp("tensor_ir_op");

  tir::PrimFunc fused_rope = MakeFusedRopePrimFunc();
  DataType f16 = DataType::Float(16);

  // ---- Actual -------------------------------------------------------
  // The forward function receives qkv (NNTensor) and offset (tir::Var
  // wrapped as a relax ShapeExpr argument via spec.Int).
  auto forward = [&fused_rope, f16](ffi::Map<ffi::String, ffi::Any> named_args) -> ffi::Any {
    NNTensor qkv = named_args.at("qkv").operator NNTensor();
    // offset arrives as a relax Var with ShapeStructInfo (from spec.Int)
    Expr offset_expr = named_args.at("offset").operator Expr();

    // Three output placeholders: [1,1,8,16] f16
    auto make_ph = [&]() -> NNTensor {
      return NNTensor(MakePlaceholder(
          ffi::Array<ffi::Any>{
              ffi::Any(int64_t(1)), ffi::Any(int64_t(1)),
              ffi::Any(int64_t(8)), ffi::Any(int64_t(16))},
          ffi::String("float16"), ffi::String("tensor")));
    };
    NNTensor ph0 = make_ph(), ph1 = make_ph(), ph2 = make_ph();

    return op_tir_op(
        fused_rope,
        ffi::String("llama_fused_rope"),
        ffi::Array<Expr>{qkv->expr, offset_expr},
        ffi::Array<ffi::Any>{
            ffi::Any(ph0), ffi::Any(ph1), ffi::Any(ph2)});
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"qkv", "offset"},
      {ffi::Any(MakeSpecTensor({1, 1, 24, 16}, "float16")),
       ffi::Any(SpecInt())});

  // ---- Expected -----------------------------------------------------
  // The expected function signature:
  //   test(qkv: R.Tensor((1,1,24,16),f16),
  //        offset: R.Shape(["offset_1"]),
  //        _io: R.Object)
  // Inside the dataflow block:
  //   lv1 = call_tir(llama_fused_rope, (qkv,),
  //                  out_sinfo=[...x3], tir_vars=R.shape([offset_1]))
  //   lv0 = lv1[0]; lv1_ = lv1[1]; lv2 = lv1[2]
  //   gv1 = (lv0, lv1_, lv2), (_io,)
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    tir::Var offset_1("offset_1", DataType::Int(64));
    Var qkv("qkv", TSInfo({1, 1, 24, 16}, f16));
    // offset parameter: R.Shape(["offset_1"])
    Var offset_param("offset", ShapeStructInfo(ffi::Array<PrimExpr>{offset_1}));
    Var io("_io", ObjectStructInfo());

    ffi::Map<ffi::String, ffi::Any> fn_attrs;
    fn_attrs.Set("num_input", ffi::Any(int64_t(3)));

    with(bb->function("test", {qkv, offset_param, io}, fn_attrs), [&]() {
      with(bb->dataflow(), [&]() {
        TensorStructInfo out_sinfo = TSInfo({1, 1, 8, 16}, f16);
        Var lv1 = EmitCallTir(
            bb, fused_rope, "llama_fused_rope",
            /*args=*/{qkv},
            /*out_sinfo_list=*/{out_sinfo, out_sinfo, out_sinfo},
            /*tir_vars=*/{offset_1},
            "lv1");
        Var r0 = bb->Emit(TupleGetItem(lv1, 0), "llama_fused_rope_0");
        Var r1 = bb->Emit(TupleGetItem(lv1, 1), "llama_fused_rope_1");
        Var r2 = bb->Emit(TupleGetItem(lv1, 2), "llama_fused_rope_2");
        Var gv1 = EmitDebugOutput(bb, relax::Tuple({r0, r1, r2}), io);
        bb->EmitFuncOutput(gv1, {qkv, offset_param, io});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestTensorIrInplaceOp
//
// Python original (param_mode="packed", effect_mode="none", no _io):
//   hidden_size = 4096
//   dtype = "float16"
//
//   @T.prim_func
//   def inplace_take(var_weight, var_pos, var_embeddings, offset: T.int64):
//       T.func_attr({"tir.noalias": True})
//       vocab_size = T.int64()
//       weight     = T.match_buffer(var_weight, (vocab_size, 4096), "float16")
//       seq_len    = T.int64()
//       total_seq_len = T.int64()
//       pos        = T.match_buffer(var_pos, (seq_len,), "int32")
//       embeddings = T.match_buffer(var_embeddings, (total_seq_len, 4096), "float16")
//       for ax0, ax1 in T.grid(seq_len, 4096):
//           with T.sblock("T_take"):
//               v0, v1 = T.axis.remap("SS", [ax0, ax1])
//               embeddings[v0+offset, v1] = weight[pos[v0], v1]
//
//   class Model(Module):
//       def test(self, embedding_table, input_ids, embedding_dst, offset: int):
//           return op.tensor_ir_inplace_op(
//               inplace_take, "inplace_take",
//               args=[embedding_table, input_ids, embedding_dst, offset],
//               inplace_indices=[2],
//               out=Tensor.placeholder(embedding_dst.shape, embedding_dst.dtype))
//
//   spec: param_mode="packed", effect_mode="none"
//
// Expected (no _io, no _initialize_effect, packed_params tuple):
//   @R.function
//   def test(
//       embedding_table: R.Tensor(("vocab_size", 4096), "float16"),
//       input_ids:        R.Tensor(("seq_len",), "int32"),
//       embedding_dst:    R.Tensor(("total_seq_len", 4096), "float16"),
//       offset:           R.Shape(["offset_1"]),
//       packed_params:    R.Tuple,
//   ) -> R.Tensor(("total_seq_len", 4096), "float16"):
//       total_seq_len = T.int64()
//       offset_1      = T.int64()
//       R.func_attr({"num_input": 4})
//       with R.dataflow():
//           lv1 = R.call_tir_inplace(
//               cls.inplace_take,
//               (embedding_table, input_ids, embedding_dst),
//               out_sinfo=R.Tensor((total_seq_len, 4096), "float16"),
//               inplace_indices=[2],
//               tir_vars=R.shape([offset_1]))
//           gv1 = lv1
//       return gv1
// ===========================================================================

// Build the inplace_take PrimFunc with symbolic vocab_size, seq_len, total_seq_len.
static tir::PrimFunc MakeInplaceTakePrimFunc() {
  const int64_t hidden_size = 4096;
  DataType f16 = DataType::Float(16);
  DataType i32 = DataType::Int(32);
  DataType idx = DataType::Int(64);

  tir::Var vocab_size    ("vocab_size",     idx);
  tir::Var seq_len       ("seq_len",        idx);
  tir::Var total_seq_len ("total_seq_len",  idx);
  tir::Var offset        ("offset",         idx);

  tir::Var h_weight    ("var_weight",     DataType::Handle());
  tir::Var h_pos       ("var_pos",        DataType::Handle());
  tir::Var h_embeddings("var_embeddings", DataType::Handle());

  auto I = [&](int64_t v) { return tir::IntImm(idx, v); };

  tir::Buffer buf_weight = tir::decl_buffer(
      {vocab_size, I(hidden_size)}, f16, "weight");
  tir::Buffer buf_pos = tir::decl_buffer(
      {seq_len}, i32, "pos");
  tir::Buffer buf_emb = tir::decl_buffer(
      {total_seq_len, I(hidden_size)}, f16, "embeddings");

  // Loop vars
  tir::Var ax0("ax0", idx), ax1("ax1", idx);
  tir::Var v0("v0", idx),   v1("v1", idx);

  // embeddings[v0+offset, v1] = weight[pos[v0], v1]
  PrimExpr pos_v0 = tir::Cast(idx, buf_pos[{v0}]);
  tir::Stmt assign = tir::BufferStore(
      buf_emb, tir::BufferLoad(buf_weight, {pos_v0, v1}),
      {v0 + offset, v1});

  // T.sblock("T_take") with axis remapping
  tir::BlockRealize block_realize = tir::BlockRealize(
      /*iter_values=*/{ax0, ax1},
      /*predicate=*/tir::const_true(),
      tir::Block(
          /*iter_vars=*/{
              tir::IterVar(Range(I(0), seq_len),        v0, tir::kDataPar, ""),
              tir::IterVar(Range(I(0), I(hidden_size)), v1, tir::kDataPar, ""),
          },
          /*reads=*/ {},
          /*writes=*/{},
          /*name_hint=*/"T_take",
          /*body=*/assign));

  // for ax0 in range(seq_len): for ax1 in range(hidden_size):
  tir::Stmt inner_loop = tir::For(
      ax1, I(0), I(hidden_size),
      tir::ForKind::kSerial, block_realize);
  tir::Stmt outer_loop = tir::For(
      ax0, I(0), seq_len,
      tir::ForKind::kSerial, inner_loop);

  ffi::Array<tir::Var> params{h_weight, h_pos, h_embeddings, offset};
  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_weight,     buf_weight);
  buf_map.Set(h_pos,        buf_pos);
  buf_map.Set(h_embeddings, buf_emb);

  tir::PrimFunc func(params, outer_loop, tvm::Type(), buf_map);
  func = WithAttr(func, tir::attr::kNoAlias, tvm::Bool(true));
  return func;
}

TEST(NNOps, TestTensorIrInplaceOp) {
  static const ffi::Function op_tir_inplace = NNOp("tensor_ir_inplace_op");

  tir::PrimFunc inplace_take = MakeInplaceTakePrimFunc();
  DataType f16 = DataType::Float(16);
  DataType idx = DataType::Int(64);

  // ---- Actual -------------------------------------------------------
  auto forward = [&inplace_take, f16](
      ffi::Map<ffi::String, ffi::Any> named_args) -> ffi::Any {
    NNTensor emb_table  = named_args.at("embedding_table").operator NNTensor();
    NNTensor input_ids  = named_args.at("input_ids").operator NNTensor();
    NNTensor emb_dst    = named_args.at("embedding_dst").operator NNTensor();
    Expr     offset_expr = named_args.at("offset").operator Expr();

    // out placeholder: same shape/dtype as embedding_dst
    // embedding_dst has symbolic shape (total_seq_len, 4096)
    NNTensor out_ph = NNTensor(MakeTensorFromStructInfo(
        Downcast<TensorStructInfo>(GetStructInfo(emb_dst->expr)),
        ffi::String("tensor")));

    return op_tir_inplace(
        inplace_take,
        ffi::String("inplace_take"),
        ffi::Array<Expr>{emb_table->expr, input_ids->expr,
                         emb_dst->expr, offset_expr},
        ffi::Array<Integer>{Integer(2)},
        ffi::Array<ffi::Any>{ffi::Any(out_ph)});
  };

  IRModule actual = ExportSingle(
      "test", ffi::Function(forward),
      {"embedding_table", "input_ids", "embedding_dst", "offset"},
      {ffi::Any(MakeSpecTensorMixed({{"vocab_size", -1}, {"", 4096}}, "float16")),
       ffi::Any(MakeSpecTensorMixed({{"seq_len", -1}},                "int32")),
       ffi::Any(MakeSpecTensorMixed({{"total_seq_len", -1}, {"", 4096}}, "float16")),
       ffi::Any(SpecInt())},
      /*named_params=*/{}, /*debug=*/false,
      /*param_mode=*/"packed", /*effect_mode=*/"none");

  // ---- Expected -----------------------------------------------------
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  {
    tir::Var vocab_size   ("vocab_size",    idx);
    tir::Var seq_len      ("seq_len",       idx);
    tir::Var total_seq_len("total_seq_len", idx);
    tir::Var offset_1     ("offset_1",      idx);

    auto I = [&](int64_t v) -> PrimExpr { return tir::IntImm(idx, v); };

    Var emb_table("embedding_table",
        TensorStructInfo(ShapeExpr({vocab_size, I(4096)}), f16));
    Var input_ids("input_ids",
        TensorStructInfo(ShapeExpr({seq_len}), DataType::Int(32)));
    Var emb_dst("embedding_dst",
        TensorStructInfo(ShapeExpr({total_seq_len, I(4096)}), f16));
    Var offset_param("offset",
        ShapeStructInfo(ffi::Array<PrimExpr>{offset_1}));
    // packed_params: R.Tuple (empty tuple struct info)
    Var packed_params("packed_params",
        TupleStructInfo(ffi::Array<StructInfo>{}));

    ffi::Map<ffi::String, ffi::Any> fn_attrs;
    fn_attrs.Set("num_input", ffi::Any(int64_t(4)));

    with(bb->function("test",
                      {emb_table, input_ids, emb_dst, offset_param, packed_params},
                      fn_attrs), [&]() {
      with(bb->dataflow(), [&]() {
        TensorStructInfo out_sinfo(
            ShapeExpr({total_seq_len, I(4096)}), f16);
        Var lv1 = EmitCallTirInplace(
            bb, inplace_take, "inplace_take",
            /*args=*/{emb_table, input_ids, emb_dst},
            /*inplace_indices=*/{Integer(2)},
            /*out_sinfo_list=*/{out_sinfo},
            /*tir_vars=*/{offset_1},
            "lv1");
        Var gv1 = bb->EmitOutput(lv1, "gv1");
        bb->EmitFuncOutput(
            gv1, {emb_table, input_ids, emb_dst, offset_param, packed_params});
      });
    });
  }
  AssertStructEqual(actual, bb->Finalize());
}
