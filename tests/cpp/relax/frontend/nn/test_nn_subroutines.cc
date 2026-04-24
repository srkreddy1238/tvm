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
 * \file tests/cpp/relax/frontend/nn/test_nn_subroutines.cc
 * \brief C++ port of tests/python/relax/test_frontend_nn_subroutines.py
 *
 * Ported test:
 *   - TestLinear  (test_linear)
 *
 * The Python test uses `define_subroutine = True` on nn.Module subclasses to
 * emit private Relax functions (subroutines) for each sub-module.  This
 * feature is Python-only (it relies on Python class attributes and the
 * SubroutineMixin metaclass).
 *
 * In C++ the equivalent is to build the expected IRModule directly with
 * private functions and verify that the C++ exporter produces the same IR
 * when the forward lambda manually calls BlockBuilder::AddFunction to emit
 * the subroutine.
 *
 * Expected IR (debug=True):
 *
 *   @R.function
 *   def forward(
 *       state: R.Tensor(("batch_size", 64), "float32"),
 *       _io: R.Object,
 *       weights: R.Tensor((64, 32), "float32"),
 *   ) -> R.Tuple(R.Tensor(("batch_size", 32), "float32"), R.Tuple(R.Object)):
 *     R.func_attr({"num_input": 2})
 *     with R.dataflow():
 *       state = Expected.layer(state, weights)
 *       dataflow_output = (state, (_io,))
 *       R.output(dataflow_output)
 *     return dataflow_output
 *
 *   @R.function(private=True)
 *   def layer(
 *       state: R.Tensor(("batch_size", 64), "float32"),
 *       weights: R.Tensor((64, 32), "float32"),
 *   ) -> R.Tensor(("batch_size", 32), "float32"):
 *     with R.dataflow():
 *       state = R.matmul(state, weights)
 *       state = Expected.activation(state)
 *       dataflow_output = state
 *       R.output(dataflow_output)
 *     return dataflow_output
 *
 *   @R.function(private=True)
 *   def activation(
 *       state: R.Tensor(("batch_size", 32), "float32"),
 *   ) -> R.Tensor(("batch_size", 32), "float32"):
 *     with R.dataflow():
 *       state = R.nn.silu(state)
 *       dataflow_output = state
 *       R.output(dataflow_output)
 *     return dataflow_output
 *
 *   @R.function
 *   def _initialize_effect() -> R.Tuple(R.Object): ...
 *
 * Design notes:
 *   - The C++ exporter does not have a `define_subroutine` flag.  Instead,
 *     the forward lambda emits the subroutine functions directly via
 *     BlockBuilder::AddFunction (no global_symbol attr → private) before
 *     emitting the call to them.  The returned GlobalVar is used as the callee.
 *   - The test verifies structural equality of the full IRModule.
 */

#include <gtest/gtest.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/op.h>

#include "../../../../../src/relax/frontend/nn/core.h"
#include "../../../../../src/relax/frontend/nn/exporter.h"
#include "../../../../../src/relax/frontend/nn/spec.h"
#include "../../../../../src/relax/op/nn/nn.h"
#include "../../../../../src/relax/op/tensor/linear_algebra.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace testing {

// ===========================================================================
// Helpers
// ===========================================================================

static TensorStructInfo TSInfo(std::initializer_list<int64_t> dims, DataType dtype) {
  ffi::Array<PrimExpr> shape_dims;
  for (int64_t d : dims) shape_dims.push_back(IntImm(DataType::Int(64), d));
  return TensorStructInfo(ShapeExpr(shape_dims), dtype);
}

static void AssertStructEqual(const IRModule& actual, const IRModule& expected) {
  EXPECT_TRUE(ffi::StructuralEqual()(actual, expected))
      << "\n=== Actual ===\n" << actual << "\n=== Expected ===\n" << expected;
}

// ===========================================================================
// TestLinear
//
// Python equivalent:
//   class Activation(nn.Module):
//       define_subroutine = True
//       def forward(self, state): return nn.op.silu(state)
//
//   class Layer(nn.Module):
//       define_subroutine = True
//       def __init__(self, in_features, out_features):
//           self.weights = nn.Parameter((in_features, out_features), "float32")
//           self.activation = Activation()
//       def forward(self, input):
//           state = nn.op.matmul(input, self.weights)
//           return self.activation(state)
//
//   mod = Layer(64, 32)
//   batch_size = tvm.tir.Var("batch_size", "int64")
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"input": nn.spec.Tensor((batch_size, 64), "float32")}},
//       debug=True)
//
// In C++ we build the subroutine functions manually inside the forward lambda
// using BlockBuilder::AddFunction with is_public=false, then emit a call to
// the private GlobalVar.
// ===========================================================================
TEST(NNSubroutines, TestLinear) {
  DataType f32 = DataType::Float(32);
  tir::Var batch_size("batch_size", DataType::Int(64));

  // Named parameter: weights (64, 32)
  NNParameter weights_param(
      Var("weights",
          TensorStructInfo(
              ShapeExpr(ffi::Array<PrimExpr>{IntImm(DataType::Int(64), 64),
                                             IntImm(DataType::Int(64), 32)}),
              f32)));
  ffi::Map<ffi::String, NNParameter> named_params;
  named_params.Set("weights", weights_param);

  // The forward lambda emits two private subroutine functions and then calls
  // them from the main dataflow block.
  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward_fn =
      [batch_size, f32, weights_param](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor input   = args.at("input").cast<NNTensor>();
    // Parameters are accessed via their NNParameter::expr, which the exporter
    // sets to the emitted Var before calling the forward lambda.
    Var weights_var  = weights_param->expr;

    BlockBuilder bb = BlockBuilder_Current();
    TVM_FFI_ICHECK(bb.defined()) << "forward called outside BlockBuilder scope";

    // --- Emit private function: activation(state) -> silu(state) ---
    // AddFunction returns the GlobalVar; no global_symbol attr → private.
    GlobalVar gv_activation;
    {
      tir::Var bs("batch_size", DataType::Int(64));
      TensorStructInfo state_sinfo(
          ShapeExpr(ffi::Array<PrimExpr>{bs, IntImm(DataType::Int(64), 32)}), f32);
      Var state_param("state", state_sinfo);
      ffi::Array<Var> act_params{state_param};
      bb->BeginScope(act_params);
      bb->BeginDataflowBlock();
      Var silu_out = bb->Emit(relax::silu(state_param), "state");
      Var df_out   = bb->EmitOutput(silu_out, "dataflow_output");
      BindingBlock df = bb->EndBlock();
      Expr act_body = bb->Normalize(SeqExpr({df}, df_out));
      bb->EndScope();
      // No global_symbol → private function.
      gv_activation = bb->AddFunction(
          Function(act_params, act_body, state_sinfo, /*is_pure=*/true, DictAttrs()),
          "activation");
    }

    // --- Emit private function: layer(state, weights) ---
    GlobalVar gv_layer;
    {
      tir::Var bs("batch_size", DataType::Int(64));
      TensorStructInfo in_sinfo(
          ShapeExpr(ffi::Array<PrimExpr>{bs, IntImm(DataType::Int(64), 64)}), f32);
      TensorStructInfo out_sinfo(
          ShapeExpr(ffi::Array<PrimExpr>{bs, IntImm(DataType::Int(64), 32)}), f32);
      TensorStructInfo w_sinfo(
          ShapeExpr(ffi::Array<PrimExpr>{IntImm(DataType::Int(64), 64),
                                         IntImm(DataType::Int(64), 32)}), f32);
      Var state_param("state", in_sinfo);
      Var w_param("weights", w_sinfo);
      ffi::Array<Var> layer_params{state_param, w_param};
      bb->BeginScope(layer_params);
      bb->BeginDataflowBlock();
      Var mm      = bb->Emit(relax::matmul(state_param, w_param, std::nullopt), "state");
      Var act_out = bb->Emit(Call(gv_activation, {mm}), "state");
      Var df_out  = bb->EmitOutput(act_out, "dataflow_output");
      BindingBlock df = bb->EndBlock();
      Expr layer_body = bb->Normalize(SeqExpr({df}, df_out));
      bb->EndScope();
      // No global_symbol → private function.
      gv_layer = bb->AddFunction(
          Function(layer_params, layer_body, out_sinfo, /*is_pure=*/true, DictAttrs()),
          "layer");
    }

    // --- Call layer from the main dataflow block ---
    Var result = bb->Emit(Call(gv_layer, {input->expr, weights_var}), "state");
    return ffi::Any(NNTensor(result));
  };

  ffi::Array<ffi::Any> spec_shape;
  spec_shape.push_back(ffi::Any(PrimExpr(batch_size)));
  spec_shape.push_back(ffi::Any(int64_t(64)));
  SpecTensor input_spec(spec_shape, "float32");

  MethodSpec ms(forward_fn, {"input"}, {ffi::Any(input_spec)}, "plain", "plain");
  ModuleSpec mod_spec({"forward"}, {ffi::Any(ms)}, named_params, {});
  
  // Create a minimal NNModule wrapper and use ExportTVM
  NNModule mod;
  ffi::Array<ffi::Any> result = mod->ExportTVM(mod_spec, /*debug=*/true, /*allow_extern=*/false);
  IRModule actual = result[0].cast<IRModule>();

  // ===========================================================================
  // Build expected IR
  // ===========================================================================
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);

  // _initialize_effect
  {
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

  // Private: activation(state: (batch_size, 32)) -> (batch_size, 32)
  // No global_symbol attr → private; AddFunction returns the GlobalVar.
  GlobalVar gv_activation;
  {
    tir::Var bs("batch_size", DataType::Int(64));
    TensorStructInfo sinfo(
        ShapeExpr(ffi::Array<PrimExpr>{bs, IntImm(DataType::Int(64), 32)}), f32);
    Var state("state", sinfo);
    ffi::Array<Var> params{state};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var silu_out = bb->Emit(relax::silu(state), "state");
    Var df_out   = bb->EmitOutput(silu_out, "dataflow_output");
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, df_out));
    bb->EndScope();
    gv_activation = bb->AddFunction(
        Function(params, body, sinfo, true, DictAttrs()), "activation");
  }

  // Private: layer(state: (batch_size, 64), weights: (64, 32)) -> (batch_size, 32)
  GlobalVar gv_layer;
  {
    tir::Var bs("batch_size", DataType::Int(64));
    TensorStructInfo in_sinfo(
        ShapeExpr(ffi::Array<PrimExpr>{bs, IntImm(DataType::Int(64), 64)}), f32);
    TensorStructInfo out_sinfo(
        ShapeExpr(ffi::Array<PrimExpr>{bs, IntImm(DataType::Int(64), 32)}), f32);
    TensorStructInfo w_sinfo(TSInfo({64, 32}, f32));
    Var state("state", in_sinfo);
    Var weights("weights", w_sinfo);
    ffi::Array<Var> params{state, weights};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var mm      = bb->Emit(relax::matmul(state, weights, std::nullopt), "state");
    Var act_out = bb->Emit(Call(gv_activation, {mm}), "state");
    Var df_out  = bb->EmitOutput(act_out, "dataflow_output");
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, df_out));
    bb->EndScope();
    gv_layer = bb->AddFunction(
        Function(params, body, out_sinfo, true, DictAttrs()), "layer");
  }

  // Public: forward(state, _io, weights)
  {
    TensorStructInfo state_in_sinfo(
        ShapeExpr(ffi::Array<PrimExpr>{batch_size, IntImm(DataType::Int(64), 64)}), f32);
    TensorStructInfo w_sinfo(TSInfo({64, 32}, f32));
    Var state("state", state_in_sinfo);
    Var io("_io", ObjectStructInfo());
    Var weights("weights", w_sinfo);
    ffi::Array<Var> params{state, io, weights};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var layer_out = bb->Emit(Call(gv_layer, {state, weights}), "state");
    Var df_out    = bb->EmitOutput(relax::Tuple({layer_out, relax::Tuple({io})}),
                                   "dataflow_output");
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, df_out));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("forward")));
    bb->AddFunction(Function(params, body, std::nullopt, true, DictAttrs(attrs)), "forward");
  }

  IRModule expected = bb->Finalize();
  AssertStructEqual(actual, expected);
}

}  // namespace testing
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
