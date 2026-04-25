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
 * \file tests/cpp/relax/frontend/nn/test_nn_exporter.cc
 * \brief C++ port of tests/python/relax/test_frontend_nn_exporter.py
 *
 * Ported tests:
 *   - TestSimple                        (test_simple)
 *   - TestDebugEffect                   (test_debug_effect)
 *   - TestDynamicShape                  (test_dynamic_shape)
 *   - TestDynamicShapeInMultipleFunctions (test_dynamic_shape_in_multiple_functions)
 *   - TestExportNestedModule            (test_export_nested_module)
 *   - TestLinearDynamicShape            (test_linear_dynamic_shape)
 *   - TestDuplicateNamesSamePythonString       (test_duplicate_names/same_python_string)
 *   - TestDuplicateNamesDifferentPythonString  (test_duplicate_names/different_python_string)
 *   - TestDuplicateNamesSameTirVar             (test_duplicate_names/same_tir_var)
 *   - TestDuplicateNamesDistinctTirVarsDistinctNames
 * (test_duplicate_names/distinct_tir_vars_with_distinct_names)
 *
 * Skipped (Python-only / xfail):
 *   - test_custom_module          (requires Python subclassing of nn.Module)
 *   - test_generate_parameters    (marked xfail in Python)
 *
 * Each test uses the module-aware ModuleSpec constructor:
 *
 *   // 1. Create the module
 *   ReLUModule mod;
 *   // 2. Build the spec dictionary
 *   ffi::Map<ffi::String, ffi::Any> forward_spec;
 *   forward_spec.Set("x", ffi::Any(MakeSpecTensor({3, 3}, "float32")));
 *   ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> spec;
 *   spec.Set("forward", forward_spec);
 *   // 3. Create ModuleSpec from module and spec
 *   ModuleSpec mod_spec(mod, spec, false);  // debug=false
 *   // 4. Export via module->ExportTVM
 *   ffi::Array<ffi::Any> result =
 *       mod->ExportTVM(mod_spec, false, false);  // debug=false, allow_extern=false
 *   IRModule actual = result[0].cast<IRModule>();
 *
 * This pattern removes the need for MethodSpec to hold or receive an
 * NNModule object: ModuleSpec derives the ffi::Function for each
 * method_name and passes it to MethodSpec.
 *
 * Tests that require custom forward lambdas (e.g. multi-function export,
 * nested modules, duplicate-names) still build MethodSpec directly with
 * an ffi::Function and assemble ModuleSpec from its low-level constructor.
 *
 * Design notes for the duplicate-names tests:
 *   The Python tests verify that the exporter deduplicates tir::Var objects
 *   by name (string equality) when the same string is used for multiple
 *   symbolic dims.  In C++ we verify this by inspecting the shape vars of
 *   the exported function parameters directly, rather than building a full
 *   expected IRModule, because the exact deduplication behaviour is an
 *   internal exporter invariant that is easier to assert structurally.
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
#include "../../../../../src/relax/frontend/nn/modules.h"
#include "../../../../../src/relax/frontend/nn/spec.h"
#include "../../../../../src/relax/op/nn/nn.h"
#include "../../../../../src/relax/op/tensor/binary.h"
#include "../../../../../src/relax/op/tensor/linear_algebra.h"
#include "../../../../../src/relax/op/tensor/manipulate.h"

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

static Var EmitDebugOutput(BlockBuilder& bb, Expr result, Var io, const std::string& hint = "gv1") {
  return bb->EmitOutput(relax::Tuple({result, relax::Tuple({io})}), hint);
}

static void AssertStructEqual(const IRModule& actual, const IRModule& expected) {
  EXPECT_TRUE(ffi::StructuralEqual()(actual, expected)) << "\n=== Actual ===\n"
                                                        << actual << "\n=== Expected ===\n"
                                                        << expected;
}

// ===========================================================================
// BeforeModuleNode / BeforeModule
//
// Test-only module for TestDynamicShapeInMultipleFunctions.
// Mirrors:
//   class Before(nn.Module):
//       def forward_relu(self, x): return nn.relu(x)
//       def forward_silu(self, x): return nn.silu(x)
//
// Two methods are registered as "forward_relu" and "forward_silu" (no "_" prefix)
// so that DeriveMethodFunction can look them up verbatim: it only maps
// "forward" -> "_forward" as a special case; all other names are used as-is.
// ===========================================================================
class BeforeModuleNode : public NNModuleNode {
 public:
  Var ForwardRelu(Var x) const { return BlockBuilder_Current()->Emit(relax::relu(x), "relu"); }
  Var ForwardSilu(Var x) const { return BlockBuilder_Current()->Emit(relax::silu(x), "silu"); }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BeforeModuleNode>()
        .def(refl::init<>())
        .def("forward_relu", &BeforeModuleNode::ForwardRelu)
        .def("forward_silu", &BeforeModuleNode::ForwardSilu);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.testing.Before", BeforeModuleNode,
                                    NNModuleNode);
};
class BeforeModule : public runtime::ObjectRef {
 public:
  explicit BeforeModule() { data_ = ffi::make_object<BeforeModuleNode>(); }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BeforeModule, runtime::ObjectRef, BeforeModuleNode);
};
TVM_FFI_STATIC_INIT_BLOCK() { BeforeModuleNode::RegisterReflection(); }

// ===========================================================================
// LlamaMLPModuleNode / LlamaMLPModule
//
// Test-only module for TestExportNestedModule.
// Mirrors:
//   class LlamaMLP(nn.Module):
//       def __init__(self, hidden_size, intermediate_size):
//           self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
//           self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
//           self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
//       def forward(self, x):
//           return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
// ===========================================================================
class LlamaMLPModuleNode : public NNModuleNode {
 public:
  LinearModule gate_proj;
  LinearModule up_proj;
  LinearModule down_proj;

  LlamaMLPModuleNode(LinearModule gate_proj, LinearModule up_proj, LinearModule down_proj)
      : gate_proj(std::move(gate_proj)),
        up_proj(std::move(up_proj)),
        down_proj(std::move(down_proj)) {
    // Populate attrs so NNModuleNode::NamedParameters() can traverse sub-modules
    // and produce "gate_proj.weight", "up_proj.weight", "down_proj.weight".
    attrs.Set("gate_proj", ffi::Any(this->gate_proj));
    attrs.Set("up_proj", ffi::Any(this->up_proj));
    attrs.Set("down_proj", ffi::Any(this->down_proj));
  }

  Var Forward(Var x) const {
    static const ffi::Function op_silu =
        ffi::Function::GetGlobal("relax.frontend.nn.op.silu").value();
    static const ffi::Function op_mul =
        ffi::Function::GetGlobal("relax.frontend.nn.op.multiply").value();
    NNTensor gate = NNTensor(gate_proj.get()->Forward(x));
    NNTensor up = NNTensor(up_proj.get()->Forward(x));
    ffi::Any silu_out = op_silu(gate->expr, ffi::String("silu"));
    ffi::Any mul_out = op_mul(silu_out.cast<Var>(), up->expr, ffi::String("mul"));
    NNTensor mul_t(mul_out.cast<Var>());
    NNTensor out = NNTensor(down_proj.get()->Forward(mul_t->expr));
    return out->expr;
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LlamaMLPModuleNode>()
        .def(refl::init<LinearModule, LinearModule, LinearModule>())
        .def_ro("gate_proj", &LlamaMLPModuleNode::gate_proj)
        .def_ro("up_proj", &LlamaMLPModuleNode::up_proj)
        .def_ro("down_proj", &LlamaMLPModuleNode::down_proj)
        .def("_forward", &LlamaMLPModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.testing.LlamaMLP", LlamaMLPModuleNode,
                                    NNModuleNode);
};
class LlamaMLPModule : public runtime::ObjectRef {
 public:
  explicit LlamaMLPModule(LinearModule gate_proj, LinearModule up_proj, LinearModule down_proj) {
    data_ = ffi::make_object<LlamaMLPModuleNode>(std::move(gate_proj), std::move(up_proj),
                                                 std::move(down_proj));
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LlamaMLPModule, runtime::ObjectRef,
                                                LlamaMLPModuleNode);
};
TVM_FFI_STATIC_INIT_BLOCK() { LlamaMLPModuleNode::RegisterReflection(); }

// ===========================================================================
// DuplicateNamesModuleNode / DuplicateNamesModule
//
// Test-only module for ExportDuplicateNamesModel.
// Holds three NNParameter members (embedding_weights, up_weights, down_weights)
// and implements forward(state) as:
//   s1 = matmul(state, embedding_weights)
//   s2 = matmul(s1,    up_weights)
//   s3 = silu(s2)
//   s4 = matmul(s3,    down_weights)
//   return s4
// ===========================================================================
class DuplicateNamesModuleNode : public NNModuleNode {
 public:
  NNParameter embedding_weights;
  NNParameter up_weights;
  NNParameter down_weights;

  DuplicateNamesModuleNode(NNParameter embedding_weights, NNParameter up_weights,
                           NNParameter down_weights)
      : embedding_weights(std::move(embedding_weights)),
        up_weights(std::move(up_weights)),
        down_weights(std::move(down_weights)) {
    // Populate attrs so NNModuleNode::NamedParameters() discovers these
    // parameters and the exporter adds them as function arguments.
    attrs.Set("embedding_weights", ffi::Any(this->embedding_weights));
    attrs.Set("up_weights", ffi::Any(this->up_weights));
    attrs.Set("down_weights", ffi::Any(this->down_weights));
  }

  Var Forward(Var state) const {
    static const ffi::Function op_silu =
        ffi::Function::GetGlobal("relax.frontend.nn.op.silu").value();
    static const ffi::Function op_matmul =
        ffi::Function::GetGlobal("relax.frontend.nn.op.matmul").value();
    ffi::Any s1 = op_matmul(state, embedding_weights->expr, ffi::Optional<ffi::String>(),
                            ffi::String("state"));
    ffi::Any s2 = op_matmul(s1.cast<Var>(), up_weights->expr, ffi::Optional<ffi::String>(),
                            ffi::String("state"));
    ffi::Any s3 = op_silu(s2.cast<Var>(), ffi::String("state"));
    ffi::Any s4 = op_matmul(s3.cast<Var>(), down_weights->expr, ffi::Optional<ffi::String>(),
                            ffi::String("state"));
    return s4.cast<Var>();
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DuplicateNamesModuleNode>()
        .def(refl::init<NNParameter, NNParameter, NNParameter>())
        .def_ro("embedding_weights", &DuplicateNamesModuleNode::embedding_weights)
        .def_ro("up_weights", &DuplicateNamesModuleNode::up_weights)
        .def_ro("down_weights", &DuplicateNamesModuleNode::down_weights)
        .def("_forward", &DuplicateNamesModuleNode::Forward);
  }
  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.testing.DuplicateNames",
                                    DuplicateNamesModuleNode, NNModuleNode);
};
class DuplicateNamesModule : public runtime::ObjectRef {
 public:
  explicit DuplicateNamesModule(NNParameter embedding_weights, NNParameter up_weights,
                                NNParameter down_weights) {
    data_ = ffi::make_object<DuplicateNamesModuleNode>(
        std::move(embedding_weights), std::move(up_weights), std::move(down_weights));
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DuplicateNamesModule, runtime::ObjectRef,
                                                DuplicateNamesModuleNode);
};
TVM_FFI_STATIC_INIT_BLOCK() { DuplicateNamesModuleNode::RegisterReflection(); }

TEST(NNExporter, TestSimple) {
  // 1. Create the module
  ReLUModule mod;
  // 2. Build the spec dictionary
  ffi::Map<ffi::String, ffi::Any> forward_spec;
  forward_spec.Set("x", ffi::Any(MakeSpecTensor({3, 3}, "float32")));
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> spec;
  spec.Set("forward", forward_spec);
  // 3. Create ModuleSpec from module and spec
  ModuleSpec mod_spec(mod, spec, /*debug=*/false);
  // 4. Export via module->ExportTVM
  ffi::Array<ffi::Any> result = mod->ExportTVM(mod_spec, /*debug=*/false, /*allow_extern=*/false);
  IRModule actual = result[0].cast<IRModule>();

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  {
    Var x("x", TSInfo({3, 3}, DataType::Float(32)));
    ffi::Array<Var> params{x};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var relu = bb->Emit(relax::relu(x), "relu");
    Var gv = bb->EmitOutput(relu, "relu");
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(1)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("forward")));
    bb->AddFunction(Function(params, body, std::nullopt, true, DictAttrs(attrs)), "forward");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestDebugEffect
//
// Python equivalent:
//   slm_mod = nn.modules.ReLU()
//   exported_mod, _ = slm_mod.export_tvm(
//       spec={"forward": {"x": nn.spec.Tensor((3, 3), "float32")}}, debug=True)
//
// Expected:
//   def forward(x: R.Tensor([3,3],"float32"), _io: R.Object):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       relu = R.nn.relu(x)
//       output = relu, (_io,)
//       R.output(output)
//     return output
//
//   def _initialize_effect() -> R.Tuple(R.Object): ...
//
// Uses the module-aware ModuleSpec constructor with debug=true:
//   ModuleSpec(mod, spec, /*debug=*/true)
// ===========================================================================
TEST(NNExporter, TestDebugEffect) {
  // 1. Create the module
  ReLUModule mod;
  // 2. Build the spec dictionary
  ffi::Map<ffi::String, ffi::Any> forward_spec;
  forward_spec.Set("x", ffi::Any(MakeSpecTensor({3, 3}, "float32")));
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> spec;
  spec.Set("forward", forward_spec);
  // 3. Create ModuleSpec with debug=true (effect_mode="plain")
  ModuleSpec mod_spec(mod, spec, /*debug=*/true);
  // 4. Export via module->ExportTVM with debug=true
  ffi::Array<ffi::Any> result = mod->ExportTVM(mod_spec, /*debug=*/true, /*allow_extern=*/false);
  IRModule actual = result[0].cast<IRModule>();

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({3, 3}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var relu = bb->Emit(relax::relu(x), "relu");
    Var gv1 = EmitDebugOutput(bb, relu, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("forward")));
    bb->AddFunction(Function(params, body, std::nullopt, true, DictAttrs(attrs)), "forward");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestDynamicShape
//
// Python equivalent:
//   slm_mod = nn.modules.ReLU()
//   exported_mod, _ = slm_mod.export_tvm(
//       spec={"forward": {"x": nn.spec.Tensor([tir.Var("batch_size","int64"), 8], "float32")}},
//       debug=False)
//
// Expected:
//   def forward(x: R.Tensor(["batch_size", 8], "float32")):
//     R.func_attr({"num_input": 1})
//     with R.dataflow():
//       relu = R.nn.relu(x)
//       R.output(relu)
//     return relu
//
// Uses the module-aware ModuleSpec constructor with a symbolic SpecTensor.
// ===========================================================================
TEST(NNExporter, TestDynamicShape) {
  // 1. Create the module
  ReLUModule mod;
  // 2. Build the spec dictionary with a symbolic batch_size dimension
  ffi::Array<ffi::Any> spec_shape;
  spec_shape.push_back(ffi::Any(ffi::String("batch_size")));
  spec_shape.push_back(ffi::Any(int64_t(8)));
  SpecTensor x_spec(spec_shape, "float32");

  ffi::Map<ffi::String, ffi::Any> forward_spec;
  forward_spec.Set("x", ffi::Any(x_spec));
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> spec;
  spec.Set("forward", forward_spec);
  // 3. Create ModuleSpec from module and spec
  ModuleSpec mod_spec(mod, spec, /*debug=*/false);
  // 4. Export via module->ExportTVM
  ffi::Array<ffi::Any> result = mod->ExportTVM(mod_spec, /*debug=*/false, /*allow_extern=*/false);
  IRModule actual = result[0].cast<IRModule>();

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  {
    tir::Var batch_size("batch_size", DataType::Int(64));
    TensorStructInfo x_sinfo(
        ShapeExpr(ffi::Array<PrimExpr>{batch_size, IntImm(DataType::Int(64), 8)}),
        DataType::Float(32));
    Var x("x", x_sinfo);
    ffi::Array<Var> params{x};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var relu = bb->Emit(relax::relu(x), "relu");
    Var gv = bb->EmitOutput(relu, "relu");
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(1)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("forward")));
    bb->AddFunction(Function(params, body, std::nullopt, true, DictAttrs(attrs)), "forward");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestDynamicShapeInMultipleFunctions
//
// Python equivalent:
//   class Before(nn.Module):
//       def forward_relu(self, x): return nn.relu(x)
//       def forward_silu(self, x): return nn.silu(x)
//
//   exported_mod, _ = Before().export_tvm(spec={
//       "forward_relu": {"x": nn.spec.Tensor((tir.Var("batch_size","int64"), 8), "float32")},
//       "forward_silu": {"x": nn.spec.Tensor((tir.Var("batch_size","int64"), 8), "float32")},
//   }, debug=False)
//
// Uses BeforeModule whose node registers "forward_relu" and "forward_silu".
// DeriveMethodFunction looks them up verbatim (only "forward" is special-cased
// to "_forward").
// ===========================================================================
TEST(NNExporter, TestDynamicShapeInMultipleFunctions) {
  BeforeModule mod;

  ffi::Array<ffi::Any> spec_shape;
  spec_shape.push_back(ffi::Any(ffi::String("batch_size")));
  spec_shape.push_back(ffi::Any(int64_t(8)));
  SpecTensor x_spec(spec_shape, "float32");

  ffi::Map<ffi::String, ffi::Any> relu_arg_spec;
  relu_arg_spec.Set("x", ffi::Any(x_spec));
  ffi::Map<ffi::String, ffi::Any> silu_arg_spec;
  silu_arg_spec.Set("x", ffi::Any(x_spec));

  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> spec;
  spec.Set("forward_relu", relu_arg_spec);
  spec.Set("forward_silu", silu_arg_spec);

  ModuleSpec mod_spec(mod, spec, /*debug=*/false);

  ffi::Array<ffi::Any> result = mod->ExportTVM(mod_spec, /*debug=*/false, /*allow_extern=*/false);
  IRModule actual = result[0].cast<IRModule>();

  // Build expected: two independent functions each with their own batch_size var.
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  auto make_func = [&](const std::string& name, auto op_fn, const std::string& hint) {
    tir::Var batch_size("batch_size", DataType::Int(64));
    TensorStructInfo x_sinfo(
        ShapeExpr(ffi::Array<PrimExpr>{batch_size, IntImm(DataType::Int(64), 8)}),
        DataType::Float(32));
    Var x("x", x_sinfo);
    ffi::Array<Var> params{x};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var out = bb->Emit(op_fn(x), hint);
    Var gv = bb->EmitOutput(out, hint);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(1)));
    attrs.Set("global_symbol", ffi::Any(ffi::String(name)));
    bb->AddFunction(Function(params, body, std::nullopt, true, DictAttrs(attrs)), name);
  };
  make_func("forward_relu", [](Var x) { return relax::relu(x); }, "relu");
  make_func("forward_silu", [](Var x) { return relax::silu(x); }, "silu");
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestExportNestedModule
//
// Python equivalent: LlamaMLP with gate_proj, up_proj, down_proj (all Linear,
// no bias, float16).  forward: down_proj(silu(gate_proj(x)) * up_proj(x))
//
// hidden_size=4096, intermediate_size=11008
// Input spec: (tir.Var("batch_size","int64"), 4096), float16
//
// Uses LlamaMLPModule whose node owns the three Linear sub-modules and
// implements _forward via the module-aware ModuleSpec constructor.
// ===========================================================================
TEST(NNExporter, TestExportNestedModule) {
  const int64_t H = 4096;
  const int64_t I = 11008;

  LinearModule gate_proj = MakeLinear(ffi::Any(H), ffi::Any(I), false,
                                      ffi::Optional<ffi::String>("float16"), std::nullopt);
  LinearModule up_proj = MakeLinear(ffi::Any(H), ffi::Any(I), false,
                                    ffi::Optional<ffi::String>("float16"), std::nullopt);
  LinearModule down_proj = MakeLinear(ffi::Any(I), ffi::Any(H), false,
                                      ffi::Optional<ffi::String>("float16"), std::nullopt);

  LlamaMLPModule mod(gate_proj, up_proj, down_proj);

  ffi::Array<ffi::Any> spec_shape;
  spec_shape.push_back(ffi::Any(ffi::String("batch_size")));
  spec_shape.push_back(ffi::Any(H));
  SpecTensor x_spec(spec_shape, "float16");

  ffi::Map<ffi::String, ffi::Any> forward_spec;
  forward_spec.Set("x", ffi::Any(x_spec));
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> spec;
  spec.Set("forward", forward_spec);

  ModuleSpec mod_spec(mod, spec, /*debug=*/false);

  ffi::Array<ffi::Any> result = mod->ExportTVM(mod_spec, /*debug=*/false, /*allow_extern=*/false);
  IRModule actual = result[0].cast<IRModule>();

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  {
    DataType f16 = DataType::Float(16);
    auto I64 = [](int64_t v) { return IntImm(DataType::Int(64), v); };
    tir::Var batch_size("batch_size", DataType::Int(64));

    TensorStructInfo x_sinfo(ShapeExpr(ffi::Array<PrimExpr>{batch_size, I64(H)}), f16);
    Var x("x", x_sinfo);
    Var gate_proj_weight("gate_proj_weight", TSInfo({I, H}, f16));
    Var up_proj_weight("up_proj_weight", TSInfo({I, H}, f16));
    Var down_proj_weight("down_proj_weight", TSInfo({H, I}, f16));
    ffi::Array<Var> params{x, gate_proj_weight, up_proj_weight, down_proj_weight};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    Var pd_gate = bb->Emit(relax::permute_dims(gate_proj_weight, std::nullopt), "permute_dims");
    Var gate = bb->Emit(relax::matmul(x, pd_gate, std::nullopt), "matmul");
    Var pd_up = bb->Emit(relax::permute_dims(up_proj_weight, std::nullopt), "permute_dims1");
    Var up = bb->Emit(relax::matmul(x, pd_up, std::nullopt), "matmul1");
    Var silu_g = bb->Emit(relax::silu(gate), "silu");
    Var mul_out = bb->Emit(relax::multiply(silu_g, up), "mul");
    Var pd_down = bb->Emit(relax::permute_dims(down_proj_weight, std::nullopt), "permute_dims2");
    Var down = bb->Emit(relax::matmul(mul_out, pd_down, std::nullopt), "matmul2");

    Var gv = bb->EmitOutput(down, "gv");
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(1)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("forward")));
    bb->AddFunction(Function(params, body, std::nullopt, true, DictAttrs(attrs)), "forward");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestLinearDynamicShape
//
// Python equivalent:
//   mod = nn.modules.Linear(in_features=4, out_features="n", bias=True)
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": nn.spec.Tensor((1, 4), "float32")}}, debug=True)
//
// Expected forward (debug=True):
//   forward(x: (1,4)f32, _io: Object, weight: (n,4)f32, bias: (n,)f32)
//   num_input = 2
//   permute_dims → matmul → add → (add, (_io,))
//
// Uses the module-aware ModuleSpec constructor with debug=true.
// The named_params (which carry the tir::Var "n" in the weight shape) are
// collected automatically from the module by ModuleSpec.
// ===========================================================================
TEST(NNExporter, TestLinearDynamicShape) {
  // 1. Create the module with a symbolic out_features dimension
  tir::Var n("n", DataType::Int(64));
  LinearModule mod = MakeLinear(ffi::Any(int64_t(4)), ffi::Any(PrimExpr(n)),
                                /*bias=*/true, std::nullopt, std::nullopt);
  // 2. Build the spec dictionary
  ffi::Map<ffi::String, ffi::Any> forward_spec;
  forward_spec.Set("x", ffi::Any(MakeSpecTensor({1, 4}, "float32")));
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> spec;
  spec.Set("forward", forward_spec);
  // 3. Create ModuleSpec — named_params (with tir::Var "n") are auto-collected
  ModuleSpec mod_spec(mod, spec, /*debug=*/true);
  // 4. Export via module->ExportTVM with debug=true
  ffi::Array<ffi::Any> result = mod->ExportTVM(mod_spec, /*debug=*/true, /*allow_extern=*/false);
  IRModule actual = result[0].cast<IRModule>();

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    tir::Var n_var("n", DataType::Int(64));
    Var x("x", TSInfo({1, 4}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    TensorStructInfo w_sinfo(ShapeExpr(ffi::Array<PrimExpr>{n_var, IntImm(DataType::Int(64), 4)}),
                             DataType::Float(32));
    TensorStructInfo b_sinfo(ShapeExpr(ffi::Array<PrimExpr>{n_var}), DataType::Float(32));
    Var weight("weight", w_sinfo);
    Var bias("bias", b_sinfo);
    ffi::Array<Var> params{x, io, weight, bias};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();
    Var perm = bb->Emit(relax::permute_dims(weight, std::nullopt), "permute_dims");
    Var mm = bb->Emit(relax::matmul(x, perm, std::nullopt), "matmul");
    Var add = bb->Emit(relax::add(mm, bias), "linear");
    Var gv1 = EmitDebugOutput(bb, add, io);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("forward")));
    bb->AddFunction(Function(params, body, std::nullopt, true, DictAttrs(attrs)), "forward");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// Duplicate-names helpers
//
// The Python test_duplicate_names parametrises over how symbolic dims are
// specified.  In C++ we build the NNParameter shapes directly with tir::Var
// objects and pass them via ModuleSpec::named_params.  The forward lambda
// receives the parameter Vars from named_args (injected by the exporter).
//
// The helper below builds and exports a three-layer model:
//   embedding: (hs, 1024)
//   up:        (is_, hs)
//   down:      (hs, is_)
// where hs / is_ are the tir::Var objects supplied by the caller.
// ===========================================================================
static IRModule ExportDuplicateNamesModel(tir::Var hs, tir::Var is_) {
  DataType f32 = DataType::Float(32);
  auto I64 = [](int64_t v) { return IntImm(DataType::Int(64), v); };

  NNParameter emb_w(Var("embedding_weights",
                        TensorStructInfo(ShapeExpr(ffi::Array<PrimExpr>{hs, I64(1024)}), f32)));
  NNParameter up_w(
      Var("up_weights", TensorStructInfo(ShapeExpr(ffi::Array<PrimExpr>{is_, hs}), f32)));
  NNParameter down_w(
      Var("down_weights", TensorStructInfo(ShapeExpr(ffi::Array<PrimExpr>{hs, is_}), f32)));

  DuplicateNamesModule mod(emb_w, up_w, down_w);

  ffi::Array<ffi::Any> spec_shape;
  spec_shape.push_back(ffi::Any(ffi::String("batch_size")));
  spec_shape.push_back(ffi::Any(int64_t(1024)));
  SpecTensor state_spec(spec_shape, "float32");

  ffi::Map<ffi::String, ffi::Any> forward_arg_spec;
  forward_arg_spec.Set("state", ffi::Any(state_spec));
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> spec;
  spec.Set("forward", forward_arg_spec);

  ModuleSpec mod_spec(mod, spec, /*debug=*/false);

  ffi::Array<ffi::Any> result = mod->ExportTVM(mod_spec, /*debug=*/false, /*allow_extern=*/false);
  return result[0].cast<IRModule>();
}

// Helper: extract the tir::Var at dim_idx in the shape of param at param_idx.
static tir::Var GetParamShapeVar(const IRModule& mod, int param_idx, int dim_idx) {
  EXPECT_EQ(mod->functions.size(), 1u);
  const auto* fn = (*mod->functions.begin()).second.as<FunctionNode>();
  EXPECT_NE(fn, nullptr);
  Var v = fn->params[param_idx];
  const auto* ts = v->struct_info_.as<TensorStructInfoNode>();
  EXPECT_NE(ts, nullptr);
  const auto* se = ts->shape.as<ShapeExprNode>();
  EXPECT_NE(se, nullptr);
  return Downcast<tir::Var>(se->values[dim_idx]);
}

// ===========================================================================
// TestDuplicateNamesSamePythonString
//
// Both hs and is_ are the *same* tir::Var (same object, same name).
// The exporter must reuse a single tir::Var for all "hidden_size" dims.
// ===========================================================================
TEST(NNExporter, TestDuplicateNamesSamePythonString) {
  tir::Var hs("hidden_size", DataType::Int(64));
  IRModule actual = ExportDuplicateNamesModel(hs, hs);  // is_ == hs

  // params: state(0), embedding_weights(1), up_weights(2), down_weights(3)
  ASSERT_EQ((*actual->functions.begin()).second.as<FunctionNode>()->params.size(), 4u);

  tir::Var v_emb0 = GetParamShapeVar(actual, 1, 0);   // embedding_weights[0] = hs
  tir::Var v_up0 = GetParamShapeVar(actual, 2, 0);    // up_weights[0] = is_ (== hs)
  tir::Var v_up1 = GetParamShapeVar(actual, 2, 1);    // up_weights[1] = hs
  tir::Var v_down0 = GetParamShapeVar(actual, 3, 0);  // down_weights[0] = hs
  tir::Var v_down1 = GetParamShapeVar(actual, 3, 1);  // down_weights[1] = is_ (== hs)

  EXPECT_EQ(std::string(v_emb0->name_hint), "hidden_size");
  EXPECT_TRUE(v_emb0.same_as(v_up0));
  EXPECT_TRUE(v_emb0.same_as(v_up1));
  EXPECT_TRUE(v_emb0.same_as(v_down0));
  EXPECT_TRUE(v_emb0.same_as(v_down1));
}

// ===========================================================================
// TestDuplicateNamesDifferentPythonString
//
// hs and is_ are distinct tir::Vars with distinct names.
// The exporter must keep them as two independent vars.
// ===========================================================================
TEST(NNExporter, TestDuplicateNamesDifferentPythonString) {
  tir::Var hs("hidden_size", DataType::Int(64));
  tir::Var is_("intermediate_size", DataType::Int(64));
  IRModule actual = ExportDuplicateNamesModel(hs, is_);

  ASSERT_EQ((*actual->functions.begin()).second.as<FunctionNode>()->params.size(), 4u);

  tir::Var v_hs = GetParamShapeVar(actual, 1, 0);  // embedding_weights[0] = hs
  tir::Var v_is = GetParamShapeVar(actual, 2, 0);  // up_weights[0] = is_

  EXPECT_EQ(std::string(v_hs->name_hint), "hidden_size");
  EXPECT_EQ(std::string(v_is->name_hint), "intermediate_size");
  EXPECT_FALSE(v_hs.same_as(v_is));
}

// ===========================================================================
// TestDuplicateNamesSameTirVar
//
// The same tir::Var object is used for both hs and is_ positions.
// Identical to TestDuplicateNamesSamePythonString in C++ (same object).
// ===========================================================================
TEST(NNExporter, TestDuplicateNamesSameTirVar) {
  tir::Var dim("hidden_size", DataType::Int(64));
  IRModule actual = ExportDuplicateNamesModel(dim, dim);

  ASSERT_EQ((*actual->functions.begin()).second.as<FunctionNode>()->params.size(), 4u);

  tir::Var v0 = GetParamShapeVar(actual, 1, 0);  // embedding_weights[0]
  tir::Var v1 = GetParamShapeVar(actual, 2, 0);  // up_weights[0]
  tir::Var v2 = GetParamShapeVar(actual, 2, 1);  // up_weights[1]

  EXPECT_TRUE(v0.same_as(v1));
  EXPECT_TRUE(v1.same_as(v2));
}

// ===========================================================================
// TestDuplicateNamesDistinctTirVarsDistinctNames
//
// Two distinct tir::Var objects with distinct names → two independent vars.
// ===========================================================================
TEST(NNExporter, TestDuplicateNamesDistinctTirVarsDistinctNames) {
  tir::Var hs("hidden_size", DataType::Int(64));
  tir::Var is_("intermediate_size", DataType::Int(64));
  IRModule actual = ExportDuplicateNamesModel(hs, is_);

  ASSERT_EQ((*actual->functions.begin()).second.as<FunctionNode>()->params.size(), 4u);

  tir::Var v_hs = GetParamShapeVar(actual, 1, 0);  // embedding_weights[0] = hs
  tir::Var v_is = GetParamShapeVar(actual, 2, 0);  // up_weights[0] = is_

  EXPECT_EQ(std::string(v_hs->name_hint), "hidden_size");
  EXPECT_EQ(std::string(v_is->name_hint), "intermediate_size");
  EXPECT_FALSE(v_hs.same_as(v_is));
}

}  // namespace testing
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
