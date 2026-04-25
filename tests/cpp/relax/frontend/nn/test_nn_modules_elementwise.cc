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
 * \file tests/cpp/relax/frontend/nn/test_nn_modules_elementwise.cc
 * \brief C++ port of selected tests from
 *        tests/python/relax/test_frontend_nn_modules.py
 *
 * Ported tests in this file:
 *   - TestReLU     (test_relu)
 *   - TestSiLU     (test_silu)
 *   - TestGELU     (test_gelu)
 *   - TestIdentity (test_identity)
 *
 * Each test:
 *   1. Instantiates the corresponding C++ module via its Make* factory.
 *   2. Wraps it in a ModuleSpec / ExportTVM(debug=true) call.
 *   3. Builds the expected IRModule manually using BlockBuilder.
 *   4. Asserts structural equality between actual and expected.
 *
 * Export pattern
 * --------------
 * The preferred pattern for simple modules (ReLU, SiLU, Linear, ...) is the
 * module-aware ModuleSpec constructor, which derives the ffi::Function for
 * each method via reflection and passes it to MethodSpec internally.
 * MethodSpec never holds or receives an NNModule object:
 *
 *   // 1. Create the module
 *   ReLUModule mod;
 *   // 2. Build the spec dictionary
 *   ffi::Map<ffi::String, ffi::Any> forward_spec;
 *   forward_spec.Set("x", ffi::Any(MakeSpecTensor({3, 3}, "float32")));
 *   ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> spec;
 *   spec.Set("forward", forward_spec);
 *   // 3. Create ModuleSpec from module and spec (debug=true -> effect_mode="plain")
 *   ModuleSpec mod_spec(mod, spec, true);
 *   // 4. Export via module->ExportTVM
 *   ffi::Array<ffi::Any> result =
 *       mod->ExportTVM(mod_spec, true, false);
 *   IRModule actual = result[0].cast<IRModule>();
 *
 * For modules whose _forward takes extra arguments beyond the spec-driven
 * inputs (GroupNorm: channel_axis + axes; Embedding: out_shape_if_nd) the
 * ExportDebug helper calls DeriveMethodFunction(mod_ref, method_name,
 * arg_names, extra_args) directly and passes the result to MethodSpec's
 * primary (ffi::Function) constructor.
 *
 * For modules with Optional<Var> arguments (Attention, TimestepEmbedding)
 * thin wrapper modules (AttentionWrapperModule, TimestepEmbeddingWrapperModule)
 * expose a plain Var signature so DeriveMethodFunction can dispatch them
 * via the module-aware ModuleSpec constructor.
 */

#include <gtest/gtest.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/op.h>

// Internal op headers
#include "../../../../../src/relax/op/nn/attention.h"
#include "../../../../../src/relax/op/nn/convolution.h"
#include "../../../../../src/relax/op/nn/nn.h"
#include "../../../../../src/relax/op/tensor/binary.h"
#include "../../../../../src/relax/op/tensor/index.h"
#include "../../../../../src/relax/op/tensor/linear_algebra.h"
#include "../../../../../src/relax/op/tensor/manipulate.h"

// nn frontend headers
#include <cmath>

#include "../../../../../src/relax/frontend/nn/core.h"
#include "../../../../../src/relax/frontend/nn/exporter.h"
#include "../../../../../src/relax/frontend/nn/modules.h"
#include "../../../../../src/relax/frontend/nn/spec.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace testing {

// ---------------------------------------------------------------------------
// Shared helpers  (mirrors test_nn_ops.cc / test_nn_debug.cc)
// ---------------------------------------------------------------------------

// Build a TensorStructInfo from a list of static dims and a dtype.
static TensorStructInfo TSInfo(std::initializer_list<int64_t> dims, DataType dtype) {
  ffi::Array<PrimExpr> shape_dims;
  for (int64_t d : dims) shape_dims.push_back(IntImm(DataType::Int(64), d));
  return TensorStructInfo(ShapeExpr(shape_dims), dtype);
}

// Build a SpecTensor from a list of static dims and a dtype string.
static SpecTensor MakeSpecTensor(std::initializer_list<int64_t> dims, const std::string& dtype) {
  ffi::Array<ffi::Any> shape;
  for (int64_t d : dims) shape.push_back(ffi::Any(d));
  return SpecTensor(shape, dtype);
}

// ---------------------------------------------------------------------------
// EmitInitEffect
//
// Emits the _initialize_effect() function into bb:
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

  ffi::Map<ffi::String, ffi::Any> attrs;
  attrs.Set("global_symbol", ffi::Any(ffi::String("_initialize_effect")));
  bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                  "_initialize_effect");
}

// ---------------------------------------------------------------------------
// EmitDebugOutput
//
// Emits the debug-mode output tuple: (result, (_io,)) and returns the Var.
// ---------------------------------------------------------------------------
static Var EmitDebugOutput(BlockBuilder& bb, Expr result, Var io, const std::string& hint = "gv1") {
  return bb->EmitOutput(relax::Tuple({result, relax::Tuple({io})}), hint);
}

// ---------------------------------------------------------------------------
// AssertStructEqual
// ---------------------------------------------------------------------------
static void AssertStructEqual(const IRModule& actual, const IRModule& expected) {
  EXPECT_TRUE(ffi::StructuralEqual()(actual, expected)) << "\n=== Actual ===\n"
                                                        << actual << "\n=== Expected ===\n"
                                                        << expected;
}

// ---------------------------------------------------------------------------
// ExportDebug
//
// Thin helper used by every test: builds MethodSpec (using the module-aware
// constructor that synthesises forward_fn via reflection), wraps it in a
// ModuleSpec, calls ExportTVM, and returns just the IRModule.
//
// Parameters:
//   mod_ref      – any NNModuleNode-derived ObjectRef
//   method_name  – exported function name (usually "forward")
//   arg_names    – ordered argument names
//   arg_specs    – SpecTensor / SpecInt / SpecTuple per argument
//   named_params – pre-collected named parameters (weight, bias, …)
//   extra_args   – trailing args appended to _forward after the spec-driven
//                  inputs (e.g. channel_axis+axes for GroupNorm, out_shape
//                  for Embedding).
// ---------------------------------------------------------------------------
static IRModule ExportDebug(runtime::ObjectRef mod_ref, const std::string& method_name,
                            ffi::Array<ffi::String> arg_names, ffi::Array<ffi::Any> arg_specs,
                            ffi::Map<ffi::String, NNParameter> named_params = {},
                            ffi::Array<ffi::Any> extra_args = {}) {
  const NNModuleNode* mod_node = mod_ref.as<NNModuleNode>();
  TVM_FFI_ICHECK(mod_node != nullptr) << "ExportDebug: object is not an NNModuleNode";

  if (extra_args.empty()) {
    // Simple case: use the module-aware ModuleSpec constructor.
    ffi::Map<ffi::String, ffi::Any> method_arg_spec;
    TVM_FFI_ICHECK_EQ(arg_names.size(), arg_specs.size());
    for (size_t i = 0; i < arg_names.size(); ++i) method_arg_spec.Set(arg_names[i], arg_specs[i]);
    ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> spec;
    spec.Set(ffi::String(method_name), method_arg_spec);
    ModuleSpec mod_spec(mod_ref, spec, /*debug=*/true);
    if (!named_params.empty()) {
      mod_spec = ModuleSpec(mod_spec->method_names, mod_spec->method_specs, named_params,
                            mod_spec->named_effects);
    }
    ffi::Array<ffi::Any> result =
        mod_node->ExportTVM(mod_spec, /*debug=*/true, /*allow_extern=*/false);
    return result[0].cast<IRModule>();
  } else {
    // Extra-args case: derive the ffi::Function via DeriveMethodFunction
    // (which appends extra_args after the spec-driven inputs) and build
    // MethodSpec with the primary (ffi::Function) constructor.
    ffi::Function forward_fn =
        DeriveMethodFunction(mod_ref, ffi::String(method_name), arg_names, extra_args);
    MethodSpec ms(forward_fn, arg_names, arg_specs, "plain", "plain");
    ModuleSpec mod_spec(ffi::Array<ffi::String>{ffi::String(method_name)},
                        ffi::Array<ffi::Any>{ffi::Any(ms)}, named_params, {});
    ffi::Array<ffi::Any> result =
        mod_node->ExportTVM(mod_spec, /*debug=*/true, /*allow_extern=*/false);
    return result[0].cast<IRModule>();
  }
}

// ---------------------------------------------------------------------------
// TestReLU
//
// Python equivalent:
//   mod = modules.ReLU()
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor((3, 3), "float32")}}, debug=True)
//
// Expected forward:
//   def forward(x: R.Tensor((3,3),"float32"), _io: R.Object)
//       -> R.Tuple(R.Tensor((3,3),"float32"), R.Tuple(R.Object)):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       relu: R.Tensor((3,3),"float32") = R.nn.relu(x)
//       gv1 = relu, (_io,)
//     return gv1
// ---------------------------------------------------------------------------
TEST(NNModules, TestReLU) {
  ReLUModule mod;

  IRModule actual =
      ExportDebug(mod, "forward", {"x"}, {ffi::Any(MakeSpecTensor({3, 3}, "float32"))});

  // Build expected IR
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
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "forward");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ---------------------------------------------------------------------------
// TestSiLU
//
// Python equivalent:
//   mod = modules.SiLU()
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor((3, 3), "float32")}}, debug=True)
//
// Expected forward:
//   def forward(x: R.Tensor((3,3),"float32"), _io: R.Object)
//       -> R.Tuple(R.Tensor((3,3),"float32"), R.Tuple(R.Object)):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       silu: R.Tensor((3,3),"float32") = R.nn.silu(x)
//       gv1 = silu, (_io,)
//     return gv1
// ---------------------------------------------------------------------------
TEST(NNModules, TestSiLU) {
  SiLUModule mod;

  IRModule actual =
      ExportDebug(mod, "forward", {"x"}, {ffi::Any(MakeSpecTensor({3, 3}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({3, 3}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    Var silu = bb->Emit(relax::silu(x), "silu");
    Var gv1 = EmitDebugOutput(bb, silu, io);

    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();

    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("forward")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "forward");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ---------------------------------------------------------------------------
// TestGELU
//
// Python equivalent:
//   mod = modules.GELU()
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor((3, 3), "float32")}}, debug=True)
//
// Expected forward:
//   def forward(x: R.Tensor((3,3),"float32"), _io: R.Object)
//       -> R.Tuple(R.Tensor((3,3),"float32"), R.Tuple(R.Object)):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       gelu: R.Tensor((3,3),"float32") = R.nn.gelu(x)
//       gv1 = gelu, (_io,)
//     return gv1
// ---------------------------------------------------------------------------
TEST(NNModules, TestGELU) {
  // Default approximate="" → exact GELU (R.nn.gelu)
  GELUModule mod;

  IRModule actual =
      ExportDebug(mod, "forward", {"x"}, {ffi::Any(MakeSpecTensor({3, 3}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({3, 3}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    Var gelu = bb->Emit(relax::gelu(x), "gelu");
    Var gv1 = EmitDebugOutput(bb, gelu, io);

    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();

    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("forward")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "forward");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ---------------------------------------------------------------------------
// TestIdentity
//
// Python equivalent:
//   mod = modules.Identity()
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor((3, 3), "float32")}}, debug=True)
//
// Expected forward:
//   def forward(x: R.Tensor((3,3),"float32"), _io: R.Object)
//       -> R.Tuple(R.Tensor((3,3),"float32"), R.Tuple(R.Object)):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       gv1 = x, (_io,)
//     return gv1
//
// Note: Identity::Forward returns x unchanged (no new binding is emitted),
// so the dataflow block contains only the output binding.
// ---------------------------------------------------------------------------
TEST(NNModules, TestIdentity) {
  IdentityModule mod;

  IRModule actual =
      ExportDebug(mod, "forward", {"x"}, {ffi::Any(MakeSpecTensor({3, 3}, "float32"))});

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({3, 3}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // Identity returns x directly — no intermediate binding.
    Var gv1 = EmitDebugOutput(bb, x, io);

    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();

    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("forward")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "forward");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ---------------------------------------------------------------------------
// TestLinear
//
// Python equivalent:
//   mod = modules.Linear(4, 8)
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor((1, 4), "float32")}}, debug=True)
//
// Expected forward:
//   def forward(x: R.Tensor((1,4),"float32"), _io: R.Object,
//               weight: R.Tensor((8,4),"float32"),
//               bias:   R.Tensor((8,),"float32"))
//       -> R.Tuple(R.Tensor((1,8),"float32"), R.Tuple(R.Object)):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       permute_dims: R.Tensor((4,8),"float32") = R.permute_dims(weight, axes=None)
//       matmul:       R.Tensor((1,8),"float32") = R.matmul(x, permute_dims, out_dtype="void")
//       add:          R.Tensor((1,8),"float32") = R.add(matmul, bias)
//       gv1 = add, (_io,)
//     return gv1

}  // namespace testing
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
