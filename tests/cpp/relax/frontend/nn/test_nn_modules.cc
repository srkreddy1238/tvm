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
 * \file tests/cpp/relax/frontend/nn/test_nn_modules.cc
 * \brief C++ port of selected tests from
 *        tests/python/relax/test_frontend_nn_modules.py
 *
 * Ported tests:
 *   - TestReLU            (test_relu)
 *   - TestSiLU            (test_silu)
 *   - TestGELU            (test_gelu)
 *   - TestIdentity        (test_identity)
 *   - TestLinear          (test_linear)
 *   - TestConv1D          (test_conv1d)
 *   - TestConv1DTranspose (test_conv1d_transpose)
 *   - TestLayerNorm       (test_layer_norm)
 *   - TestConv2D          (test_conv2d)
 *   - TestConv3D          (test_conv3d)
 *   - TestConv2DDynamic   (test_conv2d_dynamic)
 *   - TestRMSNorm         (test_rms_norm)
 *   - TestGroupNorm       (test_group_norm)
 *   - TestEmbedding1D     (test_embedding_1d)
 *   - TestKVCache         (test_kv_cache)
 *   - TestEmbedding2D     (test_embedding_2d)
 *   - TestAttention       (test_attention)
 *   - TestTimestepEmbedding  (test_timestep_embedding)
 *   - TestTimesteps       (test_timesteps)
 *   - TestNNModuleTupleInput  (test_nn_module_tuple_input)
 *   - TestNNModuleListInput   (test_nn_module_list_input)
 *   - TestModuleList          (test_module_list)
 *   - TestModuleDict          (test_module_dict)
 *
 * Each test:
 *   1. Instantiates the corresponding C++ module via its Make* factory.
 *   2. Wraps it in a ModuleSpec / ExportTVM(debug=true) call.
 *   3. Builds the expected IRModule manually using BlockBuilder.
 *   4. Asserts structural equality between actual and expected.
 *
 * The pattern mirrors test_nn_ops.cc / test_nn_debug.cc exactly:
 *   - ExportDebug() wraps the module's Forward() in a MethodSpec and
 *     calls NNModuleNode::ExportTVM(spec, debug=true).
 *   - EmitInitEffect() / EmitDebugOutput() / AssertStructEqual() are the
 *     same shared helpers used in the other test files.
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

// ===========================================================================
// Shared helpers  (mirrors test_nn_ops.cc / test_nn_debug.cc)
// ===========================================================================

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

// Build a SpecTensor with mixed static (int64) and symbolic (string) dims.
// Each element of `dims` is an ffi::Any holding either int64_t or ffi::String.
static SpecTensor MakeSpecTensorMixed(ffi::Array<ffi::Any> dims, const std::string& dtype) {
  return SpecTensor(dims, dtype);
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

  // MethodSpec(mod_ref, ...) builds the forward_fn internally via reflection.
  MethodSpec ms(mod_ref, arg_names, arg_specs, "plain", "plain", extra_args);
  ModuleSpec mod_spec(ffi::Array<ffi::String>{ffi::String(method_name)},
                      ffi::Array<ffi::Any>{ffi::Any(ms)}, named_params, {});
  ffi::Array<ffi::Any> result =
      mod_node->ExportTVM(mod_spec, /*debug=*/true, /*allow_extern=*/false);
  return result[0].cast<IRModule>();
}

// ===========================================================================
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
// ===========================================================================
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

// ===========================================================================
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
// ===========================================================================
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

// ===========================================================================
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
// ===========================================================================
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

// ===========================================================================
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
// ===========================================================================
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

// ===========================================================================
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
// ===========================================================================
TEST(NNModules, TestLinear) {
  // Linear(in_features=4, out_features=8, bias=True)
  LinearModule mod = MakeLinear(ffi::Any(int64_t(4)), ffi::Any(int64_t(8)), /*bias=*/true,
                                /*dtype=*/std::nullopt, /*out_dtype=*/std::nullopt);

  // Collect named parameters: weight [8,4] and bias [8].
  ffi::Map<ffi::String, NNParameter> named_params = mod.get()->NamedParameters("");

  IRModule actual = ExportDebug(mod, "forward", {"x"},
                                {ffi::Any(MakeSpecTensor({1, 4}, "float32"))}, named_params);

  // Build expected IR
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({1, 4}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    // Parameters appear after the user inputs and _io in the function signature.
    Var weight("weight", TSInfo({8, 4}, DataType::Float(32)));
    Var bias("bias", TSInfo({8}, DataType::Float(32)));
    ffi::Array<Var> params{x, io, weight, bias};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // permute_dims(weight, axes=None) → (4, 8)
    Var perm = bb->Emit(relax::permute_dims(weight, std::nullopt), "permute_dims");
    // matmul(x, perm, out_dtype=void) → (1, 8)
    Var mm = bb->Emit(relax::matmul(x, perm, std::nullopt), "matmul");
    // add(mm, bias) → (1, 8)
    Var add = bb->Emit(relax::add(mm, bias), "add");
    Var gv1 = EmitDebugOutput(bb, add, io);

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

// ===========================================================================
// TestConv1D
//
// Python equivalent:
//   mod = modules.Conv1D(3, 32, 3, bias=True)
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor([1, 3, 32], "float32")}}, debug=True)
//
// Expected forward:
//   def forward(x: R.Tensor((1,3,32),"float32"), _io: R.Object,
//               weight: R.Tensor((32,3,3),"float32"),
//               bias:   R.Tensor((32,),"float32"))
//       -> R.Tuple(R.Tensor((1,32,30),"float32"), R.Tuple(R.Object)):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       lv1: R.Tensor((1,32,30),"float32") = R.nn.conv1d(
//           x, weight, strides=[1], padding=[0,0], dilation=[1],
//           groups=1, data_layout="NCW", kernel_layout="OIW",
//           out_layout="NCW", out_dtype="void")
//       lv2: R.Tensor((1,32,1),"float32") = R.reshape(bias, R.shape([1,32,1]))
//       conv1d: R.Tensor((1,32,30),"float32") = R.add(lv1, lv2)
//       gv1 = conv1d, (_io,)
//     return gv1
// ===========================================================================
TEST(NNModules, TestConv1D) {
  // Conv1D(in_channels=3, out_channels=32, kernel_size=3, bias=True)
  Conv1DModule mod = MakeConv1D(ffi::Any(int64_t(3)), ffi::Any(int64_t(32)), ffi::Any(int64_t(3)),
                                /*stride=*/1, /*padding=*/0, /*dilation=*/1, /*groups=*/1,
                                /*has_bias=*/true, /*dtype=*/std::nullopt);

  ffi::Map<ffi::String, NNParameter> named_params = mod.get()->NamedParameters("");

  IRModule actual = ExportDebug(mod, "forward", {"x"},
                                {ffi::Any(MakeSpecTensor({1, 3, 32}, "float32"))}, named_params);

  // Build expected IR
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({1, 3, 32}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    Var weight("weight", TSInfo({32, 3, 3}, DataType::Float(32)));
    Var bias("bias", TSInfo({32}, DataType::Float(32)));
    ffi::Array<Var> params{x, io, weight, bias};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // conv1d(x, weight, strides=[1], padding=[0,0], dilation=[1],
    //        groups=1, data_layout="NCW", kernel_layout="OIW",
    //        out_layout=None, out_dtype=None)
    Var lv1 = bb->Emit(relax::conv1d(x, weight,
                                     /*strides=*/{1},
                                     /*padding=*/{0},
                                     /*dilation=*/{1},
                                     /*groups=*/1,
                                     /*data_layout=*/"NCW",
                                     /*kernel_layout=*/"OIW",
                                     /*out_layout=*/std::nullopt,
                                     /*out_dtype=*/std::nullopt),
                       "lv1");

    // reshape(bias, [1, 32, 1])
    ShapeExpr bias_shape(ffi::Array<PrimExpr>{
        IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 32), IntImm(DataType::Int(64), 1)});
    Var lv2 = bb->Emit(relax::reshape(bias, bias_shape), "lv2");

    // add(lv1, lv2)
    Var conv1d_out = bb->Emit(relax::add(lv1, lv2), "conv1d");
    Var gv1 = EmitDebugOutput(bb, conv1d_out, io);

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

// ===========================================================================
// TestConv1DTranspose
//
// Python equivalent:
//   mod = modules.ConvTranspose1D(3, 32, 3, bias=True)
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor([1, 3, 30], "float32")}}, debug=True)
//
// Expected forward:
//   def forward(x: R.Tensor((1,3,30),"float32"), _io: R.Object,
//               weight: R.Tensor((3,32,3),"float32"),
//               bias:   R.Tensor((32,),"float32"))
//       -> R.Tuple(R.Tensor((1,32,32),"float32"), R.Tuple(R.Object)):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       lv1: R.Tensor((1,32,32),"float32") = R.nn.conv1d_transpose(
//           x, weight, strides=[1], padding=[0,0], output_padding=[0],
//           dilation=[1], groups=1, data_layout="NCW", kernel_layout="IOW",
//           out_layout="NCW", out_dtype="void")
//       lv2: R.Tensor((1,32,1),"float32") = R.reshape(bias, R.shape([1,32,1]))
//       conv1d_transpose: R.Tensor((1,32,32),"float32") = R.add(lv1, lv2)
//       gv1 = conv1d_transpose, (_io,)
//     return gv1
// ===========================================================================
TEST(NNModules, TestConv1DTranspose) {
  // ConvTranspose1D(in_channels=3, out_channels=32, kernel_size=3, bias=True)
  ConvTranspose1DModule mod =
      MakeConvTranspose1D(ffi::Any(int64_t(3)), ffi::Any(int64_t(32)), ffi::Any(int64_t(3)),
                          /*stride=*/1, /*padding=*/0, /*output_padding=*/0,
                          /*dilation=*/1, /*groups=*/1,
                          /*has_bias=*/true, /*dtype=*/std::nullopt);

  ffi::Map<ffi::String, NNParameter> named_params = mod.get()->NamedParameters("");

  IRModule actual = ExportDebug(mod, "forward", {"x"},
                                {ffi::Any(MakeSpecTensor({1, 3, 30}, "float32"))}, named_params);

  // Build expected IR
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({1, 3, 30}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    // weight shape for ConvTranspose1D: [in_channels, out_channels/groups, kernel_size]
    //   = [3, 32, 3]
    Var weight("weight", TSInfo({3, 32, 3}, DataType::Float(32)));
    Var bias("bias", TSInfo({32}, DataType::Float(32)));
    ffi::Array<Var> params{x, io, weight, bias};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // conv1d_transpose(x, weight, strides=[1], padding=[0,0], output_padding=[0],
    //                  dilation=[1], groups=1, data_layout="NCW", kernel_layout="IOW",
    //                  out_layout=None, out_dtype=None)
    Var lv1 = bb->Emit(relax::conv1d_transpose(x, weight,
                                               /*strides=*/{1},
                                               /*padding=*/{0},
                                               /*output_padding=*/{0},
                                               /*dilation=*/{1},
                                               /*groups=*/1,
                                               /*data_layout=*/"NCW",
                                               /*kernel_layout=*/"IOW",
                                               /*out_layout=*/std::nullopt,
                                               /*out_dtype=*/std::nullopt),
                       "lv1");

    // reshape(bias, [1, 32, 1])
    ShapeExpr bias_shape(ffi::Array<PrimExpr>{
        IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 32), IntImm(DataType::Int(64), 1)});
    Var lv2 = bb->Emit(relax::reshape(bias, bias_shape), "lv2");

    // add(lv1, lv2)
    Var conv1d_t_out = bb->Emit(relax::add(lv1, lv2), "conv1d_transpose");
    Var gv1 = EmitDebugOutput(bb, conv1d_t_out, io);

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

// ===========================================================================
// TestLayerNorm
//
// Python equivalent:
//   mod = modules.LayerNorm(8)
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor((2, 4, 8), "float32")}}, debug=True)
//
// Expected forward:
//   def forward(x:      R.Tensor((2,4,8),"float32"), _io: R.Object,
//               weight: R.Tensor((8,),"float32"),
//               bias:   R.Tensor((8,),"float32"))
//       -> R.Tuple(R.Tensor((2,4,8),"float32"), R.Tuple(R.Object)):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       layer_norm: R.Tensor((2,4,8),"float32") = R.nn.layer_norm(
//           x, weight, bias, axes=[-1], epsilon=1e-5, center=True, scale=True)
//       gv1 = layer_norm, (_io,)
//     return gv1
// ===========================================================================
TEST(NNModules, TestLayerNorm) {
  // LayerNorm(normalized_shape=8, eps=1e-5, elementwise_affine=True)
  // MakeLayerNorm accepts a scalar int64 for normalized_shape.
  LayerNormModule mod = MakeLayerNorm(ffi::Any(int64_t(8)), /*eps=*/1e-5,
                                      /*elementwise_affine=*/true, /*dtype=*/std::nullopt);

  ffi::Map<ffi::String, NNParameter> named_params = mod.get()->NamedParameters("");

  IRModule actual = ExportDebug(mod, "forward", {"x"},
                                {ffi::Any(MakeSpecTensor({2, 4, 8}, "float32"))}, named_params);

  // Build expected IR
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({2, 4, 8}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    Var weight("weight", TSInfo({8}, DataType::Float(32)));
    Var bias("bias", TSInfo({8}, DataType::Float(32)));
    ffi::Array<Var> params{x, io, weight, bias};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // layer_norm(x, weight, bias, axes=[-1], epsilon=1e-5, center=True, scale=True)
    ffi::Array<Integer> axes{Integer(-1)};
    Var ln = bb->Emit(
        relax::layer_norm(x, weight, bias, axes, /*epsilon=*/1e-5, /*center=*/true, /*scale=*/true),
        "layer_norm");
    Var gv1 = EmitDebugOutput(bb, ln, io);

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

// ===========================================================================
// TestConv2D
//
// Python equivalent:
//   mod = modules.Conv2D(3, 32, 3, bias=True)
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor([1, 3, 32, 32], "float32")}}, debug=True)
//
// Expected forward:
//   def forward(x:      R.Tensor((1,3,32,32),"float32"), _io: R.Object,
//               weight: R.Tensor((32,3,3,3),"float32"),
//               bias:   R.Tensor((32,),"float32"))
//       -> R.Tuple(R.Tensor((1,32,30,30),"float32"), R.Tuple(R.Object)):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       lv1: R.Tensor((1,32,30,30),"float32") = R.nn.conv2d(x, weight)
//       lv2: R.Tensor((1,32,1,1),"float32")   = R.reshape(bias, R.shape([1,32,1,1]))
//       conv2d: R.Tensor((1,32,30,30),"float32") = R.add(lv1, lv2)
//       gv1 = conv2d, (_io,)
//     return gv1
// ===========================================================================
TEST(NNModules, TestConv2D) {
  // Conv2D(in_channels=3, out_channels=32, kernel_size=3, bias=True)
  // kernel_size=3 is broadcast to [3, 3] by the Python wrapper; in C++ we
  // pass the already-expanded Array<Integer> directly to MakeConv2D.
  Conv2DModule mod = MakeConv2D(ffi::Any(int64_t(3)), ffi::Any(int64_t(32)),
                                /*kernel_size=*/ffi::Array<Integer>{Integer(3), Integer(3)},
                                /*stride=*/1, /*padding=*/0, /*dilation=*/1, /*groups=*/1,
                                /*has_bias=*/true, /*dtype=*/std::nullopt,
                                /*data_layout=*/"NCHW");

  ffi::Map<ffi::String, NNParameter> named_params = mod.get()->NamedParameters("");

  IRModule actual = ExportDebug(
      mod, "forward", {"x"}, {ffi::Any(MakeSpecTensor({1, 3, 32, 32}, "float32"))}, named_params);

  // Build expected IR
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({1, 3, 32, 32}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    // weight shape: [out_channels, in_channels/groups, kH, kW] = [32, 3, 3, 3]
    Var weight("weight", TSInfo({32, 3, 3, 3}, DataType::Float(32)));
    Var bias("bias", TSInfo({32}, DataType::Float(32)));
    ffi::Array<Var> params{x, io, weight, bias};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // conv2d(x, weight, strides=[1], padding=[0], dilation=[1],
    //        groups=1, data_layout="NCHW", kernel_layout="OIHW",
    //        out_layout=None, out_dtype=None)
    Var lv1 = bb->Emit(relax::conv2d(x, weight,
                                     /*strides=*/{1},
                                     /*padding=*/{0},
                                     /*dilation=*/{1},
                                     /*groups=*/1,
                                     /*data_layout=*/"NCHW",
                                     /*kernel_layout=*/"OIHW",
                                     /*out_layout=*/std::nullopt,
                                     /*out_dtype=*/std::nullopt),
                       "lv1");

    // reshape(bias, [1, 32, 1, 1])
    ShapeExpr bias_shape(
        ffi::Array<PrimExpr>{IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 32),
                             IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 1)});
    Var lv2 = bb->Emit(relax::reshape(bias, bias_shape), "lv2");

    // add(lv1, lv2)
    Var conv2d_out = bb->Emit(relax::add(lv1, lv2), "conv2d");
    Var gv1 = EmitDebugOutput(bb, conv2d_out, io);

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

// ===========================================================================
// TestConv3D
//
// Python equivalent:
//   mod = modules.Conv3D(3, 32, 3, bias=True)
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor([1, 3, 32, 32, 32], "float32")}}, debug=True)
//
// Expected forward:
//   def forward(x:      R.Tensor((1,3,32,32,32),"float32"), _io: R.Object,
//               weight: R.Tensor((32,3,3,3,3),"float32"),
//               bias:   R.Tensor((32,),"float32"))
//       -> R.Tuple(R.Tensor((1,32,30,30,30),"float32"), R.Tuple(R.Object)):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       lv1: R.Tensor((1,32,30,30,30),"float32") = R.nn.conv3d(x, weight)
//       lv2: R.Tensor((1,32,1,1,1),"float32")    = R.reshape(bias, R.shape([1,32,1,1,1]))
//       conv3d: R.Tensor((1,32,30,30,30),"float32") = R.add(lv1, lv2)
//       gv1 = conv3d, (_io,)
//     return gv1
// ===========================================================================
TEST(NNModules, TestConv3D) {
  // Conv3D(in_channels=3, out_channels=32, kernel_size=3, bias=True)
  // kernel_size=3 is broadcast to [3, 3, 3] by the Python wrapper; in C++ we
  // pass the already-expanded Array<Integer> directly to MakeConv3D.
  Conv3DModule mod =
      MakeConv3D(ffi::Any(int64_t(3)), ffi::Any(int64_t(32)),
                 /*kernel_size=*/ffi::Array<Integer>{Integer(3), Integer(3), Integer(3)},
                 /*stride=*/1, /*padding=*/0, /*dilation=*/1, /*groups=*/1,
                 /*has_bias=*/true, /*dtype=*/std::nullopt,
                 /*data_layout=*/"NCDHW");

  ffi::Map<ffi::String, NNParameter> named_params = mod.get()->NamedParameters("");

  IRModule actual =
      ExportDebug(mod, "forward", {"x"}, {ffi::Any(MakeSpecTensor({1, 3, 32, 32, 32}, "float32"))},
                  named_params);

  // Build expected IR
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({1, 3, 32, 32, 32}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    // weight shape: [out_channels, in_channels, kD, kH, kW] = [32, 3, 3, 3, 3]
    Var weight("weight", TSInfo({32, 3, 3, 3, 3}, DataType::Float(32)));
    Var bias("bias", TSInfo({32}, DataType::Float(32)));
    ffi::Array<Var> params{x, io, weight, bias};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // conv3d(x, weight, strides=[1], padding=[0], dilation=[1],
    //        groups=1, data_layout="NCDHW", kernel_layout="OIDHW",
    //        out_layout=None, out_dtype=None)
    Var lv1 = bb->Emit(relax::conv3d(x, weight,
                                     /*strides=*/{1},
                                     /*padding=*/{0},
                                     /*dilation=*/{1},
                                     /*groups=*/1,
                                     /*data_layout=*/"NCDHW",
                                     /*kernel_layout=*/"OIDHW",
                                     /*out_layout=*/std::nullopt,
                                     /*out_dtype=*/std::nullopt),
                       "lv1");

    // reshape(bias, [1, 32, 1, 1, 1])
    ShapeExpr bias_shape(ffi::Array<PrimExpr>{
        IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 32), IntImm(DataType::Int(64), 1),
        IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 1)});
    Var lv2 = bb->Emit(relax::reshape(bias, bias_shape), "lv2");

    // add(lv1, lv2)
    Var conv3d_out = bb->Emit(relax::add(lv1, lv2), "conv3d");
    Var gv1 = EmitDebugOutput(bb, conv3d_out, io);

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

// ===========================================================================
// TestConv2DDynamic
//
// Python equivalent:
//   mod = modules.Conv2D(tvm.tir.Var("in_channels", "int64"), 32, 3, bias=True)
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor(["n","c","h","w"], "float32")}}, debug=True)
//
// The input spec uses symbolic string dims; the weight's in_channels dim is
// also symbolic (a tir::Var named "in_channels").  The exporter deduplicates
// symbolic vars by name so the same tir::Var appears in both x and weight.
//
// Expected forward (symbolic shapes):
//   def forward(x:      R.Tensor((n,c,h,w),"float32"),   _io: R.Object,
//               weight: R.Tensor((32,in_channels,3,3),"float32"),
//               bias:   R.Tensor((32,),"float32"))
//       -> R.Tuple(R.Tensor((n,32,h-2,w-2),"float32"), R.Tuple(R.Object)):
//     n = T.int64(); h = T.int64(); w = T.int64()
//     c = T.int64(); in_channels = T.int64()
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       lv1 = R.nn.conv2d(x, weight)
//       lv2 = R.reshape(bias, R.shape([1,32,1,1]))
//       conv2d = R.add(lv1, lv2)
//       gv1 = conv2d, (_io,)
//     return gv1
// ===========================================================================
TEST(NNModules, TestConv2DDynamic) {
  // Conv2D(in_channels=tir.Var("in_channels","int64"), out_channels=32,
  //        kernel_size=3, bias=True)
  // The Python wrapper expands kernel_size=3 to [3,3]; we do the same here.
  tir::Var in_channels_var("in_channels", DataType::Int(64));
  Conv2DModule mod = MakeConv2D(ffi::Any(PrimExpr(in_channels_var)), ffi::Any(int64_t(32)),
                                /*kernel_size=*/ffi::Array<Integer>{Integer(3), Integer(3)},
                                /*stride=*/1, /*padding=*/0, /*dilation=*/1, /*groups=*/1,
                                /*has_bias=*/true, /*dtype=*/std::nullopt,
                                /*data_layout=*/"NCHW");

  ffi::Map<ffi::String, NNParameter> named_params = mod.get()->NamedParameters("");

  // Input spec: symbolic dims ["n", "c", "h", "w"]
  ffi::Array<ffi::Any> spec_dims;
  spec_dims.push_back(ffi::Any(ffi::String("n")));
  spec_dims.push_back(ffi::Any(ffi::String("c")));
  spec_dims.push_back(ffi::Any(ffi::String("h")));
  spec_dims.push_back(ffi::Any(ffi::String("w")));
  SpecTensor x_spec = MakeSpecTensorMixed(spec_dims, "float32");

  IRModule actual = ExportDebug(mod, "forward", {"x"}, {ffi::Any(x_spec)}, named_params);

  // Build expected IR with symbolic shapes.
  // The exporter creates fresh tir::Vars for each unique string dim name;
  // we must use the same names so structural equality holds.
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    // Symbolic tir::Vars (one per unique name)
    tir::Var n("n", DataType::Int(64));
    tir::Var c("c", DataType::Int(64));
    tir::Var h("h", DataType::Int(64));
    tir::Var w("w", DataType::Int(64));
    tir::Var in_ch("in_channels", DataType::Int(64));

    // x: (n, c, h, w)
    Var x("x", TensorStructInfo(ShapeExpr(ffi::Array<PrimExpr>{n, c, h, w}), DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    // weight: (32, in_channels, 3, 3)
    Var weight("weight",
               TensorStructInfo(ShapeExpr(ffi::Array<PrimExpr>{IntImm(DataType::Int(64), 32), in_ch,
                                                               IntImm(DataType::Int(64), 3),
                                                               IntImm(DataType::Int(64), 3)}),
                                DataType::Float(32)));
    Var bias("bias", TSInfo({32}, DataType::Float(32)));
    ffi::Array<Var> params{x, io, weight, bias};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // conv2d(x, weight, strides=[1], padding=[0], dilation=[1],
    //        groups=1, data_layout="NCHW", kernel_layout="OIHW",
    //        out_layout=None, out_dtype=None)
    Var lv1 = bb->Emit(relax::conv2d(x, weight,
                                     /*strides=*/{1},
                                     /*padding=*/{0},
                                     /*dilation=*/{1},
                                     /*groups=*/1,
                                     /*data_layout=*/"NCHW",
                                     /*kernel_layout=*/"OIHW",
                                     /*out_layout=*/std::nullopt,
                                     /*out_dtype=*/std::nullopt),
                       "lv1");

    // reshape(bias, [1, 32, 1, 1])
    ShapeExpr bias_shape(
        ffi::Array<PrimExpr>{IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 32),
                             IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 1)});
    Var lv2 = bb->Emit(relax::reshape(bias, bias_shape), "lv2");

    // add(lv1, lv2)
    Var conv2d_out = bb->Emit(relax::add(lv1, lv2), "conv2d");
    Var gv1 = EmitDebugOutput(bb, conv2d_out, io);

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

// ===========================================================================
// TestRMSNorm
//
// Python equivalent:
//   mod = modules.RMSNorm(8, [2], bias=False)
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor((2, 4, 8), "float32")}}, debug=True)
//
// Expected forward:
//   def forward(x:      R.Tensor((2,4,8),"float32"), _io: R.Object,
//               weight: R.Tensor((8,),"float32"))
//       -> R.Tuple(R.Tensor((2,4,8),"float32"), R.Tuple(R.Object)):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       rms_norm: R.Tensor((2,4,8),"float32") = R.nn.rms_norm(
//           x, weight, axes=[2], epsilon=1e-5)
//       gv1 = rms_norm, (_io,)
//     return gv1
//
// Note: bias=False → no bias parameter; only weight appears in the signature.
// ===========================================================================
TEST(NNModules, TestRMSNorm) {
  // RMSNorm(hidden_size=8, axes=[2], epsilon=1e-5, has_bias=False)
  RMSNormModule mod = MakeRMSNorm(ffi::Any(int64_t(8)),
                                  /*axes=*/ffi::Array<Integer>{Integer(2)},
                                  /*epsilon=*/1e-5,
                                  /*has_bias=*/false,
                                  /*dtype=*/std::nullopt);

  ffi::Map<ffi::String, NNParameter> named_params = mod.get()->NamedParameters("");

  IRModule actual = ExportDebug(mod, "forward", {"x"},
                                {ffi::Any(MakeSpecTensor({2, 4, 8}, "float32"))}, named_params);

  // Build expected IR
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({2, 4, 8}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    // bias=False → only weight in the signature
    Var weight("weight", TSInfo({8}, DataType::Float(32)));
    ffi::Array<Var> params{x, io, weight};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // rms_norm(x, weight, axes=[2], epsilon=1e-5)
    ffi::Array<Integer> axes{Integer(2)};
    Var rn = bb->Emit(relax::rms_norm(x, weight, axes, /*epsilon=*/1e-5), "rms_norm");
    Var gv1 = EmitDebugOutput(bb, rn, io);

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

// ===========================================================================
// TestGroupNorm
//
// Python equivalent:
//   mod = modules.GroupNorm(num_groups=2, num_channels=4)
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor((2, 4, 8), "float32")}}, debug=True)
//
// Expected forward:
//   def forward(x:      R.Tensor((2,4,8),"float32"), _io: R.Object,
//               weight: R.Tensor((4,),"float32"),
//               bias:   R.Tensor((4,),"float32"))
//       -> R.Tuple(R.Tensor((2,4,8),"float32"), R.Tuple(R.Object)):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       group_norm: R.Tensor((2,4,8),"float32") = R.nn.group_norm(
//           x, weight, bias, num_groups=2, channel_axis=1, axes=[2])
//       gv1 = group_norm, (_io,)
//     return gv1
//
// GroupNormModuleNode::Forward takes (Var x, int64_t channel_axis,
// Array<Integer> axes).  The Python GroupNorm.forward() defaults are
// channel_axis=1 and axes=list(range(2, ndim)).  For input (2,4,8) with
// ndim=3 that gives channel_axis=1, axes=[2].  These extra args are passed
// via the extra_args parameter of ExportDebug.
// ===========================================================================
TEST(NNModules, TestGroupNorm) {
  // GroupNorm(num_groups=2, num_channels=4, eps=1e-5, affine=True)
  GroupNormModule mod = MakeGroupNorm(/*num_groups=*/2, ffi::Any(int64_t(4)),
                                      /*eps=*/1e-5, /*affine=*/true,
                                      /*dtype=*/std::nullopt);

  ffi::Map<ffi::String, NNParameter> named_params = mod.get()->NamedParameters("");

  // GroupNormModuleNode::Forward(Var x, int64_t channel_axis,
  //                              ffi::Array<Integer> axes)
  // Python default: channel_axis=1, axes=list(range(2, ndim)).
  // For input (2,4,8) ndim=3 → axes=[2].
  ffi::Array<ffi::Any> extra_args;
  extra_args.push_back(ffi::Any(int64_t(1)));                       // channel_axis
  extra_args.push_back(ffi::Any(ffi::Array<Integer>{Integer(2)}));  // axes

  IRModule actual =
      ExportDebug(mod, "forward", {"x"}, {ffi::Any(MakeSpecTensor({2, 4, 8}, "float32"))},
                  named_params, extra_args);

  // Build expected IR
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({2, 4, 8}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    Var weight("weight", TSInfo({4}, DataType::Float(32)));
    Var bias("bias", TSInfo({4}, DataType::Float(32)));
    ffi::Array<Var> params{x, io, weight, bias};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // group_norm(x, weight, bias, num_groups=2, channel_axis=1, axes=[2],
    //            epsilon=1e-5, center=True, scale=True)
    ffi::Array<Integer> axes{Integer(2)};
    Var gn = bb->Emit(relax::group_norm(x, weight, bias, /*num_groups=*/2, /*channel_axis=*/1, axes,
                                        /*epsilon=*/1e-5, /*center=*/true, /*scale=*/true),
                      "group_norm");
    Var gv1 = EmitDebugOutput(bb, gn, io);

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

// ===========================================================================
// TestEmbedding1D
//
// Python equivalent:
//   mod = modules.Embedding(8, 16, "float32")
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor((4,), "int32")}}, debug=True)
//
// Expected forward:
//   def forward(x:      R.Tensor((4,),"int32"),    _io: R.Object,
//               weight: R.Tensor((8,16),"float32"))
//       -> R.Tuple(R.Tensor((4,16),"float32"), R.Tuple(R.Object)):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       take: R.Tensor((4,16),"float32") = R.take(weight, x, axis=0)
//       gv1 = take, (_io,)
//     return gv1
//
// EmbeddingModuleNode::Forward takes (Var x, Array<Any> out_shape_if_nd).
// For a 1-D input the Python wrapper passes out_shape=[] (empty), which
// triggers the direct R.take path (no reshape).  The empty array is passed
// via extra_args.
// ===========================================================================
TEST(NNModules, TestEmbedding1D) {
  // Embedding(num=8, dim=16, dtype="float32")
  EmbeddingModule mod = MakeEmbedding(ffi::Any(int64_t(8)), ffi::Any(int64_t(16)),
                                      ffi::Optional<ffi::String>("float32"));

  ffi::Map<ffi::String, NNParameter> named_params = mod.get()->NamedParameters("");

  // EmbeddingModuleNode::Forward(Var x, ffi::Array<ffi::Any> out_shape_if_nd)
  // For 1-D input: out_shape_if_nd = [] (empty array → direct take path).
  ffi::Array<ffi::Any> extra_args;
  extra_args.push_back(ffi::Any(ffi::Array<ffi::Any>{}));  // out_shape_if_nd = []

  IRModule actual = ExportDebug(mod, "forward", {"x"}, {ffi::Any(MakeSpecTensor({4}, "int32"))},
                                named_params, extra_args);

  // Build expected IR
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({4}, DataType::Int(32)));
    Var io("_io", ObjectStructInfo());
    // weight: [num_embeddings, embedding_dim] = [8, 16]
    Var weight("weight", TSInfo({8, 16}, DataType::Float(32)));
    ffi::Array<Var> params{x, io, weight};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // take(weight, x, axis=0)
    Var take_out = bb->Emit(relax::take(weight, x, ffi::Optional<int64_t>(0)), "take");
    Var gv1 = EmitDebugOutput(bb, take_out, io);

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

// ===========================================================================
// TestKVCache
//
// Python equivalent:
//   class KVCacheTest(modules.Module):
//       def __init__(self):
//           self.cache = modules.KVCache(8, [2, 4])
//       def forward(self, x: core.Tensor) -> core.Tensor:
//           self.cache.append(x)
//           return self.cache.view(4)
//
//   tvm_mod, _ = KVCacheTest().export_tvm(
//       spec={"forward": {"x": spec.Tensor((2, 4), "float32")}}, debug=True)
//
// Module internals:
//   KVCacheModule(init_seq_len=8, unit_shape=[2,4], dtype="float32")
//   No trainable parameters.  One named effect: "cache".
//
// Expected _initialize_effect:
//   def _initialize_effect() -> R.Tuple(R.Object, R.Object):
//       with R.dataflow():
//           _io   = R.null_value()
//           lv    = R.zeros(R.shape([8, 2, 4]), dtype="float32")
//           cache = R.call_pure_packed(
//                       "vm.builtin.attention_kv_cache_create",
//                       lv, R.shape([8, 2, 4]), R.prim_value(0),
//                       sinfo_args=[R.Object()])
//           lv1   = (_io, cache)
//           gv    = lv1
//       return gv
//
// Expected forward:
//   def forward(x: R.Tensor((2,4),"float32"), _io: R.Object, cache: R.Object)
//       -> R.Tuple(R.Tensor((4,2,4),"float32"), R.Tuple(R.Object, R.Object)):
//     R.func_attr({"num_input": 3})
//     with R.dataflow():
//       kv_cache_append: R.Object = R.call_inplace_packed(
//           "vm.builtin.attention_kv_cache_append",
//           cache, x, inplace_indices=[0], sinfo_args=[R.Object()])
//       kv_cache_view: R.Tensor((4,2,4),"float32") = R.call_pure_packed(
//           "vm.builtin.attention_kv_cache_view",
//           kv_cache_append, R.shape([4, 2, 4]),
//           sinfo_args=(R.Tensor((4,2,4),"float32"),))
//       gv1 = kv_cache_view, (_io, kv_cache_append)
//     return gv1
//
// NOTE on binding names vs Python IR:
//   Python IR uses lv2/lv3 (sequential auto-naming).
//   C++ BlockBuilder uses the hint strings from KVCacheModuleNode:
//     Append → hint "kv_cache_append"  → binding "kv_cache_append"
//     View   → hint "kv_cache_view"    → binding "kv_cache_view"
//   The expected IR is built to match the C++ exporter output exactly.
// ===========================================================================
TEST(NNModules, TestKVCache) {
  // Create the KVCache effect: KVCache(init_seq_len=8, unit_shape=[2,4])
  // Python default dtype = get_default_dtype() = "float32"
  KVCacheModule kv(/*init_seq_len=*/8,
                   /*unit_shape=*/ffi::Array<Integer>{Integer(2), Integer(4)},
                   /*dtype=*/"float32");

  // ---------------------------------------------------------------------------
  // Build the ModuleSpec.
  //
  // The forward lambda captures the KVCacheModule by value.  Inside the
  // lambda (which runs inside a BlockBuilder dataflow scope):
  //   1. kv->Append(x_tensor)  — emits call_inplace_packed, updates kv->cache
  //   2. kv->View(4)           — emits call_pure_packed, returns NNTensor
  //
  // The lambda receives named_args = {"x": NNTensor(x_var)}.
  // ---------------------------------------------------------------------------
  // kv is a reference-counted ObjectRef: capturing it by value shares the
  // underlying KVCacheModuleNode.  kv.get() returns a non-const pointer to
  // the node, so Append() (which writes kv->cache) works without mutable.
  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward_fn =
      [kv](ffi::Map<ffi::String, ffi::Any> named_args) -> ffi::Any {
    NNTensor x_tensor = named_args.at("x").cast<NNTensor>();
    // Append x to the cache (emits call_inplace_packed, updates kv->cache)
    kv.get()->Append(x_tensor);
    // View the cache with seq_len=4 (emits call_pure_packed)
    NNTensor view = kv.get()->View(IntImm(DataType::Int(64), 4));
    return ffi::Any(view);
  };

  ffi::Array<ffi::String> arg_names{"x"};
  ffi::Array<ffi::Any> arg_specs{ffi::Any(MakeSpecTensor({2, 4}, "float32"))};

  MethodSpec ms(forward_fn, arg_names, arg_specs, "plain", "plain");

  // named_effects: {"cache": kv}
  ffi::Map<ffi::String, runtime::ObjectRef> named_effects;
  named_effects.Set("cache", kv);

    ModuleSpec mod_spec(ffi::Array<ffi::String>{ffi::String("forward")},
                      ffi::Array<ffi::Any>{ffi::Any(ms)},
                      /*named_params=*/{}, named_effects);

  
  // Create a minimal NNModule wrapper and use ExportTVM
  NNModule mod;
  ffi::Array<ffi::Any> result = mod->ExportTVM(mod_spec, /*debug=*/true, /*allow_extern=*/false);
  IRModule actual = result[0].cast<IRModule>();

  // ---------------------------------------------------------------------------
  // Build expected IRModule
  //
  // Two functions: _initialize_effect and forward.
  // ---------------------------------------------------------------------------
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);

  // ---- _initialize_effect --------------------------------------------------
  // def _initialize_effect() -> R.Tuple(R.Object, R.Object):
  //     with R.dataflow():
  //         _io   = R.null_value()
  //         lv    = R.zeros(R.shape([8, 2, 4]), dtype="float32")
  //         cache = R.call_pure_packed(
  //                     "vm.builtin.attention_kv_cache_create",
  //                     lv, R.shape([8, 2, 4]), R.prim_value(0),
  //                     sinfo_args=[R.Object()])
  //         lv1   = (_io, cache)
  //         gv    = lv1
  //     return gv
  {
    ffi::Array<Var> params;  // no parameters
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // _io = R.null_value()
    static const Op& null_value_op = Op::Get("relax.null_value");
    Var io = bb->Emit(Call(null_value_op, {}, {}, {}), "_io");

    // lv = R.zeros(R.shape([8, 2, 4]), dtype="float32")
    ShapeExpr init_shape(ffi::Array<PrimExpr>{
        IntImm(DataType::Int(64), 8), IntImm(DataType::Int(64), 2), IntImm(DataType::Int(64), 4)});
    Var lv = bb->Emit(relax::zeros(init_shape, DataType::Float(32)), "lv");

    // cache = R.call_pure_packed(
    //     "vm.builtin.attention_kv_cache_create",
    //     lv, R.shape([8, 2, 4]), R.prim_value(0),
    //     sinfo_args=[R.Object()])
    static const Op& cpp_op = Op::Get("relax.call_pure_packed");
    Expr cache_call = Call(cpp_op,
                           {ExternFunc("vm.builtin.attention_kv_cache_create"), lv, init_shape,
                            PrimValue(IntImm(DataType::Int(64), 0))},
                           {}, {ObjectStructInfo()});
    Var cache = bb->Emit(cache_call, "cache");

    // lv1 = (_io, cache)
    Var lv1 = bb->Emit(relax::Tuple({io, cache}), "lv1");
    // gv = lv1
    Var gv = bb->EmitOutput(lv1, "gv");

    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv));
    bb->EndScope();

    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("global_symbol", ffi::Any(ffi::String("_initialize_effect")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "_initialize_effect");
  }

  // ---- forward -------------------------------------------------------------
  // def forward(x: R.Tensor((2,4),"float32"), _io: R.Object, cache: R.Object)
  //     -> R.Tuple(R.Tensor((4,2,4),"float32"), R.Tuple(R.Object, R.Object)):
  //   R.func_attr({"num_input": 3})
  //   with R.dataflow():
  //     kv_cache_append: R.Object = R.call_inplace_packed(
  //         "vm.builtin.attention_kv_cache_append",
  //         cache, x, inplace_indices=[0], sinfo_args=[R.Object()])
  //     kv_cache_view: R.Tensor((4,2,4),"float32") = R.call_pure_packed(
  //         "vm.builtin.attention_kv_cache_view",
  //         kv_cache_append, R.shape([4, 2, 4]),
  //         sinfo_args=(R.Tensor((4,2,4),"float32"),))
  //     gv1 = kv_cache_view, (_io, kv_cache_append)
  //   return gv1
  {
    Var x("x", TSInfo({2, 4}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    Var cache("cache", ObjectStructInfo());
    ffi::Array<Var> params{x, io, cache};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // kv_cache_append = R.call_inplace_packed(
    //     "vm.builtin.attention_kv_cache_append",
    //     cache, x, inplace_indices=[0], sinfo_args=[R.Object()])
    ObjectPtr<CallInplacePackedAttrs> inplace_attrs = ffi::make_object<CallInplacePackedAttrs>();
    inplace_attrs->inplace_indices = {Integer(0)};
    static const Op& inplace_op = Op::Get("relax.call_inplace_packed");
    Expr append_call =
        Call(inplace_op, {ExternFunc("vm.builtin.attention_kv_cache_append"), cache, x},
             Attrs(inplace_attrs), {ObjectStructInfo()});
    Var kv_cache_append = bb->Emit(append_call, "kv_cache_append");

    // kv_cache_view = R.call_pure_packed(
    //     "vm.builtin.attention_kv_cache_view",
    //     kv_cache_append, R.shape([4, 2, 4]),
    //     sinfo_args=(R.Tensor((4,2,4),"float32"),))
    ShapeExpr view_shape(ffi::Array<PrimExpr>{
        IntImm(DataType::Int(64), 4), IntImm(DataType::Int(64), 2), IntImm(DataType::Int(64), 4)});
    TensorStructInfo view_sinfo = TSInfo({4, 2, 4}, DataType::Float(32));
    static const Op& pure_op = Op::Get("relax.call_pure_packed");
    Expr view_call = Call(
        pure_op, {ExternFunc("vm.builtin.attention_kv_cache_view"), kv_cache_append, view_shape},
        {}, {view_sinfo});
    Var kv_cache_view = bb->Emit(view_call, "kv_cache_view");

    // gv1 = (kv_cache_view, (_io, kv_cache_append))
    // Effect output tuple: (_io, kv_cache_append)
    //   - _io:              legacy debug IO effect (unchanged)
    //   - kv_cache_append:  finalized KVCache state (updated cache var)
    Var gv1 =
        bb->EmitOutput(relax::Tuple({kv_cache_view, relax::Tuple({io, kv_cache_append})}), "gv1");

    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();

    // num_input = 3: x + _io + cache
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(3)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("forward")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "forward");
  }

  AssertStructEqual(actual, bb->Finalize());
}

// ---------------------------------------------------------------------------
// MakeAttentionForwardFn
//
// AttentionModuleNode::_forward has the signature:
//   _forward(self, hidden_states: Var,
//            encoder_hidden_states: ffi::Optional<Var>)
//
// The generic MethodSpec(mod_ref, ...) constructor always unwraps each
// named arg as NNTensor → Var and passes it positionally.  That works for
// the first argument but the second must be wrapped in ffi::Optional<Var>.
//
// This factory returns a custom forward_fn that handles the wrapping, so
// the call site can still use the simple MethodSpec(forward_fn, ...) path.
// ---------------------------------------------------------------------------
static ffi::Function MakeAttentionForwardFn(runtime::ObjectRef mod_ref) {
  return ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)>(
      [mod_ref](ffi::Map<ffi::String, ffi::Any> named_args) -> ffi::Any {
        namespace refl = tvm::ffi::reflection;
        ffi::Function fwd = refl::GetMethod(mod_ref->GetTypeKey(), "_forward");
        TVM_FFI_ICHECK(fwd.defined());
        NNTensor hs_t = named_args.at("hidden_states").cast<NNTensor>();
        NNTensor enc_t = named_args.at("encoder_hidden_states").cast<NNTensor>();
        ffi::Optional<Var> opt_enc(enc_t->expr);
        std::vector<ffi::AnyView> args{ffi::AnyView(mod_ref), ffi::AnyView(hs_t->expr),
                                       ffi::AnyView(opt_enc)};
        ffi::Any rv;
        fwd.CallPacked(ffi::PackedArgs(args.data(), args.size()), &rv);
        return rv;
      });
}

// ===========================================================================
// TestAttention
//
// Python equivalent:
//   mod = modules.Attention(
//       query_dim=640, cross_attention_dim=2048, heads=10, norm_num_groups=8)
//   tvm_mod, _ = mod.export_tvm(
//       spec={
//           "forward": {
//               "hidden_states":         spec.Tensor((2, 4096, 640),  "float32"),
//               "encoder_hidden_states": spec.Tensor((2, 77,   2048), "float32"),
//           }
//       },
//       debug=True,
//   )
//   assert_structural_equal(tvm_mod["forward"], forward, True)
//
// Module internals (MakeAttention(640, 2048, 10, 64, false, 8, true)):
//   inner_dim = 64 * 10 = 640
//   cross_dim = 2048
//   head_dim  = 640 / 10 = 64
//
//   to_q       = MakeLinear(640→640,   bias=false) → weight[640,640]
//   to_k       = MakeLinear(2048→640,  bias=false) → weight[640,2048]
//   to_v       = MakeLinear(2048→640,  bias=false) → weight[640,2048]
//   group_norm = MakeGroupNorm(8, 640, 1e-5, affine=true)
//                                                  → weight[640], bias[640]
//   to_out[0]  = MakeLinear(640→640,   bias=true)  → weight[640,640], bias[640]
//
// NamedParameters traversal order (attrs: to_q, to_k, to_v, group_norm, to_out):
//   "to_q_weight"       [640, 640]
//   "to_k_weight"       [640, 2048]
//   "to_v_weight"       [640, 2048]
//   "group_norm_weight" [640]
//   "group_norm_bias"   [640]
//   "to_out_0_weight"   [640, 640]
//   "to_out_0_bias"     [640]
//
// Expected dataflow bindings (C++ binding names — see file-level comment):
//   group_norm    ← group_norm(hidden_states, gn_weight, gn_bias,
//                              num_groups=8, channel_axis=2, axes=[1], eps=1e-5)
//   permute_dims  ← permute_dims(to_q_weight)
//   linear        ← matmul(group_norm, permute_dims)          [to_q, no bias]
//   permute_dims1 ← permute_dims(to_k_weight)
//   linear1       ← matmul(enc, permute_dims1)                [to_k, no bias]
//   permute_dims2 ← permute_dims(to_v_weight)
//   linear2       ← matmul(enc, permute_dims2)                [to_v, no bias]
//   q             ← reshape(linear,  [2, 4096, 10, 64])
//   k             ← reshape(linear1, [2, 77,   10, 64])
//   v             ← reshape(linear2, [2, 77,   10, 64])
//   attn_out      ← attention(q, k, v, None, None, None, None)
//   attn_reshape  ← reshape(attn_out, [2, 4096, 640])
//   permute_dims3 ← permute_dims(to_out_0_weight)
//   matmul        ← matmul(attn_reshape, permute_dims3)       [to_out[0], with bias]
//   linear3       ← add(matmul, to_out_0_bias)
//   gv1           ← (linear3, (_io,))
// ===========================================================================
TEST(NNModules, TestAttention) {
  // Attention(query_dim=640, cross_attention_dim=2048, heads=10, norm_num_groups=8)
  // Python defaults: dim_head=64, bias=False, out_bias=True
  AttentionModule mod = MakeAttention(/*query_dim=*/640,
                                      /*cross_attention_dim=*/ffi::Optional<int64_t>(int64_t(2048)),
                                      /*heads=*/10,
                                      /*dim_head=*/64,
                                      /*bias=*/false,
                                      /*norm_num_groups=*/ffi::Optional<int64_t>(int64_t(8)),
                                      /*out_bias=*/true);

  // Collect named parameters in traversal order.
  ffi::Map<ffi::String, NNParameter> named_params = mod.get()->NamedParameters("");

  // AttentionModuleNode::_forward takes Optional<Var> for encoder_hidden_states,
  // so we supply a custom forward_fn instead of using the generic mod_ref constructor.
  ffi::Array<ffi::String> attn_arg_names{"hidden_states", "encoder_hidden_states"};
  ffi::Array<ffi::Any> attn_arg_specs{ffi::Any(MakeSpecTensor({2, 4096, 640}, "float32")),
                                      ffi::Any(MakeSpecTensor({2, 77, 2048}, "float32"))};
  MethodSpec attn_ms(MakeAttentionForwardFn(mod), attn_arg_names, attn_arg_specs, "plain", "plain");
  ModuleSpec attn_mod_spec(ffi::Array<ffi::String>{ffi::String("forward")},
                           ffi::Array<ffi::Any>{ffi::Any(attn_ms)}, named_params, {});
  ffi::Array<ffi::Any> attn_result =
      mod.get()->ExportTVM(attn_mod_spec, /*debug=*/true, /*allow_extern=*/false);
  IRModule actual = attn_result[0].cast<IRModule>();

  // ---------------------------------------------------------------------------
  // Build expected IR
  //
  // Replicate the exact sequence of bb->Emit() calls that
  // AttentionModuleNode::Forward makes, using the same hint strings so the
  // deduplication suffixes match.
  //
  // Execution trace (see file-level comment for full derivation):
  //
  //   group_norm.Forward(hs, channel_axis=2, axes=[1]):
  //     "group_norm" ← group_norm(hidden_states, gn_weight, gn_bias,
  //                               num_groups=8, channel_axis=2, axes=[1], eps=1e-5)
  //
  //   to_q.Forward(group_norm):   [no bias]
  //     "permute_dims"  ← permute_dims(to_q_weight)
  //     "linear"        ← matmul(group_norm, permute_dims)
  //
  //   enc = encoder_hidden_states
  //
  //   to_k.Forward(enc):          [no bias]
  //     "permute_dims1" ← permute_dims(to_k_weight)   [dedup suffix 1]
  //     "linear1"       ← matmul(enc, permute_dims1)  [hint "linear" → dedup "linear1"]
  //
  //   to_v.Forward(enc):          [no bias]
  //     "permute_dims2" ← permute_dims(to_v_weight)   [dedup suffix 2]
  //     "linear2"       ← matmul(enc, permute_dims2)  [hint "linear" → dedup "linear2"]
  //
  //   reshape_4d(q=linear, "q"):
  //     "q" ← reshape(linear, [2, 4096, 10, 64])
  //
  //   reshape_4d(k=linear1, "k"):
  //     "k" ← reshape(linear1, [2, 77, 10, 64])
  //
  //   reshape_4d(v=linear2, "v"):
  //     "v" ← reshape(linear2, [2, 77, 10, 64])
  //
  //   Emit(attention(q,k,v,...), "attn_out"):
  //     "attn_out" ← attention(q, k, v, None, None, None, None)
  //
  //   Emit(reshape(attn_out, [2,4096,640]), "attn_reshape"):
  //     "attn_reshape" ← reshape(attn_out, [2, 4096, 640])
  //
  //   to_out[0].Forward(attn_reshape):  [with bias]
  //     "permute_dims3" ← permute_dims(to_out_0_weight)       [dedup suffix 3]
  //     "matmul"        ← matmul(attn_reshape, permute_dims3)
  //     "linear3"       ← add(matmul, to_out_0_bias)          [hint "linear" → dedup "linear3"]
  //
  //   Output: (linear3, (_io,))
  // ---------------------------------------------------------------------------
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    DataType f32 = DataType::Float(32);
    auto I64 = [](int64_t v) { return IntImm(DataType::Int(64), v); };

    // ---- Function parameters -----------------------------------------------
    // User inputs (num_input = 3: hidden_states + encoder_hidden_states + _io)
    Var hidden_states("hidden_states", TSInfo({2, 4096, 640}, f32));
    Var encoder_hidden_states("encoder_hidden_states", TSInfo({2, 77, 2048}, f32));
    Var io("_io", ObjectStructInfo());
    // Named parameters in NamedParameters traversal order
    // (attrs insertion: to_q, to_k, to_v, group_norm, to_out)
    Var to_q_weight("to_q_weight", TSInfo({640, 640}, f32));
    Var to_k_weight("to_k_weight", TSInfo({640, 2048}, f32));
    Var to_v_weight("to_v_weight", TSInfo({640, 2048}, f32));
    Var group_norm_weight("group_norm_weight", TSInfo({640}, f32));
    Var group_norm_bias("group_norm_bias", TSInfo({640}, f32));
    Var to_out_0_weight("to_out_0_weight", TSInfo({640, 640}, f32));
    Var to_out_0_bias("to_out_0_bias", TSInfo({640}, f32));

    ffi::Array<Var> params{hidden_states,     encoder_hidden_states, io,
                           to_q_weight,       to_k_weight,           to_v_weight,
                           group_norm_weight, group_norm_bias,       to_out_0_weight,
                           to_out_0_bias};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // ---- group_norm.Forward(hidden_states, channel_axis=2, axes=[1]) ------
    // GroupNormModuleNode::Forward:
    //   Emit(group_norm(hs, weight, bias, num_groups=8,
    //                   channel_axis=2, axes=[1], eps=1e-5), "group_norm")
    ffi::Array<Integer> gn_axes{Integer(1)};
    Var gn_out = bb->Emit(relax::group_norm(hidden_states, group_norm_weight, group_norm_bias,
                                            /*num_groups=*/8, /*channel_axis=*/2, gn_axes,
                                            /*epsilon=*/1e-5, /*center=*/true, /*scale=*/true),
                          "group_norm");

    // ---- to_q.Forward(group_norm) — Linear, no bias -----------------------
    // Emit(matmul(gn_out, permute_dims(to_q_weight)), "linear")
    // BlockBuilder normalises nested exprs:
    //   "permute_dims" ← permute_dims(to_q_weight)
    Var permute_dims = bb->Emit(relax::permute_dims(to_q_weight, std::nullopt), "permute_dims");
    //   "linear" ← matmul(gn_out, permute_dims)   [hint "linear", no bias]
    Var linear_q = bb->Emit(relax::matmul(gn_out, permute_dims, std::nullopt), "linear");

    // ---- enc = encoder_hidden_states (Optional<Var> is populated) ---------

    // ---- to_k.Forward(enc) — Linear, no bias ------------------------------
    // Emit(matmul(enc, permute_dims(to_k_weight)), "linear")
    // BlockBuilder normalises:
    //   "permute_dims1" ← permute_dims(to_k_weight)   [dedup suffix 1]
    Var permute_dims1 = bb->Emit(relax::permute_dims(to_k_weight, std::nullopt), "permute_dims1");
    //   "linear1" ← matmul(enc, permute_dims1)        [hint "linear" → dedup "linear1"]
    Var linear_k =
        bb->Emit(relax::matmul(encoder_hidden_states, permute_dims1, std::nullopt), "linear1");

    // ---- to_v.Forward(enc) — Linear, no bias ------------------------------
    // Emit(matmul(enc, permute_dims(to_v_weight)), "linear")
    // BlockBuilder normalises:
    //   "permute_dims2" ← permute_dims(to_v_weight)   [dedup suffix 2]
    Var permute_dims2 = bb->Emit(relax::permute_dims(to_v_weight, std::nullopt), "permute_dims2");
    //   "linear2" ← matmul(enc, permute_dims2)        [hint "linear" → dedup "linear2"]
    Var linear_v =
        bb->Emit(relax::matmul(encoder_hidden_states, permute_dims2, std::nullopt), "linear2");

    // ---- reshape_4d(q=linear_q, "q") → [2, 4096, 10, 64] -----------------
    // AttentionModuleNode::reshape_4d uses shape [0, -1, heads, head_dim]
    // which the BlockBuilder resolves to [2, 4096, 10, 64] for this input.
    ShapeExpr q_shape(ffi::Array<PrimExpr>{I64(2), I64(4096), I64(10), I64(64)});
    Var q = bb->Emit(relax::reshape(linear_q, q_shape), "q");

    // ---- reshape_4d(k=linear_k, "k") → [2, 77, 10, 64] -------------------
    ShapeExpr k_shape(ffi::Array<PrimExpr>{I64(2), I64(77), I64(10), I64(64)});
    Var k = bb->Emit(relax::reshape(linear_k, k_shape), "k");

    // ---- reshape_4d(v=linear_v, "v") → [2, 77, 10, 64] -------------------
    ShapeExpr v_shape(ffi::Array<PrimExpr>{I64(2), I64(77), I64(10), I64(64)});
    Var v = bb->Emit(relax::reshape(linear_v, v_shape), "v");

    // ---- attention(q, k, v, bias=None, scale=None,
    //                causal_mask=None, window_size=None) → "attn_out" -------
    Var attn_out = bb->Emit(relax::attention(q, k, v,
                                             /*bias=*/std::nullopt,
                                             /*scale=*/std::nullopt,
                                             /*causal_mask=*/std::nullopt,
                                             /*window_size=*/std::nullopt),
                            "attn_out");

    // ---- reshape(attn_out, [2, 4096, 640]) → "attn_reshape" ---------------
    ShapeExpr flat_shape(ffi::Array<PrimExpr>{I64(2), I64(4096), I64(640)});
    Var attn_reshape = bb->Emit(relax::reshape(attn_out, flat_shape), "attn_reshape");

    // ---- to_out[0].Forward(attn_reshape) — Linear, with bias --------------
    // Emit(add(matmul(attn_reshape, permute_dims(to_out_0_weight)), to_out_0_bias), "linear")
    // BlockBuilder normalises:
    //   "permute_dims3" ← permute_dims(to_out_0_weight)       [dedup suffix 3]
    Var permute_dims3 =
        bb->Emit(relax::permute_dims(to_out_0_weight, std::nullopt), "permute_dims3");
    //   "matmul" ← matmul(attn_reshape, permute_dims3)
    Var matmul_out = bb->Emit(relax::matmul(attn_reshape, permute_dims3, std::nullopt), "matmul");
    //   "linear3" ← add(matmul, to_out_0_bias)                [hint "linear" → dedup "linear3"]
    Var linear3_out = bb->Emit(relax::add(matmul_out, to_out_0_bias), "linear3");

    // ---- debug output (linear3, (_io,)) ------------------------------------
    Var gv1 = EmitDebugOutput(bb, linear3_out, io);

    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();

    // num_input = 3: hidden_states + encoder_hidden_states + _io
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(3)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("forward")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "forward");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// TestEmbedding2D
//
// Python equivalent:
//   mod = modules.Embedding(4, 8, "float32")
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor((1, 4), "int32")}}, debug=True)
//
// Expected forward:
//   def forward(x:      R.Tensor((1,4),"int32"),    _io: R.Object,
//               weight: R.Tensor((4,8),"float32"))
//       -> R.Tuple(R.Tensor((1,4,8),"float32"), R.Tuple(R.Object)):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       reshape:  R.Tensor((4,),"int32")       = R.reshape(x, R.shape([4]))
//       take:     R.Tensor((4,8),"float32")    = R.take(weight, reshape, axis=0)
//       embedding:R.Tensor((1,4,8),"float32")  = R.reshape(take, R.shape([1,4,8]))
//       gv1 = embedding, (_io,)
//     return gv1
//
// EmbeddingModuleNode::Forward(Var x, Array<Any> out_shape_if_nd):
//   When out_shape_if_nd is non-empty the ND path is taken:
//     flat  = reshape(x, [-1])                  ← intermediate Expr (not Emit'd directly)
//     taken = take(weight, flat, axis=0)         ← intermediate Expr (not Emit'd directly)
//     return Emit(reshape(taken, out_shape), "embedding")
//   BlockBuilder::Emit normalises the nested Expr and auto-names the
//   intermediate bindings ("reshape", "take") before assigning the
//   caller-supplied hint ("embedding") to the outermost reshape.
//
// For input x=(1,4) and embedding_dim=8:
//   out_shape_if_nd = [1, 4, 8]   (batch × seq × dim)
//
// This is passed as extra_args = [Array<Any>{1, 4, 8}] to ExportDebug.
// ===========================================================================
TEST(NNModules, TestEmbedding2D) {
  // Embedding(num_embeddings=4, embedding_dim=8, dtype="float32")
  EmbeddingModule mod = MakeEmbedding(ffi::Any(int64_t(4)), ffi::Any(int64_t(8)),
                                      ffi::Optional<ffi::String>("float32"));

  ffi::Map<ffi::String, NNParameter> named_params = mod.get()->NamedParameters("");

  // EmbeddingModuleNode::Forward(Var x, ffi::Array<ffi::Any> out_shape_if_nd)
  // For 2-D input (1, 4) with embedding_dim=8:
  //   out_shape_if_nd = [1, 4, 8]  (the full output shape)
  ffi::Array<ffi::Any> out_shape_if_nd;
  out_shape_if_nd.push_back(ffi::Any(int64_t(1)));
  out_shape_if_nd.push_back(ffi::Any(int64_t(4)));
  out_shape_if_nd.push_back(ffi::Any(int64_t(8)));

  ffi::Array<ffi::Any> extra_args;
  extra_args.push_back(ffi::Any(out_shape_if_nd));  // out_shape_if_nd = [1, 4, 8]

  IRModule actual = ExportDebug(mod, "forward", {"x"}, {ffi::Any(MakeSpecTensor({1, 4}, "int32"))},
                                named_params, extra_args);

  // ---------------------------------------------------------------------------
  // Build expected IR
  //
  // The C++ EmbeddingModuleNode::Forward ND path calls:
  //   Expr flat  = relax::reshape(x, ShapeExpr({-1}))
  //   Expr taken = relax::take(weight->expr, flat, 0)
  //   return Emit(relax::reshape(taken, ShapeExpr({1, 4, 8})), "embedding")
  //
  // BlockBuilder::Emit normalises the nested Expr tree and emits each
  // sub-expression as a separate binding before the outermost one:
  //   reshape   ← relax::reshape(x, [-1])          auto-named "reshape"
  //   take      ← relax::take(weight, reshape, 0)   auto-named "take"
  //   embedding ← relax::reshape(take, [1,4,8])     hint "embedding"
  // ---------------------------------------------------------------------------
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    // Function parameters: x, _io, weight
    Var x("x", TSInfo({1, 4}, DataType::Int(32)));
    Var io("_io", ObjectStructInfo());
    // weight: [num_embeddings, embedding_dim] = [4, 8]
    Var weight("weight", TSInfo({4, 8}, DataType::Float(32)));
    ffi::Array<Var> params{x, io, weight};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // Step 1: reshape(x, [4])  — flatten the 2-D input to 1-D.
    // The C++ code uses ShapeExpr({-1}) which BlockBuilder resolves to the
    // concrete flat size (4) given the static input shape (1, 4).
    ShapeExpr flat_shape(ffi::Array<PrimExpr>{IntImm(DataType::Int(64), 4)});
    Var reshape_var = bb->Emit(relax::reshape(x, flat_shape), "reshape");

    // Step 2: take(weight, reshape, axis=0)  — gather embedding rows.
    Var take_var = bb->Emit(relax::take(weight, reshape_var, ffi::Optional<int64_t>(0)), "take");

    // Step 3: reshape(take, [1, 4, 8])  — restore the batch dimension.
    ShapeExpr out_shape(ffi::Array<PrimExpr>{
        IntImm(DataType::Int(64), 1), IntImm(DataType::Int(64), 4), IntImm(DataType::Int(64), 8)});
    Var embedding_var = bb->Emit(relax::reshape(take_var, out_shape), "embedding");

    // Debug output: (embedding, (_io,))
    Var gv1 = EmitDebugOutput(bb, embedding_var, io);

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
// MakeTimestepEmbeddingForwardFn
//
// TimestepEmbeddingModuleNode::_forward has the signature:
//   _forward(self, sample: Var, condition: ffi::Optional<Var>)
//
// Same situation as Attention: the second arg must be Optional<Var>, so we
// provide a custom forward_fn factory rather than using the generic
// MethodSpec(mod_ref, ...) constructor.
// ---------------------------------------------------------------------------
static ffi::Function MakeTimestepEmbeddingForwardFn(runtime::ObjectRef mod_ref) {
  return ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)>(
      [mod_ref](ffi::Map<ffi::String, ffi::Any> named_args) -> ffi::Any {
        namespace refl = tvm::ffi::reflection;
        ffi::Function fwd = refl::GetMethod(mod_ref->GetTypeKey(), "_forward");
        TVM_FFI_ICHECK(fwd.defined());
        NNTensor sample_t = named_args.at("sample").cast<NNTensor>();
        NNTensor cond_t = named_args.at("condition").cast<NNTensor>();
        ffi::Optional<Var> opt_cond(cond_t->expr);
        std::vector<ffi::AnyView> args{ffi::AnyView(mod_ref), ffi::AnyView(sample_t->expr),
                                       ffi::AnyView(opt_cond)};
        ffi::Any rv;
        fwd.CallPacked(ffi::PackedArgs(args.data(), args.size()), &rv);
        return rv;
      });
}

// ===========================================================================
// TestTimestepEmbedding
//
// Python equivalent:
//   mod = modules.TimestepEmbedding(32, 32, cond_proj_dim=16)
//   tvm_mod, _ = mod.export_tvm(
//       spec={
//           "forward": {
//               "sample":    spec.Tensor((32, 32), "float32"),
//               "condition": spec.Tensor((32, 16), "float32"),
//           }
//       },
//       debug=True,
//   )
//
// Module internals (MakeTimestepEmbedding(32, 32, "silu", nullopt, nullopt, 16)):
//   linear_1  = MakeLinear(in=32, out=32, bias=true)  → weight[32,32], bias[32]
//   cond_proj = MakeLinear(in=16, out=32, bias=false)  → weight[32,16]
//   act       = SiLU
//   linear_2  = MakeLinear(in=32, out=32, bias=true)  → weight[32,32], bias[32]
//
// NamedParameters traversal order (attrs insertion: linear_1, cond_proj, linear_2):
//   "linear_1.weight"  → [32, 32]
//   "linear_1.bias"    → [32]
//   "cond_proj.weight" → [32, 16]
//   "linear_2.weight"  → [32, 32]
//   "linear_2.bias"    → [32]
//
// Expected function signature:
//   forward(sample:           R.Tensor((32,32),"float32"),
//           condition:        R.Tensor((32,16),"float32"),
//           _io:              R.Object,
//           linear_1_weight:  R.Tensor((32,32),"float32"),
//           linear_1_bias:    R.Tensor((32,),"float32"),
//           cond_proj_weight: R.Tensor((32,16),"float32"),
//           linear_2_weight:  R.Tensor((32,32),"float32"),
//           linear_2_bias:    R.Tensor((32,),"float32"))
//   num_input = 3  (sample + condition + _io)
//
// Expected dataflow bindings (C++ binding names — see file-level comment):
//   permute_dims  ← permute_dims(cond_proj_weight)          [auto]
//   linear        ← matmul(condition, permute_dims)          [hint "linear", no bias]
//   cond_add      ← add(sample, linear)                      [hint "cond_add"]
//   permute_dims1 ← permute_dims(linear_1_weight)            [auto, dedup]
//   matmul        ← matmul(cond_add, permute_dims1)          [auto]
//   linear1       ← add(matmul, linear_1_bias)               [hint "linear" → dedup "linear1"]
//   silu          ← silu(linear1)                            [hint "silu"]
//   permute_dims2 ← permute_dims(linear_2_weight)            [auto, dedup]
//   matmul1       ← matmul(silu, permute_dims2)              [auto, dedup]
//   linear2       ← add(matmul1, linear_2_bias)              [hint "linear" → dedup "linear2"]
//   gv1           ← (linear2, (_io,))                        [output]
// ===========================================================================
TEST(NNModules, TestTimestepEmbedding) {
  // TimestepEmbedding(in_channels=32, time_embed_dim=32, cond_proj_dim=16)
  // Python: modules.TimestepEmbedding(32, 32, cond_proj_dim=16)
  // C++ defaults: act_fn="silu", out_dim=nullopt, post_act_fn=nullopt
  TimestepEmbeddingModule mod =
      MakeTimestepEmbedding(/*in_channels=*/32, /*time_embed_dim=*/32,
                            /*act_fn=*/"silu",
                            /*out_dim=*/std::nullopt,
                            /*post_act_fn=*/std::nullopt,
                            /*cond_proj_dim=*/ffi::Optional<int64_t>(int64_t(16)));

  // Collect named parameters in traversal order:
  //   linear_1.weight [32,32], linear_1.bias [32],
  //   cond_proj.weight [32,16],
  //   linear_2.weight [32,32], linear_2.bias [32]
  ffi::Map<ffi::String, NNParameter> named_params = mod.get()->NamedParameters("");

  // TimestepEmbeddingModuleNode::_forward takes Optional<Var> for condition,
  // so we supply a custom forward_fn instead of using the generic mod_ref constructor.
  ffi::Array<ffi::String> tse_arg_names{"sample", "condition"};
  ffi::Array<ffi::Any> tse_arg_specs{ffi::Any(MakeSpecTensor({32, 32}, "float32")),
                                     ffi::Any(MakeSpecTensor({32, 16}, "float32"))};
  MethodSpec tse_ms(MakeTimestepEmbeddingForwardFn(mod), tse_arg_names, tse_arg_specs, "plain",
                    "plain");
  ModuleSpec tse_mod_spec(ffi::Array<ffi::String>{ffi::String("forward")},
                          ffi::Array<ffi::Any>{ffi::Any(tse_ms)}, named_params, {});
  ffi::Array<ffi::Any> tse_result =
      mod.get()->ExportTVM(tse_mod_spec, /*debug=*/true, /*allow_extern=*/false);
  IRModule actual = tse_result[0].cast<IRModule>();

  // ---------------------------------------------------------------------------
  // Build expected IR
  //
  // The C++ TimestepEmbeddingModuleNode::Forward execution trace:
  //
  //   1. cond_proj.Forward(condition):
  //        LinearModuleNode::Forward, no bias:
  //          Emit(matmul(condition, permute_dims(cond_proj_weight)), "linear")
  //          → BlockBuilder normalises nested exprs:
  //              "permute_dims" ← permute_dims(cond_proj_weight)
  //              "linear"       ← matmul(condition, permute_dims)
  //
  //   2. Emit(add(sample, linear), "cond_add"):
  //              "cond_add" ← add(sample, linear)
  //
  //   3. linear_1.Forward(cond_add):
  //        LinearModuleNode::Forward, with bias:
  //          Emit(add(matmul(cond_add, permute_dims(linear_1_weight)), linear_1_bias), "linear")
  //          → BlockBuilder normalises:
  //              "permute_dims1" ← permute_dims(linear_1_weight)  [dedup suffix 1]
  //              "matmul"        ← matmul(cond_add, permute_dims1)
  //              "linear1"       ← add(matmul, linear_1_bias)     [hint "linear" → dedup "linear1"]
  //
  //   4. act.Forward(linear1):
  //              "silu" ← silu(linear1)
  //
  //   5. linear_2.Forward(silu):
  //        LinearModuleNode::Forward, with bias:
  //          Emit(add(matmul(silu, permute_dims(linear_2_weight)), linear_2_bias), "linear")
  //          → BlockBuilder normalises:
  //              "permute_dims2" ← permute_dims(linear_2_weight)  [dedup suffix 2]
  //              "matmul1"       ← matmul(silu, permute_dims2)    [dedup suffix 1]
  //              "linear2"       ← add(matmul1, linear_2_bias)    [hint "linear" → dedup "linear2"]
  //
  //   6. Output: (linear2, (_io,))
  // ---------------------------------------------------------------------------
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    // ---- Function parameters -----------------------------------------------
    // User inputs (num_input = 3: sample + condition + _io)
    Var sample("sample", TSInfo({32, 32}, DataType::Float(32)));
    Var condition("condition", TSInfo({32, 16}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    // Named parameters in NamedParameters traversal order
    Var linear_1_weight("linear_1_weight", TSInfo({32, 32}, DataType::Float(32)));
    Var linear_1_bias("linear_1_bias", TSInfo({32}, DataType::Float(32)));
    Var cond_proj_weight("cond_proj_weight", TSInfo({32, 16}, DataType::Float(32)));
    Var linear_2_weight("linear_2_weight", TSInfo({32, 32}, DataType::Float(32)));
    Var linear_2_bias("linear_2_bias", TSInfo({32}, DataType::Float(32)));

    ffi::Array<Var> params{sample,          condition,     io,
                           linear_1_weight, linear_1_bias, cond_proj_weight,
                           linear_2_weight, linear_2_bias};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // ---- Step 1: cond_proj.Forward(condition) — Linear, no bias -----------
    // Emit(matmul(condition, permute_dims(cond_proj_weight)), "linear")
    // BlockBuilder normalises the nested expr tree:
    //   "permute_dims" ← permute_dims(cond_proj_weight, axes=None)
    Var permute_dims =
        bb->Emit(relax::permute_dims(cond_proj_weight, std::nullopt), "permute_dims");
    //   "linear" ← matmul(condition, permute_dims)   [hint "linear", no bias]
    Var linear_out = bb->Emit(relax::matmul(condition, permute_dims, std::nullopt), "linear");

    // ---- Step 2: Emit(add(sample, linear), "cond_add") --------------------
    Var cond_add = bb->Emit(relax::add(sample, linear_out), "cond_add");

    // ---- Step 3: linear_1.Forward(cond_add) — Linear, with bias -----------
    // Emit(add(matmul(cond_add, permute_dims(linear_1_weight)), linear_1_bias), "linear")
    // BlockBuilder normalises:
    //   "permute_dims1" ← permute_dims(linear_1_weight)   [dedup: suffix 1]
    Var permute_dims1 =
        bb->Emit(relax::permute_dims(linear_1_weight, std::nullopt), "permute_dims1");
    //   "matmul" ← matmul(cond_add, permute_dims1)
    Var matmul_out = bb->Emit(relax::matmul(cond_add, permute_dims1, std::nullopt), "matmul");
    //   "linear1" ← add(matmul, linear_1_bias)   [hint "linear" → dedup "linear1"]
    Var linear1_out = bb->Emit(relax::add(matmul_out, linear_1_bias), "linear1");

    // ---- Step 4: act.Forward(linear1) — SiLU -----------------------------
    Var silu_out = bb->Emit(relax::silu(linear1_out), "silu");

    // ---- Step 5: linear_2.Forward(silu) — Linear, with bias ---------------
    // Emit(add(matmul(silu, permute_dims(linear_2_weight)), linear_2_bias), "linear")
    // BlockBuilder normalises:
    //   "permute_dims2" ← permute_dims(linear_2_weight)   [dedup: suffix 2]
    Var permute_dims2 =
        bb->Emit(relax::permute_dims(linear_2_weight, std::nullopt), "permute_dims2");
    //   "matmul1" ← matmul(silu, permute_dims2)           [dedup: suffix 1]
    Var matmul1_out = bb->Emit(relax::matmul(silu_out, permute_dims2, std::nullopt), "matmul1");
    //   "linear2" ← add(matmul1, linear_2_bias)           [hint "linear" → dedup "linear2"]
    Var linear2_out = bb->Emit(relax::add(matmul1_out, linear_2_bias), "linear2");

    // ---- Step 6: debug output (linear2, (_io,)) ---------------------------
    Var gv1 = EmitDebugOutput(bb, linear2_out, io);

    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();

    // num_input = 3: sample + condition + _io
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(3)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("forward")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "forward");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ---------------------------------------------------------------------------
// MakeF32Const
//
// Build a scalar float32 constant: mirrors R.const(v, "float32").
// Matches the MakeF32Const helper used inside NNGetTimestepEmbedding in op.cc.
// ---------------------------------------------------------------------------
static Expr MakeF32Const(float v) {
  auto tensor = runtime::Tensor::Empty(ffi::Shape({}), DLDataType{kDLFloat, 32, 1},
                                       DLDevice{kDLCPU, 0}, std::nullopt);
  tensor.CopyFromBytes(&v, sizeof(float));
  return relax::Constant(tensor, std::nullopt);
}

// ===========================================================================
// TestTimesteps
//
// Python equivalent:
//   mod = modules.Timesteps(10)
//   tvm_mod, _ = mod.export_tvm(
//       spec={"forward": {"x": spec.Tensor((3,), "float32")}}, debug=True)
//
// Module internals:
//   TimestepsModule(num_channels=10, flip_sin_to_cos=false,
//                   downscale_freq_shift=1.0)
//   No trainable parameters.
//
// TimestepsModuleNode::Forward calls NNGetTimestepEmbedding with:
//   embedding_dim=10, flip_sin_to_cos=false, downscale_freq_shift=1.0,
//   scale=1.0, max_period=10000, out_dtype="float32",
//   name="get_timestep_embedding"
//
// NNGetTimestepEmbedding emits the following bindings
// (half_dim=5, log_val=-log(10000), denom=4.0):
//
//   "timesteps"  ← astype(x, float32)
//   "timesteps1" ← expand_dims(timesteps, [1])
//   "arange"     ← arange(0, 5, 1, float32)
//   "exponent"   ← multiply(const(-log(10000)), arange)
//   "exponent1"  ← divide(exponent, const(4.0))
//   "emb"        ← exp(exponent1)
//   "emb1"       ← expand_dims(emb, [0])
//   "emb2"       ← multiply(timesteps1, emb1)
//   (scale==1.0 → no scale multiply emitted)
//   "sin"        ← sin(emb2)
//   "cos"        ← cos(emb2)
//   (flip_sin_to_cos=false → concat([sin, cos], axis=-1))
//   "emb3"       ← concat((sin, cos), axis=-1)
//   (embedding_dim=10 is even → no pad)
//   "get_timestep_embedding" ← astype(emb3, float32)
//
// Function signature:
//   forward(x: R.Tensor((3,),"float32"), _io: R.Object)
//   num_input = 2
// ===========================================================================
TEST(NNModules, TestTimesteps) {
  // Timesteps(num_channels=10)
  // Python defaults: flip_sin_to_cos=False, downscale_freq_shift=1
  TimestepsModule mod(/*num_channels=*/10, /*flip_sin_to_cos=*/false,
                      /*downscale_freq_shift=*/1.0);

  // Timesteps has no trainable parameters.
  ffi::Map<ffi::String, NNParameter> named_params = mod.get()->NamedParameters("");
  EXPECT_TRUE(named_params.empty());

  IRModule actual = ExportDebug(mod, "forward", {"x"}, {ffi::Any(MakeSpecTensor({3}, "float32"))});

  // ---------------------------------------------------------------------------
  // Build expected IR
  //
  // Replicate the exact sequence of bb->Emit() calls that NNGetTimestepEmbedding
  // makes, using the same hint strings so the deduplication suffixes match.
  //
  // Parameters:
  //   embedding_dim        = 10
  //   flip_sin_to_cos      = false
  //   downscale_freq_shift = 1.0
  //   scale                = 1.0   (no scale multiply)
  //   max_period           = 10000
  //   out_dtype            = "float32"
  //
  // Derived:
  //   half_dim  = 10 / 2 = 5
  //   log_val   = -log(10000)
  //   denom     = 5 - 1.0 = 4.0
  // ---------------------------------------------------------------------------
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    DataType f32 = DataType::Float(32);
    auto I64 = [](int64_t v) { return IntImm(DataType::Int(64), v); };

    // Numeric constants matching NNGetTimestepEmbedding exactly.
    const int64_t half_dim = 5;                 // embedding_dim / 2
    const double log_val = -std::log(10000.0);  // -log(max_period)
    const double denom = half_dim - 1.0;        // half_dim - downscale_freq_shift

    Var x("x", TSInfo({3}, f32));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // "timesteps" ← astype(x, float32)
    // Always emitted even when x is already float32.
    Var timesteps = bb->Emit(relax::astype(x, f32), "timesteps");

    // "timesteps1" ← expand_dims(timesteps, [1])
    // Dedup: "timesteps" already used → suffix 1.
    Var timesteps1 = bb->Emit(relax::expand_dims(timesteps, {1}), "timesteps1");

    // "arange" ← arange(0, half_dim, 1, float32)
    Var arange_v =
        bb->Emit(relax::arange(PrimValue(I64(0)), PrimValue(I64(half_dim)), PrimValue(I64(1)), f32),
                 "arange");

    // "exponent" ← multiply(const(log_val), arange)
    Var exponent =
        bb->Emit(relax::multiply(MakeF32Const(static_cast<float>(log_val)), arange_v), "exponent");

    // "exponent1" ← divide(exponent, const(denom))
    // Dedup: "exponent" already used → suffix 1.
    Var exponent1 =
        bb->Emit(relax::divide(exponent, MakeF32Const(static_cast<float>(denom))), "exponent1");

    // "emb" ← exp(exponent1)
    Var emb = bb->Emit(relax::exp(exponent1), "emb");

    // "emb1" ← expand_dims(emb, [0])
    // Dedup: "emb" already used → suffix 1.
    Var emb1 = bb->Emit(relax::expand_dims(emb, {0}), "emb1");

    // "emb2" ← multiply(timesteps1, emb1)
    // Dedup: "emb" already used twice → suffix 2.
    Var emb2 = bb->Emit(relax::multiply(timesteps1, emb1), "emb2");

    // scale == 1.0 → NNGetTimestepEmbedding skips the scale multiply entirely.

    // "sin" ← sin(emb2)
    Var sin_emb = bb->Emit(relax::sin(emb2), "sin");

    // "cos" ← cos(emb2)
    Var cos_emb = bb->Emit(relax::cos(emb2), "cos");

    // flip_sin_to_cos=false → concat([sin, cos], axis=-1)
    // "emb3" ← concat((sin, cos), axis=-1)
    // Dedup: "emb" already used three times → suffix 3.
    Var emb3 = bb->Emit(relax::concat(relax::Tuple({sin_emb, cos_emb}), ffi::Optional<int64_t>(-1)),
                        "emb3");

    // embedding_dim=10 is even → NNGetTimestepEmbedding skips the pad.

    // "get_timestep_embedding" ← astype(emb3, float32)
    // out_dtype="float32" matches x's dtype, but the op always emits the cast.
    // The hint "get_timestep_embedding" is passed by TimestepsModuleNode::Forward.
    Var out = bb->Emit(relax::astype(emb3, f32), "get_timestep_embedding");

    Var gv1 = EmitDebugOutput(bb, out, io);

    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();

    // num_input = 2: x + _io
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("forward")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "forward");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ---------------------------------------------------------------------------
// ExportTupleInputDebug
//
// Export helper for modules whose single input "x" is a SpecTuple.
// ---------------------------------------------------------------------------
static IRModule ExportTupleInputDebug(
    std::function<ffi::Array<ffi::Any>(NNTensor, NNTensor)> body_fn, SpecTensor elem_spec,
    bool is_tuple) {
  ffi::Array<ffi::Any> elements{ffi::Any(elem_spec), ffi::Any(elem_spec)};
  SpecTuple x_spec("x", elements, is_tuple);

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward_fn =
      [body_fn](ffi::Map<ffi::String, ffi::Any> named_args) -> ffi::Any {
    ffi::Array<ffi::Any> x_arr = named_args.at("x").cast<ffi::Array<ffi::Any>>();
    NNTensor x0 = x_arr[0].cast<NNTensor>();
    NNTensor x1 = x_arr[1].cast<NNTensor>();
    ffi::Array<ffi::Any> result = body_fn(x0, x1);
    return ffi::Any(result);
  };

  ffi::Array<ffi::String> arg_names{"x"};
  ffi::Array<ffi::Any> arg_specs{ffi::Any(x_spec)};

    MethodSpec ms(forward_fn, arg_names, arg_specs, "plain", "plain");
  ModuleSpec mod_spec(ffi::Array<ffi::String>{ffi::String("forward")},
                                            ffi::Array<ffi::Any>{ffi::Any(ms)},
                      /*named_params=*/{},
                      /*named_effects=*/{});
  
  // Create a minimal NNModule wrapper and use ExportTVM
  NNModule mod;
  ffi::Array<ffi::Any> result = mod->ExportTVM(mod_spec, /*debug=*/true, /*allow_extern=*/false);
  return result[0].cast<IRModule>();
}

// ===========================================================================
// TestNNModuleTupleInput
// ===========================================================================
TEST(NNModules, TestNNModuleTupleInput) {
  DataType f32 = DataType::Float(32);
  SpecTensor elem_spec = MakeSpecTensor({10, 5}, "float32");

  auto body_fn = [](NNTensor x0, NNTensor x1) -> ffi::Array<ffi::Any> {
    BlockBuilder bb = BlockBuilder_Current();
    TVM_FFI_ICHECK(bb.defined());
    Var add_out = bb->Emit(relax::add(x0->expr, x1->expr), "add");
    Var sub_out = bb->Emit(relax::subtract(x0->expr, x1->expr), "subtract");
    return {ffi::Any(NNTensor(add_out)), ffi::Any(NNTensor(sub_out))};
  };

  IRModule actual = ExportTupleInputDebug(body_fn, elem_spec, /*is_tuple=*/true);

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    TupleStructInfo x_sinfo(ffi::Array<StructInfo>{TSInfo({10, 5}, f32), TSInfo({10, 5}, f32)});
    Var x("x", x_sinfo);
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    Var x_0 = bb->Emit(TupleGetItem(x, 0), "x_0");
    Var x_1 = bb->Emit(TupleGetItem(x, 1), "x_1");
    Var add_out = bb->Emit(relax::add(x_0, x_1), "add");
    Var sub_out = bb->Emit(relax::subtract(x_0, x_1), "subtract");
    Var gv1 =
        bb->EmitOutput(relax::Tuple({relax::Tuple({add_out, sub_out}), relax::Tuple({io})}), "gv1");

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

// ===========================================================================
// TestNNModuleListInput
// ===========================================================================
TEST(NNModules, TestNNModuleListInput) {
  DataType f32 = DataType::Float(32);
  SpecTensor elem_spec = MakeSpecTensor({10, 5}, "float32");

  auto body_fn = [](NNTensor x0, NNTensor x1) -> ffi::Array<ffi::Any> {
    BlockBuilder bb = BlockBuilder_Current();
    TVM_FFI_ICHECK(bb.defined());
    Var add_out = bb->Emit(relax::add(x0->expr, x1->expr), "add");
    Var sub_out = bb->Emit(relax::subtract(x0->expr, x1->expr), "subtract");
    return {ffi::Any(NNTensor(add_out)), ffi::Any(NNTensor(sub_out))};
  };

  IRModule actual = ExportTupleInputDebug(body_fn, elem_spec, /*is_tuple=*/false);

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    TupleStructInfo x_sinfo(ffi::Array<StructInfo>{TSInfo({10, 5}, f32), TSInfo({10, 5}, f32)});
    Var x("x", x_sinfo);
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    Var x_0 = bb->Emit(TupleGetItem(x, 0), "x_0");
    Var x_1 = bb->Emit(TupleGetItem(x, 1), "x_1");
    Var add_out = bb->Emit(relax::add(x_0, x_1), "add");
    Var sub_out = bb->Emit(relax::subtract(x_0, x_1), "subtract");
    Var gv1 =
        bb->EmitOutput(relax::Tuple({relax::Tuple({add_out, sub_out}), relax::Tuple({io})}), "gv1");

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

// ===========================================================================
// TestModuleList
// ===========================================================================
TEST(NNModules, TestModuleList) {
  LinearModule l0 = MakeLinear(ffi::Any(int64_t(4)), ffi::Any(int64_t(4)),
                               /*bias=*/false, std::nullopt, std::nullopt);
  LinearModule l1 = MakeLinear(ffi::Any(int64_t(4)), ffi::Any(int64_t(4)),
                               /*bias=*/false, std::nullopt, std::nullopt);
  ModuleList inner({ffi::Any(l0), ffi::Any(l1)});
  ModuleList outer({ffi::Any(inner)});

  NNModule mod;
  mod->attrs.Set("layers", ffi::Any(outer));

  ffi::Map<ffi::String, NNParameter> named_params = mod->NamedParameters("");

  std::vector<std::string> keys;
  for (const auto& [k, _] : named_params) keys.push_back(std::string(k));
  std::sort(keys.begin(), keys.end());

  ASSERT_EQ(keys.size(), 2u);
  EXPECT_EQ(keys[0], "layers.0.0.weight");
  EXPECT_EQ(keys[1], "layers.0.1.weight");
}

// ===========================================================================
// TestModuleDict
// ===========================================================================
TEST(NNModules, TestModuleDict) {
  LinearModule ld0 = MakeLinear(ffi::Any(int64_t(4)), ffi::Any(int64_t(4)),
                                /*bias=*/false, std::nullopt, std::nullopt);
  LinearModule ld1 = MakeLinear(ffi::Any(int64_t(4)), ffi::Any(int64_t(4)),
                                /*bias=*/false, std::nullopt, std::nullopt);

  ffi::Map<ffi::String, ffi::Any> dict_map;
  dict_map.Set("linear0", ffi::Any(ld0));
  dict_map.Set("linear1", ffi::Any(ld1));
  ModuleDict layers(dict_map);

  NNModule mod;
  mod->attrs.Set("layers", ffi::Any(layers));

  ffi::Map<ffi::String, NNParameter> named_params = mod->NamedParameters("");

  std::vector<std::string> keys;
  for (const auto& [k, _] : named_params) keys.push_back(std::string(k));
  std::sort(keys.begin(), keys.end());

  ASSERT_EQ(keys.size(), 2u);
  EXPECT_EQ(keys[0], "layers.linear0.weight");
  EXPECT_EQ(keys[1], "layers.linear1.weight");
}

}  // namespace testing
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
