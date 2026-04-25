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
 * \file tests/cpp/relax/frontend/nn/test_nn_modules_nn.cc
 * \brief C++ port of selected tests from
 *        tests/python/relax/test_frontend_nn_modules.py
 *
 * Ported tests in this file:
 *   - TestLinear          (test_linear)
 *   - TestConv1D          (test_conv1d)
 *   - TestConv1DTranspose (test_conv1d_transpose)
 *   - TestLayerNorm       (test_layer_norm)
 *   - TestConv2D          (test_conv2d)
 *   - TestConv3D          (test_conv3d)
 *   - TestConv2DDynamic   (test_conv2d_dynamic)
 *   - TestRMSNorm         (test_rms_norm)
 *   - TestGroupNorm       (test_group_norm)
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
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

}  // namespace testing
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
