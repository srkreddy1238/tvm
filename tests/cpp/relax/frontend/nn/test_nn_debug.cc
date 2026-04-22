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
 * \file tests/cpp/relax/frontend/nn/test_nn_debug.cc
 * \brief C++ port of tests/python/relax/test_frontend_nn_debug.py
 *
 * Tests the debug-mode IR emission for:
 *   - op.print_  (TestDebugPrint)
 *   - op.debug_func  (TestDebugFunc)
 *
 * Both tests export a single-method module with debug=true and verify the
 * resulting IRModule structure using ffi::StructuralEqual.
 */

#include <gtest/gtest.h>
#include <tvm/driver/compile.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/tir/expr.h>

#include <atomic>

// nn frontend headers
#include "../../../../../src/relax/frontend/nn/core.h"
#include "../../../../../src/relax/frontend/nn/cpp_module.h"
#include "../../../../../src/relax/frontend/nn/exporter.h"
#include "../../../../../src/relax/frontend/nn/spec.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace testing {

// ===========================================================================
// Shared helpers
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

// ---------------------------------------------------------------------------
// TestModule: a concrete NNModuleNode subclass used by all tests.
//
// Holds a forward ffi::Function and a MethodSpec so that ExportTVM / Jit
// can be called directly on the module, mirroring Python's Module.jit().
//
// Usage:
//   TestModule mod(forward_fn, arg_names, arg_specs);
//   IRModule ir  = mod.ExportTVM(mod.MakeSpec("forward"), /*debug=*/true);
//   CppModule vm = mod.JitCpp("forward", {kDLCPU,0}, "cpu_generic", true);
// ---------------------------------------------------------------------------
class TestModuleNode : public NNModuleNode {
 public:
  ffi::Function forward_fn;
  ffi::Array<ffi::String> arg_names;
  ffi::Array<ffi::Any> arg_specs;

  TestModuleNode(ffi::Function forward_fn, ffi::Array<ffi::String> arg_names,
                 ffi::Array<ffi::Any> arg_specs)
      : forward_fn(std::move(forward_fn)),
        arg_names(std::move(arg_names)),
        arg_specs(std::move(arg_specs)) {}

  // Build a ModuleSpec for the given method name.
  ModuleSpec MakeSpec(const std::string& method_name,
                      ffi::Map<ffi::String, NNParameter> named_params = {},
                      ffi::Map<ffi::String, runtime::ObjectRef> named_effects = {}) const {
    MethodSpec ms(forward_fn, arg_names, arg_specs, "plain", "plain");
    return ModuleSpec(ffi::Array<ffi::String>{ffi::String(method_name)},
                      ffi::Array<ffi::Any>{ffi::Any(ms)}, named_params, named_effects);
  }

  // Convenience: ExportTVM with debug=true.
  IRModule ExportDebug(const std::string& method_name = "forward") const {
    return ExportTVM(MakeSpec(method_name), /*debug=*/true);
  }

  // Convenience: Jit to CppModule.
  CppModule JitCpp(const std::string& method_name = "forward", tvm::Device device = {kDLCPU, 0},
                   ffi::String pipeline = "cpu_generic", bool debug = false) const {
    auto result = NNModuleNode::Jit(MakeSpec(method_name), device, pipeline, debug);
    return Downcast<CppModule>(result);
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.nn.TestModule", TestModuleNode, NNModuleNode);
};

// ---------------------------------------------------------------------------
// ExportSingle: thin wrapper kept for the IR-structure tests that build
// expected IR manually and compare with AssertStructEqual.
// Now delegates to TestModuleNode::ExportDebug.
// ---------------------------------------------------------------------------
static IRModule ExportSingle(const std::string& method_name, ffi::Function forward_fn,
                             ffi::Array<ffi::String> arg_names, ffi::Array<ffi::Any> arg_specs) {
  TestModuleNode mod(forward_fn, arg_names, arg_specs);
  return mod.ExportTVM(mod.MakeSpec(method_name), /*debug=*/true);
}

// Build the _initialize_effect function and add it to bb (public, is_pure=true).
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

// Emit the debug-mode output tuple: (result, (_io,)).
static Var EmitDebugOutput(BlockBuilder& bb, Expr result, Var io, const std::string& hint = "gv1") {
  return bb->EmitOutput(relax::Tuple({result, relax::Tuple({io})}), hint);
}

// Assert structural equality between actual and expected IRModules.
static void AssertStructEqual(const IRModule& actual, const IRModule& expected) {
  EXPECT_TRUE(ffi::StructuralEqual()(actual, expected)) << "\n=== Actual ===\n"
                                                        << actual << "\n=== Expected ===\n"
                                                        << expected;
}

// ===========================================================================
// TestDebugPrint
//
// Python equivalent:
//   class Layer(nn.Module):
//       def forward(self, x: nn.Tensor):
//           op.print_(x)
//           return x
//
// op.print_(x) calls:
//   debug_func("vm.builtin.debug_print", x, _line_info="<file>:<line>")
//
// Which emits into the dataflow block:
//   _io = call_pure_packed(ExternFunc("vm.builtin.debug_print"),
//                          StringImm("<file>:<line>"), x)
//   gv1 = (x, (_io,))
//
// The line_info string is not known at compile time, so we use a sentinel
// constant "__line_info__" in the forward closure and verify the rest of
// the IR structure matches exactly.
// ===========================================================================
TEST(NNDebug, TestDebugPrint) {
  static const ffi::Function op_debug_func = NNOp("debug_func");

  // Use a fixed line_info sentinel so the expected IR can be constructed.
  const ffi::String kLineInfo = "test_nn_debug.cc:0";

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [kLineInfo](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    // Retrieve the current _io var from the C++ thread-local.
    auto get_io = ffi::Function::GetGlobal("relax.frontend.nn.GetCurrentIOVar");
    TVM_FFI_ICHECK(get_io.has_value());
    Var io_var = (*get_io)().cast<Var>();
    // Emit: _io = call_pure_packed(ExternFunc("vm.builtin.debug_print"),
    //                              StringImm(line_info), x)
    op_debug_func(ffi::String("vm.builtin.debug_print"), ffi::Array<ffi::Any>{ffi::Any(x->expr)},
                  io_var, kLineInfo);
    return ffi::Any(x->expr);
  };

  IRModule actual =
      ExportSingle("forward", forward, {"x"}, {ffi::Any(MakeSpecTensor({10, 5}, "float32"))});

  // ---- Build expected IR ----
  //
  // @I.ir_module
  // class Module:
  //   @R.function
  //   def _initialize_effect() -> R.Tuple(R.Object): ...
  //
  //   @R.function
  //   def forward(x: R.Tensor((10,5),"float32"), _io: R.Object)
  //       -> R.Tuple(R.Tensor((10,5),"float32"), R.Tuple(R.Object)):
  //     R.func_attr({"num_input": 2})
  //     with R.dataflow():
  //       _io1 = R.call_pure_packed(
  //                  "vm.builtin.debug_print",
  //                  R.str("test_nn_debug.cc:0"),
  //                  x,
  //                  sinfo_args=[R.Object])
  //       gv1 = (x, (_io1,))
  //       R.output(gv1)
  //     return gv1

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    Var x("x", TSInfo({10, 5}, DataType::Float(32)));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // _io1 = call_pure_packed(ExternFunc("vm.builtin.debug_print"),
    //                         StringImm(kLineInfo), x)
    static const Op& call_pure_packed_op = Op::Get("relax.call_pure_packed");
    Expr call =
        Call(call_pure_packed_op, {ExternFunc("vm.builtin.debug_print"), StringImm(kLineInfo), x},
             tvm::Attrs(), {ObjectStructInfo()});
    Var io1 = bb->Emit(call, "_io");

    Var gv1 = EmitDebugOutput(bb, x, io1);
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
// TestDebugFunc
//
// Python equivalent:
//   @tvm.register_global_func("testing.relax.frontend.nn.test_debug_func")
//   def _debug(lineno, tensor, const_int, const_float, const_str, var_int): ...
//
//   class Layer(nn.Module):
//       def forward(self, x: nn.Tensor, v: tir.Var):
//           op.debug_func("testing.relax.frontend.nn.test_debug_func",
//                         x, 1, 2.0, "test", v)
//           return x
//
// debug_func emits:
//   _io = call_pure_packed(
//             ExternFunc("testing.relax.frontend.nn.test_debug_func"),
//             StringImm("<file>:<line>"),
//             x,
//             PrimValue(int64(1)),
//             PrimValue(float64(2.0)),
//             StringImm("test"),
//             v_shape_var)   -- v is a SpecInt, so its relax Var has ShapeStructInfo
//
// The forward function signature is:
//   forward(x: Tensor((10,5),"float32"), v: Shape([v_sym]), _io: Object)
//   -> Tuple(Tensor((10,5),"float32"), Tuple(Object))
// with num_input=3 (x, v, _io).
// ===========================================================================
TEST(NNDebug, TestDebugFunc) {
  static const ffi::Function op_debug_func = NNOp("debug_func");

  // Register the debug callback so it can be called at runtime.
  // In the IR test we only verify structure, but register it anyway to match
  // the Python test's intent.
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("testing.relax.frontend.nn.test_debug_func_cpp",
                        [](ffi::String lineno, ffi::Any /*tensor*/, int64_t const_int,
                           double const_float, ffi::String const_str, int64_t var_int) {
                          EXPECT_NE(std::string(lineno).find("test_nn_debug.cc"),
                                    std::string::npos);
                          EXPECT_EQ(const_int, 1);
                          EXPECT_DOUBLE_EQ(const_float, 2.0);
                          EXPECT_EQ(std::string(const_str), "test");
                          EXPECT_EQ(var_int, 8);
                        });

  const ffi::String kFuncName = "testing.relax.frontend.nn.test_debug_func_cpp";
  const ffi::String kLineInfo = "test_nn_debug.cc:0";

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [kFuncName, kLineInfo](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    // v is a SpecInt → passed as tir::Var to forward()
    tir::Var v_var = args.at("v").cast<tir::Var>();

    auto get_io = ffi::Function::GetGlobal("relax.frontend.nn.GetCurrentIOVar");
    TVM_FFI_ICHECK(get_io.has_value());
    Var io_var = (*get_io)().cast<Var>();

    // debug_func(name, [x, 1, 2.0, "test", v], io, line_info)
    op_debug_func(
        kFuncName,
        ffi::Array<ffi::Any>{ffi::Any(x->expr), ffi::Any(int64_t(1)), ffi::Any(double(2.0)),
                             ffi::Any(ffi::String("test")), ffi::Any(v_var)},
        io_var, kLineInfo);
    return ffi::Any(x->expr);
  };

  IRModule actual =
      ExportSingle("forward", forward, {"x", "v"},
                   {ffi::Any(MakeSpecTensor({10, 5}, "float32")), ffi::Any(SpecInt())});

  // ---- Build expected IR ----
  //
  // @R.function
  // def forward(x: R.Tensor((10,5),"float32"),
  //             v: R.Shape([v_sym]),
  //             _io: R.Object)
  //     -> R.Tuple(R.Tensor((10,5),"float32"), R.Tuple(R.Object)):
  //   R.func_attr({"num_input": 3})
  //   with R.dataflow():
  //     _io1 = R.call_pure_packed(
  //                ExternFunc("testing..."),
  //                R.str("test_nn_debug.cc:0"),
  //                x,
  //                R.prim_value(T.int64(1)),
  //                R.prim_value(T.float64(2.0)),
  //                R.str("test"),
  //                v,
  //                sinfo_args=[R.Object])
  //     gv1 = (x, (_io1,))
  //     R.output(gv1)
  //   return gv1

  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  EmitInitEffect(bb);
  {
    tir::Var v_sym("v", DataType::Int(64));
    Var x("x", TSInfo({10, 5}, DataType::Float(32)));
    // SpecInt produces a ShapeStructInfo Var with a single symbolic dim.
    // The exporter names it after the arg_name ("v"), but since "v" is also
    // used as the tir::Var name, the relax Var gets name "v_1" to avoid clash.
    Var v("v_1", ShapeStructInfo(ffi::Array<PrimExpr>{v_sym}));
    Var io("_io", ObjectStructInfo());
    ffi::Array<Var> params{x, v, io};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // _io1 = call_pure_packed(ExternFunc(kFuncName), StringImm(kLineInfo),
    //                         x, PrimValue(1i64), PrimValue(2.0f64),
    //                         StringImm("test"), PrimValue(v_sym))
    static const Op& call_pure_packed_op = Op::Get("relax.call_pure_packed");
    Expr call = Call(
        call_pure_packed_op,
        {ExternFunc(kFuncName), StringImm(kLineInfo), x, PrimValue(IntImm(DataType::Int(64), 1)),
         PrimValue(FloatImm(DataType::Float(64), 2.0)), StringImm("test"),
         PrimValue(v_sym)},  // Pass tir::Var wrapped in PrimValue, not relax::Var
        tvm::Attrs(), {ObjectStructInfo()});
    Var io1 = bb->Emit(call, "_io");

    Var gv1 = EmitDebugOutput(bb, x, io1);
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();

    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(3)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("forward")));
    bb->AddFunction(Function(params, body, std::nullopt, /*is_pure=*/true, DictAttrs(attrs)),
                    "forward");
  }
  AssertStructEqual(actual, bb->Finalize());
}

// ===========================================================================
// Runtime helpers shared by the JIT tests
//
// The low-level CompileToVM / GetOutputRec / InitEffect / RunForward helpers
// are kept for the IR-structure tests (TestDebugPrintJIT / TestDebugFuncJIT)
// that build their IRModule manually and need fine-grained control.
//
// The new Jit-based tests (TestDebugPrintJIT_Jit / TestDebugFuncJIT_Jit) use
// the high-level CppModule / Jit() API instead.
// ===========================================================================

// Compile an IRModule to a VM using tvm::driver::Compile (llvm/cpu_generic).
static ffi::Module CompileToVM(const IRModule& mod) {
  tvm::Target tgt = tvm::Target::WithHost(tvm::Target("llvm"), tvm::Target("llvm"));
  ffi::Map<ffi::Any, ffi::ObjectRef> empty_params;
  auto vm_mod = tvm::driver::Compile(mod, tgt, empty_params,
                                     ffi::Optional<ffi::String>(ffi::String("cpu_generic")),
                                     ffi::Optional<ffi::String>(ffi::String("generic")));
  auto vm_ex = vm_mod.as<runtime::vm::VMExecutable>();
  TVM_FFI_ICHECK(vm_ex) << "Compile did not return a VMExecutable";
  ffi::Module vm = vm_ex->VMLoadExecutable();

  // Initialize the VM for CPU execution.
  std::vector<ffi::AnyView> init_args = {
      kDLCPU, int(0), runtime::memory::AllocatorType::kPooled,
      kDLCPU, int(0), runtime::memory::AllocatorType::kPooled,
  };
  ffi::Any rv;
  vm->GetFunction("vm_initialization")
      .value()
      .CallPacked(ffi::PackedArgs(init_args.data(), init_args.size()), &rv);
  return vm;
}

// Recursively fetch a (potentially nested) output from the VM using
// get_output_arity + get_output, mirroring Python's get_output_rec().
// Returns ffi::Any which is either a runtime::Tensor (leaf) or
// ffi::Array<ffi::Any> (tuple node).
static ffi::Any GetOutputRec(const ffi::Module& vm, const std::string& func_name,
                             const std::vector<int>& indices) {
  return CppModuleNode::GetOutputRec(vm, func_name, indices);
}

// Call _initialize_effect() and return the effect state as Array<Any>.
// _initialize_effect returns Tuple(Object) i.e. a 1-element tuple containing
// the _io null object. We return it as Array<Any> so it can be spread into
// the forward call's effect arguments.
static ffi::Array<ffi::Any> InitEffect(const ffi::Module& vm) {
  std::vector<ffi::AnyView> set_args = {ffi::String("_initialize_effect")};
  ffi::Any rv;
  vm->GetFunction("set_input")
      .value()
      .CallPacked(ffi::PackedArgs(set_args.data(), set_args.size()), &rv);
  vm->GetFunction("invoke_stateful").value()("_initialize_effect");

  // _initialize_effect returns Tuple(Object): fetch element 0 individually.
  ffi::Any io_obj = GetOutputRec(vm, "_initialize_effect", {0});
  ffi::Array<ffi::Any> effects;
  effects.push_back(io_obj);
  return effects;
}

// Run the "forward" function: forward(*user_args, *effects)
// The debug-mode return is Tuple(output_tensor, Tuple(_io_new)).
// Fetches output_tensor via index 0 and the new _io via index (1, 0),
// updates effects in-place, and returns the output tensor.
static runtime::Tensor RunForward(const ffi::Module& vm, const std::string& func_name,
                                  const std::vector<ffi::AnyView>& user_args,
                                  ffi::Array<ffi::Any>& effects) {
  // set_input(func_name, *user_args, *effects)
  std::vector<ffi::AnyView> set_args;
  set_args.push_back(ffi::String(func_name));
  for (const auto& a : user_args) set_args.push_back(a);
  for (const auto& e : effects) set_args.push_back(ffi::AnyView(e));

  ffi::Any rv;
  vm->GetFunction("set_input")
      .value()
      .CallPacked(ffi::PackedArgs(set_args.data(), set_args.size()), &rv);
  vm->GetFunction("invoke_stateful").value()(func_name);

  // Result is Tuple(output_tensor, Tuple(_io_new)).
  // Fetch each element individually to avoid the "cannot return a tuple" error.
  runtime::Tensor output = GetOutputRec(vm, func_name, {0}).cast<runtime::Tensor>();
  ffi::Any new_io = GetOutputRec(vm, func_name, {1, 0});
  effects = ffi::Array<ffi::Any>{new_io};
  return output;
}

// ===========================================================================
// TestDebugPrintJIT  (low-level: manual ExportSingle + CompileToVM)
// ===========================================================================
TEST(NNDebug, TestDebugPrintJIT) {
  static const ffi::Function op_debug_func = NNOp("debug_func");

  const ffi::String kLineInfo = "test_nn_debug.cc:0";
  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [kLineInfo](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    auto get_io = ffi::Function::GetGlobal("relax.frontend.nn.GetCurrentIOVar");
    TVM_FFI_ICHECK(get_io.has_value());
    Var io_var = (*get_io)().cast<Var>();
    op_debug_func(ffi::String("vm.builtin.debug_print"), ffi::Array<ffi::Any>{ffi::Any(x->expr)},
                  io_var, kLineInfo);
    return ffi::Any(x->expr);
  };

  // Use TestModuleNode::ExportDebug — the module-centric API.
  TestModuleNode mod(forward, {"x"}, {ffi::Any(MakeSpecTensor({10, 5}, "float32"))});
  IRModule ir_mod = mod.ExportDebug("forward");

  ffi::Module vm = CompileToVM(ir_mod);
  ffi::Array<ffi::Any> effects = InitEffect(vm);

  tvm::Device cpu{kDLCPU, 0};
  runtime::Tensor x_tensor =
      runtime::Tensor::Empty(ffi::Shape({10, 5}), DLDataType{kDLFloat, 32, 1}, cpu, std::nullopt);
  std::vector<float> x_data(10 * 5, 1.0f);
  x_tensor.CopyFromBytes(x_data.data(), x_data.size() * sizeof(float));

  runtime::Tensor y = RunForward(vm, "forward", {x_tensor}, effects);
  ASSERT_EQ(y.Shape().size(), 2);
  EXPECT_EQ(y.Shape()[0], 10);
  EXPECT_EQ(y.Shape()[1], 5);
}

// ===========================================================================
// TestDebugFuncJIT  (low-level: manual ExportSingle + CompileToVM)
// ===========================================================================
TEST(NNDebug, TestDebugFuncJIT) {
  static const ffi::Function op_debug_func = NNOp("debug_func");

  static std::atomic<bool> g_debug_called{false};
  g_debug_called = false;

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "testing.relax.frontend.nn.test_debug_func_jit",
      [](ffi::String lineno, runtime::Tensor tensor, int64_t const_int, double const_float,
         ffi::String const_str, int64_t var_int) {
        EXPECT_FALSE(std::string(lineno).empty());
        ASSERT_EQ(tensor.Shape().size(), 2);
        EXPECT_EQ(tensor.Shape()[0], 10);
        EXPECT_EQ(tensor.Shape()[1], 5);
        EXPECT_EQ(const_int, 1);
        EXPECT_DOUBLE_EQ(const_float, 2.0);
        EXPECT_EQ(std::string(const_str), "test");
        EXPECT_EQ(var_int, 8);
        g_debug_called = true;
      });

  const ffi::String kFuncName = "testing.relax.frontend.nn.test_debug_func_jit";
  const ffi::String kLineInfo = "test_nn_debug.cc:0";

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [kFuncName, kLineInfo](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    tir::Var v_var = args.at("v").cast<tir::Var>();
    auto get_io = ffi::Function::GetGlobal("relax.frontend.nn.GetCurrentIOVar");
    TVM_FFI_ICHECK(get_io.has_value());
    Var io_var = (*get_io)().cast<Var>();
    op_debug_func(
        kFuncName,
        ffi::Array<ffi::Any>{ffi::Any(x->expr), ffi::Any(int64_t(1)), ffi::Any(double(2.0)),
                             ffi::Any(ffi::String("test")), ffi::Any(v_var)},
        io_var, kLineInfo);
    return ffi::Any(x->expr);
  };

  // Use TestModuleNode::ExportDebug.
  TestModuleNode mod(forward, {"x", "v"},
                     {ffi::Any(MakeSpecTensor({10, 5}, "float32")), ffi::Any(SpecInt())});
  IRModule ir_mod = mod.ExportDebug("forward");

  ffi::Module vm = CompileToVM(ir_mod);
  ffi::Array<ffi::Any> effects = InitEffect(vm);

  tvm::Device cpu{kDLCPU, 0};
  runtime::Tensor x_tensor =
      runtime::Tensor::Empty(ffi::Shape({10, 5}), DLDataType{kDLFloat, 32, 1}, cpu, std::nullopt);
  std::vector<float> x_data(10 * 5, 1.0f);
  x_tensor.CopyFromBytes(x_data.data(), x_data.size() * sizeof(float));

  ffi::Shape v_shape({8});
  runtime::Tensor y = RunForward(vm, "forward", {x_tensor, v_shape}, effects);

  EXPECT_TRUE(g_debug_called.load()) << "Debug callback was not called during VM execution";
  ASSERT_EQ(y.Shape().size(), 2);
  EXPECT_EQ(y.Shape()[0], 10);
  EXPECT_EQ(y.Shape()[1], 5);
}

// ===========================================================================
// TestDebugPrintJIT_Jit  (high-level: NNModuleNode::Jit → CppModule)
//
// Mirrors Python:
//   model = Layer().jit(spec={"forward": {"x": spec.Tensor([10,5],"float32")}},
//                       debug=True)
//   y = model["forward"](x)
// ===========================================================================
TEST(NNDebug, TestDebugPrintJIT_Jit) {
  static const ffi::Function op_debug_func = NNOp("debug_func");

  const ffi::String kLineInfo = "test_nn_debug.cc:0";
  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [kLineInfo](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    auto get_io = ffi::Function::GetGlobal("relax.frontend.nn.GetCurrentIOVar");
    TVM_FFI_ICHECK(get_io.has_value());
    Var io_var = (*get_io)().cast<Var>();
    op_debug_func(ffi::String("vm.builtin.debug_print"), ffi::Array<ffi::Any>{ffi::Any(x->expr)},
                  io_var, kLineInfo);
    return ffi::Any(x->expr);
  };

  // Create the module and call Jit() — mirrors Python's Layer().jit(...).
  TestModuleNode mod(forward, {"x"}, {ffi::Any(MakeSpecTensor({10, 5}, "float32"))});
  CppModule model = mod.JitCpp("forward", {kDLCPU, 0}, "cpu_generic", /*debug=*/true);

  tvm::Device cpu{kDLCPU, 0};
  runtime::Tensor x_tensor =
      runtime::Tensor::Empty(ffi::Shape({10, 5}), DLDataType{kDLFloat, 32, 1}, cpu, std::nullopt);
  std::vector<float> x_data(10 * 5, 1.0f);
  x_tensor.CopyFromBytes(x_data.data(), x_data.size() * sizeof(float));

  // model["forward"](x) — CppMethodCaller handles effect threading.
  ffi::Any result = model["forward"]({ffi::Any(x_tensor)});
  runtime::Tensor y = result.cast<runtime::Tensor>();
  ASSERT_EQ(y.Shape().size(), 2);
  EXPECT_EQ(y.Shape()[0], 10);
  EXPECT_EQ(y.Shape()[1], 5);
}

// ===========================================================================
// TestDebugFuncJIT_Jit  (high-level: NNModuleNode::Jit → CppModule)
//
// Verifies:
//   - Debug callback invoked with correct arguments on each call.
//   - Effect state correctly threaded between successive calls.
//
// Python equivalent:
//   model = Layer().jit(spec={"forward": {"x": ..., "v": "int"}}, debug=True)
//   y = model["forward"](x, 8)
//   y = model["forward"](x, 42)
// ===========================================================================
TEST(NNDebug, TestDebugFuncJIT_Jit) {
  static const ffi::Function op_debug_func = NNOp("debug_func");

  static std::atomic<int> g_call_count{0};
  g_call_count = 0;
  static std::atomic<int64_t> g_last_var_int{-1};

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "testing.relax.frontend.nn.test_debug_func_jit2",
      [](ffi::String lineno, runtime::Tensor tensor, int64_t const_int, double const_float,
         ffi::String const_str, int64_t var_int) {
        EXPECT_FALSE(std::string(lineno).empty());
        ASSERT_EQ(tensor.Shape().size(), 2);
        EXPECT_EQ(tensor.Shape()[0], 10);
        EXPECT_EQ(tensor.Shape()[1], 5);
        EXPECT_EQ(const_int, 1);
        EXPECT_DOUBLE_EQ(const_float, 2.0);
        EXPECT_EQ(std::string(const_str), "test");
        g_last_var_int = var_int;
        g_call_count++;
      });

  const ffi::String kFuncName = "testing.relax.frontend.nn.test_debug_func_jit2";
  const ffi::String kLineInfo = "test_nn_debug.cc:0";

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [kFuncName, kLineInfo](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    tir::Var v_var = args.at("v").cast<tir::Var>();
    auto get_io = ffi::Function::GetGlobal("relax.frontend.nn.GetCurrentIOVar");
    TVM_FFI_ICHECK(get_io.has_value());
    Var io_var = (*get_io)().cast<Var>();
    op_debug_func(
        kFuncName,
        ffi::Array<ffi::Any>{ffi::Any(x->expr), ffi::Any(int64_t(1)), ffi::Any(double(2.0)),
                             ffi::Any(ffi::String("test")), ffi::Any(v_var)},
        io_var, kLineInfo);
    return ffi::Any(x->expr);
  };

  TestModuleNode mod(forward, {"x", "v"},
                     {ffi::Any(MakeSpecTensor({10, 5}, "float32")), ffi::Any(SpecInt())});
  CppModule model = mod.JitCpp("forward", {kDLCPU, 0}, "cpu_generic", /*debug=*/true);

  tvm::Device cpu{kDLCPU, 0};
  runtime::Tensor x_tensor =
      runtime::Tensor::Empty(ffi::Shape({10, 5}), DLDataType{kDLFloat, 32, 1}, cpu, std::nullopt);
  std::vector<float> x_data(10 * 5, 1.0f);
  x_tensor.CopyFromBytes(x_data.data(), x_data.size() * sizeof(float));

  CppMethodCaller caller = model["forward"];

  // First call: v=8
  {
    ffi::Shape v_shape({8});
    ffi::Any result = caller({ffi::Any(x_tensor), ffi::Any(v_shape)});
    runtime::Tensor y = result.cast<runtime::Tensor>();
    ASSERT_EQ(y.Shape().size(), 2);
    EXPECT_EQ(y.Shape()[0], 10);
    EXPECT_EQ(y.Shape()[1], 5);
    EXPECT_EQ(g_call_count.load(), 1) << "Debug callback not called on first invocation";
    EXPECT_EQ(g_last_var_int.load(), 8);
  }

  // Second call: v=42 — verifies effect state is correctly threaded.
  {
    ffi::Shape v_shape({42});
    ffi::Any result = caller({ffi::Any(x_tensor), ffi::Any(v_shape)});
    runtime::Tensor y = result.cast<runtime::Tensor>();
    ASSERT_EQ(y.Shape().size(), 2);
    EXPECT_EQ(y.Shape()[0], 10);
    EXPECT_EQ(y.Shape()[1], 5);
    EXPECT_EQ(g_call_count.load(), 2) << "Debug callback not called on second invocation";
    EXPECT_EQ(g_last_var_int.load(), 42);
  }
}

}  // namespace testing
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
