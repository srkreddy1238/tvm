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
 * \file tests/cpp/relax/frontend/nn/test_nn_extern_module.cc
 * \brief C++ port of tests/python/relax/test_frontend_nn_extern_module.py
 *
 * Ported tests:
 *   - TestExternObject  (test_extern_object)
 *   - TestExternSource  (test_extern_source)
 *
 * Both tests:
 *   1. Build a ModuleSpec whose forward functions call nn.op.extern() to
 *      emit call_dps_packed nodes for "ext_scalar_add" and "ext_test_sym".
 *   2. Export to IRModule via ExportToIRModule (allow_extern=true).
 *   3. Assert structural equality of the IRModule against the expected IR.
 *   4. Attach the external object/source module via AttachExternModules.
 *   5. Compile with tvm::driver::Compile and run via VirtualMachine.
 *   6. Verify numerical results.
 *
 * The external C++ kernel lives in
 *   tests/python/relax/frontend_nn_extern_module.cc
 * (shared with the Python tests).
 *
 * Design notes:
 *   - nn.op.extern() is registered as "relax.frontend.nn.op.extern" and
 *     takes (name: String, args: Array<Any>, out: Array<Any>) -> Any.
 *     The "out" array contains placeholder NNTensor Vars whose struct_info
 *     describes the output shape/dtype.
 *   - The infer-shape callbacks (_infer_scalar_add, _infer_test_sym) are
 *     implemented inline in C++ rather than as Python callables.
 *   - TestExternObject compiles the .cc file to a .o and passes it as an
 *     ObjectModule; TestExternSource passes the .cc path as a SourceModule.
 *     Both use the same ExternModule FFI path (relax.frontend.nn.AddExtern).
 *   - The IR equality check mirrors _check_ir_equality() from the Python test.
 */

#include <gtest/gtest.h>
#include <tvm/driver/compile.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/tensor.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/tir/op.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "../../../../../src/relax/frontend/nn/core.h"
#include "../../../../../src/relax/frontend/nn/exporter.h"
#include "../../../../../src/relax/frontend/nn/extern.h"
#include "../../../../../src/relax/frontend/nn/spec.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace testing {

// ===========================================================================
// Shared helpers
// ===========================================================================

// Path to the shared extern module source (relative to workspace root).
static std::filesystem::path ExternModuleSourcePath() {
  return std::filesystem::path("tests") / "python" / "relax" /
         "frontend_nn_extern_module.cc";
}

// Build a placeholder NNTensor Var with the given struct_info (used as the
// "out" argument to nn.op.extern()).
static NNTensor MakePlaceholderTensor(TensorStructInfo sinfo, const std::string& name) {
  Var v(name, sinfo);
  return NNTensor(v);
}

// Retrieve the registered nn.op FFI function by short name.
static ffi::Function NNOp(const std::string& name) {
  auto f = ffi::Function::GetGlobal("relax.frontend.nn.op." + name);
  TVM_FFI_ICHECK(f.has_value()) << "nn.op not found: " << name;
  return f.value();
}

// ===========================================================================
// Expected IR builder
//
// Mirrors _check_ir_equality() from the Python test:
//
//   @R.function
//   def scalar_add(a: R.Tensor((), "float32"), b: R.Tensor((), "float32"))
//       -> R.Tensor((), "float32"):
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       ext_scalar_add = R.call_dps_packed("ext_scalar_add", (a, b),
//                                          out_sinfo=R.Tensor((), "float32"))
//       gv = ext_scalar_add
//       R.output(gv)
//     return gv
//
//   @R.function
//   def test_sym(a: R.Tensor(("x","y",1), "float32"),
//                b: R.Tensor(("y","z",5), "float32"))
//       -> R.Tensor(("x","y","z",9), "float32"):
//     x = T.int64(); y = T.int64(); z = T.int64()
//     R.func_attr({"num_input": 2})
//     with R.dataflow():
//       ext_test_sym = R.call_dps_packed("ext_test_sym", (a, b),
//                                        out_sinfo=R.Tensor((x,y,z,9), "float32"))
//       gv1 = ext_test_sym
//       R.output(gv1)
//     return gv1
// ===========================================================================
static IRModule BuildExpectedIR() {
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  DataType f32 = DataType::Float(32);
  auto I64 = [](int64_t v) { return IntImm(DataType::Int(64), v); };

  // --- scalar_add ---
  {
    TensorStructInfo scalar_sinfo(ShapeExpr(ffi::Array<PrimExpr>{}), f32);
    Var a("a", scalar_sinfo);
    Var b("b", scalar_sinfo);
    ffi::Array<Var> params{a, b};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    static const Op& call_dps = Op::Get("relax.call_dps_packed");
    Expr func_name = relax::StringImm("ext_scalar_add");
    Expr args_tuple = relax::Tuple({a, b});
    Call call(call_dps, {func_name, args_tuple}, Attrs(),
              ffi::Array<StructInfo>{scalar_sinfo});
    Var ext_sa = bb->Emit(call, "ext_scalar_add");
    Var gv = bb->EmitOutput(ext_sa, "gv");
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("scalar_add")));
    bb->AddFunction(Function(params, body, std::nullopt, true, DictAttrs(attrs)), "scalar_add");
  }

  // --- test_sym ---
  {
    tir::Var x("x", DataType::Int(64));
    tir::Var y("y", DataType::Int(64));
    tir::Var z("z", DataType::Int(64));
    TensorStructInfo a_sinfo(
        ShapeExpr(ffi::Array<PrimExpr>{x, y, I64(1)}), f32);
    TensorStructInfo b_sinfo(
        ShapeExpr(ffi::Array<PrimExpr>{y, z, I64(5)}), f32);
    TensorStructInfo out_sinfo(
        ShapeExpr(ffi::Array<PrimExpr>{x, y, z, I64(9)}), f32);
    Var a("a", a_sinfo);
    Var b("b", b_sinfo);
    ffi::Array<Var> params{a, b};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    static const Op& call_dps = Op::Get("relax.call_dps_packed");
    Expr func_name = relax::StringImm("ext_test_sym");
    Expr args_tuple = relax::Tuple({a, b});
    Call call(call_dps, {func_name, args_tuple}, Attrs(),
              ffi::Array<StructInfo>{out_sinfo});
    Var ext_ts = bb->Emit(call, "ext_test_sym");
    Var gv1 = bb->EmitOutput(ext_ts, "gv1");
    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv1));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(2)));
    attrs.Set("global_symbol", ffi::Any(ffi::String("test_sym")));
    bb->AddFunction(Function(params, body, std::nullopt, true, DictAttrs(attrs)), "test_sym");
  }

  return bb->Finalize();
}

// ===========================================================================
// Build the ModuleSpec for the two extern methods.
//
// The forward lambdas call nn.op.extern() which emits call_dps_packed nodes.
// The "out" placeholder tensors carry the output shape/dtype inferred by the
// shape-inference callbacks (_infer_scalar_add / _infer_test_sym in Python).
// In C++ we compute the output shape directly.
// ===========================================================================
static ModuleSpec BuildExternModuleSpec() {
  const ffi::Function op_extern = NNOp("extern");
  DataType f32 = DataType::Float(32);
  auto I64 = [](int64_t v) { return IntImm(DataType::Int(64), v); };

  // scalar_add: () x () -> ()
  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> scalar_add_fn =
      [f32, op_extern](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor a = args.at("a").cast<NNTensor>();
    NNTensor b = args.at("b").cast<NNTensor>();
    // Output placeholder: scalar float32
    TensorStructInfo out_sinfo(ShapeExpr(ffi::Array<PrimExpr>{}), f32);
    NNTensor out_ph = MakePlaceholderTensor(out_sinfo, "out");
    ffi::Array<ffi::Any> in_args{ffi::Any(a), ffi::Any(b)};
    ffi::Array<ffi::Any> out_args{ffi::Any(out_ph)};
    return op_extern(ffi::String("ext_scalar_add"), in_args, out_args);
  };

  // test_sym: (x,y,1) x (y,z,5) -> (x,y,z,9)
  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> test_sym_fn =
      [f32, I64, op_extern](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor a = args.at("a").cast<NNTensor>();
    NNTensor b = args.at("b").cast<NNTensor>();
    // Infer output shape: (x, y, z, 9) from a.shape[0], a.shape[1], b.shape[1]
    ffi::Array<PrimExpr> a_shape = a->GetShape();
    ffi::Array<PrimExpr> b_shape = b->GetShape();
    PrimExpr x = a_shape[0];
    PrimExpr y = a_shape[1];
    PrimExpr z = b_shape[1];
    TensorStructInfo out_sinfo(ShapeExpr(ffi::Array<PrimExpr>{x, y, z, I64(9)}), f32);
    NNTensor out_ph = MakePlaceholderTensor(out_sinfo, "out");
    ffi::Array<ffi::Any> in_args{ffi::Any(a), ffi::Any(b)};
    ffi::Array<ffi::Any> out_args{ffi::Any(out_ph)};
    return op_extern(ffi::String("ext_test_sym"), in_args, out_args);
  };

  // SpecTensors
  SpecTensor scalar_spec(ffi::Array<ffi::Any>{}, "float32");

  ffi::Array<ffi::Any> a_sym_shape;
  a_sym_shape.push_back(ffi::Any(ffi::String("x")));
  a_sym_shape.push_back(ffi::Any(ffi::String("y")));
  a_sym_shape.push_back(ffi::Any(int64_t(1)));
  SpecTensor a_sym_spec(a_sym_shape, "float32");

  ffi::Array<ffi::Any> b_sym_shape;
  b_sym_shape.push_back(ffi::Any(ffi::String("y")));
  b_sym_shape.push_back(ffi::Any(ffi::String("z")));
  b_sym_shape.push_back(ffi::Any(int64_t(5)));
  SpecTensor b_sym_spec(b_sym_shape, "float32");

  MethodSpec ms_scalar(scalar_add_fn, {"a", "b"},
                       {ffi::Any(scalar_spec), ffi::Any(scalar_spec)}, "plain", "none");
  MethodSpec ms_sym(test_sym_fn, {"a", "b"},
                    {ffi::Any(a_sym_spec), ffi::Any(b_sym_spec)}, "plain", "none");

  return ModuleSpec(
      ffi::Array<ffi::String>{"scalar_add", "test_sym"},
      ffi::Array<ffi::Any>{ffi::Any(ms_scalar), ffi::Any(ms_sym)},
      {}, {});
}

// ===========================================================================
// Compile the extern .cc file to a .o and return the path.
// Mirrors _compile_cc() from the Python test.
// ===========================================================================
static std::string CompileExternCC(const std::filesystem::path& src) {
  // Use a temp file in the system temp directory.
  auto tmp = std::filesystem::temp_directory_path() / "frontend_nn_extern_module.o";

  // Retrieve TVM include paths via the FFI.
  auto find_include = ffi::Function::GetGlobal("tvm.contrib.utils.find_include_path");
  TVM_FFI_ICHECK(find_include.has_value()) << "tvm.contrib.utils.find_include_path not found";
  ffi::Array<ffi::String> include_paths = (*find_include)().cast<ffi::Array<ffi::String>>();

  std::string cmd = "g++ " + src.string();
  for (const auto& p : include_paths) cmd += " -I " + std::string(p);
  cmd += " -c -std=c++17 -fPIC -o " + tmp.string();

  int ret = std::system(cmd.c_str());
  TVM_FFI_ICHECK(ret == 0) << "Compilation failed: " << cmd;
  return tmp.string();
}

// ===========================================================================
// Load and initialise a VM from a compiled VMExecutable module.
// Mirrors LoadAndInitVM() in cpp_module.cc.
// ===========================================================================
static ffi::Module LoadAndInitVM(const ffi::Module& compiled) {
  using namespace tvm::runtime;
  auto vm_ex = compiled.as<vm::VMExecutable>();
  TVM_FFI_ICHECK(vm_ex) << "Expected VMExecutable from Compile()";
  ffi::Module vm = vm_ex->VMLoadExecutable();

  Device cpu{kDLCPU, 0};
  std::vector<ffi::AnyView> init_args = {
      cpu.device_type,
      static_cast<int>(cpu.device_id),
      memory::AllocatorType::kPooled,
      cpu.device_type,
      static_cast<int>(cpu.device_id),
      memory::AllocatorType::kPooled,
  };
  ffi::Any rv;
  vm->GetFunction("vm_initialization")
      .value()
      .CallPacked(ffi::PackedArgs(init_args.data(), init_args.size()), &rv);
  return vm;
}

// Invoke a no-effect VM function by name with the given tensor inputs and
// return the output tensor.  Uses the stateful set_input/invoke/get_output API.
static runtime::Tensor InvokeVM(const ffi::Module& vm, const std::string& func_name,
                                 std::vector<runtime::Tensor> inputs) {
  // set_input(func_name, *inputs)
  std::vector<ffi::AnyView> set_args;
  set_args.push_back(ffi::String(func_name));
  for (const auto& t : inputs) set_args.push_back(ffi::AnyView(t));
  ffi::Any rv;
  vm->GetFunction("set_input")
      .value()
      .CallPacked(ffi::PackedArgs(set_args.data(), set_args.size()), &rv);

  // invoke_stateful(func_name)
  vm->GetFunction("invoke_stateful").value()(ffi::String(func_name));

  // get_output(func_name) -> Tensor (arity == -1 means scalar/leaf output)
  std::vector<ffi::AnyView> out_args = {ffi::String(func_name)};
  ffi::Any out_rv;
  vm->GetFunction("get_output")
      .value()
      .CallPacked(ffi::PackedArgs(out_args.data(), out_args.size()), &out_rv);
  return out_rv.cast<runtime::Tensor>();
}

// ===========================================================================
// Run the compiled VM and verify scalar_add and test_sym results.
// ===========================================================================
static void RunAndVerify(const ffi::Module& compiled) {
  ffi::Module vm = LoadAndInitVM(compiled);
  Device cpu{kDLCPU, 0};

  // --- scalar_add: 1.0 + 3.0 = 4.0 ---
  {
    runtime::Tensor a = runtime::Tensor::Empty(
        ffi::Shape{}, DLDataType{kDLFloat, 32, 1}, cpu, std::nullopt);
    runtime::Tensor b = runtime::Tensor::Empty(
        ffi::Shape{}, DLDataType{kDLFloat, 32, 1}, cpu, std::nullopt);
    float va = 1.0f, vb = 3.0f;
    a.CopyFromBytes(&va, sizeof(float));
    b.CopyFromBytes(&vb, sizeof(float));
    runtime::Tensor c = InvokeVM(vm, "scalar_add", {a, b});
    float vc = 0.0f;
    c.CopyToBytes(&vc, sizeof(float));
    EXPECT_FLOAT_EQ(vc, 4.0f);
  }

  // --- test_sym: shapes (3,4,1) x (4,2,5) -> (3,4,2,9) ---
  {
    runtime::Tensor a = runtime::Tensor::Empty(
        ffi::Shape{3, 4, 1}, DLDataType{kDLFloat, 32, 1}, cpu, std::nullopt);
    runtime::Tensor b = runtime::Tensor::Empty(
        ffi::Shape{4, 2, 5}, DLDataType{kDLFloat, 32, 1}, cpu, std::nullopt);
    runtime::Tensor c = InvokeVM(vm, "test_sym", {a, b});
    ASSERT_EQ(c.ndim(), 4);
    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 4);
    EXPECT_EQ(c.shape()[2], 2);
    EXPECT_EQ(c.shape()[3], 9);
  }
}

// ===========================================================================
// TestExternObject
//
// Python equivalent: test_extern_object()
//   - Compile frontend_nn_extern_module.cc to a .o
//   - Wrap in nn.ObjectModule
//   - Export, attach, compile, run
// ===========================================================================
TEST(NNExternModule, TestExternObject) {
  std::filesystem::path src = ExternModuleSourcePath();
  if (!std::filesystem::exists(src)) {
    GTEST_SKIP() << "Extern module source not found: " << src;
  }

  // Build the symbols map for the extern functions.
  ffi::Map<ffi::String, ffi::Function> symbols;
  symbols.Set("ext_scalar_add",
              ffi::Function([](ffi::PackedArgs args, ffi::Any* rv) {
                *rv = ffi::Any(NNTensor(
                    Var("out", TensorStructInfo(
                        ShapeExpr(ffi::Array<PrimExpr>{}), DataType::Float(32)))));
              }));
  symbols.Set("ext_test_sym",
              ffi::Function([](ffi::PackedArgs args, ffi::Any* rv) {
                NNTensor a = args[0].cast<NNTensor>();
                NNTensor b = args[1].cast<NNTensor>();
                ffi::Array<PrimExpr> as = a->GetShape();
                ffi::Array<PrimExpr> bs = b->GetShape();
                TensorStructInfo out_sinfo(ShapeExpr(ffi::Array<PrimExpr>{
                    as[0], as[1], bs[1], IntImm(DataType::Int(64), 9)}),
                    DataType::Float(32));
                *rv = ffi::Any(NNTensor(Var("out", out_sinfo)));
              }));

  // Compile the .cc to a .o and create an ObjectModule.
  std::string obj_path = CompileExternCC(src);
  auto make_obj_mod = ffi::Function::GetGlobal("relax.frontend.nn.MakeObjectModule");
  TVM_FFI_ICHECK(make_obj_mod.has_value());
  ObjectModule obj_mod = (*make_obj_mod)(symbols, ffi::String(obj_path)).cast<ObjectModule>();
  ffi::Array<runtime::ObjectRef> ext_mods_arr{obj_mod};

  // Export to IRModule and verify IR equality.
  ModuleSpec mod_spec = BuildExternModuleSpec();
  Exporter exporter(/*debug=*/false);
  IRModule mod = exporter.get()->Build(mod_spec)[0].cast<IRModule>();
  IRModule expected = BuildExpectedIR();
  EXPECT_TRUE(ffi::StructuralEqual()(mod, expected))
      << "\n=== Actual ===\n" << mod << "\n=== Expected ===\n" << expected;

  // Attach and compile.
  relax::transform::Pass attach_pass =
      relax::transform::AttachExternModules(ext_mods_arr);
  IRModule mod_with_ext = attach_pass(mod);

  // Compile and run.
  ffi::Module compiled = tvm::driver::Compile(mod_with_ext, ffi::String("llvm"));
  RunAndVerify(compiled);
}

// ===========================================================================
// TestExternSource
//
// Python equivalent: test_extern_source()
//   - Pass the .cc path as nn.SourceModule (source_format="cpp")
//   - Export, attach, compile, run
// ===========================================================================
TEST(NNExternModule, TestExternSource) {
  std::filesystem::path src = ExternModuleSourcePath();
  if (!std::filesystem::exists(src)) {
    GTEST_SKIP() << "Extern module source not found: " << src;
  }

  // Build the symbols map (same as TestExternObject).
  ffi::Map<ffi::String, ffi::Function> symbols;
  symbols.Set("ext_scalar_add",
              ffi::Function([](ffi::PackedArgs args, ffi::Any* rv) {
                *rv = ffi::Any(NNTensor(
                    Var("out", TensorStructInfo(
                        ShapeExpr(ffi::Array<PrimExpr>{}), DataType::Float(32)))));
              }));
  symbols.Set("ext_test_sym",
              ffi::Function([](ffi::PackedArgs args, ffi::Any* rv) {
                NNTensor a = args[0].cast<NNTensor>();
                NNTensor b = args[1].cast<NNTensor>();
                ffi::Array<PrimExpr> as = a->GetShape();
                ffi::Array<PrimExpr> bs = b->GetShape();
                TensorStructInfo out_sinfo(ShapeExpr(ffi::Array<PrimExpr>{
                    as[0], as[1], bs[1], IntImm(DataType::Int(64), 9)}),
                    DataType::Float(32));
                *rv = ffi::Any(NNTensor(Var("out", out_sinfo)));
              }));

  // Read the source file and create a SourceModule.
  auto make_src_mod = ffi::Function::GetGlobal("relax.frontend.nn.MakeSourceModule");
  TVM_FFI_ICHECK(make_src_mod.has_value());
  // MakeSourceModule(symbols, source_code, source_format, compile_options, compiler, output_format)
  SourceModule src_mod =
      (*make_src_mod)(symbols, ffi::String(src.string()), ffi::String("cpp"),
                      ffi::Optional<ffi::Array<ffi::String>>(),
                      ffi::Optional<ffi::String>(),
                      ffi::String(".o")).cast<SourceModule>();
  ffi::Array<runtime::ObjectRef> ext_mods_arr{src_mod};

  // Export to IRModule and verify IR equality.
  ModuleSpec mod_spec = BuildExternModuleSpec();
  Exporter exporter(/*debug=*/false);
  IRModule mod = exporter.get()->Build(mod_spec)[0].cast<IRModule>();
  IRModule expected = BuildExpectedIR();
  EXPECT_TRUE(ffi::StructuralEqual()(mod, expected))
      << "\n=== Actual ===\n" << mod << "\n=== Expected ===\n" << expected;

  // Attach and compile.
  relax::transform::Pass attach_pass2 =
      relax::transform::AttachExternModules(ext_mods_arr);
  IRModule mod_with_ext = attach_pass2(mod);

  ffi::Module compiled = tvm::driver::Compile(mod_with_ext, ffi::String("llvm"));
  RunAndVerify(compiled);
}

}  // namespace testing
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
