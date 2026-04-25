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
 * Tests the debug-mode IR emission and JIT execution for:
 *   - op.print_  (test_debug_print)
 *   - op.debug_func  (test_debug_func)
 *
 * Both tests use the high-level NNModuleNode::Jit() → CppModule API,
 * mirroring the Python test pattern:
 *   model = Layer().jit(spec={...}, debug=True)
 *   y = model["forward"](x)
 */

#include <gtest/gtest.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>
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

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

static ffi::Function NNOp(const std::string& name) {
  const std::string key = "relax.frontend.nn.op." + name;
  auto f = ffi::Function::GetGlobal(key);
  TVM_FFI_ICHECK(f.has_value()) << "nn.op not found: " << key;
  return f.value();
}

static SpecTensor MakeSpecTensor(std::initializer_list<int64_t> dims, const std::string& dtype) {
  ffi::Array<ffi::Any> shape;
  for (int64_t d : dims) shape.push_back(ffi::Any(d));
  return SpecTensor(shape, dtype);
}

// ---------------------------------------------------------------------------
// TestModuleNode: a concrete NNModuleNode subclass used by all tests.
//
// Holds a forward ffi::Function and a MethodSpec so that ExportTVM / Jit
// can be called directly on the module, mirroring Python's Module.jit().
//
// Usage:
//   TestModuleNode mod(forward_fn, arg_names, arg_specs);
//   CppModule model = mod.JitCpp("forward", {kDLCPU,0}, "cpu_generic", true);
//   ffi::Any result = model["forward"]({x_tensor});
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
// test_debug_print
//
// Python equivalent:
//   class Layer(nn.Module):
//       def forward(self, x: nn.Tensor):
//           op.print_(x)
//           return x
//
//   model = Layer().jit(
//       spec={"forward": {"x": spec.Tensor([10, 5], dtype="float32")}},
//       debug=True,
//   )
//   x = torch.rand((10, 5), dtype=torch.float32)
//   y = model["forward"](x)
//   assert isinstance(y, torch.Tensor)
// ---------------------------------------------------------------------------
TEST(NNDebug, test_debug_print) {
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

// ---------------------------------------------------------------------------
// test_debug_func
//
// Python equivalent:
//   @tvm.register_global_func("testing.relax.frontend.nn.test_debug_func")
//   def _debug(lineno, tensor, const_int, const_float, const_str, var_int):
//       assert "test_frontend_nn_debug.py" in lineno
//       assert tensor.shape == (10, 5)
//       assert const_int == 1
//       assert const_float == 2.0
//       assert const_str == "test"
//       assert var_int == 8
//
//   class Layer(nn.Module):
//       def forward(self, x: nn.Tensor, v: tir.Var):
//           op.debug_func("testing.relax.frontend.nn.test_debug_func",
//                         x, 1, 2.0, "test", v)
//           return x
//
//   model = Layer().jit(
//       spec={
//           "forward": {
//               "x": spec.Tensor([10, 5], dtype="float32"),
//               "v": "int",
//           },
//       },
//       debug=True,
//   )
//   x = torch.rand((10, 5), dtype=torch.float32)
//   y = model["forward"](x, 8)
//   assert isinstance(y, torch.Tensor)
// ---------------------------------------------------------------------------
TEST(NNDebug, test_debug_func) {
  static const ffi::Function op_debug_func = NNOp("debug_func");

  static std::atomic<bool> g_debug_called{false};
  g_debug_called = false;

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("testing.relax.frontend.nn.test_debug_func_cpp",
                        [](ffi::String lineno, runtime::Tensor tensor, int64_t const_int,
                           double const_float, ffi::String const_str, int64_t var_int) {
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

  const ffi::String kFuncName = "testing.relax.frontend.nn.test_debug_func_cpp";
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

  ffi::Shape v_shape({8});
  ffi::Any result = model["forward"]({ffi::Any(x_tensor), ffi::Any(v_shape)});
  runtime::Tensor y = result.cast<runtime::Tensor>();

  EXPECT_TRUE(g_debug_called.load()) << "Debug callback was not called during VM execution";
  ASSERT_EQ(y.Shape().size(), 2);
  EXPECT_EQ(y.Shape()[0], 10);
  EXPECT_EQ(y.Shape()[1], 5);
}

}  // namespace testing
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
