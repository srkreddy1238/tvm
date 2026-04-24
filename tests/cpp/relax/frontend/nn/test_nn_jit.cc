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
 * \file tests/cpp/relax/frontend/nn/test_nn_jit.cc
 * \brief C++ port of tests/python/relax/test_frontend_nn_jit.py
 *
 * Ported tests:
 *   - TestJit                  (test_jit)
 *   - TestJitIntInput          (test_jit_int_input)
 *   - TestJitWithEffect        (test_jit_with_effect)
 *   - TestJitTupleInput        (test_jit_tuple_input)
 *   - TestJitListInput         (test_jit_list_input)
 *   - TestJitTupleInputWithInt (test_jit_tuple_input_with_int)
 *
 * Each test:
 *   1. Builds a MethodSpec / ModuleSpec describing the forward function.
 *   2. Calls nn::Jit() to compile to a CppModule.
 *   3. Runs the method with runtime::Tensor inputs.
 *   4. Asserts numerical correctness against a reference computation.
 *
 * The Python tests use torch.Tensor; here we use tvm::runtime::Tensor
 * (backed by DLPack / NDArray) and compare with numpy-style loops.
 */

#include <gtest/gtest.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/tensor.h>
#include <tvm/tir/op.h>

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

// nn frontend headers
#include "../../../../../src/relax/frontend/nn/core.h"
#include "../../../../../src/relax/frontend/nn/cpp_module.h"
#include "../../../../../src/relax/frontend/nn/exporter.h"
#include "../../../../../src/relax/frontend/nn/modules.h"
#include "../../../../../src/relax/frontend/nn/spec.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace testing {

// ===========================================================================
// Helpers
// ===========================================================================

// Retrieve the registered nn.op FFI function by short name.
static ffi::Function NNOp(const std::string& name) {
  auto f = ffi::Function::GetGlobal("relax.frontend.nn.op." + name);
  TVM_FFI_ICHECK(f.has_value()) << "nn.op not found: " << name;
  return f.value();
}

// Build a SpecTensor with all-static dims.
static SpecTensor MakeSpecTensor(std::initializer_list<int64_t> dims, const std::string& dtype) {
  ffi::Array<ffi::Any> shape;
  for (int64_t d : dims) shape.push_back(ffi::Any(d));
  return SpecTensor(shape, dtype);
}

// Allocate a CPU float32 Tensor filled with the given flat values.
static runtime::Tensor MakeF32Tensor(const std::vector<int64_t>& shape,
                                     const std::vector<float>& data) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  EXPECT_EQ(static_cast<int64_t>(data.size()), n);
  auto t = runtime::Tensor::Empty(ffi::Shape(shape), DLDataType{kDLFloat, 32, 1},
                                  DLDevice{kDLCPU, 0}, std::nullopt);
  t.CopyFromBytes(data.data(), n * sizeof(float));
  return t;
}

// Allocate a CPU float32 Tensor filled with a constant value.
static runtime::Tensor MakeF32Const(const std::vector<int64_t>& shape, float val) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  std::vector<float> data(n, val);
  return MakeF32Tensor(shape, data);
}

// Copy a Tensor's data into a flat float vector.
static std::vector<float> ToFloatVec(const runtime::Tensor& t) {
  int64_t n = 1;
  for (int i = 0; i < t.ndim(); ++i) n *= t.shape()[i];
  std::vector<float> out(n);
  t.CopyToBytes(out.data(), n * sizeof(float));
  return out;
}

// Element-wise approximate equality for float vectors.
static bool AllClose(const std::vector<float>& a, const std::vector<float>& b, float atol = 1e-5f) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(a[i] - b[i]) > atol) return false;
  }
  return true;
}

// ===========================================================================
// TestJit
//
// Python equivalent:
//   class Layer(nn.Module):
//       def forward(self, x: nn.Tensor):
//           y = nn.add(x, x)
//           return y
//
//   model = Layer().jit(spec={"forward": {"x": spec.Tensor([10, 5], "float32")}})
//   y = model["forward"](x)
//   assert torch.allclose(x + x, y)
//
// Parametrised over debug=True and debug=False.
// ===========================================================================
class TestJit : public ::testing::TestWithParam<bool> {};

TEST_P(TestJit, AddSelf) {
  const bool debug = GetParam();

  static const ffi::Function op_add = NNOp("add");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward_fn =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    return op_add(x->expr, x->expr, ffi::String("add"));
  };

  MethodSpec ms(forward_fn, {"x"}, {ffi::Any(MakeSpecTensor({10, 5}, "float32"))}, "plain",
                "plain");
  ModuleSpec mod_spec({"forward"}, {ffi::Any(ms)}, {}, {});

  CppModule model = Jit(mod_spec, {kDLCPU, 0}, "cpu_generic", debug);

  // Input: x[i] = i * 0.1f
  std::vector<float> x_data(50);
  for (int i = 0; i < 50; ++i) x_data[i] = i * 0.1f;
  runtime::Tensor x = MakeF32Tensor({10, 5}, x_data);

  ffi::Any result = model["forward"]({ffi::Any(x)});
  runtime::Tensor y = result.cast<runtime::Tensor>();

  // Expected: x + x = 2 * x
  std::vector<float> y_data = ToFloatVec(y);
  std::vector<float> expected(50);
  for (int i = 0; i < 50; ++i) expected[i] = x_data[i] * 2.0f;

  EXPECT_TRUE(AllClose(y_data, expected)) << "TestJit(debug=" << debug << "): output mismatch";
}

INSTANTIATE_TEST_SUITE_P(DebugModes, TestJit, ::testing::Values(true, false));

// ===========================================================================
// TestJitIntInput
//
// Python equivalent:
//   class Layer(nn.Module):
//       def forward(self, x: nn.Tensor, i: tir.Var):
//           y = nn.add(x, x)
//           y = nn.reshape(y, (i, 5, 5))
//           return y
//
//   model = Layer().jit(spec={"forward": {
//       "x": spec.Tensor([10, 5], "float32"), "i": int}})
//   y = model["forward"](x, 2)
//   assert torch.allclose(torch.reshape(x + x, (2, 5, 5)), y)
//
// The SpecInt argument is passed as ffi::Shape({value}) to the CppModule.
// ===========================================================================
class TestJitIntInput : public ::testing::TestWithParam<bool> {};

TEST_P(TestJitIntInput, AddAndReshape) {
  const bool debug = GetParam();

  static const ffi::Function op_add = NNOp("add");
  static const ffi::Function op_reshape = NNOp("reshape");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward_fn =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    tir::Var i = args.at("i").cast<tir::Var>();
    ffi::Any y = op_add(x->expr, x->expr, ffi::String("add"));
    // reshape(y, [i, 5, 5])
    ffi::Array<ffi::Any> new_shape;
    new_shape.push_back(ffi::Any(PrimExpr(i)));
    new_shape.push_back(ffi::Any(int64_t(5)));
    new_shape.push_back(ffi::Any(int64_t(5)));
    return op_reshape(y.cast<Var>(), new_shape, ffi::String("reshape"));
  };

  SpecInt spec_int;
  MethodSpec ms(forward_fn, {"x", "i"},
                {ffi::Any(MakeSpecTensor({10, 5}, "float32")), ffi::Any(spec_int)}, "plain",
                "plain");
  ModuleSpec mod_spec({"forward"}, {ffi::Any(ms)}, {}, {});

  CppModule model = Jit(mod_spec, {kDLCPU, 0}, "cpu_generic", debug);

  std::vector<float> x_data(50);
  for (int i = 0; i < 50; ++i) x_data[i] = i * 0.1f;
  runtime::Tensor x = MakeF32Tensor({10, 5}, x_data);

  // Pass i=2 as ffi::Shape({2}) — mirrors Python's ShapeTuple
  ffi::Any result = model["forward"]({ffi::Any(x), ffi::Any(ffi::Shape({2}))});
  runtime::Tensor y = result.cast<runtime::Tensor>();

  // Shape should be [2, 5, 5]
  ASSERT_EQ(y.ndim(), 3);
  EXPECT_EQ(y.shape()[0], 2);
  EXPECT_EQ(y.shape()[1], 5);
  EXPECT_EQ(y.shape()[2], 5);

  // Values: reshape(x + x, [2, 5, 5])
  std::vector<float> y_data = ToFloatVec(y);
  std::vector<float> expected(50);
  for (int i = 0; i < 50; ++i) expected[i] = x_data[i] * 2.0f;

  EXPECT_TRUE(AllClose(y_data, expected))
      << "TestJitIntInput(debug=" << debug << "): output mismatch";
}

INSTANTIATE_TEST_SUITE_P(DebugModes, TestJitIntInput, ::testing::Values(true, false));

// ===========================================================================
// TestJitWithEffect
//
// Python equivalent:
//   class Layer(nn.Module):
//       def __init__(self):
//           self.cache = nn.KVCache(10, [10, 5])
//       def forward(self, x: nn.Tensor, total_seq_len: tir.Var):
//           self.cache.append(x)
//           y = self.cache.view(total_seq_len)
//           return y
//
//   model = Layer().jit(spec={"forward": {
//       "x": spec.Tensor([1, 10, 5], "float32"), "total_seq_len": int}})
//
//   x0 = rand(1, 10, 5); y = model["forward"](x0, 1)
//   assert allclose(x0, y)
//   x1 = rand(1, 10, 5); y = model["forward"](x1, 2)
//   assert allclose(concat([x0, x1], dim=0), y)
//   x2 = rand(1, 10, 5); y = model["forward"](x2, 3)
//   assert allclose(concat([x0, x1, x2], dim=0), y)
// ===========================================================================
class TestJitWithEffect : public ::testing::TestWithParam<bool> {};

TEST_P(TestJitWithEffect, KVCacheAppendView) {
  const bool debug = GetParam();

  // KVCache(init_seq_len=10, unit_shape=[10, 5], dtype="float32")
  KVCacheModule kv(/*init_seq_len=*/10,
                   /*unit_shape=*/ffi::Array<Integer>{Integer(10), Integer(5)},
                   /*dtype=*/"float32");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward_fn =
      [kv](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x = args.at("x").cast<NNTensor>();
    tir::Var total_seq_len = args.at("total_seq_len").cast<tir::Var>();
    kv.get()->Append(x);
    NNTensor view = kv.get()->View(PrimExpr(total_seq_len));
    return ffi::Any(view);
  };

  SpecInt spec_int;
  MethodSpec ms(forward_fn, {"x", "total_seq_len"},
                {ffi::Any(MakeSpecTensor({1, 10, 5}, "float32")), ffi::Any(spec_int)}, "plain",
                "plain");

  ffi::Map<ffi::String, runtime::ObjectRef> named_effects;
  named_effects.Set("cache", kv);

  ModuleSpec mod_spec({"forward"}, {ffi::Any(ms)}, {}, named_effects);

  CppModule model = Jit(mod_spec, {kDLCPU, 0}, "cpu_generic", debug);

  // Helper: fill a [1, 10, 5] tensor with a constant value.
  auto make_input = [](float val) -> runtime::Tensor { return MakeF32Const({1, 10, 5}, val); };

  // --- Step 1: append x0, view seq_len=1 → should equal x0 ---
  runtime::Tensor x0 = make_input(1.0f);
  ffi::Any r0 = model["forward"]({ffi::Any(x0), ffi::Any(ffi::Shape({1}))});
  runtime::Tensor y0 = r0.cast<runtime::Tensor>();

  ASSERT_EQ(y0.ndim(), 3);
  EXPECT_EQ(y0.shape()[0], 1);
  EXPECT_EQ(y0.shape()[1], 10);
  EXPECT_EQ(y0.shape()[2], 5);
  EXPECT_TRUE(AllClose(ToFloatVec(y0), ToFloatVec(x0))) << "TestJitWithEffect step 1 mismatch";

  // --- Step 2: append x1, view seq_len=2 → should equal concat([x0, x1]) ---
  runtime::Tensor x1 = make_input(2.0f);
  ffi::Any r1 = model["forward"]({ffi::Any(x1), ffi::Any(ffi::Shape({2}))});
  runtime::Tensor y1 = r1.cast<runtime::Tensor>();

  ASSERT_EQ(y1.shape()[0], 2);
  std::vector<float> y1_data = ToFloatVec(y1);
  // First half should be x0 (all 1.0), second half x1 (all 2.0)
  std::vector<float> expected1(2 * 10 * 5);
  std::fill(expected1.begin(), expected1.begin() + 50, 1.0f);
  std::fill(expected1.begin() + 50, expected1.end(), 2.0f);
  EXPECT_TRUE(AllClose(y1_data, expected1)) << "TestJitWithEffect step 2 mismatch";

  // --- Step 3: append x2, view seq_len=3 → concat([x0, x1, x2]) ---
  runtime::Tensor x2 = make_input(3.0f);
  ffi::Any r2 = model["forward"]({ffi::Any(x2), ffi::Any(ffi::Shape({3}))});
  runtime::Tensor y2 = r2.cast<runtime::Tensor>();

  ASSERT_EQ(y2.shape()[0], 3);
  std::vector<float> y2_data = ToFloatVec(y2);
  std::vector<float> expected2(3 * 10 * 5);
  std::fill(expected2.begin(), expected2.begin() + 50, 1.0f);
  std::fill(expected2.begin() + 50, expected2.begin() + 100, 2.0f);
  std::fill(expected2.begin() + 100, expected2.end(), 3.0f);
  EXPECT_TRUE(AllClose(y2_data, expected2)) << "TestJitWithEffect step 3 mismatch";
}

INSTANTIATE_TEST_SUITE_P(DebugModes, TestJitWithEffect, ::testing::Values(true, false));

// ===========================================================================
// TestJitTupleInput
//
// Python equivalent:
//   class Layer(nn.Module):
//       def forward(self, x: tuple[nn.Tensor, nn.Tensor]):
//           x0, x1 = x
//           return (nn.add(x0, x1), nn.subtract(x0, x1))
//
//   model = Layer().jit(spec={"forward": {"x": (
//       spec.Tensor([10, 5], "float32"), spec.Tensor([10, 5], "float32"))}})
//   y = model["forward"](x)
//   assert allclose(x0 + x1, y[0])
//   assert allclose(x0 - x1, y[1])
// ===========================================================================
class TestJitTupleInput : public ::testing::TestWithParam<bool> {};

TEST_P(TestJitTupleInput, AddSubtract) {
  const bool debug = GetParam();

  static const ffi::Function op_add = NNOp("add");
  static const ffi::Function op_sub = NNOp("subtract");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward_fn =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    ffi::Array<ffi::Any> x_arr = args.at("x").cast<ffi::Array<ffi::Any>>();
    NNTensor x0 = x_arr[0].cast<NNTensor>();
    NNTensor x1 = x_arr[1].cast<NNTensor>();
    ffi::Array<ffi::Any> out;
    out.push_back(op_add(x0->expr, x1->expr, ffi::String("add")));
    out.push_back(op_sub(x0->expr, x1->expr, ffi::String("subtract")));
    return ffi::Any(out);
  };

  SpecTensor elem_spec = MakeSpecTensor({10, 5}, "float32");
  ffi::Array<ffi::Any> tuple_elems{ffi::Any(elem_spec), ffi::Any(elem_spec)};
  SpecTuple x_spec("x", tuple_elems, /*is_tuple=*/true);

  MethodSpec ms(forward_fn, {"x"}, {ffi::Any(x_spec)}, "plain", "plain");
  ModuleSpec mod_spec({"forward"}, {ffi::Any(ms)}, {}, {});

  CppModule model = Jit(mod_spec, {kDLCPU, 0}, "cpu_generic", debug);

  std::vector<float> a_data(50), b_data(50);
  for (int i = 0; i < 50; ++i) {
    a_data[i] = i * 0.1f;
    b_data[i] = i * 0.05f;
  }
  runtime::Tensor x0 = MakeF32Tensor({10, 5}, a_data);
  runtime::Tensor x1 = MakeF32Tensor({10, 5}, b_data);

  // Pass the tuple as ffi::Array<ffi::Any>
  ffi::Array<ffi::Any> x_input{ffi::Any(x0), ffi::Any(x1)};
  ffi::Any result = model["forward"]({ffi::Any(x_input)});
  ffi::Array<ffi::Any> outputs = result.cast<ffi::Array<ffi::Any>>();

  ASSERT_EQ(outputs.size(), 2u);
  runtime::Tensor y0 = outputs[0].cast<runtime::Tensor>();
  runtime::Tensor y1 = outputs[1].cast<runtime::Tensor>();

  std::vector<float> y0_data = ToFloatVec(y0);
  std::vector<float> y1_data = ToFloatVec(y1);

  std::vector<float> expected_add(50), expected_sub(50);
  for (int i = 0; i < 50; ++i) {
    expected_add[i] = a_data[i] + b_data[i];
    expected_sub[i] = a_data[i] - b_data[i];
  }

  EXPECT_TRUE(AllClose(y0_data, expected_add))
      << "TestJitTupleInput(debug=" << debug << "): add mismatch";
  EXPECT_TRUE(AllClose(y1_data, expected_sub))
      << "TestJitTupleInput(debug=" << debug << "): subtract mismatch";
}

INSTANTIATE_TEST_SUITE_P(DebugModes, TestJitTupleInput, ::testing::Values(true, false));

// ===========================================================================
// TestJitListInput
//
// Python equivalent:
//   class Layer(nn.Module):
//       def forward(self, x: list[nn.Tensor]):
//           x0, x1 = x
//           return (nn.add(x0, x1), nn.subtract(x0, x1))
//
// Identical to TestJitTupleInput except SpecTuple is constructed with
// is_tuple=false (list semantics).  The runtime behaviour is the same.
// ===========================================================================
class TestJitListInput : public ::testing::TestWithParam<bool> {};

TEST_P(TestJitListInput, AddSubtract) {
  const bool debug = GetParam();

  static const ffi::Function op_add = NNOp("add");
  static const ffi::Function op_sub = NNOp("subtract");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward_fn =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    ffi::Array<ffi::Any> x_arr = args.at("x").cast<ffi::Array<ffi::Any>>();
    NNTensor x0 = x_arr[0].cast<NNTensor>();
    NNTensor x1 = x_arr[1].cast<NNTensor>();
    ffi::Array<ffi::Any> out;
    out.push_back(op_add(x0->expr, x1->expr, ffi::String("add")));
    out.push_back(op_sub(x0->expr, x1->expr, ffi::String("subtract")));
    return ffi::Any(out);
  };

  SpecTensor elem_spec = MakeSpecTensor({10, 5}, "float32");
  ffi::Array<ffi::Any> list_elems{ffi::Any(elem_spec), ffi::Any(elem_spec)};
  // is_tuple=false → list semantics (same IR, different Python annotation)
  SpecTuple x_spec("x", list_elems, /*is_tuple=*/false);

  MethodSpec ms(forward_fn, {"x"}, {ffi::Any(x_spec)}, "plain", "plain");
  ModuleSpec mod_spec({"forward"}, {ffi::Any(ms)}, {}, {});

  CppModule model = Jit(mod_spec, {kDLCPU, 0}, "cpu_generic", debug);

  std::vector<float> a_data(50), b_data(50);
  for (int i = 0; i < 50; ++i) {
    a_data[i] = i * 0.1f;
    b_data[i] = i * 0.05f;
  }
  runtime::Tensor x0 = MakeF32Tensor({10, 5}, a_data);
  runtime::Tensor x1 = MakeF32Tensor({10, 5}, b_data);

  ffi::Array<ffi::Any> x_input{ffi::Any(x0), ffi::Any(x1)};
  ffi::Any result = model["forward"]({ffi::Any(x_input)});
  ffi::Array<ffi::Any> outputs = result.cast<ffi::Array<ffi::Any>>();

  ASSERT_EQ(outputs.size(), 2u);
  runtime::Tensor y0 = outputs[0].cast<runtime::Tensor>();
  runtime::Tensor y1 = outputs[1].cast<runtime::Tensor>();

  std::vector<float> y0_data = ToFloatVec(y0);
  std::vector<float> y1_data = ToFloatVec(y1);

  std::vector<float> expected_add(50), expected_sub(50);
  for (int i = 0; i < 50; ++i) {
    expected_add[i] = a_data[i] + b_data[i];
    expected_sub[i] = a_data[i] - b_data[i];
  }

  EXPECT_TRUE(AllClose(y0_data, expected_add))
      << "TestJitListInput(debug=" << debug << "): add mismatch";
  EXPECT_TRUE(AllClose(y1_data, expected_sub))
      << "TestJitListInput(debug=" << debug << "): subtract mismatch";
}

INSTANTIATE_TEST_SUITE_P(DebugModes, TestJitListInput, ::testing::Values(true, false));

// ===========================================================================
// TestJitTupleInputWithInt
//
// Python equivalent:
//   class Layer(nn.Module):
//       def forward(self, x: tuple[nn.Tensor, nn.Tensor, int]):
//           x0, x1, i = x
//           y0 = nn.add(x0, x1)
//           y1 = nn.subtract(x0, x1)
//           y2 = nn.reshape(x0, (5, i, 5))
//           return (y0, y1, y2)
//
//   model = Layer().jit(spec={"forward": {"x": (
//       spec.Tensor([10, 5], "float32"),
//       spec.Tensor([10, 5], "float32"),
//       int)}})
//   x = (x0, x1, 2)
//   y0, y1, y2 = model["forward"](x)
//   assert allclose(x0 + x1, y0)
//   assert allclose(x0 - x1, y1)
//   assert allclose(reshape(x0, (5, 2, 5)), y2)
// ===========================================================================
class TestJitTupleInputWithInt : public ::testing::TestWithParam<bool> {};

TEST_P(TestJitTupleInputWithInt, AddSubtractReshape) {
  const bool debug = GetParam();

  static const ffi::Function op_add = NNOp("add");
  static const ffi::Function op_sub = NNOp("subtract");
  static const ffi::Function op_reshape = NNOp("reshape");

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward_fn =
      [](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    ffi::Array<ffi::Any> x_arr = args.at("x").cast<ffi::Array<ffi::Any>>();
    NNTensor x0 = x_arr[0].cast<NNTensor>();
    NNTensor x1 = x_arr[1].cast<NNTensor>();
    tir::Var i = x_arr[2].cast<tir::Var>();

    ffi::Any y0 = op_add(x0->expr, x1->expr, ffi::String("add"));
    ffi::Any y1 = op_sub(x0->expr, x1->expr, ffi::String("subtract"));

    // reshape(x0, [5, i, 5])
    ffi::Array<ffi::Any> new_shape;
    new_shape.push_back(ffi::Any(int64_t(5)));
    new_shape.push_back(ffi::Any(PrimExpr(i)));
    new_shape.push_back(ffi::Any(int64_t(5)));
    ffi::Any y2 = op_reshape(x0->expr, new_shape, ffi::String("reshape"));

    ffi::Array<ffi::Any> out;
    out.push_back(y0);
    out.push_back(y1);
    out.push_back(y2);
    return ffi::Any(out);
  };

  SpecTensor tensor_spec = MakeSpecTensor({10, 5}, "float32");
  SpecInt int_spec;
  ffi::Array<ffi::Any> tuple_elems{ffi::Any(tensor_spec), ffi::Any(tensor_spec),
                                   ffi::Any(int_spec)};
  SpecTuple x_spec("x", tuple_elems, /*is_tuple=*/true);

  MethodSpec ms(forward_fn, {"x"}, {ffi::Any(x_spec)}, "plain", "plain");
  ModuleSpec mod_spec({"forward"}, {ffi::Any(ms)}, {}, {});

  CppModule model = Jit(mod_spec, {kDLCPU, 0}, "cpu_generic", debug);

  std::vector<float> a_data(50), b_data(50);
  for (int i = 0; i < 50; ++i) {
    a_data[i] = i * 0.1f;
    b_data[i] = i * 0.05f;
  }
  runtime::Tensor x0 = MakeF32Tensor({10, 5}, a_data);
  runtime::Tensor x1 = MakeF32Tensor({10, 5}, b_data);

  // Tuple: (x0, x1, 2)  — int 2 passed as ffi::Shape({2})
  ffi::Array<ffi::Any> x_input{ffi::Any(x0), ffi::Any(x1), ffi::Any(ffi::Shape({2}))};
  ffi::Any result = model["forward"]({ffi::Any(x_input)});
  ffi::Array<ffi::Any> outputs = result.cast<ffi::Array<ffi::Any>>();

  ASSERT_EQ(outputs.size(), 3u);
  runtime::Tensor y0 = outputs[0].cast<runtime::Tensor>();
  runtime::Tensor y1 = outputs[1].cast<runtime::Tensor>();
  runtime::Tensor y2 = outputs[2].cast<runtime::Tensor>();

  // y0 = x0 + x1
  std::vector<float> y0_data = ToFloatVec(y0);
  std::vector<float> expected_add(50);
  for (int i = 0; i < 50; ++i) expected_add[i] = a_data[i] + b_data[i];
  EXPECT_TRUE(AllClose(y0_data, expected_add))
      << "TestJitTupleInputWithInt(debug=" << debug << "): add mismatch";

  // y1 = x0 - x1
  std::vector<float> y1_data = ToFloatVec(y1);
  std::vector<float> expected_sub(50);
  for (int i = 0; i < 50; ++i) expected_sub[i] = a_data[i] - b_data[i];
  EXPECT_TRUE(AllClose(y1_data, expected_sub))
      << "TestJitTupleInputWithInt(debug=" << debug << "): subtract mismatch";

  // y2 = reshape(x0, [5, 2, 5])
  ASSERT_EQ(y2.ndim(), 3);
  EXPECT_EQ(y2.shape()[0], 5);
  EXPECT_EQ(y2.shape()[1], 2);
  EXPECT_EQ(y2.shape()[2], 5);
  std::vector<float> y2_data = ToFloatVec(y2);
  // reshape is a view — values are the same as x0 in row-major order
  EXPECT_TRUE(AllClose(y2_data, a_data))
      << "TestJitTupleInputWithInt(debug=" << debug << "): reshape mismatch";
}

INSTANTIATE_TEST_SUITE_P(DebugModes, TestJitTupleInputWithInt, ::testing::Values(true, false));

}  // namespace testing
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
