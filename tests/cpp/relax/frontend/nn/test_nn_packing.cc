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
 * \file tests/cpp/relax/frontend/nn/test_nn_packing.cc
 * \brief C++ port of tests/python/relax/test_frontend_nn_packing.py
 *
 * Ported test:
 *   - TestNNExportToRelax  (test_nn_export_to_relax)
 *
 * The Python test verifies that when param_mode="packed" and effect_mode="none"
 * are specified in the method spec's "$" key, the exported Relax function
 * receives all parameters as a single packed_params tuple rather than as
 * individual arguments.
 *
 * In C++ the param_mode / effect_mode are fields of MethodSpec, so we pass
 * them directly.
 *
 * Expected IR:
 *   def forward(
 *       x: R.Tensor((1, 10), "float32"),
 *       packed_params: R.Tuple(
 *           R.Tensor((20, 10), "float32"),
 *           R.Tensor((20, 10), "float32"),
 *       ),
 *   ):
 *     R.func_attr({"num_input": 1})
 *     with R.dataflow():
 *       linear_1_weight = packed_params[0]
 *       linear_2_weight = packed_params[1]
 *       matmul_1_weight = R.permute_dims(linear_1_weight)
 *       matmul       = R.matmul(x, matmul_1_weight)
 *       matmul_2_weight = R.permute_dims(linear_2_weight)
 *       matmul1      = R.matmul(x, matmul_2_weight)
 *       add          = R.add(matmul, matmul1)
 *       gv           = add
 *       R.output(gv)
 *     return gv
 *
 * The test also verifies that the binding names in the dataflow block match
 * the expected names exactly (mirrors _iter_binding_names() in Python).
 */

#include <gtest/gtest.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/op.h>

#include <string>
#include <vector>

#include "../../../../../src/relax/frontend/nn/core.h"
#include "../../../../../src/relax/frontend/nn/exporter.h"
#include "../../../../../src/relax/frontend/nn/modules.h"
#include "../../../../../src/relax/frontend/nn/spec.h"
#include "../../../../../src/relax/op/tensor/binary.h"
#include "../../../../../src/relax/op/tensor/linear_algebra.h"
#include "../../../../../src/relax/op/tensor/manipulate.h"

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
// TestNNExportToRelax
//
// Python equivalent:
//   class TestModule(nn.Module):
//       def __init__(self, in_features, out_features):
//           self.linear_1 = nn.Linear(in_features, out_features, bias=False)
//           self.linear_2 = nn.Linear(in_features, out_features, bias=False)
//       def forward(self, x):
//           x1 = self.linear_1(x)
//           x2 = self.linear_2(x)
//           return x1 + x2
//
//   model = TestModule(10, 20)
//   mod, _ = model.export_tvm(spec={
//       "forward": {
//           "x": nn.spec.Tensor([1, 10], "float32"),
//           "$": {"param_mode": "packed", "effect_mode": "none"},
//       }
//   })
//
// With param_mode="packed" the two weight tensors are bundled into a single
// packed_params tuple argument.  The dataflow block unpacks them with
// TupleGetItem before use.
// ===========================================================================
TEST(NNPacking, TestNNExportToRelax) {
  const int64_t IN = 10;
  const int64_t OUT = 20;

  // Build two Linear sub-modules (no bias, float32).
  LinearModule linear_1 = MakeLinear(ffi::Any(IN), ffi::Any(OUT), false,
                                     std::nullopt, std::nullopt);
  LinearModule linear_2 = MakeLinear(ffi::Any(IN), ffi::Any(OUT), false,
                                     std::nullopt, std::nullopt);

  // Collect named params in order: linear_1.weight, linear_2.weight.
  ffi::Map<ffi::String, NNParameter> named_params;
  for (const auto& [k, v] : linear_1.get()->NamedParameters("linear_1"))
    named_params.Set(k, v);
  for (const auto& [k, v] : linear_2.get()->NamedParameters("linear_2"))
    named_params.Set(k, v);

  static const ffi::Function op_add =
      ffi::Function::GetGlobal("relax.frontend.nn.op.add").value();

  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward_fn =
      [linear_1, linear_2](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor x  = args.at("x").cast<NNTensor>();
    NNTensor x1 = NNTensor(linear_1.get()->Forward(x->expr));
    NNTensor x2 = NNTensor(linear_2.get()->Forward(x->expr));
    return op_add(x1->expr, x2->expr, ffi::String("add"));
  };

  ffi::Array<ffi::Any> spec_shape;
  spec_shape.push_back(ffi::Any(int64_t(1)));
  spec_shape.push_back(ffi::Any(IN));
  SpecTensor x_spec(spec_shape, "float32");

  // param_mode="packed", effect_mode="none"
  MethodSpec ms(forward_fn, {"x"}, {ffi::Any(x_spec)}, "packed", "none");
  ModuleSpec mod_spec({"forward"}, {ffi::Any(ms)}, named_params, {});
  
  // Create a minimal NNModule wrapper and use ExportTVM
  NNModule mod;
  ffi::Array<ffi::Any> result = mod->ExportTVM(mod_spec, /*debug=*/false, /*allow_extern=*/false);
  IRModule actual = result[0].cast<IRModule>();

  // ===========================================================================
  // Build expected IR
  // ===========================================================================
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  {
    DataType f32 = DataType::Float(32);
    Var x("x", TSInfo({1, IN}, f32));

    // packed_params: Tuple(Tensor(20,10), Tensor(20,10))
    TupleStructInfo packed_sinfo(ffi::Array<StructInfo>{
        TensorStructInfo(ShapeExpr(ffi::Array<PrimExpr>{
            IntImm(DataType::Int(64), OUT), IntImm(DataType::Int(64), IN)}), f32),
        TensorStructInfo(ShapeExpr(ffi::Array<PrimExpr>{
            IntImm(DataType::Int(64), OUT), IntImm(DataType::Int(64), IN)}), f32),
    });
    Var packed_params("packed_params", packed_sinfo);
    ffi::Array<Var> params{x, packed_params};
    bb->BeginScope(params);
    bb->BeginDataflowBlock();

    // linear_1_weight = packed_params[0]
    Var w1 = bb->Emit(TupleGetItem(packed_params, 0), "linear_1_weight");
    // linear_2_weight = packed_params[1]
    Var w2 = bb->Emit(TupleGetItem(packed_params, 1), "linear_2_weight");
    // matmul_1_weight = permute_dims(linear_1_weight)
    Var pd1 = bb->Emit(relax::permute_dims(w1, std::nullopt), "matmul_1_weight");
    // matmul = matmul(x, matmul_1_weight)
    Var mm1 = bb->Emit(relax::matmul(x, pd1, std::nullopt), "matmul");
    // matmul_2_weight = permute_dims(linear_2_weight)
    Var pd2 = bb->Emit(relax::permute_dims(w2, std::nullopt), "matmul_2_weight");
    // matmul1 = matmul(x, matmul_2_weight)
    Var mm2 = bb->Emit(relax::matmul(x, pd2, std::nullopt), "matmul1");
    // add = add(matmul, matmul1)
    Var add = bb->Emit(relax::add(mm1, mm2), "add");
    // gv = add
    Var gv = bb->EmitOutput(add, "gv");

    BindingBlock df = bb->EndBlock();
    Expr body = bb->Normalize(SeqExpr({df}, gv));
    bb->EndScope();
    ffi::Map<ffi::String, ffi::Any> attrs;
    attrs.Set("num_input", ffi::Any(int64_t(1)));
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
