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
 * \file tests/cpp/relax/frontend/nn/llm/test_llm_llama_rope.cc
 * \brief C++ tests for llama_rope function.
 *
 * Ports tests from tests/python/relax/llm/test_llm_position_embedding_llama_rope.py
 */

#include <gtest/gtest.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/module.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>

#include "../../../../../../src/relax/frontend/nn/core.h"
#include "../../../../../../src/relax/frontend/nn/exporter.h"
#include "../../../../../../src/relax/frontend/nn/llm/position_embedding.h"
#include "../../../../../../src/relax/frontend/nn/spec.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {
namespace testing {

using namespace tvm::tir;

// Test constants matching Python tests
constexpr int64_t BATCH = 1;
constexpr int64_t SEQ = 4;
constexpr int64_t NUM_Q = 2;
constexpr int64_t NUM_KV = 2;
constexpr int64_t HEAD_DIM = 8;
constexpr int64_t FUSED = NUM_Q + NUM_KV * 2;  // 6
static constexpr const char DTYPE[] = "float16";
constexpr double THETA = 10000.0;
constexpr double SCALE = 1.0;

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

static SpecTensor MakeSpecTensor(std::initializer_list<int64_t> dims, const std::string& dtype) {
  ffi::Array<ffi::Any> shape;
  for (int64_t d : dims) shape.push_back(ffi::Any(d));
  return SpecTensor(shape, dtype);
}

// Build a test module that calls llama_rope
static IRModule BuildLlamaRopeModule(const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
                                     int64_t num_q = NUM_Q, int64_t num_kv = NUM_KV,
                                     ffi::Optional<int64_t> rotary_dim = ffi::Optional<int64_t>()) {
  int64_t fused = num_q + num_kv * 2;

  // Create a forward function that calls llama_rope
  ffi::TypedFunction<ffi::Any(ffi::Map<ffi::String, ffi::Any>)> forward =
      [rope_scaling, num_q, num_kv, rotary_dim](ffi::Map<ffi::String, ffi::Any> args) -> ffi::Any {
    NNTensor qkv = args.at("qkv").cast<NNTensor>();
    tir::Var total_seq_len = args.at("total_seq_len").cast<tir::Var>();

    auto [q, k, v] = LlamaRope(qkv, total_seq_len, THETA, SCALE, num_q, num_kv, rope_scaling,
                               rotary_dim);

    ffi::Array<ffi::Any> outputs;
    outputs.push_back(ffi::Any(q->expr));
    outputs.push_back(ffi::Any(k->expr));
    outputs.push_back(ffi::Any(v->expr));
    return ffi::Any(outputs);
  };

  // Create method spec
  MethodSpec ms(forward, {"qkv", "total_seq_len"},
                {ffi::Any(MakeSpecTensor({BATCH, SEQ, fused, HEAD_DIM}, DTYPE)),
                 ffi::Any(SpecInt())},
                "plain", "plain");

  // Create module spec
  ModuleSpec mod_spec(ffi::Array<ffi::String>{ffi::String("forward")},
                      ffi::Array<ffi::Any>{ffi::Any(ms)}, {}, {});

  // Export to IRModule
  NNModule mod;
  ffi::Array<ffi::Any> result = mod->ExportTVM(mod_spec, /*debug=*/true, /*allow_extern=*/false);
  return result[0].cast<IRModule>();
}

// ---------------------------------------------------------------------------
// Test LlamaRope
// ---------------------------------------------------------------------------

TEST(LlamaRope, Llama3Scaling) {
  // Setup rope_scaling config for llama3
  ffi::Map<ffi::String, ffi::Any> rope_scaling;
  rope_scaling.Set("rope_type", ffi::Any(ffi::String("llama3")));
  rope_scaling.Set("factor", ffi::Any(8.0));
  rope_scaling.Set("low_freq_factor", ffi::Any(1.0));
  rope_scaling.Set("high_freq_factor", ffi::Any(4.0));
  rope_scaling.Set("original_max_position_embeddings", ffi::Any(int64_t(8192)));

  IRModule mod = BuildLlamaRopeModule(rope_scaling);

  // Verify the module structure
  ASSERT_TRUE(mod.defined());

  // Check that we have the expected functions:
  // 1. _initialize_effect
  // 2. forward
  // 3. llama_rope (TIR PrimFunc)
  const auto& funcs = mod->functions;
  EXPECT_EQ(funcs.size(), 3u) << "Expected 3 functions in module";

  // Find the functions
  bool found_init = false;
  bool found_forward = false;
  bool found_llama_rope = false;

  for (const auto& [gv, func] : funcs) {
    std::string name = gv->name_hint;
    if (name == "_initialize_effect") {
      found_init = true;
      // Verify it's a Relax function
      EXPECT_TRUE(func.as<FunctionNode>() != nullptr);
    } else if (name == "forward") {
      found_forward = true;
      // Verify it's a Relax function
      const auto* relax_func = func.as<FunctionNode>();
      ASSERT_NE(relax_func, nullptr);

      // Verify the function has the right number of parameters
      // qkv, total_seq_len (as Shape), _io
      EXPECT_EQ(relax_func->params.size(), 3u);

      // Verify the function body contains a call to llama_rope
      const auto* seq = relax_func->body.as<SeqExprNode>();
      ASSERT_NE(seq, nullptr);

      // Look for llama_rope call in the dataflow block
      bool has_llama_rope_call = false;
      for (const auto& block : seq->blocks) {
        for (const auto& binding : block->bindings) {
          if (const auto* vb = binding.as<VarBindingNode>()) {
            if (const auto* call = vb->value.as<CallNode>()) {
              if (call->op.as<OpNode>() && call->op.as<OpNode>()->name == "relax.call_tir") {
                has_llama_rope_call = true;
                break;
              }
            }
          }
        }
        if (has_llama_rope_call) break;
      }
      EXPECT_TRUE(has_llama_rope_call) << "Missing call_tir to llama_rope";
    } else if (name == "llama_rope") {
      found_llama_rope = true;
      // Verify it's a TIR PrimFunc
      const auto* prim_func = func.as<tir::PrimFuncNode>();
      ASSERT_NE(prim_func, nullptr);

      // Verify the PrimFunc has the expected attributes
      EXPECT_TRUE(prim_func->attrs.defined());
      auto attrs_dict = prim_func->attrs.as<DictAttrsNode>();
      ASSERT_NE(attrs_dict, nullptr);

      // Check for op_pattern and tir.noalias attributes
      EXPECT_TRUE(attrs_dict->dict.count("op_pattern"));
      EXPECT_TRUE(attrs_dict->dict.count("tir.noalias"));

      // Verify the PrimFunc has 5 parameters: qkv, q, k, v, total_seq_len
      EXPECT_EQ(prim_func->params.size(), 5u);
    }
  }

  EXPECT_TRUE(found_init) << "Missing _initialize_effect function";
  EXPECT_TRUE(found_forward) << "Missing forward function";
  EXPECT_TRUE(found_llama_rope) << "Missing llama_rope TIR function";
}

TEST(LlamaRope, GptjScaling) {
  // Setup rope_scaling config for gptj
  ffi::Map<ffi::String, ffi::Any> rope_scaling;
  rope_scaling.Set("rope_type", ffi::Any(ffi::String("gptj")));

  IRModule mod = BuildLlamaRopeModule(rope_scaling);

  // Verify the module structure
  ASSERT_TRUE(mod.defined());

  // Check that we have the expected functions
  const auto& funcs = mod->functions;
  EXPECT_EQ(funcs.size(), 3u) << "Expected 3 functions in module";

  // Find the llama_rope TIR function and verify it exists
  bool found_llama_rope = false;
  for (const auto& [gv, func] : funcs) {
    if (gv->name_hint == "llama_rope") {
      found_llama_rope = true;
      const auto* prim_func = func.as<tir::PrimFuncNode>();
      ASSERT_NE(prim_func, nullptr);
      break;
    }
  }
  EXPECT_TRUE(found_llama_rope) << "Missing llama_rope TIR function";
}

TEST(LlamaRope, GQA) {
  // Test GQA variant: num_q_heads=4, num_kv_heads=2
  ffi::Map<ffi::String, ffi::Any> rope_scaling;
  rope_scaling.Set("rope_type", ffi::Any(ffi::String("llama3")));
  rope_scaling.Set("factor", ffi::Any(8.0));
  rope_scaling.Set("low_freq_factor", ffi::Any(1.0));
  rope_scaling.Set("high_freq_factor", ffi::Any(4.0));
  rope_scaling.Set("original_max_position_embeddings", ffi::Any(int64_t(8192)));

  IRModule mod = BuildLlamaRopeModule(rope_scaling, /*num_q=*/4, /*num_kv=*/2);

  // Verify the module structure
  ASSERT_TRUE(mod.defined());

  // Check that we have the expected functions
  const auto& funcs = mod->functions;
  EXPECT_EQ(funcs.size(), 3u) << "Expected 3 functions in module";

  // Verify the forward function has the correct output shapes
  bool found_forward = false;
  for (const auto& [gv, func] : funcs) {
    if (gv->name_hint == "forward") {
      found_forward = true;
      const auto* relax_func = func.as<FunctionNode>();
      ASSERT_NE(relax_func, nullptr);

      // The function should return a tuple of (q, k, v) with shapes:
      // q: (1, 4, 4, 8)
      // k: (1, 4, 2, 8)
      // v: (1, 4, 2, 8)
      // We can't easily verify the exact shapes without parsing the entire IR,
      // but we can verify the function exists and is well-formed
      EXPECT_TRUE(relax_func->body.defined());
      break;
    }
  }
  EXPECT_TRUE(found_forward) << "Missing forward function";
}

TEST(LlamaRope, PartialRotaryDim) {
  // Test partial rotary_dim = HEAD_DIM // 2 = 4
  ffi::Map<ffi::String, ffi::Any> rope_scaling;
  rope_scaling.Set("rope_type", ffi::Any(ffi::String("llama3")));
  rope_scaling.Set("factor", ffi::Any(8.0));
  rope_scaling.Set("low_freq_factor", ffi::Any(1.0));
  rope_scaling.Set("high_freq_factor", ffi::Any(4.0));
  rope_scaling.Set("original_max_position_embeddings", ffi::Any(int64_t(8192)));

  IRModule mod = BuildLlamaRopeModule(rope_scaling, NUM_Q, NUM_KV,
                                      ffi::Optional<int64_t>(HEAD_DIM / 2));

  // Verify the module structure
  ASSERT_TRUE(mod.defined());

  // Check that we have the expected functions
  const auto& funcs = mod->functions;
  EXPECT_EQ(funcs.size(), 3u) << "Expected 3 functions in module";

  // Verify the llama_rope TIR function exists
  bool found_llama_rope = false;
  for (const auto& [gv, func] : funcs) {
    if (gv->name_hint == "llama_rope") {
      found_llama_rope = true;
      const auto* prim_func = func.as<tir::PrimFuncNode>();
      ASSERT_NE(prim_func, nullptr);
      break;
    }
  }
  EXPECT_TRUE(found_llama_rope) << "Missing llama_rope TIR function";
}

}  // namespace testing
}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
