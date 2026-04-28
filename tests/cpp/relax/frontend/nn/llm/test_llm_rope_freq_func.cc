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
 * \file tests/cpp/relax/frontend/nn/llm/test_llm_rope_freq_func.cc
 * \brief C++ tests for RoPE frequency functions.
 *
 * Ports tests from tests/python/relax/llm/test_llm_position_embedding_switch_rope_freq_func.py
 */

#include <gtest/gtest.h>
#include <tvm/relax/expr.h>
#include <tvm/tir/expr.h>

#include <cmath>

#include "../../../../../../src/relax/frontend/nn/llm/position_embedding.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {
namespace testing {

using namespace tvm::tir;

// Test constants matching Python tests
constexpr int64_t D_RANGE = 8;
constexpr double THETA = 10000.0;
static constexpr const char DTYPE[] = "float16";

// ---------------------------------------------------------------------------
// Test RoPE frequency functions
// ---------------------------------------------------------------------------

TEST(RopeFreqFunc, RopeFreqDefault) {
  Var s("s", DataType::Int(64));
  Var d("d", DataType::Int(64));
  int64_t d_range = 8;
  double theta = 10000.0;
  std::string dtype = "float32";
  ffi::Map<ffi::String, ffi::Any> extra_args;

  auto result = RopeFreqDefault(s, d, d_range, theta, dtype, extra_args);

  // Verify that cos_freq and sin_freq are defined
  EXPECT_TRUE(result.cos_freq.defined());
  EXPECT_TRUE(result.sin_freq.defined());
  EXPECT_EQ(result.var_map.size(), 1);  // Should have freq_var
}

TEST(RopeFreqFunc, RopeFreqGptj) {
  Var s("s", DataType::Int(64));
  Var d("d", DataType::Int(64));
  int64_t d_range = 8;
  double theta = 10000.0;
  std::string dtype = "float32";
  ffi::Map<ffi::String, ffi::Any> extra_args;

  auto result = RopeFreqGptj(s, d, d_range, theta, dtype, extra_args);

  EXPECT_TRUE(result.cos_freq.defined());
  EXPECT_TRUE(result.sin_freq.defined());
  EXPECT_EQ(result.var_map.size(), 1);
}

TEST(RopeFreqFunc, RopeFreqLlama3) {
  Var s("s", DataType::Int(64));
  Var d("d", DataType::Int(64));
  int64_t d_range = 8;
  double theta = 10000.0;
  std::string dtype = "float32";

  ffi::Map<ffi::String, ffi::Any> extra_args;
  extra_args.Set("factor", ffi::Any(8.0));
  extra_args.Set("low_freq_factor", ffi::Any(1.0));
  extra_args.Set("high_freq_factor", ffi::Any(4.0));
  extra_args.Set("original_max_position_embeddings", ffi::Any(int64_t(8192)));

  auto result = RopeFreqLlama3(s, d, d_range, theta, dtype, extra_args);

  EXPECT_TRUE(result.cos_freq.defined());
  EXPECT_TRUE(result.sin_freq.defined());
  EXPECT_EQ(result.var_map.size(), 2);  // Should have smoothed_freq_var and orig_freq_var
}

TEST(RopeFreqFunc, RopeFreqLlama4) {
  Var s("s", DataType::Int(64));
  Var d("d", DataType::Int(64));
  int64_t d_range = 8;
  double theta = 10000.0;
  std::string dtype = "float32";

  ffi::Map<ffi::String, ffi::Any> extra_args;
  extra_args.Set("factor", ffi::Any(8.0));
  extra_args.Set("low_freq_factor", ffi::Any(1.0));
  extra_args.Set("high_freq_factor", ffi::Any(4.0));
  extra_args.Set("original_max_position_embeddings", ffi::Any(int64_t(8192)));

  auto result = RopeFreqLlama4(s, d, d_range, theta, dtype, extra_args);

  EXPECT_TRUE(result.cos_freq.defined());
  EXPECT_TRUE(result.sin_freq.defined());
  EXPECT_EQ(result.var_map.size(), 2);  // Should have smoothed_freq_var and orig_freq_var
}

TEST(RopeFreqFunc, RopeFreqLongrope) {
  Var s("s", DataType::Int(64));
  Var d("d", DataType::Int(64));
  int64_t d_range = 8;
  double theta = 10000.0;
  std::string dtype = "float32";

  ffi::Map<ffi::String, ffi::Any> extra_args;
  extra_args.Set("max_position_embeddings", ffi::Any(int64_t(131072)));
  extra_args.Set("original_max_position_embeddings", ffi::Any(int64_t(4096)));

  auto result = RopeFreqLongrope(s, d, d_range, theta, dtype, extra_args);

  EXPECT_TRUE(result.cos_freq.defined());
  EXPECT_TRUE(result.sin_freq.defined());
  EXPECT_EQ(result.var_map.size(), 1);

  // Verify scaling factor is applied (sqrt(1 + log(scale) / log(original_max)))
  // scale = 131072 / 4096 = 32
  // scaling_factor = sqrt(1 + log(32) / log(4096)) ≈ 1.19
  // The cos/sin should be multiplied by this factor
}

TEST(RopeFreqFunc, RopeFreqYarn) {
  Var s("s", DataType::Int(64));
  Var d("d", DataType::Int(64));
  int64_t d_range = 8;
  double theta = 10000.0;
  std::string dtype = "float32";

  ffi::Map<ffi::String, ffi::Any> extra_args;
  extra_args.Set("original_max_position_embeddings", ffi::Any(int64_t(8192)));
  extra_args.Set("scaling_factor", ffi::Any(2.0));
  extra_args.Set("beta_fast", ffi::Any(int64_t(32)));
  extra_args.Set("beta_slow", ffi::Any(int64_t(1)));
  extra_args.Set("inv_theta_log_scale", ffi::Any(1.0 / (2.0 * std::log(theta))));

  auto result = RopeFreqYarn(s, d, d_range, theta, dtype, extra_args);

  EXPECT_TRUE(result.cos_freq.defined());
  EXPECT_TRUE(result.sin_freq.defined());
  EXPECT_EQ(result.var_map.size(), 1);
}

// ---------------------------------------------------------------------------
// Test SwitchRopeFreqFunc
// ---------------------------------------------------------------------------

TEST(RopeFreqFunc, SwitchRopeFreqFuncDefault) {
  ffi::Map<ffi::String, ffi::Any> rope_scaling;
  // No rope_type key -> should return RopeFreqDefault

  auto func = SwitchRopeFreqFunc(rope_scaling);
  EXPECT_NE(func, nullptr);

  // Verify it returns the default function by calling it
  Var s("s", DataType::Int(64));
  Var d("d", DataType::Int(64));
  auto result = func(s, d, 8, 10000.0, "float32", rope_scaling);
  EXPECT_TRUE(result.cos_freq.defined());
}

TEST(RopeFreqFunc, SwitchRopeFreqFuncGptj) {
  ffi::Map<ffi::String, ffi::Any> rope_scaling;
  rope_scaling.Set("rope_type", ffi::Any(ffi::String("gptj")));

  auto func = SwitchRopeFreqFunc(rope_scaling);
  EXPECT_NE(func, nullptr);

  Var s("s", DataType::Int(64));
  Var d("d", DataType::Int(64));
  auto result = func(s, d, 8, 10000.0, "float32", rope_scaling);
  EXPECT_TRUE(result.cos_freq.defined());
}

TEST(RopeFreqFunc, SwitchRopeFreqFuncLlama3) {
  ffi::Map<ffi::String, ffi::Any> rope_scaling;
  rope_scaling.Set("rope_type", ffi::Any(ffi::String("llama3")));
  rope_scaling.Set("factor", ffi::Any(8.0));
  rope_scaling.Set("low_freq_factor", ffi::Any(1.0));
  rope_scaling.Set("high_freq_factor", ffi::Any(4.0));
  rope_scaling.Set("original_max_position_embeddings", ffi::Any(int64_t(8192)));

  auto func = SwitchRopeFreqFunc(rope_scaling);
  EXPECT_NE(func, nullptr);

  Var s("s", DataType::Int(64));
  Var d("d", DataType::Int(64));
  auto result = func(s, d, 8, 10000.0, "float32", rope_scaling);
  EXPECT_TRUE(result.cos_freq.defined());
  EXPECT_EQ(result.var_map.size(), 2);  // llama3 has 2 intermediate vars
}

TEST(RopeFreqFunc, SwitchRopeFreqFuncLlama4) {
  ffi::Map<ffi::String, ffi::Any> rope_scaling;
  rope_scaling.Set("rope_type", ffi::Any(ffi::String("llama4")));
  rope_scaling.Set("factor", ffi::Any(8.0));
  rope_scaling.Set("low_freq_factor", ffi::Any(1.0));
  rope_scaling.Set("high_freq_factor", ffi::Any(4.0));
  rope_scaling.Set("original_max_position_embeddings", ffi::Any(int64_t(8192)));

  auto func = SwitchRopeFreqFunc(rope_scaling);
  EXPECT_NE(func, nullptr);

  Var s("s", DataType::Int(64));
  Var d("d", DataType::Int(64));
  auto result = func(s, d, 8, 10000.0, "float32", rope_scaling);
  EXPECT_TRUE(result.cos_freq.defined());
}

TEST(RopeFreqFunc, SwitchRopeFreqFuncLongrope) {
  ffi::Map<ffi::String, ffi::Any> rope_scaling;
  rope_scaling.Set("rope_type", ffi::Any(ffi::String("longrope")));
  rope_scaling.Set("max_position_embeddings", ffi::Any(int64_t(131072)));
  rope_scaling.Set("original_max_position_embeddings", ffi::Any(int64_t(4096)));

  auto func = SwitchRopeFreqFunc(rope_scaling);
  EXPECT_NE(func, nullptr);

  Var s("s", DataType::Int(64));
  Var d("d", DataType::Int(64));
  auto result = func(s, d, 8, 10000.0, "float32", rope_scaling);
  EXPECT_TRUE(result.cos_freq.defined());
}

TEST(RopeFreqFunc, SwitchRopeFreqFuncYarn) {
  ffi::Map<ffi::String, ffi::Any> rope_scaling;
  rope_scaling.Set("rope_type", ffi::Any(ffi::String("yarn")));
  rope_scaling.Set("original_max_position_embeddings", ffi::Any(int64_t(8192)));
  rope_scaling.Set("scaling_factor", ffi::Any(2.0));
  rope_scaling.Set("beta_fast", ffi::Any(int64_t(32)));
  rope_scaling.Set("beta_slow", ffi::Any(int64_t(1)));
  rope_scaling.Set("inv_theta_log_scale", ffi::Any(1.0 / (2.0 * std::log(10000.0))));

  auto func = SwitchRopeFreqFunc(rope_scaling);
  EXPECT_NE(func, nullptr);

  Var s("s", DataType::Int(64));
  Var d("d", DataType::Int(64));
  auto result = func(s, d, 8, 10000.0, "float32", rope_scaling);
  EXPECT_TRUE(result.cos_freq.defined());
}

}  // namespace testing
}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
