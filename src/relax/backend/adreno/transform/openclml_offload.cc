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
 * \file src/relax/backend/adreno/transform/openclml_offload.cc
 * \brief OpenCLML Offloading based on pattern matching
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/attrs/statistical.h>
#include <tvm/relax/backend/adreno/transform.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/stmt_functor.h>

#include <utility>

#define RET_FALSE \
  {               \
    *ret = false; \
    return;       \
  }

namespace tvm {
namespace relax {
namespace backend {
namespace adreno {

using tvm::relax::transform::FusionPattern;
using tvm::relax::transform::PatternCheckContext;

namespace {

template <typename... Args>
ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> PopulatePatterns(
    const ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>>& patterns, ffi::String name,
    ExprPattern op, ffi::Map<ffi::String, DFPattern> annotations, Args&&... args) {
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> ret_patterns;

  for (const auto& it : patterns) {
    ffi::Map<ffi::String, DFPattern> ret_ann(
        it.second["annotation"].cast<ffi::Map<ffi::String, DFPattern>>());

    for (const auto& jt : annotations) {
      ret_ann.Set(jt.first, jt.second);
    }

    ret_patterns.Set(
        name + "." + it.first,
        {{"pattern", op(it.second["pattern"].cast<DFPattern>(), std::forward<Args>(args)...)},
         {"annotation", ffi::Map<ffi::String, DFPattern>(ret_ann)}});
  }

  return ret_patterns;
}

void AppendPatterns(ffi::Array<FusionPattern>* clml_patterns,
                    const ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>>& patterns,
                    ffi::Optional<ffi::Function> check_func) {
  // Build FusionPatterns for all
  ffi::Array<FusionPattern> f_patterns;
  for (const auto& it : patterns) {
    auto ret_annotations = it.second["annotation"].cast<ffi::Map<ffi::String, DFPattern>>();
    ret_annotations.Set("root", it.second["pattern"].cast<DFPattern>());
    f_patterns.push_back(
        FusionPattern("openclml." + it.first, it.second["pattern"].cast<DFPattern>(),
                      ffi::Map<ffi::String, DFPattern>(ret_annotations), check_func, std::nullopt));
  }

  // Reverse to ensure longer patterns to be selected first
  for (size_t i = f_patterns.size(); i > 0; --i) {
    clml_patterns->push_back(f_patterns[i - 1]);
  }
}

// Conv2D
void Conv2DPatterns(ffi::Array<FusionPattern>* clml_patterns) {
  DFPattern data = Wildcard();
  DFPattern weight = Wildcard();
  DFPattern bias = IsConst();
  DFPattern bn_scale = IsConst();
  DFPattern bn_bias = IsConst();
  DFPattern bn_mean = IsConst();
  DFPattern bn_variance = IsConst();
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> patterns;

  // Check function
  auto CheckConv2D = ffi::Function::FromPacked(
      [](const ffi::AnyView* args, int32_t num_args, ffi::Any* ret) mutable {
        TVM_FFI_ICHECK(num_args == 1) << "Expected 1 argument PatternCheckContext";
        auto context = args[0].cast<PatternCheckContext>();
        *ret = true;

        if (context->annotated_expr.find("root") != context->annotated_expr.end()) {
          auto root_call = Downcast<Call>(context->annotated_expr["root"]);
          if (root_call->op == Op::Get("relax.nn.conv2d")) {
            const auto* attrs = root_call->attrs.as<Conv2DAttrs>();
            if (attrs->data_layout != "NCHW" || attrs->kernel_layout != "OIHW") {
              *ret = false;
            }
          } else if (root_call->op == Op::Get("relax.nn.conv2d_transpose")) {
            const auto* attrs = root_call->attrs.as<Conv2DTransposeAttrs>();
            if (attrs->data_layout != "NCHW" || attrs->kernel_layout != "OIHW") {
              *ret = false;
            }
          }
        }

        for (const auto& arg : {"data", "weight"}) {
          if (context->annotated_expr.find(arg) != context->annotated_expr.end()) {
            auto tsinfo = Downcast<TensorStructInfo>(GetStructInfo(context->annotated_expr[arg]));
            if (tsinfo->dtype != DataType::Float(32) && tsinfo->dtype != DataType::Float(16)) {
              *ret = false;
            }
          }
        }
      });

  // Initial annotations
  ffi::Map<ffi::String, DFPattern> annotations = {{"data", data}, {"weight", weight}};

  // Patterns: Conv2D, Pad+Conv2D, Conv2DTranspose
  patterns.Set("nn.conv2d", {{"pattern", IsOp("relax.nn.conv2d")(data, weight)},
                             {"annotation", ffi::Map<ffi::String, DFPattern>(annotations)}});
  ffi::Map<ffi::String, DFPattern> pad_annotations(annotations);
  patterns.Set("pad.nn.conv2d",
               {{"pattern", IsOp("relax.nn.conv2d")(IsOp("relax.nn.pad")(data), weight)},
                {"annotation", pad_annotations}});
  patterns.Set("nn.conv2d_transpose",
               {{"pattern", IsOp("relax.nn.conv2d_transpose")(data, weight)},
                {"annotation", ffi::Map<ffi::String, DFPattern>(annotations)}});

  // Optional Bias addition for all previous patterns
  auto bias_patterns = PopulatePatterns(patterns, "bias", IsOp("relax.add"),
                                        ffi::Map<ffi::String, DFPattern>({{"bias", bias}}), bias);
  for (const auto& it : bias_patterns) patterns.Set(it.first, it.second);

  // Optional Batchnorm addition for all previous patterns
  auto bn_patterns = PopulatePatterns(patterns, "bn", IsOp("relax.nn.batch_norm"),
                                      ffi::Map<ffi::String, DFPattern>({{"bn_scale", bn_scale},
                                                                        {"bn_bias", bn_bias},
                                                                        {"bn_mean", bn_mean},
                                                                        {"bn_var", bn_variance}}),
                                      bn_scale, bn_bias, bn_mean, bn_variance);
  for (const auto& it : bn_patterns) patterns.Set(it.first, it.second);

  // Optional TuplegetItem after Batchnorm
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> tuple_patterns;
  for (const auto& it : patterns) {
    ffi::Map<ffi::String, DFPattern> tuple_ann(
        it.second["annotation"].cast<ffi::Map<ffi::String, DFPattern>>());
    tuple_patterns.Set("tuple." + it.first,
                       {{"pattern", IsTupleGetItem(it.second["pattern"].cast<DFPattern>(), 0)},
                        {"annotation", ffi::Map<ffi::String, DFPattern>(tuple_ann)}});
  }
  for (const auto& it : tuple_patterns) patterns.Set(it.first, it.second);

  // Optional Relu or Clip (Relu6 realized as Clip by frontend).
  auto relu_patterns = PopulatePatterns(patterns, "relu", IsOp("relax.nn.relu"),
                                        ffi::Map<ffi::String, DFPattern>({}));
  auto clip_patterns =
      PopulatePatterns(patterns, "clip", IsOp("relax.clip"), ffi::Map<ffi::String, DFPattern>({}));
  for (const auto& it : relu_patterns) patterns.Set(it.first, it.second);
  for (const auto& it : clip_patterns) patterns.Set(it.first, it.second);

  AppendPatterns(clml_patterns, patterns, CheckConv2D);
}

// Batchnorm
void BatchNormPatterns(ffi::Array<FusionPattern>* clml_patterns) {
  DFPattern data = Wildcard();
  DFPattern bn_scale = IsConst();
  DFPattern bn_bias = IsConst();
  DFPattern bn_mean = IsConst();
  DFPattern bn_variance = IsConst();
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> patterns;

  // Check function
  auto CheckBatchNorm = ffi::Function::FromPacked(
      [](const ffi::AnyView* args, int32_t num_args, ffi::Any* ret) mutable {
        TVM_FFI_ICHECK(num_args == 1) << "Expected 1 argument PatternCheckContext";
        auto context = args[0].cast<PatternCheckContext>();
        tvm::arith::Analyzer analyzer;
        *ret = true;

        if (context->annotated_expr.find("root") != context->annotated_expr.end()) {
          auto root_call = Downcast<Call>(context->annotated_expr["root"]);

          if (root_call->op != Op::Get("relax.reshape")) {
            RET_FALSE
          }
        }

        ffi::Array<PrimExpr> base_shape;
        for (const auto& param : {"moving_var", "gamma", "moving_mean", "beta"}) {
          if (context->annotated_expr.find(param) == context->annotated_expr.end()) {
            RET_FALSE
          }

          if (!context->annotated_expr[param].as<ConstantNode>()) {
            RET_FALSE
          }
          auto tsinfo = Downcast<TensorStructInfo>(GetStructInfo(context->annotated_expr[param]));
          if (tsinfo->dtype != DataType::Float(32)) {
            RET_FALSE
          }
          if (!strcmp(param, "moving_var")) {  // Should be 1 st key on above list.
            base_shape = tsinfo->GetShape().value();
          } else {
            auto arg_shape = tsinfo->GetShape().value();

            if (arg_shape.size() != base_shape.size()) {
              RET_FALSE
            }

            for (size_t i = 0; i < arg_shape.size(); ++i) {
              if (!analyzer.CanProveEqual(base_shape[i], arg_shape[i])) {
                RET_FALSE
              }
            }
          }
        }
      });

  // Annotations
  ffi::Map<ffi::String, DFPattern> annotations = {{"data", data},
                                                  {"gamma", bn_scale},
                                                  {"beta", bn_bias},
                                                  {"moving_mean", bn_mean},
                                                  {"moving_var", bn_variance}};

  // Pattern
  patterns.Set(
      "nn.batch_norm",
      {{"pattern",
        IsOp("relax.reshape")(
            IsTupleGetItem(
                IsOp("relax.nn.batch_norm")(data, bn_scale, bn_bias, bn_mean, bn_variance), 0),
            Wildcard())},
       {"annotation", annotations}});

  AppendPatterns(clml_patterns, patterns, CheckBatchNorm);
}

// Pool
void PoolPatterns(ffi::Array<FusionPattern>* clml_patterns) {
  DFPattern data = Wildcard();
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> patterns;

  // Check function
  auto CheckPool = ffi::Function::FromPacked(
      [](const ffi::AnyView* args, int32_t num_args, ffi::Any* ret) mutable {
        TVM_FFI_ICHECK(num_args == 1) << "Expected 1 argument PatternCheckContext";
        auto context = args[0].cast<PatternCheckContext>();
        tvm::arith::Analyzer analyzer;
        *ret = true;

        if (context->annotated_expr.find("root") != context->annotated_expr.end()) {
          auto root_call = Downcast<Call>(context->annotated_expr["root"]);

          if ((root_call->op != Op::Get("relax.nn.max_pool2d")) &&
              (root_call->op != Op::Get("relax.nn.avg_pool2d"))) {
            RET_FALSE
          }

          const auto* attrs = root_call->attrs.as<Pool2DAttrs>();

          if (attrs->pool_size.size() != 2) *ret = false;
          for (const auto& dim : attrs->pool_size) {
            if (dim <= 0) *ret = false;
          }

          if (attrs->strides.size() != 2) *ret = false;
          for (const auto& dim : attrs->strides) {
            if (dim <= 0) *ret = false;
          }

          if (attrs->padding.size() != 4) *ret = false;
          for (const auto& dim : attrs->padding) {
            if (dim < 0) *ret = false;
          }
        } else {
          *ret = false;
        }

        if (!(ret->cast<bool>())) return;

        if (context->annotated_expr.find("data") != context->annotated_expr.end()) {
          auto tsinfo = Downcast<TensorStructInfo>(GetStructInfo(context->annotated_expr["data"]));
          if (tsinfo->ndim != 4) *ret = false;
          for (const auto& dim : tsinfo->GetShape().value()) {
            if (!analyzer.CanProveGreaterEqual(dim, 0)) *ret = false;
          }
        } else {
          *ret = false;
        }
      });

  // Patterns
  patterns.Set("nn.max_pool2d",
               {{"pattern", IsOp("relax.nn.max_pool2d")(data)},
                {"annotation", ffi::Map<ffi::String, DFPattern>({{"data", data}})}});
  patterns.Set("nn.avg_pool2d",
               {{"pattern", IsOp("relax.nn.avg_pool2d")(data)},
                {"annotation", ffi::Map<ffi::String, DFPattern>({{"data", data}})}});

  AppendPatterns(clml_patterns, patterns, CheckPool);
}

// Global Avg Pool
void GlobalAvgPatterns(ffi::Array<FusionPattern>* clml_patterns) {
  DFPattern data = Wildcard();
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> patterns;

  // Check function
  auto CheckGlobalPool = ffi::Function::FromPacked(
      [](const ffi::AnyView* args, int32_t num_args, ffi::Any* ret) mutable {
        TVM_FFI_ICHECK(num_args == 1) << "Expected 1 argument PatternCheckContext";
        auto context = args[0].cast<PatternCheckContext>();
        tvm::arith::Analyzer analyzer;
        *ret = true;

        if (context->annotated_expr.find("root") != context->annotated_expr.end()) {
          auto root_call = Downcast<Call>(context->annotated_expr["root"]);

          if (root_call->op != Op::Get("relax.mean")) {
            RET_FALSE
          }

          const auto* attrs = root_call->attrs.as<StatisticalAttrs>();

          if (!attrs->axis.has_value()) {
            RET_FALSE
          }

          if (attrs->axis.value().size() != 2) {
            *ret = false;
          }

          if (!analyzer.CanProveEqual(attrs->axis.value()[0], 2)) *ret = false;
          if (!analyzer.CanProveEqual(attrs->axis.value()[1], 3)) *ret = false;

        } else {
          *ret = false;
        }

        if (!(ret->cast<bool>())) return;

        if (context->annotated_expr.find("data") != context->annotated_expr.end()) {
          auto tsinfo = Downcast<TensorStructInfo>(GetStructInfo(context->annotated_expr["data"]));
          if (tsinfo->ndim != 4) *ret = false;
          for (size_t i = 1; i < 4; ++i) {
            if (!analyzer.CanProveGreaterEqual(tsinfo->GetShape().value()[i], 0)) *ret = false;
          }
        } else {
          *ret = false;
        }
      });

  // Patterns
  patterns.Set("nn.global_avg_pool2d",
               {{"pattern", IsOp("relax.mean")(data)},
                {"annotation", ffi::Map<ffi::String, DFPattern>({{"data", data}})}});

  AppendPatterns(clml_patterns, patterns, CheckGlobalPool);
}

// Reshape
void ReshapePatterns(ffi::Array<FusionPattern>* clml_patterns) {
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> patterns;

  // Check function
  auto CheckReshape = ffi::Function::FromPacked(
      [](const ffi::AnyView* args, int32_t num_args, ffi::Any* ret) mutable {
        TVM_FFI_ICHECK(num_args == 1) << "Expected 1 argument PatternCheckContext";
        auto context = args[0].cast<PatternCheckContext>();
        tvm::arith::Analyzer analyzer;
        *ret = true;

        if (context->annotated_expr.find("root") != context->annotated_expr.end()) {
          auto root_call = Downcast<Call>(context->annotated_expr["root"]);

          if (root_call->op != Op::Get("relax.reshape")) {
            RET_FALSE
          }
          if (!root_call->args[1].as<ShapeExprNode>()) {
            RET_FALSE
          }
        } else {
          *ret = false;
        }
      });

  // Patterns
  patterns.Set("reshape", {{"pattern", IsOp("relax.reshape")(Wildcard(), Wildcard())},
                           {"annotation", ffi::Map<ffi::String, DFPattern>({})}});

  AppendPatterns(clml_patterns, patterns, CheckReshape);
}

// Binary
void BinaryPatterns(ffi::Array<FusionPattern>* clml_patterns) {
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> patterns;

  // Check function
  auto CheckBinary = ffi::Function::FromPacked([](const ffi::AnyView* args, int32_t num_args,
                                                  ffi::Any* ret) mutable {
    TVM_FFI_ICHECK(num_args == 1) << "Expected 1 argument PatternCheckContext";
    auto context = args[0].cast<PatternCheckContext>();
    tvm::arith::Analyzer analyzer;
    *ret = true;

    if (context->annotated_expr.find("root") != context->annotated_expr.end()) {
      auto root_call = Downcast<Call>(context->annotated_expr["root"]);

      bool is_bop = false;
      ffi::Array<ffi::String> bops = {"relax.add",    "relax.subtract", "relax.multiply",
                                      "relax.divide", "relax.maximum",  "relax.minimum"};
      for (const auto& op : bops) {
        if (root_call->op == Op::Get(op)) {
          is_bop = true;
          break;
        }
      }
      *ret = is_bop;

      if (context->annotated_expr.find("lhs") != context->annotated_expr.end() &&
          context->annotated_expr.find("rhs") != context->annotated_expr.end()) {
        auto lhs_tsinfo = Downcast<TensorStructInfo>(GetStructInfo(context->annotated_expr["lhs"]));
        auto rhs_tsinfo = Downcast<TensorStructInfo>(GetStructInfo(context->annotated_expr["rhs"]));
        auto lhs_shape = lhs_tsinfo->GetShape().value();
        auto rhs_shape = rhs_tsinfo->GetShape().value();

        if (lhs_shape.size() == 0 || rhs_shape.size() == 0 ||
            lhs_tsinfo->dtype == DataType::Int(64) || rhs_tsinfo->dtype == DataType::Int(64) ||
            lhs_shape.size() != rhs_shape.size()) {
          RET_FALSE
        }

        if (analyzer.CanProveGreaterEqual(lhs_shape[0], 2) ||
            analyzer.CanProveGreaterEqual(rhs_shape[0], 2)) {
          RET_FALSE
        }

        for (size_t i = 0; i < lhs_shape.size(); ++i) {
          if (!analyzer.CanProveEqual(lhs_shape[i], rhs_shape[i])) {
            RET_FALSE
          }
        }

      } else
        RET_FALSE
    } else {
      RET_FALSE
    }
  });

  DFPattern lhs = Wildcard();
  DFPattern rhs = Wildcard();
  ffi::Map<ffi::String, DFPattern> annotations = {{"lhs", lhs}, {"rhs", rhs}};

  // Patterns
  ffi::Array<ffi::String> ops = {"relax.add",    "relax.subtract", "relax.multiply",
                                 "relax.divide", "relax.maximum",  "relax.minimum"};
  for (const auto& op : ops) {
    patterns.Set(op, {{"pattern", IsOp(op)(lhs, rhs)}, {"annotation", annotations}});
  }

  AppendPatterns(clml_patterns, patterns, CheckBinary);
}

// Unary
void UnaryPatterns(ffi::Array<FusionPattern>* clml_patterns) {
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> patterns;

  // Check function
  auto CheckUnary = ffi::Function::FromPacked([](const ffi::AnyView* args, int32_t num_args,
                                                 ffi::Any* ret) mutable {
    TVM_FFI_ICHECK(num_args == 1) << "Expected 1 argument PatternCheckContext";
    auto context = args[0].cast<PatternCheckContext>();
    tvm::arith::Analyzer analyzer;
    *ret = true;

    if (context->annotated_expr.find("root") != context->annotated_expr.end()) {
      auto root_call = Downcast<Call>(context->annotated_expr["root"]);

      bool is_bop = false;
      ffi::Array<ffi::String> uops = {"relax.nn.softmax", "relax.nn.relu", "relax.clip"};
      for (const auto& op : uops) {
        if (root_call->op == Op::Get(op)) {
          is_bop = true;
          break;
        }
      }
      *ret = is_bop;

      if (context->annotated_expr.find("lhs") != context->annotated_expr.end()) {
        auto lhs_tsinfo = Downcast<TensorStructInfo>(GetStructInfo(context->annotated_expr["lhs"]));
        auto lhs_shape = lhs_tsinfo->GetShape().value();

        if (lhs_shape.size() == 0 || lhs_tsinfo->dtype == DataType::Int(64)) {
          RET_FALSE
        }

        if (analyzer.CanProveGreaterEqual(lhs_shape[0], 2)) {
          RET_FALSE
        }
      } else
        RET_FALSE
    } else {
      RET_FALSE
    }
  });

  DFPattern lhs = Wildcard();
  ffi::Map<ffi::String, DFPattern> annotations = {{"lhs", lhs}};

  // Patterns
  ffi::Array<ffi::String> ops = {"relax.nn.softmax", "relax.nn.relu", "relax.clip"};
  for (const auto& op : ops) {
    patterns.Set(op, {{"pattern", IsOp(op)(lhs)}, {"annotation", annotations}});
  }

  AppendPatterns(clml_patterns, patterns, CheckUnary);
}

ffi::Array<FusionPattern> CreatePatterns() {
  static ffi::Array<FusionPattern> clml_patterns;
  if (clml_patterns.size() > 0) {
    // Initialize only once
    return clml_patterns;
  }

  Conv2DPatterns(&clml_patterns);
  BatchNormPatterns(&clml_patterns);
  PoolPatterns(&clml_patterns);
  GlobalAvgPatterns(&clml_patterns);
  ReshapePatterns(&clml_patterns);
  BinaryPatterns(&clml_patterns);
  UnaryPatterns(&clml_patterns);

  return clml_patterns;
}

// LLM
void LLMPatterns(ffi::Array<FusionPattern>* clml_patterns) {
  ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>> patterns;

  // Check function
  auto CheckLLM = ffi::Function::FromPacked(
      [](const ffi::AnyView* args, int32_t num_args, ffi::Any* ret) mutable {
        TVM_FFI_ICHECK(num_args == 1) << "Expected 1 argument PatternCheckContext";
        auto context = args[0].cast<PatternCheckContext>();
        tvm::arith::Analyzer analyzer;
        *ret = true;

        if (context->annotated_expr.find("root") == context->annotated_expr.end() ||
            context->annotated_expr.find("lhs") == context->annotated_expr.end() ||
            context->annotated_expr.find("w_decoded") == context->annotated_expr.end() ||
            context->annotated_expr.find("w_encoded") == context->annotated_expr.end()) {
          RET_FALSE
        }

        auto input = context->annotated_expr["lhs"];
        auto root = context->annotated_expr["root"];
        auto wdq = context->annotated_expr["w_decoded"];
        auto w_pack = context->annotated_expr["w_encoded"];

        if (Downcast<TensorStructInfo>(GetStructInfo(input))->dtype != DataType::Float(16)) {
          RET_FALSE
        }

        if (auto wdq_call = wdq.as<CallNode>()) {
          if (auto g_var = wdq_call->args[0].as<GlobalVarNode>()) {
            if (g_var->name_hint != "dequantize") {
              RET_FALSE
            }
          } else {
            RET_FALSE
          }
        } else {
          RET_FALSE
        }

        auto root_tsinfo = Downcast<TensorStructInfo>(GetStructInfo(root));
        auto root_shape = root_tsinfo->GetShape().value();
        if (!(root_shape.size() == 3 && root_shape[0].as<IntImmNode>() &&
              root_tsinfo->dtype == DataType::Float(16) &&
              analyzer.CanProveGreaterEqual(root_shape[0], 1))) {
          RET_FALSE
        }

        auto wdq_shape = Downcast<TensorStructInfo>(GetStructInfo(wdq))->GetShape().value();
        auto w_pack_shape = Downcast<TensorStructInfo>(GetStructInfo(w_pack))->GetShape().value();
        auto input_shape = Downcast<TensorStructInfo>(GetStructInfo(input))->GetShape().value();

        if (!(wdq_shape.size() == 2 &&
              analyzer.CanProveEqual(w_pack_shape[w_pack_shape.size() - 1],
                                     root_shape[root_shape.size() - 1]) &&
              analyzer.CanProveEqual(wdq_shape[wdq_shape.size() - 2],
                                     input_shape[input_shape.size() - 1]))) {
          RET_FALSE
        }
      });

  DFPattern scales = Wildcard();
  DFPattern x = Wildcard();
  DFPattern w_packed = Wildcard();
  DFPattern g_var = GlobalVarPattern("dequantize");

  auto w_decoded = IsOp("relax.call_tir")(g_var, TuplePattern({w_packed, scales}));
  auto matmul = IsOp("relax.matmul")(x, w_decoded);

  ffi::Map<ffi::String, DFPattern> annotations = {
      {"lhs", x}, {"w_encoded", w_packed}, {"w_decoded", w_decoded}, {"scales", scales}};

  // Patterns
  patterns.Set("dequant_matmul", {{"pattern", matmul}, {"annotation", annotations}});

  AppendPatterns(clml_patterns, patterns, CheckLLM);
}

ffi::Array<FusionPattern> CreateLLMPatterns() {
  static ffi::Array<FusionPattern> clml_patterns;
  if (clml_patterns.size() > 0) {
    // Initialize only once
    return clml_patterns;
  }

  LLMPatterns(&clml_patterns);

  return clml_patterns;
}

}  // namespace

namespace transform {

Pass OpenCLMLOffLoad() {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    ffi::Array<FusionPattern> patterns = CreatePatterns();
    ffi::Map<ffi::String, ffi::Array<ffi::String>> desired_layouts = {
        {"relax.nn.conv2d", {"NCHW", "OIHW", "NCHW"}},
        {"relax.nn.conv2d_transpose", {"NCHW", "OIHW", "NCHW"}}};
    mod = relax::transform::ConvertLayout(desired_layouts, nullptr)(mod);
    mod = relax::transform::Normalize()(mod);
    mod = relax::transform::FoldBatchnormToConv2D()(mod);
    mod = relax::backend::adreno::transform::AppendReshapeToBatchnorm()(mod);
    mod = relax::transform::FoldConstant()(mod);
    mod = relax::transform::FuseOpsByPattern(patterns)(mod);
    mod = relax::transform::MergeCompositeFunctions()(mod);
    mod = relax::transform::RunCodegen(std::nullopt, {})(mod);
    return mod;
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"OpenCLMLOffLoad",
                          /*required=*/{});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.backend.adreno.transform.OpenCLMLOffLoad", OpenCLMLOffLoad);
}

Pass OpenCLMLOffLoadForLLM() {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    ffi::Array<FusionPattern> patterns = CreateLLMPatterns();
    mod = relax::transform::Normalize()(mod);
    mod = relax::transform::FuseOpsByPattern(patterns)(mod);
    mod = relax::transform::RunCodegen(std::nullopt, {})(mod);
    return mod;
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"OpenCLMLOffLoadForLLM",
                          /*required=*/{});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.backend.adreno.transform.OpenCLMLOffLoadForLLM",
                        OpenCLMLOffLoadForLLM);
}

}  // namespace transform
}  // namespace adreno
}  // namespace backend
}  // namespace relax
}  // namespace tvm
