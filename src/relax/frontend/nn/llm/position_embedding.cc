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
 * \file src/relax/frontend/nn/llm/position_embedding.cc
 * \brief Implementation of rotary position embedding functions.
 */

#include "position_embedding.h"

#include <tvm/ir/type.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

#include "../../../../tir/ir/script/script_complete.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <cmath>

// spec.h must be included before core.h because core.h registers FFI methods
// that take ModuleSpec by value.
#include "../spec.h"
#include "../core.h"
#include "../op.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

using namespace tvm::tir;

// Helper to cast a PrimExpr to a specific dtype
static PrimExpr CastTo(PrimExpr expr, const std::string& dtype) {
  return tir::Cast(DataType(runtime::StringToDLDataType(dtype)), expr);
}

// Helper to create a float32 constant
static PrimExpr MakeFloat32(double value) {
  return FloatImm(DataType::Float(32), value);
}

// Helper to create an int64 constant
static PrimExpr MakeInt64(int64_t value) { return IntImm(DataType::Int(64), value); }

RopeFreqResult RopeFreqDefault(PrimExpr s, PrimExpr d, int64_t d_range, PrimExpr theta,
                               const std::string& dtype,
                               const ffi::Map<ffi::String, ffi::Any>& extra_args) {
  PrimExpr exponent = CastTo(tir::FloorMod(d * IntImm(d.dtype(), 2), IntImm(d.dtype(), d_range)), "float32") /
                      MakeFloat32(static_cast<double>(d_range));
  PrimExpr freq = s / tvm::pow(theta, exponent);

  tir::Var freq_var("freq", DataType::Float(32));
  // cos/sin of a float32 var are already float32; only cast if dtype != float32
  PrimExpr cos_val = tvm::cos(freq_var);
  PrimExpr sin_val = tvm::sin(freq_var);
  PrimExpr cos_freq = (dtype == "float32") ? cos_val : CastTo(cos_val, dtype);
  PrimExpr sin_freq = (dtype == "float32") ? sin_val : CastTo(sin_val, dtype);

  std::vector<std::pair<Var, PrimExpr>> var_map;
  var_map.push_back({freq_var, freq});

  return {cos_freq, sin_freq, var_map};
}

RopeFreqResult RopeFreqGptj(PrimExpr s, PrimExpr d, int64_t d_range, PrimExpr theta,
                            const std::string& dtype,
                            const ffi::Map<ffi::String, ffi::Any>& extra_args) {
  // freq = s / (theta ^ (2 * (d // 2) % d_range / d_range))
  // s is already float32 by convention
  PrimExpr exponent = CastTo(tir::FloorMod(IntImm(d.dtype(), 2) * tir::FloorDiv(d, IntImm(d.dtype(), 2)), IntImm(d.dtype(), d_range)),
                              "float32") /
                      MakeFloat32(static_cast<double>(d_range));
  PrimExpr freq = s / tvm::pow(theta, exponent);

  tir::Var freq_var("freq", DataType::Float(32));
  PrimExpr cos_val = tvm::cos(freq_var);
  PrimExpr sin_val = tvm::sin(freq_var);
  PrimExpr cos_freq = (dtype == "float32") ? cos_val : CastTo(cos_val, dtype);
  PrimExpr sin_freq = (dtype == "float32") ? sin_val : CastTo(sin_val, dtype);

  std::vector<std::pair<Var, PrimExpr>> var_map;
  var_map.push_back({freq_var, freq});

  return {cos_freq, sin_freq, var_map};
}

RopeFreqResult RopeFreqLlama3(PrimExpr s, PrimExpr d, int64_t d_range, PrimExpr theta,
                              const std::string& dtype,
                              const ffi::Map<ffi::String, ffi::Any>& extra_args) {
  // Extract required parameters
  double factor = extra_args.at("factor").cast<double>();
  double low_freq_factor = extra_args.at("low_freq_factor").cast<double>();
  double high_freq_factor = extra_args.at("high_freq_factor").cast<double>();
  int64_t original_max_position_embeddings =
      extra_args.at("original_max_position_embeddings").cast<int64_t>();

  // orig_freq = 1 / (theta ^ ((d * 2 % d_range) / d_range))
  PrimExpr exponent = CastTo(tir::FloorMod(d * IntImm(d.dtype(), 2), IntImm(d.dtype(), d_range)), "float32") /
                      MakeFloat32(static_cast<double>(d_range));
  PrimExpr orig_freq = MakeFloat32(1.0) / tvm::pow(theta, exponent);

  tir::Var orig_freq_var("orig_freq", DataType::Float(32));

  // Compute smooth interpolation factor
  double inv_diff_freq_factor = 1.0 / (high_freq_factor - low_freq_factor);
  double llama3_inv_scaling_factor = 1.0 / factor;
  double llama3_alpha =
      static_cast<double>(original_max_position_embeddings) / (2.0 * M_PI) * inv_diff_freq_factor;
  double llama3_beta = low_freq_factor * inv_diff_freq_factor;

  PrimExpr smooth =
      tvm::max(MakeFloat32(0.0),
               tvm::min(MakeFloat32(1.0),
                        MakeFloat32(llama3_alpha) * orig_freq_var - MakeFloat32(llama3_beta)));

  // smoothed_freq = s * ((1 - smooth) * orig_freq * inv_scaling_factor + smooth * orig_freq)
  // s is already float32 by convention
  PrimExpr smoothed_freq =
      s *
      ((MakeFloat32(1.0) - smooth) * orig_freq_var * MakeFloat32(llama3_inv_scaling_factor) +
       smooth * orig_freq_var);

  tir::Var smoothed_freq_var("smoothed_freq", DataType::Float(32));
  PrimExpr cos_val = tvm::cos(smoothed_freq_var);
  PrimExpr sin_val = tvm::sin(smoothed_freq_var);
  PrimExpr cos_freq = (dtype == "float32") ? cos_val : CastTo(cos_val, dtype);
  PrimExpr sin_freq = (dtype == "float32") ? sin_val : CastTo(sin_val, dtype);

  // Insert smoothed_freq first, then orig_freq - matches Python var_map ordering
  // (test expects var_map_list[0] = smoothed_freq, var_map_list[1] = orig_freq)
  std::vector<std::pair<Var, PrimExpr>> var_map;
  var_map.push_back({smoothed_freq_var, smoothed_freq});
  var_map.push_back({orig_freq_var, orig_freq});

  return {cos_freq, sin_freq, var_map};
}

RopeFreqResult RopeFreqLlama4(PrimExpr s, PrimExpr d, int64_t d_range, PrimExpr theta,
                              const std::string& dtype,
                              const ffi::Map<ffi::String, ffi::Any>& extra_args) {
  // Extract required parameters
  double factor = extra_args.at("factor").cast<double>();
  double low_freq_factor = extra_args.at("low_freq_factor").cast<double>();
  double high_freq_factor = extra_args.at("high_freq_factor").cast<double>();
  int64_t original_max_position_embeddings =
      extra_args.at("original_max_position_embeddings").cast<int64_t>();

  // orig_freq = 1 / (theta ^ (2 * (d // 2) / d_range))
  PrimExpr exponent = CastTo(IntImm(d.dtype(), 2) * tir::FloorDiv(d, IntImm(d.dtype(), 2)), "float32") /
                      MakeFloat32(static_cast<double>(d_range));
  PrimExpr orig_freq = MakeFloat32(1.0) / tvm::pow(theta, exponent);

  tir::Var orig_freq_var("orig_freq", DataType::Float(32));

  double llama4_inv_scaling_factor = 1.0 / factor;

  PrimExpr smoothed_freq;
  if (high_freq_factor == low_freq_factor) {
    // Uniform scaling with threshold
    PrimExpr wavelength = tvm::floordiv(MakeFloat32(2.0 * M_PI), orig_freq_var);
    PrimExpr threshold_wavelen =
        MakeFloat32(static_cast<double>(original_max_position_embeddings) / low_freq_factor);

    PrimExpr scaled_freq = tvm::if_then_else(wavelength > threshold_wavelen,
                                              orig_freq_var / MakeFloat32(factor), orig_freq_var);
    // s is already float32 by convention
    smoothed_freq = s * scaled_freq;
  } else {
    // Smooth interpolation
    double inv_diff_freq_factor = 1.0 / (high_freq_factor - low_freq_factor);
    double llama4_alpha = static_cast<double>(original_max_position_embeddings) / (2.0 * M_PI) *
                          inv_diff_freq_factor;
    double llama4_beta = low_freq_factor * inv_diff_freq_factor;

    PrimExpr smooth =
        tvm::max(MakeFloat32(0.0),
                 tvm::min(MakeFloat32(1.0),
                          MakeFloat32(llama4_alpha) * orig_freq_var - MakeFloat32(llama4_beta)));

    // s is already float32 by convention
    smoothed_freq =
        s *
        ((MakeFloat32(1.0) - smooth) * orig_freq_var * MakeFloat32(llama4_inv_scaling_factor) +
         smooth * orig_freq_var);
  }

  tir::Var smoothed_freq_var("smoothed_freq", DataType::Float(32));
  PrimExpr cos_val_l4 = tvm::cos(smoothed_freq_var);
  PrimExpr sin_val_l4 = tvm::sin(smoothed_freq_var);
  PrimExpr cos_freq = (dtype == "float32") ? cos_val_l4 : CastTo(cos_val_l4, dtype);
  PrimExpr sin_freq = (dtype == "float32") ? sin_val_l4 : CastTo(sin_val_l4, dtype);

  // Insert smoothed_freq first, then orig_freq - matches Python var_map ordering
  std::vector<std::pair<Var, PrimExpr>> var_map;
  var_map.push_back({smoothed_freq_var, smoothed_freq});
  var_map.push_back({orig_freq_var, orig_freq});

  return {cos_freq, sin_freq, var_map};
}

RopeFreqResult RopeFreqLongrope(PrimExpr s, PrimExpr d, int64_t d_range, PrimExpr theta,
                                const std::string& dtype,
                                const ffi::Map<ffi::String, ffi::Any>& extra_args) {
  int64_t max_position_embeddings = extra_args.at("max_position_embeddings").cast<int64_t>();
  int64_t original_max_position_embeddings =
      extra_args.at("original_max_position_embeddings").cast<int64_t>();

  double scale = static_cast<double>(max_position_embeddings) /
                 static_cast<double>(original_max_position_embeddings);
  double scaling_factor = (scale > 1.0) ? std::sqrt(1.0 + std::log(scale) /
                                                               std::log(original_max_position_embeddings))
                                        : 1.0;

  // divisor = theta ^ ((d * 2 % d_range) / d_range)
  PrimExpr exponent = CastTo(tir::FloorMod(d * IntImm(d.dtype(), 2), IntImm(d.dtype(), d_range)), "float32") /
                      MakeFloat32(static_cast<double>(d_range));
  PrimExpr divisor = tvm::pow(theta, exponent);

  // Apply extension factors if provided
  if (extra_args.count("ext_factors")) {
    // ext_factors is a buffer, index it by d % (d_range // 2)
    auto ext_factors_any = extra_args.at("ext_factors");
    // In the Python code, ext_factors is accessed as ext_factors[d % (d_range // 2)]
    // For C++, we need to handle this as a buffer access in TIR
    // This will be handled in the TIR function generation
  }

  // s is already float32 by convention
  PrimExpr freq = s / divisor;

  tir::Var freq_var("freq", DataType::Float(32));
  PrimExpr cos_val_lr = tvm::cos(freq_var);
  PrimExpr sin_val_lr = tvm::sin(freq_var);
  PrimExpr cos_freq = (dtype == "float32") ?
      cos_val_lr * MakeFloat32(scaling_factor) :
      CastTo(cos_val_lr * MakeFloat32(scaling_factor), dtype);
  PrimExpr sin_freq = (dtype == "float32") ?
      sin_val_lr * MakeFloat32(scaling_factor) :
      CastTo(sin_val_lr * MakeFloat32(scaling_factor), dtype);

  std::vector<std::pair<Var, PrimExpr>> var_map;
  var_map.push_back({freq_var, freq});

  return {cos_freq, sin_freq, var_map};
}

// Helper for YaRN: find correction dimension
static PrimExpr YarnFindCorrectionDim(PrimExpr d, int64_t num_rotations,
                                      int64_t max_position_embeddings,
                                      double inv_theta_log_scale) {
  double log_val = std::log(static_cast<double>(max_position_embeddings) /
                             (static_cast<double>(num_rotations) * 2.0 * M_PI));
  return CastTo(d, "float32") * MakeFloat32(log_val * inv_theta_log_scale);
}

// Helper for YaRN: find correction range
static std::pair<PrimExpr, PrimExpr> YarnFindCorrectionRange(
    PrimExpr d, int64_t low_rot, int64_t high_rot, int64_t d_range,
    int64_t max_position_embeddings, double inv_theta_log_scale) {
  PrimExpr low = YarnFindCorrectionDim(d, low_rot, max_position_embeddings, inv_theta_log_scale);
  PrimExpr high = YarnFindCorrectionDim(d, high_rot, max_position_embeddings, inv_theta_log_scale);

  low = tvm::max(low, MakeFloat32(0.0));
  high = tvm::min(high, MakeFloat32(static_cast<double>(d_range - 1)));

  return {low, high};
}

RopeFreqResult RopeFreqYarn(PrimExpr s, PrimExpr d, int64_t d_range, PrimExpr theta,
                            const std::string& dtype,
                            const ffi::Map<ffi::String, ffi::Any>& extra_args) {
  int64_t original_max_position_embeddings =
      extra_args.at("original_max_position_embeddings").cast<int64_t>();
  double scaling_factor = extra_args.at("scaling_factor").cast<double>();
  int64_t beta_fast = extra_args.at("beta_fast").cast<int64_t>();
  int64_t beta_slow = extra_args.at("beta_slow").cast<int64_t>();
  double inv_theta_log_scale = extra_args.at("inv_theta_log_scale").cast<double>();

  // Compute base frequencies
  PrimExpr exponent = CastTo(tir::FloorMod(d * IntImm(d.dtype(), 2), IntImm(d.dtype(), d_range)), "float32") /
                      MakeFloat32(static_cast<double>(d_range));
  PrimExpr freq_power = tvm::pow(theta, exponent);
  PrimExpr freq_extra = MakeFloat32(1.0) / freq_power;
  PrimExpr freq_inter = MakeFloat32(1.0) / (MakeFloat32(scaling_factor) * freq_power);

  // Compute correction range
  auto [low, high] = YarnFindCorrectionRange(d, beta_fast, beta_slow, d_range,
                                              original_max_position_embeddings, inv_theta_log_scale);

  // Avoid division by zero
  high = tvm::if_then_else(low == high, high + MakeFloat32(0.001), high);

  // Compute mask
  PrimExpr inv_freq_mask =
      MakeFloat32(1.0) -
      tvm::max(tvm::min((CastTo(d, "float32") - low) / (high - low), MakeFloat32(1.0)),
               MakeFloat32(0.0));

  // Interpolate frequencies
  PrimExpr inv_freq = freq_inter * (MakeFloat32(1.0) - inv_freq_mask) + freq_extra * inv_freq_mask;

  // s is already float32 by convention
  PrimExpr freq = s * inv_freq;

  tir::Var freq_var("freq", DataType::Float(32));
  PrimExpr cos_val_yarn = tvm::cos(freq_var);
  PrimExpr sin_val_yarn = tvm::sin(freq_var);
  PrimExpr cos_freq = (dtype == "float32") ? cos_val_yarn : CastTo(cos_val_yarn, dtype);
  PrimExpr sin_freq = (dtype == "float32") ? sin_val_yarn : CastTo(sin_val_yarn, dtype);

  std::vector<std::pair<Var, PrimExpr>> var_map;
  var_map.push_back({freq_var, freq});

  return {cos_freq, sin_freq, var_map};
}

RopeFreqFunc SwitchRopeFreqFunc(const ffi::Map<ffi::String, ffi::Any>& rope_scaling) {
  if (rope_scaling.count("rope_type") == 0) {
    return RopeFreqDefault;
  }

  std::string rope_type = rope_scaling.at("rope_type").cast<ffi::String>().operator std::string();

  if (rope_type == "gptj") {
    return RopeFreqGptj;
  } else if (rope_type == "llama3") {
    return RopeFreqLlama3;
  } else if (rope_type == "llama4") {
    return RopeFreqLlama4;
  } else if (rope_type == "longrope") {
    return RopeFreqLongrope;
  } else if (rope_type == "yarn") {
    return RopeFreqYarn;
  } else {
    LOG(FATAL) << "Unsupported RoPE scaling type: " << rope_type;
    return nullptr;
  }
}

std::tuple<NNTensor, NNTensor, NNTensor> LlamaRope(
    NNTensor qkv, tir::Var total_seq_len, double theta, double scale, int64_t num_q_heads,
    int64_t num_kv_heads, const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
    ffi::Optional<int64_t> rotary_dim) {
  auto qkv_sinfo = qkv->expr->struct_info_.as<TensorStructInfoNode>();
  TVM_FFI_ICHECK(qkv_sinfo) << "QKV must be a tensor";
  TVM_FFI_ICHECK(qkv_sinfo->shape.defined()) << "QKV shape must be defined";
  auto shape_expr = qkv_sinfo->shape.value().as<ShapeExprNode>();
  TVM_FFI_ICHECK(shape_expr && shape_expr->values.size() == 4)
      << "QKV must be 4-D: [batch, seq_len, fused_heads, head_dim]";

  PrimExpr batch_size_expr = shape_expr->values[0];
  PrimExpr seq_len_expr    = shape_expr->values[1];
  int64_t fused_heads      = num_q_heads + num_kv_heads * 2;
  auto head_dim_int        = shape_expr->values[3].as<IntImmNode>();
  TVM_FFI_ICHECK(head_dim_int) << "head_dim must be a static integer";
  int64_t head_dim       = head_dim_int->value;
  int64_t rotary_dim_val = rotary_dim.value_or(head_dim);

  DataType dtype = qkv_sinfo->dtype;
  std::string dtype_str = runtime::DLDataTypeToString(dtype);

  RopeFreqFunc rope_freq_func = SwitchRopeFreqFunc(rope_scaling);
  std::string rope_type = "default";
  if (rope_scaling.count("rope_type"))
    rope_type = rope_scaling.at("rope_type").cast<ffi::String>().operator std::string();

  // int32 constant helper for static dims (matches plain integer literals in TVMScript)
  auto I32 = [](int64_t v) { return IntImm(DataType::Int(32), v); };

  // Symbolic vars for dynamic dims (int64)
  tir::Var batch_var("batch_size", DataType::Int(64));
  tir::Var seq_var("seq_len",     DataType::Int(64));

  // Axis vars: b,s are int64 (dynamic); h,d are int32 (static ranges)
  tir::Var b("b", DataType::Int(64));
  tir::Var s("s", DataType::Int(64));
  tir::Var h("h", DataType::Int(32));
  tir::Var d("d", DataType::Int(32));

  // Plain handle vars for function params and buffer_map keys.
  // These have PrimType(handle) = T.handle, matching TVMScript's plain T.handle params.
  // decl_buffer creates a separate internal data var with PointerType (buf.data),
  // which is what the Buffer constructor requires internally.
  // The buffer_map maps plain_handle_var -> buffer (where buffer.data is the PointerType var).
  tir::Var h_qkv("var_qkv", DataType::Handle());
  tir::Var h_q("var_q",   DataType::Handle());
  tir::Var h_k("var_k",   DataType::Handle());
  tir::Var h_v("var_v",   DataType::Handle());

  tir::Buffer qkv_buf = tir::decl_buffer(
      {batch_var, seq_var, I32(fused_heads), I32(head_dim)}, dtype, "qkv");
  tir::Buffer q_buf = tir::decl_buffer(
      {batch_var, seq_var, I32(num_q_heads),  I32(head_dim)}, dtype, "q");
  tir::Buffer k_buf = tir::decl_buffer(
      {batch_var, seq_var, I32(num_kv_heads), I32(head_dim)}, dtype, "k");
  tir::Buffer v_buf = tir::decl_buffer(
      {batch_var, seq_var, I32(num_kv_heads), I32(head_dim)}, dtype, "v");

  // Create a FRESH total_seq_len var for the PrimFunc param and body.
  // The original total_seq_len (passed in) is used only at the Relax level
  // in tir_vars=R.shape([total_seq_len]). If we reuse the same C++ object
  // as both the PrimFunc param and the Relax shape var, the SE engine sees
  // them as the same bound var across both functions, causing SE failure.
  tir::Var prim_total_seq_len("total_seq_len", DataType::Int(64));

  // Build RoPE expression: d is int32, arithmetic stays int32
  auto build_rope_expr = [&]() -> PrimExpr {
    PrimExpr offset    = prim_total_seq_len - seq_var;
    PrimExpr pos_int   = s + offset;
    PrimExpr pos_dtype = tir::Cast(dtype, pos_int);
    PrimExpr pos_f32   = tir::Cast(DataType::Float(32), pos_dtype);
    if (scale != 1.0) pos_f32 = pos_f32 * MakeFloat32(scale);

    auto freq_result = rope_freq_func(pos_f32, d, rotary_dim_val, MakeFloat32(theta), dtype_str, rope_scaling);

    PrimExpr x        = tir::BufferLoad(qkv_buf, {b, s, h, d});
    PrimExpr cos_part = freq_result.cos_freq * x;
    PrimExpr neg_one  = tir::make_const(dtype, -1.0);
    PrimExpr sin_part;
    if (rope_type == "gptj") {
      sin_part = freq_result.sin_freq *
          tvm::if_then_else(
              tir::FloorMod(d, I32(2)) == I32(0),
              tir::BufferLoad(qkv_buf, {b, s, h, d + I32(1)}) * neg_one,
              tir::BufferLoad(qkv_buf, {b, s, h, d - I32(1)}));
    } else {
      sin_part = freq_result.sin_freq *
          tvm::if_then_else(
              d < I32(rotary_dim_val / 2),
              tir::BufferLoad(qkv_buf, {b, s, h, d + I32(rotary_dim_val / 2)}) * neg_one,
              tir::BufferLoad(qkv_buf, {b, s, h, d - I32(rotary_dim_val / 2)}));
    }
    PrimExpr result = cos_part + sin_part;
    for (const auto& [var, expr] : freq_result.var_map)
      result = tir::Let(var, expr, result);
    return result;
  };

  Stmt q_store = tir::BufferStore(
      q_buf,
      tvm::if_then_else(d < I32(rotary_dim_val), build_rope_expr(),
                        tir::BufferLoad(qkv_buf, {b, s, h, d})),
      {b, s, h, d});
  Stmt k_store = tir::BufferStore(
      k_buf,
      tvm::if_then_else(d < I32(rotary_dim_val), build_rope_expr(),
                        tir::BufferLoad(qkv_buf, {b, s, h, d})),
      {b, s, h - I32(num_q_heads), d});
  Stmt v_store = tir::BufferStore(
      v_buf, tir::BufferLoad(qkv_buf, {b, s, h, d}),
      {b, s, h - I32(num_q_heads + num_kv_heads), d});

  Stmt inner_body = tir::IfThenElse(
      h < I32(num_q_heads), q_store,
      tir::IfThenElse(h < I32(num_q_heads + num_kv_heads), k_store, v_store));

  // Loop vars: iters_0,1 are int64 (dynamic); iters_2,3 are int32 (static)
  tir::Var iters_0("iters_0", DataType::Int(64));
  tir::Var iters_1("iters_1", DataType::Int(64));
  tir::Var iters_2("iters_2", DataType::Int(32));
  tir::Var iters_3("iters_3", DataType::Int(32));

  ffi::Array<tir::IterVar> iter_vars = {
      tir::IterVar(Range::FromMinExtent(IntImm(DataType::Int(64), 0), batch_var),
                   b, tir::kDataPar, ""),
      tir::IterVar(Range::FromMinExtent(IntImm(DataType::Int(64), 0), seq_var),
                   s, tir::kDataPar, ""),
      tir::IterVar(Range::FromMinExtent(I32(0), I32(fused_heads)), h, tir::kDataPar, ""),
      tir::IterVar(Range::FromMinExtent(I32(0), I32(head_dim)),    d, tir::kDataPar, "")};

  ffi::Map<ffi::String, ffi::Any> sblock_annots;
  sblock_annots.Set("tir.script_parsing_detect_access", IntImm(DataType::Int(32), 3));

  Stmt sblock_realize = tir::SBlockRealize(
      {PrimExpr(iters_0), PrimExpr(iters_1), PrimExpr(iters_2), PrimExpr(iters_3)},
      tir::const_true(),
      tir::SBlock(iter_vars, {}, {}, "llama_fused_rope", inner_body,
                  std::nullopt, {}, {}, sblock_annots));

  Stmt loop_body = sblock_realize;
  loop_body = tir::For(iters_3, I32(0), I32(head_dim),    tir::ForKind::kSerial, loop_body);
  loop_body = tir::For(iters_2, I32(0), I32(fused_heads), tir::ForKind::kSerial, loop_body);
  loop_body = tir::For(iters_1, IntImm(DataType::Int(64), 0), seq_var,
                       tir::ForKind::kSerial, loop_body);
  loop_body = tir::For(iters_0, IntImm(DataType::Int(64), 0), batch_var,
                       tir::ForKind::kSerial, loop_body);

  // Root sblock to match TVMScript-generated IR
  Stmt root_block = tir::SBlockRealize(
      {}, tir::const_true(),
      tir::SBlock({}, {}, {}, "root", loop_body));

  // Buffer map: plain handle var -> buffer (buf.data is the internal PointerType var)
  ffi::Map<tir::Var, tir::Buffer> buf_map;
  buf_map.Set(h_qkv, qkv_buf);
  buf_map.Set(h_q,   q_buf);
  buf_map.Set(h_k,   k_buf);
  buf_map.Set(h_v,   v_buf);

  ffi::Array<tir::Var> params = {h_qkv, h_q, h_k, h_v, prim_total_seq_len};
  // Build without attrs first, then use WithAttr with raw bool/int64_t so that
  // the stored types match Python T.func_attr({"tir.noalias": True, "op_pattern": 8})
  // which stores plain Python bool and int (not IntImm nodes).
  tir::PrimFunc prim_func(params, root_block, VoidType(), buf_map);
  prim_func = WithAttr(prim_func, "tir.noalias", ffi::Any(true));
  prim_func = WithAttr(prim_func, "op_pattern", ffi::Any(static_cast<int64_t>(8)));
  prim_func = tir::ScriptComplete(prim_func, {});

  TensorStructInfo q_sinfo(
      ShapeExpr({batch_size_expr, seq_len_expr, MakeInt64(num_q_heads),  MakeInt64(head_dim)}),
      dtype);
  TensorStructInfo k_sinfo(
      ShapeExpr({batch_size_expr, seq_len_expr, MakeInt64(num_kv_heads), MakeInt64(head_dim)}),
      dtype);
  TensorStructInfo v_sinfo(
      ShapeExpr({batch_size_expr, seq_len_expr, MakeInt64(num_kv_heads), MakeInt64(head_dim)}),
      dtype);
  relax::Var q_out_ph("q", q_sinfo);
  relax::Var k_out_ph("k", k_sinfo);
  relax::Var v_out_ph("v", v_sinfo);

  ffi::Any result_any = NNTensorIrOp(
      prim_func, "llama_rope", {qkv->expr, PrimValue(total_seq_len)},
      {ffi::Any(q_out_ph), ffi::Any(k_out_ph), ffi::Any(v_out_ph)});

  ffi::Array<ffi::Any> result_array = result_any.cast<ffi::Array<ffi::Any>>();
  TVM_FFI_ICHECK_EQ(result_array.size(), 3);
  return {NNTensor(result_array[0].cast<relax::Var>()),
          NNTensor(result_array[1].cast<relax::Var>()),
          NNTensor(result_array[2].cast<relax::Var>())};
}

// ---------------------------------------------------------------------------
// LlamaRopeWithPositionMap
// ---------------------------------------------------------------------------

static tir::PrimFunc BuildLlamaRopeWithPositionMapFunc(
    double theta, double scale, int64_t head_dim, int64_t num_q_heads, int64_t num_kv_heads,
    const std::string& dtype, const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
    int64_t rotary_dim) {
  int64_t fused_heads = num_q_heads + num_kv_heads * 2;
  DataType elem_dtype = DataType(runtime::StringToDLDataType(dtype));

  std::string rope_type = "default";
  if (rope_scaling.count("rope_type")) {
    rope_type = rope_scaling.at("rope_type").cast<ffi::String>().operator std::string();
  }

  RopeFreqFunc rope_freq_func = SwitchRopeFreqFunc(rope_scaling);

  // int32 constant helper — static dims are int32 in TVMScript
  auto I32 = [](int64_t v) { return IntImm(DataType::Int(32), v); };

  // Dynamic symbolic vars (int32 — matches Python T.int32())
  tir::Var seq_len("seq_len", DataType::Int(32));
  tir::Var position_map_elem_offset("position_map_elem_offset", DataType::Int(32));

  // Plain handle params (T.handle)
  tir::Var h_qkv("var_qkv",          DataType::Handle());
  tir::Var h_pos("var_position_map",  DataType::Handle());
  tir::Var h_q("var_q",              DataType::Handle());
  tir::Var h_k("var_k",              DataType::Handle());
  tir::Var h_v("var_v",              DataType::Handle());
  tir::Var apply_rope_var("apply_rope", DataType::Int(32));

  // Buffers with int32 static dims to match TVMScript plain integer literals
  tir::Buffer qkv_buf = tir::decl_buffer(
      {seq_len, I32(fused_heads), I32(head_dim)}, elem_dtype, "qkv");
  tir::Buffer q_buf = tir::decl_buffer(
      {seq_len, I32(num_q_heads),  I32(head_dim)}, elem_dtype, "q");
  tir::Buffer k_buf = tir::decl_buffer(
      {seq_len, I32(num_kv_heads), I32(head_dim)}, elem_dtype, "k");
  tir::Buffer v_buf = tir::decl_buffer(
      {seq_len, I32(num_kv_heads), I32(head_dim)}, elem_dtype, "v");
  // position_map: construct with PointerType data var (required by Buffer constructor),
  // elem_offset=position_map_elem_offset, offset_factor=1.
  // Map both h_pos (plain handle param) and pos_buf->data (PointerType internal var)
  // so ScriptComplete can find pos_buf when inferring reads/writes.
  tir::Var pos_data("position_map", PointerType(PrimType(DataType::Int(32))));
  tir::Buffer pos_buf = tir::Buffer(
      pos_data,
      DataType::Int(32),
      ffi::Array<PrimExpr>{seq_len},
      ffi::Array<PrimExpr>{},
      position_map_elem_offset,
      "position_map",
      /*data_alignment=*/0,
      /*offset_factor=*/1,
      tir::kDefault);

  // Loop / axis vars — all int32 (seq_len is int32)
  tir::Var iters_0("iters_0", DataType::Int(32));
  tir::Var iters_1("iters_1", DataType::Int(32));
  tir::Var iters_2("iters_2", DataType::Int(32));
  tir::Var s("s", DataType::Int(32));
  tir::Var h("h", DataType::Int(32));
  tir::Var d("d", DataType::Int(32));

  // Build RoPE value for a single element.
  // pos = cast<float32>(position_map[s]) * scale  (scale=1.0 → no multiply)
  // freq_func receives pos as float32, d as int32 (same dtype as in Python)
  // We pass "float32" to rope_freq_func so cos/sin stay float32 (no extra cast),
  // then cast the final result to elem_dtype.
  auto make_rope_val = [&]() -> PrimExpr {
    PrimExpr pos_f32 = CastTo(tir::BufferLoad(pos_buf, {s}), "float32");
    if (scale != 1.0) pos_f32 = pos_f32 * MakeFloat32(scale);

    // Use "float32" so cos_freq/sin_freq are plain float32 (no Cast wrapper)
    auto freq_result = rope_freq_func(pos_f32, d, rotary_dim, MakeFloat32(theta), "float32", rope_scaling);

    PrimExpr x = tir::BufferLoad(qkv_buf, {s, h, d});
    // cos_freq and sin_freq are float32; cast x to float32 for arithmetic
    PrimExpr cos_part = freq_result.cos_freq * CastTo(x, "float32");
    PrimExpr sin_part;
    if (rope_type == "gptj") {
      sin_part = freq_result.sin_freq *
                 CastTo(tvm::if_then_else(
                     tir::FloorMod(d, I32(2)) == I32(0),
                     tir::BufferLoad(qkv_buf, {s, h, d + I32(1)}) *
                         tir::make_const(elem_dtype, -1.0),
                     tir::BufferLoad(qkv_buf, {s, h, d - I32(1)})),
                 "float32");
    } else {
      sin_part = freq_result.sin_freq *
                 CastTo(tvm::if_then_else(
                     d < I32(rotary_dim / 2),
                     tir::BufferLoad(qkv_buf, {s, h, d + I32(rotary_dim / 2)}) *
                         tir::make_const(elem_dtype, -1.0),
                     tir::BufferLoad(qkv_buf, {s, h, d - I32(rotary_dim / 2)})),
                 "float32");
    }
    // Result is float32 arithmetic; cast to elem_dtype at the outermost level
    PrimExpr rope_val = CastTo(cos_part + sin_part, dtype);
    for (const auto& [var, expr] : freq_result.var_map)
      rope_val = tir::Let(var, expr, rope_val);
    return rope_val;
  };

  // Condition: apply_rope > 0 && d < rotary_dim
  PrimExpr cond = tir::And(apply_rope_var > I32(0), d < I32(rotary_dim));

  // Q store
  Stmt q_store = tir::BufferStore(
      q_buf,
      tvm::if_then_else(cond, make_rope_val(),
                        tir::BufferLoad(qkv_buf, {s, h, d})),
      {s, h, d});
  // K store: h - num_q_heads
  Stmt k_store = tir::BufferStore(
      k_buf,
      tvm::if_then_else(cond, make_rope_val(),
                        tir::BufferLoad(qkv_buf, {s, h, d})),
      {s, h - I32(num_q_heads), d});
  // V store: no RoPE
  Stmt v_store = tir::BufferStore(
      v_buf,
      tir::BufferLoad(qkv_buf, {s, h, d}),
      {s, h - I32(num_q_heads + num_kv_heads), d});

  Stmt inner_body = tir::IfThenElse(
      h < I32(num_q_heads), q_store,
      tir::IfThenElse(h < I32(num_q_heads + num_kv_heads), k_store, v_store));

  // Inner sblock: "llama_fused_rope" with T.axis.spatial for s, h, d
  // Add tir.script_parsing_detect_access=3 so ScriptComplete infers reads AND writes.
  // ScriptComplete removes this annotation after filling reads/writes.
  ffi::Array<tir::IterVar> iter_vars = {
      tir::IterVar(Range::FromMinExtent(I32(0), seq_len),          s, tir::kDataPar, ""),
      tir::IterVar(Range::FromMinExtent(I32(0), I32(fused_heads)), h, tir::kDataPar, ""),
      tir::IterVar(Range::FromMinExtent(I32(0), I32(head_dim)),    d, tir::kDataPar, "")};

  ffi::Map<ffi::String, ffi::Any> sblock_annots;
  sblock_annots.Set("tir.script_parsing_detect_access", IntImm(DataType::Int(32), 3));

  Stmt sblock_realize = tir::SBlockRealize(
      {PrimExpr(iters_0), PrimExpr(iters_1), PrimExpr(iters_2)},
      tir::const_true(),
      tir::SBlock(iter_vars, {}, {}, "llama_fused_rope", inner_body,
                  std::nullopt, {}, {}, sblock_annots));

  // Nested For loops wrapping the sblock
  Stmt loop_body = sblock_realize;
  loop_body = tir::For(iters_2, I32(0), I32(head_dim),    tir::ForKind::kSerial, loop_body);
  loop_body = tir::For(iters_1, I32(0), I32(fused_heads), tir::ForKind::kSerial, loop_body);
  loop_body = tir::For(iters_0, I32(0), seq_len,          tir::ForKind::kSerial, loop_body);

  // Root sblock
  Stmt root_block = tir::SBlockRealize(
      {}, tir::const_true(),
      tir::SBlock({}, {}, {}, "root", loop_body));

  // Build buf_map with pos_buf->data (pos_data) so ScriptComplete can find pos_buf
  // when inferring reads/writes from BufferLoad(pos_buf, {s}) in the body.
  // After ScriptComplete, rebuild the PrimFunc with only h_pos as the key
  // (matching TVMScript's buffer_map which uses plain handle params as keys).
  ffi::Map<tir::Var, tir::Buffer> buf_map_complete;
  buf_map_complete.Set(h_qkv,    qkv_buf);
  buf_map_complete.Set(pos_data, pos_buf);  // pos_data = pos_buf->data (PointerType)
  buf_map_complete.Set(h_q,      q_buf);
  buf_map_complete.Set(h_k,      k_buf);
  buf_map_complete.Set(h_v,      v_buf);

  ffi::Array<tir::Var> params = {h_qkv, h_pos, h_q, h_k, h_v, apply_rope_var};
  tir::PrimFunc prim_func(params, root_block, VoidType(), buf_map_complete);
  prim_func = WithAttr(prim_func, "tir.noalias",   ffi::Any(true));
  prim_func = WithAttr(prim_func, "op_pattern",    ffi::Any(static_cast<int64_t>(8)));
  prim_func = WithAttr(prim_func, "global_symbol", ffi::Any(ffi::String("fused_rope")));
  // ScriptComplete infers reads/writes (triggered by tir.script_parsing_detect_access=3)
  // and removes that annotation from the sblock.
  prim_func = tir::ScriptComplete(prim_func, {});

  // Rebuild buffer_map with h_pos (plain handle) as key instead of pos_data (PointerType),
  // matching TVMScript's buffer_map layout.
  ffi::Map<tir::Var, tir::Buffer> buf_map_final;
  buf_map_final.Set(h_qkv, qkv_buf);
  buf_map_final.Set(h_pos, pos_buf);
  buf_map_final.Set(h_q,   q_buf);
  buf_map_final.Set(h_k,   k_buf);
  buf_map_final.Set(h_v,   v_buf);
  auto fptr = prim_func.CopyOnWrite();
  fptr->buffer_map = buf_map_final;
  return prim_func;
}

// ---------------------------------------------------------------------------
// LlamaRopeWithPositionMap (longrope variant)
// ---------------------------------------------------------------------------
// Builds the longrope-specific PrimFunc matching the expected TVMScript:
//   - 6th param: ext_factors_handle (T.handle) -> ext_factors buffer (8,) float32
//   - seq_len is int64
//   - root body: if seq_len > original_max_pos: long_loop else short_loop
//   - each sblock has a local sub-buffer (long_factors / short_factors) aliasing ext_factors
//   - condition: d < rotary_dim (no apply_rope)
//   - freq: pos / (factors[d % (rotary_dim/2)] * theta^(...))
// ---------------------------------------------------------------------------
static tir::PrimFunc BuildLlamaRopeWithPositionMapLongropeFunc(
    double theta, double scale, int64_t head_dim, int64_t num_q_heads, int64_t num_kv_heads,
    const std::string& dtype, const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
    int64_t rotary_dim) {
  int64_t fused_heads = num_q_heads + num_kv_heads * 2;
  DataType elem_dtype = DataType(runtime::StringToDLDataType(dtype));

  int64_t original_max_position_embeddings =
      rope_scaling.at("original_max_position_embeddings").cast<int64_t>();
  int64_t max_position_embeddings =
      rope_scaling.at("max_position_embeddings").cast<int64_t>();

  double scale_val = static_cast<double>(max_position_embeddings) /
                     static_cast<double>(original_max_position_embeddings);
  double scaling_factor =
      (scale_val > 1.0)
          ? std::sqrt(1.0 + std::log(scale_val) / std::log(original_max_position_embeddings))
          : 1.0;

  int64_t half_rotary = rotary_dim / 2;  // = 4 for rotary_dim=8

  auto I32 = [](int64_t v) { return IntImm(DataType::Int(32), v); };
  auto I64 = [](int64_t v) { return IntImm(DataType::Int(64), v); };

  // seq_len is int64 for longrope
  tir::Var seq_len("seq_len", DataType::Int(64));
  tir::Var position_map_elem_offset("position_map_elem_offset", DataType::Int(64));

  // Plain handle params
  tir::Var h_qkv("var_qkv",           DataType::Handle());
  tir::Var h_pos("var_position_map",   DataType::Handle());
  tir::Var h_q("var_q",               DataType::Handle());
  tir::Var h_k("var_k",               DataType::Handle());
  tir::Var h_v("var_v",               DataType::Handle());
  tir::Var h_ext("ext_factors_handle", DataType::Handle());

  // Buffers — seq_len is int64, static dims are int32
  tir::Buffer qkv_buf = tir::decl_buffer(
      {seq_len, I32(fused_heads), I32(head_dim)}, elem_dtype, "qkv");
  tir::Buffer q_buf = tir::decl_buffer(
      {seq_len, I32(num_q_heads),  I32(head_dim)}, elem_dtype, "q");
  tir::Buffer k_buf = tir::decl_buffer(
      {seq_len, I32(num_kv_heads), I32(head_dim)}, elem_dtype, "k");
  tir::Buffer v_buf = tir::decl_buffer(
      {seq_len, I32(num_kv_heads), I32(head_dim)}, elem_dtype, "v");

  // position_map: int32 elements, offset_factor=1
  tir::Var pos_data("position_map", PointerType(PrimType(DataType::Int(32))));
  tir::Buffer pos_buf = tir::Buffer(
      pos_data, DataType::Int(32), {seq_len}, {},
      position_map_elem_offset, "position_map", 0, 1, tir::kDefault);

  // ext_factors: float32 buffer of size (rotary_dim,) = (8,)
  tir::Buffer ext_buf = tir::decl_buffer({I32(rotary_dim)}, DataType::Float(32), "ext_factors");

  // Build one loop nest body given a factors sub-buffer (long or short).
  // Each call gets its OWN fresh loop/axis vars so the two branches are
  // structurally independent (SE maps vars by position within each branch).
  auto make_loop_body = [&](tir::Buffer factors_buf,
                            const std::string& factors_name) -> Stmt {
    // Fresh vars for this branch
    tir::Var li0("iters_0", DataType::Int(64));
    tir::Var li1("iters_1", DataType::Int(32));
    tir::Var li2("iters_2", DataType::Int(32));
    tir::Var ls("s", DataType::Int(64));
    tir::Var lh("h", DataType::Int(32));
    tir::Var ld("d", DataType::Int(32));
    (void)factors_name;  // used only for documentation
    auto make_rope_val = [&]() -> PrimExpr {
      PrimExpr pos_f32 = CastTo(tir::BufferLoad(pos_buf, {ls}), "float32");
      if (scale != 1.0) pos_f32 = pos_f32 * MakeFloat32(scale);

      PrimExpr exponent =
          CastTo(tir::FloorMod(ld * I32(2), I32(rotary_dim)), "float32") /
          MakeFloat32(static_cast<double>(rotary_dim));
      PrimExpr divisor =
          tir::BufferLoad(factors_buf, {tir::FloorMod(ld, I32(half_rotary))}) *
          tvm::pow(MakeFloat32(theta), exponent);
      PrimExpr freq_expr = pos_f32 / divisor;

      tir::Var freq_var("freq", DataType::Float(32));
      PrimExpr cos_v = tvm::cos(freq_var) * MakeFloat32(scaling_factor);
      PrimExpr sin_v = tvm::sin(freq_var) * MakeFloat32(scaling_factor);

      PrimExpr x = tir::BufferLoad(qkv_buf, {ls, lh, ld});
      PrimExpr cos_part = cos_v * CastTo(x, "float32");
      PrimExpr sin_part = sin_v *
          CastTo(tvm::if_then_else(
              ld < I32(half_rotary),
              tir::BufferLoad(qkv_buf, {ls, lh, ld + I32(half_rotary)}) *
                  tir::make_const(elem_dtype, -1.0),
              tir::BufferLoad(qkv_buf, {ls, lh, ld - I32(half_rotary)})),
          "float32");
      PrimExpr rope_val = CastTo(cos_part + sin_part, dtype);
      rope_val = tir::Let(freq_var, freq_expr, rope_val);
      return rope_val;
    };

    PrimExpr cond = ld < I32(rotary_dim);
    Stmt q_store = tir::BufferStore(
        q_buf,
        tvm::if_then_else(cond, make_rope_val(), tir::BufferLoad(qkv_buf, {ls, lh, ld})),
        {ls, lh, ld});
    Stmt k_store = tir::BufferStore(
        k_buf,
        tvm::if_then_else(cond, make_rope_val(), tir::BufferLoad(qkv_buf, {ls, lh, ld})),
        {ls, lh - I32(num_q_heads), ld});
    Stmt v_store = tir::BufferStore(
        v_buf, tir::BufferLoad(qkv_buf, {ls, lh, ld}),
        {ls, lh - I32(num_q_heads + num_kv_heads), ld});

    Stmt inner_body = tir::IfThenElse(
        lh < I32(num_q_heads), q_store,
        tir::IfThenElse(lh < I32(num_q_heads + num_kv_heads), k_store, v_store));

    ffi::Array<tir::IterVar> iter_vars = {
        tir::IterVar(Range::FromMinExtent(I64(0), seq_len),          ls, tir::kDataPar, ""),
        tir::IterVar(Range::FromMinExtent(I32(0), I32(fused_heads)), lh, tir::kDataPar, ""),
        tir::IterVar(Range::FromMinExtent(I32(0), I32(head_dim)),    ld, tir::kDataPar, "")};

    ffi::Map<ffi::String, ffi::Any> annots;
    annots.Set("tir.script_parsing_detect_access", IntImm(DataType::Int(32), 3));

    Stmt sblock = tir::SBlockRealize(
        {PrimExpr(li0), PrimExpr(li1), PrimExpr(li2)},
        tir::const_true(),
        tir::SBlock(iter_vars, {}, {}, "llama_fused_rope", inner_body,
                    std::nullopt, {}, {}, annots));

    Stmt loop = sblock;
    loop = tir::For(li2, I32(0), I32(head_dim),    tir::ForKind::kSerial, loop);
    loop = tir::For(li1, I32(0), I32(fused_heads), tir::ForKind::kSerial, loop);
    loop = tir::For(li0, I64(0), seq_len,          tir::ForKind::kSerial, loop);
    return loop;
  };

  // long_factors: sub-buffer aliasing ext_factors.data, offset=0, size=half_rotary
  tir::Buffer long_factors_buf = tir::Buffer(
      ext_buf->data, DataType::Float(32), {I32(half_rotary)}, {},
      I32(0), "long_factors", 0, 0, tir::kDefault);
  // short_factors: sub-buffer aliasing ext_factors.data, offset=half_rotary
  tir::Buffer short_factors_buf = tir::Buffer(
      ext_buf->data, DataType::Float(32), {I32(half_rotary)}, {},
      I32(half_rotary), "short_factors", 0, 0, tir::kDefault);

  Stmt long_loop  = make_loop_body(long_factors_buf,  "long_factors");
  Stmt short_loop = make_loop_body(short_factors_buf, "short_factors");

  // Root body: if seq_len > original_max_position_embeddings: long else short
  Stmt root_body = tir::IfThenElse(
      seq_len > I64(original_max_position_embeddings), long_loop, short_loop);

  Stmt root_block = tir::SBlockRealize(
      {}, tir::const_true(),
      tir::SBlock({}, {}, {}, "root", root_body));

  // buf_map for ScriptComplete: use PointerType data vars as keys
  ffi::Map<tir::Var, tir::Buffer> buf_map_complete;
  buf_map_complete.Set(h_qkv,         qkv_buf);
  buf_map_complete.Set(pos_data,       pos_buf);
  buf_map_complete.Set(h_q,            q_buf);
  buf_map_complete.Set(h_k,            k_buf);
  buf_map_complete.Set(h_v,            v_buf);
  buf_map_complete.Set(ext_buf->data,  ext_buf);

  ffi::Array<tir::Var> params = {h_qkv, h_pos, h_q, h_k, h_v, h_ext};
  tir::PrimFunc prim_func(params, root_block, VoidType(), buf_map_complete);
  prim_func = WithAttr(prim_func, "tir.noalias",   ffi::Any(true));
  prim_func = WithAttr(prim_func, "op_pattern",    ffi::Any(static_cast<int64_t>(8)));
  prim_func = WithAttr(prim_func, "global_symbol",
                       ffi::Any(ffi::String("fused_rope_longrope_scaling")));
  prim_func = tir::ScriptComplete(prim_func, {});

  // Rebuild buffer_map with plain handle keys (matching TVMScript)
  ffi::Map<tir::Var, tir::Buffer> buf_map_final;
  buf_map_final.Set(h_qkv, qkv_buf);
  buf_map_final.Set(h_pos, pos_buf);
  buf_map_final.Set(h_q,   q_buf);
  buf_map_final.Set(h_k,   k_buf);
  buf_map_final.Set(h_v,   v_buf);
  buf_map_final.Set(h_ext, ext_buf);
  auto fptr2 = prim_func.CopyOnWrite();
  fptr2->buffer_map = buf_map_final;
  return prim_func;
}

tir::PrimFunc LlamaRopeWithPositionMap(
    double theta, double scale, int64_t head_dim, int64_t num_q_heads, int64_t num_kv_heads,
    const std::string& dtype, const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
    ffi::Optional<int64_t> rotary_dim) {
  int64_t rotary_dim_val = rotary_dim.value_or(head_dim);
  // Dispatch to longrope-specific builder when rope_type == "longrope"
  if (rope_scaling.count("rope_type")) {
    std::string rope_type =
        rope_scaling.at("rope_type").cast<ffi::String>().operator std::string();
    if (rope_type == "longrope") {
      return BuildLlamaRopeWithPositionMapLongropeFunc(
          theta, scale, head_dim, num_q_heads, num_kv_heads, dtype, rope_scaling, rotary_dim_val);
    }
  }
  return BuildLlamaRopeWithPositionMapFunc(theta, scale, head_dim, num_q_heads, num_kv_heads,
                                           dtype, rope_scaling, rotary_dim_val);
}

tir::PrimFunc Llama4RopeWithPositionMap(
    double theta, double scale, int64_t head_dim, int64_t num_q_heads, int64_t num_kv_heads,
    const std::string& dtype, const ffi::Map<ffi::String, ffi::Any>& rope_scaling,
    ffi::Optional<int64_t> rotary_dim) {
  int64_t rotary_dim_val = rotary_dim.value_or(head_dim);
  return BuildLlamaRopeWithPositionMapFunc(theta, scale, head_dim, num_q_heads, num_kv_heads,
                                           dtype, rope_scaling, rotary_dim_val);
}

// ---------------------------------------------------------------------------
// FFI registrations
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.frontend.nn.llm.position_embedding.switch_rope_freq_func",
           [](ffi::Map<ffi::String, ffi::Any> rope_scaling, PrimExpr s, PrimExpr d,
              int64_t d_range, PrimExpr theta,
              ffi::String dtype) -> ffi::Array<ffi::Any> {
             RopeFreqFunc fn = SwitchRopeFreqFunc(rope_scaling);
             RopeFreqResult res = fn(s, d, d_range, theta, std::string(dtype), rope_scaling);
             ffi::Array<ffi::Any> keys, vals;
             for (const auto& [var, expr] : res.var_map) {
               keys.push_back(ffi::Any(var));
               vals.push_back(ffi::Any(expr));
             }
             return {ffi::Any(res.cos_freq), ffi::Any(res.sin_freq),
                     ffi::Any(keys), ffi::Any(vals)};
           })
      .def("relax.frontend.nn.llm.position_embedding.llama_rope",
           [](NNTensor qkv, tir::Var total_seq_len, double theta, double scale,
              int64_t num_q_heads, int64_t num_kv_heads,
              ffi::Map<ffi::String, ffi::Any> rope_scaling,
              ffi::Optional<int64_t> rotary_dim) -> ffi::Array<ffi::Any> {
             auto [q, k, v] = LlamaRope(qkv, total_seq_len, theta, scale, num_q_heads,
                                         num_kv_heads, rope_scaling, rotary_dim);
             return {ffi::Any(q), ffi::Any(k), ffi::Any(v)};
           })
      .def("relax.frontend.nn.llm.position_embedding.llama_rope_with_position_map",
           [](double theta, double scale, int64_t head_dim, int64_t num_q_heads,
              int64_t num_kv_heads, ffi::String dtype,
              ffi::Map<ffi::String, ffi::Any> rope_scaling,
              ffi::Optional<int64_t> rotary_dim) -> tir::PrimFunc {
             return LlamaRopeWithPositionMap(theta, scale, head_dim, num_q_heads, num_kv_heads,
                                             std::string(dtype), rope_scaling, rotary_dim);
           })
      .def("relax.frontend.nn.llm.position_embedding.llama4_rope_with_position_map",
           [](double theta, double scale, int64_t head_dim, int64_t num_q_heads,
              int64_t num_kv_heads, ffi::String dtype,
              ffi::Map<ffi::String, ffi::Any> rope_scaling,
              ffi::Optional<int64_t> rotary_dim) -> tir::PrimFunc {
             return Llama4RopeWithPositionMap(theta, scale, head_dim, num_q_heads, num_kv_heads,
                                              std::string(dtype), rope_scaling, rotary_dim);
           });
}

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

