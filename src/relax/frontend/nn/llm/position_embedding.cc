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

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

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

RopeFreqResult RopeFreqDefault(tir::Var s, tir::Var d, int64_t d_range, double theta,
                               const std::string& dtype,
                               const ffi::Map<ffi::String, ffi::Any>& extra_args) {
  // freq = s / (theta ^ ((d * 2 % d_range) / d_range))
  PrimExpr exponent = CastTo(tir::FloorMod(d * 2, MakeInt64(d_range)), "float32") /
                      MakeFloat32(static_cast<double>(d_range));
  PrimExpr freq = CastTo(s, "float32") / tvm::pow(MakeFloat32(theta), exponent);

  tir::Var freq_var("freq", DataType::Float(32));
  PrimExpr cos_freq = CastTo(tvm::cos(freq_var), dtype);
  PrimExpr sin_freq = CastTo(tvm::sin(freq_var), dtype);

  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> var_map;
  var_map[freq_var] = freq;

  return {cos_freq, sin_freq, var_map};
}

RopeFreqResult RopeFreqGptj(tir::Var s, tir::Var d, int64_t d_range, double theta,
                            const std::string& dtype,
                            const ffi::Map<ffi::String, ffi::Any>& extra_args) {
  // freq = s / (theta ^ (2 * (d // 2) % d_range / d_range))
  PrimExpr exponent = CastTo(tir::FloorMod(2 * tir::FloorDiv(d, MakeInt64(2)), MakeInt64(d_range)),
                              "float32") /
                      MakeFloat32(static_cast<double>(d_range));
  PrimExpr freq = CastTo(s, "float32") / tvm::pow(MakeFloat32(theta), exponent);

  tir::Var freq_var("freq", DataType::Float(32));
  PrimExpr cos_freq = CastTo(tvm::cos(freq_var), dtype);
  PrimExpr sin_freq = CastTo(tvm::sin(freq_var), dtype);

  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> var_map;
  var_map[freq_var] = freq;

  return {cos_freq, sin_freq, var_map};
}

RopeFreqResult RopeFreqLlama3(tir::Var s, tir::Var d, int64_t d_range, double theta,
                              const std::string& dtype,
                              const ffi::Map<ffi::String, ffi::Any>& extra_args) {
  // Extract required parameters
  double factor = extra_args.at("factor").cast<double>();
  double low_freq_factor = extra_args.at("low_freq_factor").cast<double>();
  double high_freq_factor = extra_args.at("high_freq_factor").cast<double>();
  int64_t original_max_position_embeddings =
      extra_args.at("original_max_position_embeddings").cast<int64_t>();

  // orig_freq = 1 / (theta ^ ((d * 2 % d_range) / d_range))
  PrimExpr exponent = CastTo(tir::FloorMod(d * 2, MakeInt64(d_range)), "float32") /
                      MakeFloat32(static_cast<double>(d_range));
  PrimExpr orig_freq = MakeFloat32(1.0) / tvm::pow(MakeFloat32(theta), exponent);

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
  PrimExpr smoothed_freq =
      CastTo(s, "float32") *
      ((MakeFloat32(1.0) - smooth) * orig_freq_var * MakeFloat32(llama3_inv_scaling_factor) +
       smooth * orig_freq_var);

  tir::Var smoothed_freq_var("smoothed_freq", DataType::Float(32));
  PrimExpr cos_freq = CastTo(tvm::cos(smoothed_freq_var), dtype);
  PrimExpr sin_freq = CastTo(tvm::sin(smoothed_freq_var), dtype);

  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> var_map;
  var_map[smoothed_freq_var] = smoothed_freq;
  var_map[orig_freq_var] = orig_freq;

  return {cos_freq, sin_freq, var_map};
}

RopeFreqResult RopeFreqLlama4(tir::Var s, tir::Var d, int64_t d_range, double theta,
                              const std::string& dtype,
                              const ffi::Map<ffi::String, ffi::Any>& extra_args) {
  // Extract required parameters
  double factor = extra_args.at("factor").cast<double>();
  double low_freq_factor = extra_args.at("low_freq_factor").cast<double>();
  double high_freq_factor = extra_args.at("high_freq_factor").cast<double>();
  int64_t original_max_position_embeddings =
      extra_args.at("original_max_position_embeddings").cast<int64_t>();

  // orig_freq = 1 / (theta ^ (2 * (d // 2) / d_range))
  PrimExpr exponent = CastTo(2 * tir::FloorDiv(d, MakeInt64(2)), "float32") /
                      MakeFloat32(static_cast<double>(d_range));
  PrimExpr orig_freq = MakeFloat32(1.0) / tvm::pow(MakeFloat32(theta), exponent);

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
    smoothed_freq = CastTo(s, "float32") * scaled_freq;
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

    smoothed_freq =
        CastTo(s, "float32") *
        ((MakeFloat32(1.0) - smooth) * orig_freq_var * MakeFloat32(llama4_inv_scaling_factor) +
         smooth * orig_freq_var);
  }

  tir::Var smoothed_freq_var("smoothed_freq", DataType::Float(32));
  PrimExpr cos_freq = CastTo(tvm::cos(smoothed_freq_var), dtype);
  PrimExpr sin_freq = CastTo(tvm::sin(smoothed_freq_var), dtype);

  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> var_map;
  var_map[smoothed_freq_var] = smoothed_freq;
  var_map[orig_freq_var] = orig_freq;

  return {cos_freq, sin_freq, var_map};
}

RopeFreqResult RopeFreqLongrope(tir::Var s, tir::Var d, int64_t d_range, double theta,
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
  PrimExpr exponent = CastTo(tir::FloorMod(d * 2, MakeInt64(d_range)), "float32") /
                      MakeFloat32(static_cast<double>(d_range));
  PrimExpr divisor = tvm::pow(MakeFloat32(theta), exponent);

  // Apply extension factors if provided
  if (extra_args.count("ext_factors")) {
    // ext_factors is a buffer, index it by d % (d_range // 2)
    auto ext_factors_any = extra_args.at("ext_factors");
    // In the Python code, ext_factors is accessed as ext_factors[d % (d_range // 2)]
    // For C++, we need to handle this as a buffer access in TIR
    // This will be handled in the TIR function generation
  }

  PrimExpr freq = CastTo(s, "float32") / divisor;

  tir::Var freq_var("freq", DataType::Float(32));
  PrimExpr cos_freq = CastTo(tvm::cos(freq_var) * MakeFloat32(scaling_factor), dtype);
  PrimExpr sin_freq = CastTo(tvm::sin(freq_var) * MakeFloat32(scaling_factor), dtype);

  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> var_map;
  var_map[freq_var] = freq;

  return {cos_freq, sin_freq, var_map};
}

// Helper for YaRN: find correction dimension
static PrimExpr YarnFindCorrectionDim(tir::Var d, int64_t num_rotations,
                                      int64_t max_position_embeddings,
                                      double inv_theta_log_scale) {
  // d * log(max_position_embeddings / (num_rotations * 2 * pi)) * inv_theta_log_scale
  double log_val = std::log(static_cast<double>(max_position_embeddings) /
                             (static_cast<double>(num_rotations) * 2.0 * M_PI));
  return CastTo(d, "float32") * MakeFloat32(log_val * inv_theta_log_scale);
}

// Helper for YaRN: find correction range
static std::pair<PrimExpr, PrimExpr> YarnFindCorrectionRange(
    tir::Var d, int64_t low_rot, int64_t high_rot, int64_t d_range,
    int64_t max_position_embeddings, double inv_theta_log_scale) {
  PrimExpr low = YarnFindCorrectionDim(d, low_rot, max_position_embeddings, inv_theta_log_scale);
  PrimExpr high = YarnFindCorrectionDim(d, high_rot, max_position_embeddings, inv_theta_log_scale);

  low = tvm::max(low, MakeFloat32(0.0));
  high = tvm::min(high, MakeFloat32(static_cast<double>(d_range - 1)));

  return {low, high};
}

RopeFreqResult RopeFreqYarn(tir::Var s, tir::Var d, int64_t d_range, double theta,
                            const std::string& dtype,
                            const ffi::Map<ffi::String, ffi::Any>& extra_args) {
  int64_t original_max_position_embeddings =
      extra_args.at("original_max_position_embeddings").cast<int64_t>();
  double scaling_factor = extra_args.at("scaling_factor").cast<double>();
  int64_t beta_fast = extra_args.at("beta_fast").cast<int64_t>();
  int64_t beta_slow = extra_args.at("beta_slow").cast<int64_t>();
  double inv_theta_log_scale = extra_args.at("inv_theta_log_scale").cast<double>();

  // Compute base frequencies
  PrimExpr exponent = CastTo(tir::FloorMod(d * 2, MakeInt64(d_range)), "float32") /
                      MakeFloat32(static_cast<double>(d_range));
  PrimExpr freq_power = tvm::pow(MakeFloat32(theta), exponent);
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

  PrimExpr freq = CastTo(s, "float32") * inv_freq;

  tir::Var freq_var("freq", DataType::Float(32));
  PrimExpr cos_freq = CastTo(tvm::cos(freq_var), dtype);
  PrimExpr sin_freq = CastTo(tvm::sin(freq_var), dtype);

  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> var_map;
  var_map[freq_var] = freq;

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
  // Get the QKV tensor shape
  auto qkv_sinfo = qkv->expr->struct_info_.as<TensorStructInfoNode>();
  TVM_FFI_ICHECK(qkv_sinfo != nullptr) << "QKV must be a tensor";
  TVM_FFI_ICHECK(qkv_sinfo->shape.defined()) << "QKV shape must be defined";

  auto shape_expr = qkv_sinfo->shape.value().as<ShapeExprNode>();
  TVM_FFI_ICHECK(shape_expr != nullptr) << "QKV shape must be a ShapeExpr";
  TVM_FFI_ICHECK(shape_expr->values.size() == 4)
      << "QKV must be 4-D: [batch, seq_len, fused_heads, head_dim]";

  PrimExpr batch_size = shape_expr->values[0];
  PrimExpr seq_len = shape_expr->values[1];
  int64_t fused_heads = num_q_heads + num_kv_heads * 2;
  PrimExpr head_dim_expr = shape_expr->values[3];

  // Extract head_dim as int64
  auto head_dim_int = head_dim_expr.as<IntImmNode>();
  TVM_FFI_ICHECK(head_dim_int != nullptr) << "head_dim must be a static integer";
  int64_t head_dim = head_dim_int->value;

  // Set rotary_dim
  int64_t rotary_dim_val = rotary_dim.value_or(head_dim);

  DataType dtype = qkv_sinfo->dtype;
  std::string dtype_str = runtime::DLDataTypeToString(dtype);

  // Get the RoPE frequency function
  RopeFreqFunc rope_freq_func = SwitchRopeFreqFunc(rope_scaling);

  // Determine rope_type for rotation pattern
  std::string rope_type = "default";
  if (rope_scaling.count("rope_type")) {
    rope_type = rope_scaling.at("rope_type").cast<ffi::String>().operator std::string();
  }

  // Build the TIR PrimFunc using the C++ TIR builder
  // We'll construct it similar to the Python @T.prim_func decorator

  // Create symbolic dimension variables
  tir::Var batch_var("batch_size", DataType::Int(64));
  tir::Var seq_var("seq_len", DataType::Int(64));

  // Create buffer declarations
  tir::Buffer qkv_buf =
      tir::decl_buffer({batch_var, seq_var, MakeInt64(fused_heads), MakeInt64(head_dim)}, dtype,
                       "qkv");
  tir::Buffer q_buf = tir::decl_buffer(
      {batch_var, seq_var, MakeInt64(num_q_heads), MakeInt64(head_dim)}, dtype, "q");
  tir::Buffer k_buf = tir::decl_buffer(
      {batch_var, seq_var, MakeInt64(num_kv_heads), MakeInt64(head_dim)}, dtype, "k");
  tir::Buffer v_buf = tir::decl_buffer(
      {batch_var, seq_var, MakeInt64(num_kv_heads), MakeInt64(head_dim)}, dtype, "v");

  // Create loop iteration variables
  tir::Var b("b", DataType::Int(64));
  tir::Var s("s", DataType::Int(64));
  tir::Var h("h", DataType::Int(64));
  tir::Var d_var("d", DataType::Int(64));

  // Helper lambda to build the RoPE expression
  auto build_rope_expr = [&](PrimExpr d_idx) -> PrimExpr {
    // Compute position: (s + (total_seq_len - seq_len)) * scale
    PrimExpr offset = total_seq_len - seq_var;
    PrimExpr pos = (CastTo(s, "float32") + CastTo(offset, "float32")) * MakeFloat32(scale);

    // Get frequency computation result
    auto freq_result = rope_freq_func(tir::Var("_pos", DataType::Int(64)), d_var, rotary_dim_val,
                                      theta, "float32", rope_scaling);

    // Build cos and sin parts
    PrimExpr x = tir::BufferLoad(qkv_buf, {b, s, h, d_idx});
    PrimExpr cos_part = freq_result.cos_freq * CastTo(x, "float32");

    PrimExpr sin_part;
    if (rope_type == "gptj") {
      // GPT-J: rotate adjacent pairs
      sin_part = freq_result.sin_freq *
                 tvm::if_then_else(tir::FloorMod(d_idx, MakeInt64(2)) == MakeInt64(0),
                                   -CastTo(tir::BufferLoad(qkv_buf, {b, s, h, d_idx + MakeInt64(1)}), "float32"),
                                   CastTo(tir::BufferLoad(qkv_buf, {b, s, h, d_idx - MakeInt64(1)}), "float32"));
    } else {
      // Default/LLaMA: rotate first half with second half
      sin_part =
          freq_result.sin_freq *
          tvm::if_then_else(
              d_idx < MakeInt64(rotary_dim_val / 2),
              -CastTo(tir::BufferLoad(qkv_buf, {b, s, h, d_idx + MakeInt64(rotary_dim_val / 2)}), "float32"),
              CastTo(tir::BufferLoad(qkv_buf, {b, s, h, d_idx - MakeInt64(rotary_dim_val / 2)}), "float32"));
    }

    PrimExpr result = CastTo(cos_part + sin_part, dtype_str);

    // Bind intermediate variables from frequency computation
    for (const auto& [var, expr] : freq_result.var_map) {
      result = tir::Let(var, expr, result);
    }

    return result;
  };

  // Build the store statements with conditionals
  // Q: if h < num_q_heads
  PrimExpr q_value =
      tvm::if_then_else(d_var < MakeInt64(rotary_dim_val), build_rope_expr(d_var),
                        tir::BufferLoad(qkv_buf, {b, s, h, d_var}));
  Stmt q_store = tir::BufferStore(q_buf, q_value, {b, s, h, d_var});

  // K: if num_q_heads <= h < num_q_heads + num_kv_heads
  PrimExpr k_value =
      tvm::if_then_else(d_var < MakeInt64(rotary_dim_val), build_rope_expr(d_var),
                        tir::BufferLoad(qkv_buf, {b, s, h, d_var}));
  Stmt k_store =
      tir::BufferStore(k_buf, k_value, {b, s, h - MakeInt64(num_q_heads), d_var});

  // V: if h >= num_q_heads + num_kv_heads
  Stmt v_store = tir::BufferStore(
      v_buf, tir::BufferLoad(qkv_buf, {b, s, h, d_var}),
      {b, s, h - MakeInt64(num_q_heads + num_kv_heads), d_var});

  // Build the conditional: if h < num_q_heads then Q else if h < num_q_heads + num_kv_heads then K else V
  Stmt body = tir::IfThenElse(
      h < MakeInt64(num_q_heads), q_store,
      tir::IfThenElse(h < MakeInt64(num_q_heads + num_kv_heads), k_store, v_store));

  // Wrap in a block with axis bindings
  ffi::Array<tir::IterVar> iter_vars = {
      tir::IterVar(Range::FromMinExtent(0, batch_var), b, tir::kDataPar, "b"),
      tir::IterVar(Range::FromMinExtent(0, seq_var), s, tir::kDataPar, "s"),
      tir::IterVar(Range::FromMinExtent(0, MakeInt64(fused_heads)), h, tir::kDataPar, "h"),
      tir::IterVar(Range::FromMinExtent(0, MakeInt64(head_dim)), d_var, tir::kDataPar, "d")};

  tir::SBlock block(iter_vars, {}, {}, "llama_fused_rope", body);
  body = tir::SBlockRealize({b, s, h, d_var}, Bool(true), block);

  // Create nested for loops (innermost to outermost)
  body = tir::For(d_var, MakeInt64(0), MakeInt64(head_dim), tir::ForKind::kSerial, body);
  body = tir::For(h, MakeInt64(0), MakeInt64(fused_heads), tir::ForKind::kSerial, body);
  body = tir::For(s, MakeInt64(0), seq_var, tir::ForKind::kSerial, body);
  body = tir::For(b, MakeInt64(0), batch_var, tir::ForKind::kSerial, body);

  // Create function parameters
  ffi::Array<Var> params = {qkv_buf->data, q_buf->data, k_buf->data, v_buf->data, total_seq_len};

  // Create function attributes
  ffi::Map<ffi::String, ObjectRef> attrs;
  attrs.Set("op_pattern", Integer(8));  // opaque
  attrs.Set("tir.noalias", Bool(true));

  // Create the PrimFunc
    ffi::Map<Var, tir::Buffer> buffer_map;
  buffer_map.Set(qkv_buf->data, qkv_buf);
  buffer_map.Set(q_buf->data, q_buf);
  buffer_map.Set(k_buf->data, k_buf);
  buffer_map.Set(v_buf->data, v_buf);
  tir::PrimFunc prim_func(params, body, VoidType(), buffer_map, DictAttrs(attrs));
  prim_func = WithAttr(prim_func, "global_symbol", ffi::String("llama_rope"));

  // Now use tensor_ir_op to call this function
  BlockBuilder bb = BlockBuilder_Current();
  TVM_FFI_ICHECK(bb.defined()) << "BlockBuilder must be defined";

  // Create output placeholders
  TensorStructInfo q_sinfo(
      ShapeExpr({batch_size, seq_len, MakeInt64(num_q_heads), MakeInt64(head_dim)}), dtype);
  TensorStructInfo k_sinfo(
      ShapeExpr({batch_size, seq_len, MakeInt64(num_kv_heads), MakeInt64(head_dim)}), dtype);
  TensorStructInfo v_sinfo(
      ShapeExpr({batch_size, seq_len, MakeInt64(num_kv_heads), MakeInt64(head_dim)}), dtype);
  relax::Var q_out_ph("q", q_sinfo);
  relax::Var k_out_ph("k", k_sinfo);
  relax::Var v_out_ph("v", v_sinfo);

  // Call tensor_ir_op
  ffi::Any result_any = NNTensorIrOp(
      prim_func, "llama_rope", {qkv->expr, PrimValue(total_seq_len)},
      {ffi::Any(q_out_ph), ffi::Any(k_out_ph), ffi::Any(v_out_ph)});

  // Extract the three outputs from the Array<Any>
  ffi::Array<ffi::Any> result_array = result_any.cast<ffi::Array<ffi::Any>>();
  TVM_FFI_ICHECK_EQ(result_array.size(), 3) << "Expected 3 outputs from llama_rope";

  relax::Var q_out = result_array[0].cast<relax::Var>();
  relax::Var k_out = result_array[1].cast<relax::Var>();
  relax::Var v_out = result_array[2].cast<relax::Var>();

  return {NNTensor(q_out), NNTensor(k_out), NNTensor(v_out)};
}

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm






