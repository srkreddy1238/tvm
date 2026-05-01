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
 * \file src/relax/frontend/nn/llm/kv_cache_attention_prefill_gpu_helpers.cc
 * \brief Implementations of shared helpers for GPU prefill attention kernels.
 */

#include "kv_cache_attention_prefill_gpu_helpers.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace llm {

using namespace tvm::tir;

// ============================================================================
// ComputePrefillKernelConfig  (mirrors Python _get_prefill_kernel_config)
// ============================================================================

PrefillKernelConfig ComputePrefillKernelConfig(int64_t h_kv, int64_t h_q, int64_t d,
                                               const std::string& dtype, Target target) {
  int64_t NUM_BLKS = 16;
  int64_t dtype_bits = DataType(runtime::StringToDLDataType(dtype)).bits();
  int64_t dtype_bytes = (dtype_bits + 7) / 8;
  int64_t LOAD_VEC = 8 / dtype_bytes;  // 8 bytes
  int64_t group_size = h_q / h_kv;

  int64_t bdx = 32;
  int64_t num_warps = 4;

  int64_t d_factor = std::max(d / 128, (int64_t)1);
  int64_t base_tile = 64 / dtype_bytes / d_factor;
  int64_t tile_x = base_tile;
  int64_t tile_y = d;
  int64_t tile_z = base_tile;
  int64_t original_tile_y = tile_y;
  int64_t original_tile_z = tile_z;

  while ((tile_x * tile_z) % (bdx * num_warps) != 0) tile_z += original_tile_z;
  while ((tile_x * tile_y) % (bdx * num_warps) != 0) tile_y += original_tile_y;

  std::string target_str = target->str();

  // webgpu: lower tile_z / num_warps to avoid exceeding maxComputeWorkgroupStorageSize
  bool is_webgpu = (target_str.find("webgpu") != std::string::npos);
  if (is_webgpu) {
    int64_t d_factor2 = (d + 127) / 128;
    int64_t bits_factor = (dtype_bits + 15) / 16;
    if (d_factor2 * bits_factor >= 4) {
      tile_z = 8;
      num_warps = 2;
    }
  }

  // Adreno / Android mobile GPU
  bool is_opencl = (target_str.find("opencl") != std::string::npos);
  bool is_vulkan = (target_str.find("vulkan") != std::string::npos);
  bool is_android = (target_str.find("android") != std::string::npos);
  bool is_adreno = (target_str.find("adreno") != std::string::npos);
  if ((is_opencl || is_vulkan) && (is_android || is_adreno)) {
    if (is_opencl) LOAD_VEC = 16 / dtype_bytes;
    NUM_BLKS = group_size * 8;
    tile_x = 32;
    tile_z = 4;
    if ((tile_y * tile_z) % (bdx * num_warps) != 0) tile_z = 16;
  }

  return {NUM_BLKS, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z};
}

// ============================================================================
// BuildPrefillRopeExpr
// ============================================================================

PrimExpr BuildPrefillRopeExpr(tir::Buffer buf, ffi::Array<PrimExpr> base_indices,
                               tir::Var d_idx, int64_t d, PrimExpr pos_expr,
                               tir::Var rope_scale, tir::Var rope_theta, tir::Var rotary_mode,
                               const std::string& dtype,
                               const ffi::Map<ffi::String, ffi::Any>& rope_scaling) {
  bool is_f16 = (dtype == "float16");
  int64_t half_d = d / 2;
  DataType dt = DataType(runtime::StringToDLDataType(dtype));

  // Current element: buf[base_indices..., d_idx]
  ffi::Array<PrimExpr> full_idx = base_indices;
  full_idx.push_back(d_idx);
  PrimExpr elem = tir::BufferLoad(buf, full_idx);

  // Partner: if d_idx < d/2: buf[..., d_idx+d/2]*(-1)  else: buf[..., d_idx-d/2]
  ffi::Array<PrimExpr> idx_plus = base_indices;
  idx_plus.push_back(d_idx + I32(half_d));
  ffi::Array<PrimExpr> idx_minus = base_indices;
  idx_minus.push_back(d_idx - I32(half_d));
  PrimExpr neg_one = tir::make_const(dt, -1.0);
  PrimExpr partner = tvm::if_then_else(d_idx < I32(half_d),
                                        tir::BufferLoad(buf, idx_plus) * neg_one,
                                        tir::BufferLoad(buf, idx_minus));

  // Rope frequency  always computed in float32 (matches Python which passes "float32")
  PrimExpr pos_f32 = CastTo(pos_expr, "float32") * rope_scale;
  RopeFreqFunc rope_freq_func = SwitchRopeFreqFunc(rope_scaling);
  auto freq = rope_freq_func(pos_f32, d_idx, d, rope_theta, "float32", rope_scaling);

  // Build rope value
  PrimExpr rope_val;
  if (is_f16) {
    // cos_freq / sin_freq are float32; cast elem/partner to float32, then cast result to f16
    rope_val = CastTo(freq.cos_freq * CastTo(elem, "float32") +
                      freq.sin_freq * CastTo(partner, "float32"), dtype);
  } else {
    rope_val = freq.cos_freq * elem + freq.sin_freq * partner;
  }

  // Wrap in Let bindings (forward order: first var_map entry is innermost Let)
  for (auto it = freq.var_map.begin(); it != freq.var_map.end(); ++it) {
    rope_val = tir::Let(it->first, it->second, rope_val);
  }

  return tvm::if_then_else(rotary_mode == I32(1), rope_val, elem);
}

// ============================================================================
// Paged KV helpers
// ============================================================================

PrimExpr GetKvChunkLen(PrimExpr num_pages, int64_t page_size, PrimExpr seq_id,
                       tir::Buffer length_info, bool sliding_window) {
  if (!sliding_window) {
    return (num_pages - I32(1)) * I32(page_size) +
           tir::BufferLoad(length_info, {seq_id});
  } else {
    return (num_pages - I32(1)) * I32(page_size) +
           tir::BufferLoad(length_info, {I32(0), seq_id}) -
           tir::BufferLoad(length_info, {I32(1), seq_id}) +
           tir::BufferLoad(length_info, {I32(2), seq_id});
  }
}

PrimExpr GetSeqOffset(PrimExpr pos, PrimExpr seq_id, tir::Buffer length_info,
                      bool sliding_window) {
  if (!sliding_window) {
    return pos;
  } else {
    PrimExpr sink_size = tir::BufferLoad(length_info, {I32(2), seq_id});
    PrimExpr sw_offset = tir::BufferLoad(length_info, {I32(1), seq_id});
    return tvm::if_then_else(pos < sink_size, pos, pos - sink_size + sw_offset);
  }
}

tir::Buffer DeclLengthInfo(tir::Var var_length_info, PrimExpr batch_size, bool sliding_window,
                           tir::Var elem_offset_var) {
  ffi::Array<PrimExpr> shape = sliding_window
      ? ffi::Array<PrimExpr>{I32(3), batch_size}
      : ffi::Array<PrimExpr>{batch_size};
  // Use decl_buffer to obtain a PointerType-annotated data var (required by Buffer ctor).
  tir::Var data_var = tir::decl_buffer(shape, DataType::Int(32), "length_info")->data;
  return tir::Buffer(data_var, DataType::Int(32), shape, {}, elem_offset_var,
                     "length_info", 0, 0, tir::kDefault);
}

}  // namespace llm
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
