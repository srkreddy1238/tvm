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
 * \file src/relax/backend/adreno/target_kind.cc
 * \brief Target kind registry for Adreno GPU
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {

TVM_REGISTER_TARGET_KIND("opencl-clml", kDLOpenCL)
    .add_attr_option<int64_t>("max_threads_per_block", 256)
    .add_attr_option<int64_t>("max_shared_memory_per_block", 16384)
    .add_attr_option<int64_t>("max_num_threads", 256)
    .add_attr_option<int64_t>("thread_warp_size", 1)
    .add_attr_option<int64_t>("texture_spatial_limit", 16384)
    .add_attr_option<int64_t>("texture_depth_limit", 2048)
    .add_attr_option<int64_t>("max_function_args", 128)
    .add_attr_option<int64_t>("image_base_address_alignment", 64)
    .add_attr_option<ffi::String>("relax_pipeline", "adreno")
    .add_attr_option<ffi::String>("tir_pipeline", "adreno")
    .add_attr_option<ffi::String>("target_host", "llvm -mtriple=aarch64-linux-gnu")
    .set_default_keys({"adreno", "opencl", "gpu", "clml"});

TVM_REGISTER_TARGET_KIND("adreno-opencl", kDLOpenCL)
    .add_attr_option<int64_t>("max_threads_per_block", 256)
    .add_attr_option<int64_t>("max_shared_memory_per_block", 16384)
    .add_attr_option<int64_t>("max_num_threads", 256)
    .add_attr_option<int64_t>("thread_warp_size", 1)
    .add_attr_option<int64_t>("texture_spatial_limit", 16384)
    .add_attr_option<int64_t>("texture_depth_limit", 2048)
    .add_attr_option<int64_t>("max_function_args", 128)
    .add_attr_option<int64_t>("image_base_address_alignment", 64)
    .add_attr_option<ffi::String>("relax_pipeline", "adreno")
    .add_attr_option<ffi::String>("tir_pipeline", "adreno")
    .add_attr_option<ffi::String>("target_host", "llvm -mtriple=aarch64-linux-gnu")
    .set_default_keys({"adreno", "opencl", "gpu", "texture"});

TVM_REGISTER_TARGET_KIND("adreno-opencl-clml", kDLOpenCL)
    .add_attr_option<int64_t>("max_threads_per_block", 256)
    .add_attr_option<int64_t>("max_shared_memory_per_block", 16384)
    .add_attr_option<int64_t>("max_num_threads", 256)
    .add_attr_option<int64_t>("thread_warp_size", 1)
    .add_attr_option<int64_t>("texture_spatial_limit", 16384)
    .add_attr_option<int64_t>("texture_depth_limit", 2048)
    .add_attr_option<int64_t>("max_function_args", 128)
    .add_attr_option<int64_t>("image_base_address_alignment", 64)
    .add_attr_option<ffi::String>("relax_pipeline", "adreno")
    .add_attr_option<ffi::String>("tir_pipeline", "adreno")
    .add_attr_option<ffi::String>("target_host", "llvm -mtriple=aarch64-linux-gnu")
    .set_default_keys({"adreno", "opencl", "gpu", "clml", "texture"});

TVM_REGISTER_TARGET_KIND("adreno-vulkan", kDLVulkan)
    .add_attr_option<ffi::Array<ffi::String>>("mattr")
    // Feature support
    .add_attr_option<bool>("supports_float16")
    .add_attr_option<bool>("supports_float32", true)
    .add_attr_option<bool>("supports_float64")
    .add_attr_option<bool>("supports_int8")
    .add_attr_option<bool>("supports_int16")
    .add_attr_option<bool>("supports_int32", true)
    .add_attr_option<bool>("supports_int64")
    .add_attr_option<bool>("supports_8bit_buffer")
    .add_attr_option<bool>("supports_16bit_buffer")
    .add_attr_option<bool>("supports_storage_buffer_storage_class")
    .add_attr_option<bool>("supports_push_descriptor")
    .add_attr_option<bool>("supports_dedicated_allocation")
    .add_attr_option<bool>("supports_integer_dot_product")
    .add_attr_option<bool>("supports_cooperative_matrix")
    .add_attr_option<int64_t>("supported_subgroup_operations")
    // Physical device limits
    .add_attr_option<int64_t>("max_num_threads", 256)
    .add_attr_option<int64_t>("max_threads_per_block", 256)
    .add_attr_option<int64_t>("thread_warp_size", 1)
    .add_attr_option<int64_t>("max_block_size_x")
    .add_attr_option<int64_t>("max_block_size_y")
    .add_attr_option<int64_t>("max_block_size_z")
    .add_attr_option<int64_t>("max_push_constants_size")
    .add_attr_option<int64_t>("max_uniform_buffer_range")
    .add_attr_option<int64_t>("max_storage_buffer_range")
    .add_attr_option<int64_t>("max_per_stage_descriptor_storage_buffer")
    .add_attr_option<int64_t>("max_shared_memory_per_block")
    // Other device properties
    .add_attr_option<ffi::String>("device_type")
    .add_attr_option<ffi::String>("device_name")
    .add_attr_option<ffi::String>("driver_name")
    .add_attr_option<int64_t>("driver_version")
    .add_attr_option<int64_t>("vulkan_api_version")
    .add_attr_option<int64_t>("max_spirv_version")
    // Compilation settings
    .add_attr_option<ffi::String>("relax_pipeline", "adreno")
    .add_attr_option<ffi::String>("tir_pipeline", "adreno")
    .add_attr_option<ffi::String>("target_host", "llvm -mtriple=aarch64-linux-gnu")
    // Tags
    .set_default_keys({"adreno", "vulkan", "gpu", "texture"});

}  // namespace tvm
