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
 * \file src/relax/backend/adreno/target_tag.cc
 * \brief Target tags for Adreno GPU
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/tag.h>

namespace tvm {

TVM_REGISTER_TARGET_TAG("qcom/adreno-opencl")
    .set_config(ffi::Map<ffi::String, Any>({{"kind", "opencl"},
                                            {"device", "adreno"},
                                            {"relax_pipeline", "adreno"},
                                            {"tir_pipeline", "adreno"},
                                            {"keys", ffi::Array<ffi::String>({"adreno", "opencl",
                                                                              "gpu"})}}));

TVM_REGISTER_TARGET_TAG("qcom/adreno-opencl-clml")
    .set_config(ffi::Map<ffi::String, Any>({{"kind", "opencl"},
                                            {"device", "adreno"},
                                            {"relax_pipeline", "adreno"},
                                            {"tir_pipeline", "adreno"},
                                            {"keys", ffi::Array<ffi::String>({"adreno", "opencl",
                                                                              "gpu", "clml"})}}));

TVM_REGISTER_TARGET_TAG("qcom/adreno-opencl-texture")
    .set_config(ffi::Map<ffi::String, Any>(
        {{"kind", "opencl"},
         {"device", "adreno"},
         {"relax_pipeline", "adreno"},
         {"tir_pipeline", "adreno"},
         {"keys", ffi::Array<ffi::String>({"adreno", "opencl", "gpu", "texture"})}}));

TVM_REGISTER_TARGET_TAG("qcom/adreno-opencl-texture-clml")
    .set_config(ffi::Map<ffi::String, Any>(
        {{"kind", "opencl"},
         {"device", "adreno"},
         {"relax_pipeline", "adreno"},
         {"tir_pipeline", "adreno"},
         {"keys", ffi::Array<ffi::String>({"adreno", "opencl", "gpu", "texture", "clml"})}}));

TVM_REGISTER_TARGET_TAG("qcom/adreno-vulkan")
    .set_config(ffi::Map<ffi::String, Any>({{"kind", "vulkan"},
                                            {"device", "adreno"},
                                            {"relax_pipeline", "adreno"},
                                            {"tir_pipeline", "adreno"},
                                            {"keys", ffi::Array<ffi::String>({"adreno", "vulkan",
                                                                              "gpu"})}}));

TVM_REGISTER_TARGET_TAG("qcom/adreno-vulkan-texture")
    .set_config(ffi::Map<ffi::String, Any>(
        {{"kind", "vulkan"},
         {"device", "adreno"},
         {"relax_pipeline", "adreno"},
         {"tir_pipeline", "adreno"},
         {"keys", ffi::Array<ffi::String>({"adreno", "vulkan", "gpu", "texture"})}}));
}  // namespace tvm
