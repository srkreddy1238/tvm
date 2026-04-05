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

#include <gtest/gtest.h>
#include <tvm/driver/compile.h>

#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_set>

class CPPCompilerBase {
 public:
  bool TargetSetup(void);

  relax::BlockBuilder BuilderSetup(const ffi::Array<relax::Var>& args);

  tvm::IRModule BuilderFinalize(const relax::BlockBuilder& ctx_, const ffi::Array<relax::Var>& args,
                                const relax::Var& out);

  ffi::Module Compile(tvm::IRModule mod_, tvm::Target& tgt, DLDeviceType dl_dev_type,
                      ffi::String relax_pipeline = "cpu_generic",
                      ffi::String tir_pipeline = "generic",
                      ffi::Map<ffi::Any, ffi::ObjectRef> params = {});

  ffi::Array<runtime::Tensor> VMRun(const ffi::Module& vm, const std::vector<ffi::AnyView>& args,
                                    int ret_count = 1);

  std::vector<runtime::Tensor> InitRandomInputs(const tvm::IRModule& mod, const tvm::Device& dev);

  DLDataType dl_type;
  runtime::DataType dtype;
  DLDeviceType dl_dev_type;
  std::string dev_name;
  tvm::Device device;
  tvm::Target tgt;
  tvm::Target tgt_host;
  ffi::String relax_pipeline;
  ffi::String tir_pipeline;
};
