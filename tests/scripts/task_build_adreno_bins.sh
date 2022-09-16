#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -e
set -u

output_directory=$(realpath ${PWD}/build-adreno-target)
rm -rf ${output_directory}

mkdir -p ${output_directory}
cd ${output_directory}

cp ../cmake/config.cmake .

echo set\(USE_CLML ON\) >> config.cmake
echo set\(USE_RPC ON\) >> config.cmake
echo set\(USE_CLML_GRAPH_EXECUTOR ${ADRENO_OPENCL}\) >> config.cmake
echo set\(USE_LIBBACKTRACE AUTO\) >> config.cmake
echo set\(CMAKE_TOOLCHAIN_FILE "${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake"\) >> config.cmake
echo set\(ANDROID_ABI arm64-v8a\) >> config.cmake
echo set\(ANDROID_PLATFORM android-28\) >> config.cmake
echo set\(OS Linux\) >> config.cmake
echo set\(CMAKE_CXX_COMPILER "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++"\) >> config.cmake

cmake ..

make -j$(nproc) tvm_rpc
