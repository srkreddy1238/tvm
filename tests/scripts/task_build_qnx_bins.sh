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
set -x

output_directory=$(realpath ${PWD}/build-qnx-target)
rm -rf ${output_directory}

mkdir -p ${output_directory}
cd ${output_directory}

cp ../cmake/config.cmake .

#if [ -f "${ADRENO_OPENCL}/CL/cl_qcom_ml_ops.h" ] ; then
#echo set\(USE_CLML "${ADRENO_OPENCL}"\) >> config.cmake
#echo set\(USE_CLML_GRAPH_EXECUTOR "${ADRENO_OPENCL}"\) >> config.cmake
#fi
#if [ -f "${ADRENO_OPENCL}/CL/cl.h" ] ; then
#echo set\(USE_OPENCL "${ADRENO_OPENCL}"\) >> config.cmake
#else
#echo set\(USE_OPENCL ON\) >> config.cmake
#fi
echo set\(USE_RPC ON\) >> config.cmake
echo set\(USE_CPP_RPC ON\) >> config.cmake
echo set\(USE_CPP_RTVM ON\) >> config.cmake
echo set\(USE_GRAPH_EXECUTOR ON\) >> config.cmake
echo set\(USE_LIBBACKTRACE OFF\) >> config.cmake
echo set\(USE_KALLOC_ALIGNMENT 16\) >> config.cmake

#echo set\(USE_OPENCL_EXTN_QCOM ON\) >> config.cmake
# Enable for OpenCL Shader dumps
#echo set\(PROFILE_SHADER_DUMP ON\) >> config.cmake


echo add_definitions\(-D_XOPEN_SOURCE=700\) >> config.cmake
echo add_definitions\(-D__USE_GNU\) >> config.cmake

export DOTNET_ROOT=""
export CRM_BUILDID=""
cd $QNX_BASE/qnx_ap
source setenv_qos222.sh -np 8 -qp -ex $QNX_BASE/qnx_bins/prebuilt_QOS222 || true
cd -

$CONDA_PREFIX/bin/cmake -DCMAKE_INSTALL_PREFIX=install \
                        -DCMAKE_TOOLCHAIN_FILE="$QNX_BASE/qnx_ap/compute/qml/cmake/QNXToolchain.cmake" \
                        -DBUILD_SHARED_LIBS=1 \
                        -DCMAKE_BUILD_TYPE=RELWITHDEBINF ..

make -j$(nproc) rtvm
