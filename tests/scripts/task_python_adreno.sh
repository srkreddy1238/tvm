#!/usr/bin/env bash
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

set -euxo pipefail

export TVM_TEST_TARGETS="opencl -device=adreno"

source tests/scripts/setup-pytest-env.sh
export LD_LIBRARY_PATH="build:${LD_LIBRARY_PATH:-}"
export TVM_INTEGRATION_TESTSUITE_NAME=python-integration-rpc-adreno

sudo apt install net-tools -y

# to avoid CI CPU thread throttling.
export TVM_BIND_THREADS=0
export TVM_NUM_THREADS=2

find_free_port () {
    port=$1
    RANGE=$2
    isfree=0
    while [ 0 -eq "$isfree" ]; do
        for ii in `seq 0 $RANGE` ; do
            sport=$((port+ii))
            netstat -taln | grep LISTEN | grep $sport > /dev/null
            isfree=$?
            if [ 0 -eq "$isfree" ] ; then
                port=$((port+1))
                break
            fi
        done
        sleep 1
    done
    echo $port
    return 0
}

export TVM_TRACKER_HOST=127.0.0.1
FREE_PORT=`find_free_port 9000 1`
export TVM_TRACKER_PORT=$FREE_PORT
export RPC_DEVICE_KEY="android"
export ADRENO_TARGET="adreno"
export TVM_NDK_CC="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"

export ANDROID_SERIAL=$1
TARGET_FOLDER=/data/local/tmp/tvm_ci-${USER}-${TVM_TRACKER_PORT}
adb shell "mkdir -p ${TARGET_FOLDER}"
adb push build-adreno-target/tvm_rpc ${TARGET_FOLDER}/tvm_rpc-${USER}-${TVM_TRACKER_PORT}
adb push build-adreno-target/libtvm_runtime.so ${TARGET_FOLDER}
adb push build-adreno-target/lib/libtvm_ffi.so ${TARGET_FOLDER}
CPP_LIB=`find ${ANDROID_NDK_HOME} -name libc++_shared.so | grep aarch64`
if [ -f ${CPP_LIB} ] ; then
    adb push ${CPP_LIB} ${TARGET_FOLDER}
fi

# CPP Tests

if [ -f build-adreno-target/opencl-cpptest ] ; then
  adb push build-adreno-target/opencl-cpptest ${TARGET_FOLDER}
  adb shell "cd ${TARGET_FOLDER};LD_LIBRARY_PATH=${TARGET_FOLDER}/ ./opencl-cpptest"
fi

if [ -f build-adreno-target/vulkan-cpptest ] ; then
  adb push build-adreno-target/vulkan-cpptest ${TARGET_FOLDER}
  adb shell "cd ${TARGET_FOLDER};LD_LIBRARY_PATH=${TARGET_FOLDER}/ ./vulkan-cpptest"
fi

if [ -f build-adreno-compiler/cpp-compiler-test ] ; then
  adb push build-adreno-compiler/libtvm.so ${TARGET_FOLDER}
  adb push build-adreno-compiler/cpp-compiler-test ${TARGET_FOLDER}
  adb shell "cd ${TARGET_FOLDER};LD_LIBRARY_PATH=${TARGET_FOLDER}/ ./cpp-compiler-test"
fi

env PYTHONPATH=python python3 -m tvm.exec.rpc_tracker --host "${TVM_TRACKER_HOST}" --port "${TVM_TRACKER_PORT}" &
TRACKER_PID=$!
sleep 5   # Wait for tracker to bind

adb reverse tcp:${TVM_TRACKER_PORT} tcp:${TVM_TRACKER_PORT}
ADB_PORTS_RANGE=4
RPC_LISTEN_PORT=`find_free_port 6000 ${ADB_PORTS_RANGE}`
export DEVICE_LISTEN_PORT=${RPC_LISTEN_PORT}
for ii in `seq 0 ${ADB_PORTS_RANGE}` ; do
  adb forward tcp:$((RPC_LISTEN_PORT+ii)) tcp:$((RPC_LISTEN_PORT+ii))
done
env adb shell "cd ${TARGET_FOLDER}; killall -9 tvm_rpc-${USER}-${TVM_TRACKER_PORT}; sleep 2; TARGET_RPC_TMP=${TARGET_FOLDER}/rpc_tmp LD_LIBRARY_PATH=${TARGET_FOLDER}/ ./tvm_rpc-${USER}-${TVM_TRACKER_PORT} server --host=0.0.0.0 --port=${RPC_LISTEN_PORT} --port-end=$((RPC_LISTEN_PORT+${ADB_PORTS_RANGE})) --tracker=127.0.0.1:${TVM_TRACKER_PORT} --key=${RPC_DEVICE_KEY}" &
DEVICE_PID=$!
sleep 5 # Wait for the device connections
clean_ports() {
    for ii in `seq 0 ${ADB_PORTS_RANGE}` ; do
        adb forward --remove tcp:$((RPC_LISTEN_PORT+${ii})) || true
    done;
}
trap "{ kill ${TRACKER_PID} || true; kill ${DEVICE_PID} || true; clean_ports; cleanup;}" 0

# cleanup pycache
find . -type f -path "*.pyc" | xargs rm -f
# setup tvm-ffi into python folder
python3 -m pip install --target=python -v ./3rdparty/tvm-ffi/

# Relax test
run_pytest -s tests/python/relax/backend/adreno

kill ${TRACKER_PID} || true
kill ${DEVICE_PID} || true
clean_ports
