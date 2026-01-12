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
# ruff: noqa: F401

import os

import numpy as np
import tvm
import tempfile

import tvm.testing
from tvm import relax
from tvm.contrib import dlpack as dl
from tvm.contrib import ndk, utils
from tvm.relax.transform.legalize_ops import adreno as legalize_adreno
from tvm.rpc import connect_tracker
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T


class RemoteConnection:
    """
    RPC class handling creation/destruction of remote connection.
    """

    def __init__(self):
        self.RPC_TRACKER_HOST = os.getenv("TVM_TRACKER_HOST", "localhost")
        self.RPC_TRACKER_PORT = int(os.getenv("TVM_TRACKER_PORT", 7979))
        self.RPC_KEY = os.getenv("RPC_DEVICE_KEY", "android")

    def __enter__(self):
        self.tracker = tvm.rpc.connect_tracker(self.RPC_TRACKER_HOST, self.RPC_TRACKER_PORT)
        self.remote = self.tracker.request(self.RPC_KEY, priority=0, session_timeout=600)
        return self.remote

    def __exit__(self, exc_type, exc_value, traceback):
        self.remote.get_function("CloseRPCConnection")()


def is_target_rpc():
    """
    Checks the system env if the target is a remote device

    Returns
    -------
    bool: True if RPC_TARGET is set, False otherwise
    """
    return "RPC_TARGET" in os.environ


def get_target(backend, is_adreno=False):
    """
    Get the target for the Adreno GPU.

    Returns
    -------
    tvm.target.Target
        The target for the Adreno GPU.
    """
    _TAG_MAP = {
        ("opencl", False): "qcom/adreno-opencl",
        ("opencl", True): "qcom/adreno-opencl-texture",
        ("vulkan", False): "qcom/adreno-vulkan",
        ("vulkan", True): "qcom/adreno-vulkan-texture",
    }
    return tvm.target.Target(_TAG_MAP[(backend, is_adreno)])


def get_unique_dso_lib():
    """
    Generate a unique shared library filename based on environment variables.

    Returns
    -------
    str
        The unique shared library filename.
    """
    rpc_tracker_port = os.getenv("TVM_TRACKER_PORT", "")
    device_port = os.getenv("DEVICE_LISTEN_PORT", "")
    return f"dev_lib_cl-{rpc_tracker_port}-{device_port}.so"


def run_cpu(mod, inputs, save_lib=False):
    """
    Run the Relax module on the local CPU for verification.

    Parameters
    ----------
    mod : tvm.IRModule
        The Relax IRModule to execute.
    inputs : list of numpy.ndarray
        The input data for the module.
    save_lib : bool, optional
        Whether to save the compiled library. Default is False.

    Returns
    -------
    tvm.runtime.NDArray or tuple of tvm.runtime.NDArray
        The output from the module execution.
    """
    print("Running on local CPU for verification")
    target = tvm.target.Target("llvm")
    ex = relax.build(mod, target)
    if save_lib:
        ex.export_library("mod.so")
    dev = tvm.cpu()
    vm = relax.VirtualMachine(ex, dev)
    inputs = [tvm.nd.array(inp, dev) for inp in inputs]
    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    return tvm_output


def build_run(mod, inputs, backend, is_adreno=False):
    is_rpc = is_target_rpc()
    target = get_target(backend, is_adreno)
    if remote is None:
        tgt = tvm.target.Target(target, host="llvm")
    else:
        tgt = tvm.target.Target(target, host={"kind": "llvm", "mtriple": "aarch64-linux-gnu"})
    relax_pipeline = relax.pipeline.get_default_pipeline(tgt)
    tir_pipeline = tvm.tir.get_default_tir_pipeline(tgt)
    mod = relax_pipeline(mod)
    ex = tvm.compile(mod, tgt, tir_pipeline=tir_pipeline)

    if is_rpc:
        with RemoteConnection() as remote:
            with tempfile.TemporaryDirectory() as temp_dir:
                filename = get_unique_dso_lib()
                file_path = os.path.join(temp_dir, filename)
                ex.export_library(
                    file_path, fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"]
                )

                remote.upload(file_path)
                rexec = remote.load_module(filename)
                dev = remote.device(str(target.kind))

                if "vdevice" in mod.global_infos:
                    device_arr = [dev for ii in range(len(mod.global_infos["vdevice"]))]
                else:
                    device_arr = [dev]

                vm = relax.VirtualMachine(rexec, device_arr)

                inputs = [tvm.runtime.tensor(inp, dev) for inp in inputs]
                vm.set_input("main", *inputs)
                vm.invoke_stateful("main")
                tvm_output = vm.get_outputs("main")

                if isinstance(tvm_output, tuple):
                    tvm_output = (out.numpy() for out in tvm_output)
                else:
                    tvm_output = tvm_output.numpy()
    else:
        dev = tvm.device(str(target.kind))
        if "vdevice" in mod.global_infos:
            device_arr = [dev for ii in range(len(mod.global_infos["vdevice"]))]
        else:
            device_arr = [dev]
        vm = relax.VirtualMachine(ex, device_arr)

        inputs = [tvm.runtime.tensor(inp, dev) for inp in inputs]
        vm.set_input("main", *inputs)
        vm.invoke_stateful("main")
        tvm_output = vm.get_outputs("main")

        if isinstance(tvm_output, tuple):
            tvm_output = (out.numpy() for out in tvm_output)
        else:
            tvm_output = tvm_output.numpy()

    return tvm_output


def verify(mod, backend):
    if backend not in ["opencl", "vulkan"]:
        raise ValueError(f"Unsupported API: {backend}. Must be 'opencl' or 'vulkan'.")

    inputs = []
    for arg in mod["main"].params:
        shape = tuple(shape_val.value for shape_val in arg.struct_info.shape.values)
        inputs.append(np.random.uniform(0, 1, size=shape).astype(arg.struct_info.dtype))

    ret1 = build_run(mod, inputs, backend, True)
    ret2 = build_run(mod, inputs, backend)

    if isinstance(ret1, tuple):
        for val1, val2 in zip(ret1, ret2):
            tvm.testing.assert_allclose(val1, ret2, rtol=1e-3, atol=1e-3)
    else:
        tvm.testing.assert_allclose(ret1, ret2, rtol=1e-3, atol=1e-3)
