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

import os
import copy
import tempfile
import numpy as np

from typing import Literal

import tvm
import tvm.testing

from tvm import relax
from tvm.contrib import utils, ndk, dlpack as dl
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.relax.transform.legalize_ops import adreno as legalize_adreno


class SessionManager:
    def __init__(self):
        self.is_remote = SessionManager.is_target_rpc()

    def __enter__(self):
        if self.is_remote:
            self.RPC_TRACKER_HOST = os.getenv("TVM_TRACKER_HOST", "localhost")
            self.RPC_TRACKER_PORT = int(os.getenv("TVM_TRACKER_PORT", 7979))
            self.RPC_DEVICE_KEY = os.getenv("RPC_DEVICE_KEY", "android")

            self.tracker = tvm.rpc.connect_tracker(self.RPC_TRACKER_HOST, self.RPC_TRACKER_PORT)
            self.rpc = self.tracker.request(self.RPC_DEVICE_KEY, priority=0, session_timeout=600)
        else:
            self.rpc = tvm.rpc.LocalSession()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.rpc.get_function("CloseRPCConnection")()

    def load_module(self, ex: relax.VMExecutable):
        with tempfile.TemporaryDirectory() as tempdir:
            file_name = "vm_library.so"
            file_path = os.path.join(tempdir, file_name)
            if self.is_remote:
                ex.export_library(
                    file_path, fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"]
                )
            else:
                ex.export_library(file_path)

            self.rpc.upload(file_path)
            rexec = self.rpc.load_module(file_name)
        return rexec

    def device(self, backend: str):
        return self.rpc.device(backend)

    @staticmethod
    def is_target_rpc():
        """
        Checks if the target is a remote device.

        Returns
        -------
        bool: True if RPC_TARGET is set, False otherwise
        """
        return os.environ.get("RPC_TARGET") == "adreno"


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
    target = tvm.target.Target("llvm")
    ex = relax.build(mod, target)
    if save_lib:
        ex.export_library("mod.so")
    dev = tvm.cpu()
    vm = relax.VirtualMachine(ex, dev)
    inputs = [tvm.runtime.tensor(inp, dev) for inp in inputs]
    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    if isinstance(tvm_output, tuple):
        tvm_output = tuple(out.numpy() for out in tvm_output)
    else:
        tvm_output = (tvm_output.numpy(),)
    return tvm_output


def build_and_run(mod, inputs, backend: Literal["opencl", "vulkan", "llvm"], cfg, opts):
    tgt = tvm.target.adreno(backend=backend, cfg=cfg, options=opts)
    if SessionManager.is_target_rpc():
        tgt = tvm.target.Target(tgt, host="llvm -mtriple=aarch64-linux-gnu")
    else:
        tgt = tvm.target.Target(tgt, host="llvm -march=native")

    relax_pipeline = relax.pipeline.get_default_pipeline(tgt)
    tir_pipeline = tvm.tir.get_default_tir_pipeline(tgt)
    mod = relax_pipeline(mod)

    ex = tvm.compile(mod, tgt, tir_pipeline=tir_pipeline)

    with SessionManager() as sess:
        rexec = sess.load_module(ex)
        dev = sess.device(backend)

        if "vdevice" in mod.global_infos:
            device_arr = [dev for ii in range(len(mod.global_infos["vdevice"]))]
        else:
            device_arr = [dev]

        vm = relax.VirtualMachine(rexec, device_arr)
        inputs = [tvm.runtime.tensor(ip, dev) for ip in inputs]
        vm.set_input("main", *inputs)

        vm.invoke_stateful("main")

        tvm_output = vm.get_outputs("main")
        if isinstance(tvm_output, tuple):
            tvm_output = tuple(out.numpy() for out in tvm_output)
        else:
            tvm_output = (tvm_output.numpy(),)

    return tvm_output


def verify_results(mod, backend, cfg, opts, use_cpu: bool = False):
    if not tvm.testing.adreno_target_exists():
        print("Skipping Eval Tests", flush=True)
        return

    backend = backend.split()[0]
    if backend not in ["opencl", "vulkan"]:
        raise ValueError(f"Unsupported API: {backend}. Must be 'opencl' or 'vulkan'.")

    inputs = []
    for arg in mod["main"].params:
        shape = tuple(shape_val.value for shape_val in arg.struct_info.shape.values)
        inputs.append(np.random.uniform(0, 1, size=shape).astype(arg.struct_info.dtype))

    mod_org, mod_ref = mod, copy.deepcopy(mod)
    rs_org = build_and_run(mod_org, inputs, backend, cfg, opts)
    if use_cpu:
        rs_ref = run_cpu(mod_ref, inputs)
    else:
        rs_ref = build_and_run(mod_ref, inputs, backend, "", "")

    for vl_org, vl_ref in zip(rs_org, rs_ref):
        tvm.testing.assert_allclose(vl_org, vl_ref, rtol=1e-3, atol=1e-3)
