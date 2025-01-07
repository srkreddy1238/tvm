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

import pytest
import numpy as np
import tvm
import tvm.testing
import onnx

from tvm import relax
from tvm.script import relax as R
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import relax as relax_builder
from tvm.relax.frontend.onnx import from_onnx

from tvm import relay
from tvm.relay import testing
import pytest
import json
import copy

from tvm import dlight as dl
from tvm.contrib import utils, ndk

from tvm import topi

def customize_legalize_conv2d(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    return bb.call_te(
        #topi.adreno.conv2d_nchw.conv_nchwc_oihwo,
        topi.nn.conv2d_NCHWc_OIHWo,
        data=call.args[0],
        kernel=call.args[1],
        stride=call.attrs.strides,
        padding=call.attrs.padding,
        dilation=call.attrs.dilation,
        layout=call.attrs.data_layout,
        out_layout=call.attrs.out_layout,
        #out_dtype=call.attrs.out_dtype,
        primfunc_name_hint="conv2d_NCHWc_OIHWo",
    )


def customize_legalize_conv2d_legacy(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    return bb.call_te(
        topi.adreno.conv2d_nchw.conv_nchwc_oihwo,
        inp=call.args[0],
        filt=call.args[1],
        stride=call.attrs.strides,
        padding=call.attrs.padding,
        dilation=call.attrs.dilation,
        out_dtype=call.attrs.out_dtype,
        primfunc_name_hint="conv2d_NCHWc",
    )


def build_and_run(mod, inputs_np, target, rpc=None, legalize=False, params_np={}, load_path="vm_library.so", is_adreno=False):
    if legalize:
        mod = relax.transform.LegalizeOps()(mod)

    tgt = tvm.target.Target(target, host="llvm -mtriple=aarch64-linux-gnu")

    with tgt:
      mod = tvm.tir.transform.BindTarget(tvm.target.Target.current(allow_none=False))(mod)
      desired_layouts = {"relax.nn.conv2d": ["NCHW", "OIHW", "NCHW"]}
      mod = tvm.relax.transform.ConvertLayout(desired_layouts)(mod)
      mod = tvm.relax.transform.Normalize()(mod)
      if is_adreno:
          mod = tvm.relax.transform.OptimizeBatchnorm()(mod)
          mod = tvm.relax.transform.FoldConstant()(mod)
          mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
          mod = tvm.relax.transform.FoldConstant()(mod)

          desired_layouts = {"relax.nn.conv2d": ["NCHW4c", "OIHW4o", "NCHW4c"]}
          mod = tvm.relax.transform.ConvertLayout(desired_layouts)(mod)
          mod = tvm.relax.transform.Normalize()(mod)
          mod = tvm.relax.transform.FoldConstant()(mod)
          mod = tvm.relax.transform.Normalize()(mod)

      mod = tvm.relax.transform.LegalizeOps()(mod)
      if is_adreno:
          mod = tvm.relax.transform.LegalizeOps({"relax.nn.conv2d": customize_legalize_conv2d})(mod)
      mod = tvm.relax.transform.AnnotateTIROpPattern()(mod)
      mod = tvm.relax.transform.FoldConstant()(mod)
      mod = tvm.relax.transform.FuseOps()(mod)
      mod = tvm.relax.transform.FuseTIR()(mod)
      mod = tvm.relax.transform.DeadCodeElimination()(mod)

      if is_adreno:
          mod = tvm.relax.transform.AnnotateCustomMemoryScope(tgt)(mod)
   
      mod =  dl.ApplyDefaultSchedule(
          #dl.gpu.Matmul(),
          #dl.gpu.GEMV(),
          dl.adreno.Conv2d(),
          dl.gpu.Reduction(),
          dl.gpu.GeneralReduction(),
          dl.gpu.Fallback(),
      )(mod)

      mod = tvm.relax.transform.ToNonDataflow()(mod)
      mod = tvm.relax.transform.RemovePurityChecking()(mod)
      mod = tvm.relax.transform.CallTIRRewrite()(mod)
      mod = tvm.relax.transform.Normalize()(mod)
      mod = tvm.relax.transform.StaticPlanBlockMemory()(mod)
      mod = tvm.relax.transform.LowerAllocTensor()(mod)
      mod = tvm.relax.transform.KillAfterLastUse()(mod)
      mod = tvm.relax.transform.VMBuiltinLower()(mod)
      mod = tvm.relax.transform.VMShapeLower()(mod)
      mod = tvm.relax.transform.AttachGlobalSymbol()(mod)

    print("Transformed:", mod)
    #exit(0)
    for k, v in mod.functions.items():
      print("K:", k, " - V:", v.params)
      for p in v.params:
        print("param:", type(p))
      print(type(v))
      if isinstance(v, tvm.tir.function.PrimFunc):
        for k1, v1 in v.buffer_map.items():
          print("Map:", type(k1), " - ", v1.scope()) # VarNode # BufferNode
    """
    with tgt:
        seq = tvm.transform.Sequential(
            [
                #tvm.tir.transform.BindTarget(tvm.target.Target.current(allow_none=False)),
                #tvm.relax.transform.LegalizeOps(),
                #tvm.relax.transform.AnnotateTIROpPattern(),
                #tvm.relax.transform.FoldConstant(),
                #tvm.relax.transform.FuseOps(),
                #tvm.relax.transform.FuseTIR(),
                #tvm.relax.transform.DeadCodeElimination(),
                #dl.ApplyDefaultSchedule(
                    #dl.gpu.Matmul(),
                    #dl.gpu.GEMV(),
                #    dl.gpu.Reduction(),
                #    dl.gpu.GeneralReduction(),
                #    dl.gpu.Fallback(),
                #),
                #tvm.relax.transform.RewriteDataflowReshape(),
                tvm.relax.transform.ToNonDataflow(),
                tvm.relax.transform.RemovePurityChecking(),
                tvm.relax.transform.CallTIRRewrite(),
                tvm.relax.transform.StaticPlanBlockMemory(),
                tvm.relax.transform.RewriteCUDAGraph(),
                tvm.relax.transform.LowerAllocTensor(),
                tvm.relax.transform.KillAfterLastUse(),
                tvm.relax.transform.VMBuiltinLower(),
                tvm.relax.transform.VMShapeLower(),
                tvm.relax.transform.AttachGlobalSymbol(),
            ]
        )
        mod = seq(mod)
    """

    print("About to Relax.build:", mod)
    #exit(0)
    if rpc:
        ex = relax.build(mod, tgt)
        #for smod in ex.mod.imported_modules:
        #  print("Mod:", smod.type_key)
        #  print(smod.imported_modules[0].get_source())
        temp = utils.tempdir()
        path = temp.relpath(load_path)
        path = "./" + load_path
        ex.export_library(path, fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"])
        rpc.upload(path)
        rexec = rpc.load_module(load_path)
        dev = rpc.cl(0)
        vm = relax.VirtualMachine(rexec, [dev, dev])
    else:
        ex = relax.build(mod, target)
        dev = tvm.device(target, 0)
        vm = relax.VirtualMachine(ex, dev)

    params_dev = []
    for k, v in params_np.items():
        params_dev.append(tvm.nd.array(v, dev))

    f = vm["main"]
    inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]

    vm.set_input("main", *inputs)

    vm.invoke_stateful("main")

    tvm_output = vm.get_outputs("main")
    return tvm_output.numpy()

import os
from tvm.autotvm.measure import request_remote
def get_rpc():
    rpc_target = os.getenv("RPC_TARGET", None)
    if rpc_target:
        connection_type = "tracker"
        host = os.getenv("TVM_TRACKER_HOST", "localhost")
        port = int(os.getenv("TVM_TRACKER_PORT", 9090))
        target = "opencl"
        target_host = "llvm -mtriple=aarch64-linux-gnu"
        device_key = os.getenv("RPC_DEVICE_KEY", "android")
        cross_compile = os.getenv("TVM_NDK_CC", "aarch64-linux-android-g++")
        return request_remote(device_key, host, port, timeout=1000)
    else:
        return None

@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "url, shape_dict",
    [
      #("mobilenetv2-12.onnx", {"input": [1, 3, 224, 224]}),
      #("densenet-12.onnx", {"data_0": [1, 3, 224, 224]}),
      #("inception-v2-9.onnx", {"data_0": [1, 3, 224, 224]}),
      ("resnet18-v2-7.onnx", {"data": [1, 3, 224, 224]}), 
      #("resnet50-v2-7.onnx", {"data": [1, 3, 224, 224]}),
    ],
)
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_network(url, shape_dict, dtype, target, rpc):
    print("Network evaluating .. " + url + " " + dtype)
    model = onnx.load("./"+url)
    mod = from_onnx(model, shape_dict)
    mod1 = from_onnx(model, shape_dict)
    print("Frontend Mod:", mod)

    inputs = []
    for key, val in shape_dict.items():
        inputs.append(np.random.randint(0, 1, size=val).astype(dtype))

    ref1 = build_and_run(mod1, inputs, "opencl", rpc=rpc, params_np={}, load_path="vm_library_opencl-texture.so", is_adreno=False)
    print("Result:", ref1.shape)
    ref = build_and_run(mod, inputs, "opencl -device=adreno", rpc=rpc, params_np={}, load_path="vm_library_opencl.so", is_adreno=True)
    print("Result:", ref.shape)
    tvm.testing.assert_allclose(ref, ref1, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    tvm.testing.main()
