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
# under the License


import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax, tir
from tvm.relax import TensorStructInfo


# TODO ir_module
def build_module(
    N: int,
    C1: int,
    C2: int,
    H: int,
    in_scales,
    in_zero_points,
    out_scale,
    out_zero_point,
    axis: int = -1,
) -> tvm.IRModule:
    """Build a Relax module that performs qnn.concatenate along the given axis."""

    dtype = "int8"
    N_ = tir.IntImm("int64", N)
    C1_ = tir.IntImm("int64", C1)
    C2_ = tir.IntImm("int64", C2)
    H_ = tir.IntImm("int64", H)

    x1 = relax.Var("x1", TensorStructInfo(relax.ShapeExpr([N_, C1_, H_]), dtype))
    x2 = relax.Var("x2", TensorStructInfo(relax.ShapeExpr([N_, C2_, H_]), dtype))

    # Convert Python floats/ints to Relax consts
    s1 = relax.const(float(in_scales[0]), dtype="float32")
    s2 = relax.const(float(in_scales[1]), dtype="float32")
    z1 = relax.const(int(in_zero_points[0]), dtype="int8")
    z2 = relax.const(int(in_zero_points[1]), dtype="int8")

    out_s = relax.const(float(out_scale), dtype="float32")
    out_z = relax.const(int(out_zero_point), dtype="int8")

    def mod_gen():
        bb = relax.BlockBuilder()
        with bb.function("main", params=[x1, x2]):
            with bb.dataflow():
                # Relax concatenate op takes a list/tuple of tensors and an axis
                lv = relax.qnn.op.concat([x1, x2], [s1, s2], [z1, z2], out_s, out_z, axis=1)
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)

        return bb.get()

    def ref_mod_gen():
        bb = relax.BlockBuilder()
        with bb.function("main", params=[x1, x2]):
            with bb.dataflow():
                # Relax concatenate op takes a list/tuple of tensors and an axis
                lv0 = relax.op.dequantize(x1, s1, z1)
                lv1 = relax.op.dequantize(x2, s2, z2)
                lv2 = relax.op.concat([lv0, lv1], axis=axis)
                lv3 = relax.op.quantize(lv2, out_s, out_z)
                gv = bb.emit_output(lv3)
            bb.emit_func_output(gv)

        return bb.get()

    return ref_mod_gen(), mod_gen()


def run_on_cpu(mod: tvm.IRModule, x1_np: np.ndarray, x2_np: np.ndarray) -> np.ndarray:
    """Build and run the Relax module on CPU and return numpy output."""
    target = tvm.target.Target("llvm")
    rt_mod = relax.build(mod, target=target)

    dev = tvm.device("llvm", 0)
    vm = relax.VirtualMachine(rt_mod, dev)

    out = vm["main"](x1_np, x2_np)
    return out.numpy() if hasattr(out, "numpy") else out.asnumpy()


@pytest.mark.parametrize(
    "N, C1, C2, H, in_scales, in_zero_points, out_scale, out_zero_point, axis",
    [
        (1, 2, 1, 3, [0.10, 0.20], [10, 5], 0.25, 0, 1),
        (2, 3, 2, 4, [0.05, 0.30], [0, 15], 0.10, -3, 1),
        (1, 4, 4, 5, [0.125, 0.0625], [12, 7], 0.08, 5, 1),
        (3, 1, 2, 2, [0.2, 0.05], [0, 20], 0.1, 10, 1),
    ],
)
def test_qnn_concatenate(N, C1, C2, H, in_scales, in_zero_points, out_scale, out_zero_point, axis):
    x1_np = np.random.randint(-128, 127, (N, C1, H)).astype("int8")
    x2_np = np.random.randint(-128, 127, (N, C2, H)).astype("int8")

    ref_mod, mod = build_module(
        N,
        C1,
        C2,
        H,
        in_scales=in_scales,
        in_zero_points=in_zero_points,
        out_scale=out_scale,
        out_zero_point=out_zero_point,
        axis=axis,
    )

    ref_out = run_on_cpu(ref_mod, x1_np, x2_np)
    out = run_on_cpu(mod, x1_np, x2_np)
    np.testing.assert_allclose(out, ref_out, atol=1)


if __name__ == "__main__":
    tvm.testing.main()
