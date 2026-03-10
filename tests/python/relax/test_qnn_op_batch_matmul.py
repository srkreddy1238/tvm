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


def build_modules(
    B: int,
    M: int,
    K: int,
    N: int,
    x_scale,
    y_scale,
    x_zero_point,
    y_zero_point,
    out_dtype: str,
):
    """
    Build (ref_mod, qnn_mod) Relax modules for qnn.batch_matmul.

    ref_mod:
        cast(x, int32) -> cast(y, int32)
        -> (x_int32 - x_zero_point) @ (y_int32 - y_zero_point)^T (via matmul)
    qnn_mod:
        qnn.batch_matmul(x, y, x_zp, y_zp, x_scale, y_scale, out_dtype)
    """

    dtype = "int8"
    zp_dtype = "int32"  # integer zero-points for reference
    scale_dtype = "float32"

    B_ = tir.IntImm("int64", B)
    M_ = tir.IntImm("int64", M)
    K_ = tir.IntImm("int64", K)
    N_ = tir.IntImm("int64", N)

    # Relax matmul convention:
    # x: (B, M, K)
    # y: (B, K, N)
    x_shape = [B_, M_, K_]
    y_shape = [B_, K_, N_]

    x = relax.Var("x", TensorStructInfo(relax.ShapeExpr(x_shape), dtype))
    y = relax.Var("y", TensorStructInfo(relax.ShapeExpr(y_shape), dtype))

    # consts
    x_s = relax.const(float(x_scale), dtype=scale_dtype)
    y_s = relax.const(float(y_scale), dtype=scale_dtype)
    x_z = relax.const(int(x_zero_point), dtype=zp_dtype)
    y_z = relax.const(int(y_zero_point), dtype=zp_dtype)

    # Reference module: integer matmul on (x - x_zp) and (y - y_zp)
    def ref_mod_gen():
        bb = relax.BlockBuilder()
        with bb.function("main", params=[x, y]):
            with bb.dataflow():
                # Cast inputs to int32
                x_i32 = relax.op.astype(x, "int32")
                y_i32 = relax.op.astype(y, "int32")

                # Subtract zero points (both sides int32)
                x_centered = relax.op.subtract(x_i32, x_z)
                y_centered = relax.op.subtract(y_i32, y_z)

                # Batched matmul in int32
                mm = relax.op.matmul(x_centered, y_centered, out_dtype="int32")
                gv = bb.emit_output(mm)
            bb.emit_func_output(gv)
        return bb.get()

    # QNN module: qnn.batch_matmul
    def qnn_mod_gen():
        bb = relax.BlockBuilder()
        with bb.function("main", params=[x, y]):
            with bb.dataflow():
                out = relax.qnn.op.batch_matmul(
                    x,
                    y,
                    x_z,
                    y_z,
                    x_s,
                    y_s,
                    out_dtype,
                )
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        return bb.get()

    return ref_mod_gen(), qnn_mod_gen()


def run_on_cpu(mod: tvm.IRModule, x_np: np.ndarray, y_np: np.ndarray) -> np.ndarray:
    """Build and run the Relax module on CPU and return numpy output."""
    target = tvm.target.Target("llvm")
    rt_mod = relax.build(mod, target=target)

    dev = tvm.device("llvm", 0)
    vm = relax.VirtualMachine(rt_mod, dev)

    out = vm["main"](x_np, y_np)
    return out.numpy() if hasattr(out, "numpy") else out.asnumpy()


@pytest.mark.parametrize(
    "B, M, K, N, x_scale, y_scale, x_zero_point, y_zero_point, out_dtype",
    [
        (1, 4, 8, 3, 0.1, 0.2, 0, 0, "int32"),
        (32, 64, 128, 64, 0.05, 0.05, 0, 0, "int32"),
        (1, 256, 128, 256, 0.1, 0.1, 0, 0, "int32"),
        (1, 16, 512, 16, 0.1, 0.1, 0, 0, "int32"),
        (16, 128, 256, 128, 0.05, 0.05, 0, 0, "int32"),
        (8, 64, 128, 64, 0.1, 0.1, 127, -128, "int32"),
        (8, 64, 128, 64, 0.1, 0.1, -128, 127, "int32"),
        (4, 64, 128, 64, 1e-5, 1e-5, 0, 0, "int32"),
        (4, 64, 128, 64, 127.0, 127.0, 0, 0, "int32"),
        (7, 127, 255, 129, 0.1, 0.2, 3, -3, "int32"),
    ],
)
def test_qnn_batch_matmul(
    B,
    M,
    K,
    N,
    x_scale,
    y_scale,
    x_zero_point,
    y_zero_point,
    out_dtype,
):
    x_shape = (B, M, K)
    y_shape = (B, K, N)

    x_np = np.random.randint(-128, 127, size=x_shape).astype("int8")
    y_np = np.random.randint(-128, 127, size=y_shape).astype("int8")

    ref_mod, qnn_mod = build_modules(
        B,
        M,
        K,
        N,
        x_scale,
        y_scale,
        x_zero_point,
        y_zero_point,
        out_dtype,
    )
    ref_out = run_on_cpu(ref_mod, x_np, y_np)
    out = run_on_cpu(qnn_mod, x_np, y_np)

    np.testing.assert_allclose(out, ref_out, atol=1)


if __name__ == "__main__":
    tvm.testing.main()
