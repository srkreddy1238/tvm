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


def build_module(
    N: int,
    C: int,
    H: int,
    W: int,
    in_scales,
    in_zero_points,
    out_scale,
    out_zero_point,
    inp_dtype,
    out_dtype,
    axis: int = -1,
) -> tvm.IRModule:
    """Build a Relax module that performs qnn.requantize along the given axis."""

    N_ = tir.IntImm("int64", N)
    C_ = tir.IntImm("int64", C)
    H_ = tir.IntImm("int64", H)
    W_ = tir.IntImm("int64", W)

    x = relax.Var("x", TensorStructInfo(relax.ShapeExpr([N_, C_, H_, W_]), inp_dtype))
    s = relax.const(in_scales, dtype="float32")
    z = relax.const(in_zero_points, dtype="int32")
    out_s = relax.const(float(out_scale), dtype="float32")
    out_z = relax.const(int(out_zero_point), dtype="int32")

    # Relax mod gen
    bb = relax.BlockBuilder()
    with bb.function("main", params=[x]):  # params set here
        with bb.dataflow():
            lv0 = relax.qnn.op.requantize(x, s, z, out_s, out_z, axis=axis, out_dtype=out_dtype)
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)
    return bb.get()


def requantize_reference(
    data: np.ndarray,
    input_scale,
    input_zero_point,
    output_scale: float,
    output_zero_point: int,
    out_dtype: str = "int8",
    axis: int = -1,
) -> np.ndarray:
    """
    Reference implementation of per-axis requantization.

    Formula:
        real_value = (data - input_zero_point) * input_scale
        output = clip(round(real_value / output_scale) + output_zero_point, dtype_min, dtype_max)

    Args:
        data: Input quantized tensor (int8/uint8)
        input_scale: Per-axis input scales (scalar, list, or 1D array matching axis dimension)
        input_zero_point: Per-axis input zero points (scalar, list, or 1D array)
        output_scale: Output scale (scalar)
        output_zero_point: Output zero point (scalar)
        axis: Axis along which per-axis quantization is applied
        out_dtype: Output dtype ("int8", "uint8", "int16", "uint16")

    Returns:
        Requantized output tensor
    """
    # Normalize axis
    if axis < 0:
        axis = axis + data.ndim

    # Convert to numpy arrays (handles scalars, lists, and arrays uniformly)
    input_scale = np.asarray(input_scale, dtype=np.float32)
    input_zero_point = np.asarray(input_zero_point, dtype=np.int32)

    if input_scale.ndim == 0:
        pass
    elif input_scale.ndim == 1:
        # Reshape for broadcasting if needed (per-axis quantization)
        if input_scale.shape[0] != data.shape[axis]:
            raise ValueError(
                f"Per-axis quantization: parameter length ({input_scale.shape[0]}) "
                f"must match data.shape[{axis}] ({data.shape[axis]})"
            )
        # Reshape to (1, ..., C, ..., 1) for broadcasting along axis
        broadcast_shape = [1] * data.ndim
        broadcast_shape[axis] = input_scale.shape[0]
        input_scale = input_scale.reshape(broadcast_shape)
        input_zero_point = input_zero_point.reshape(broadcast_shape)
    else:
        raise ValueError(
            f"input_scale and input_zero_point must be scalars or 1D arrays, "
            f"got ndim {input_scale.ndim}"
        )

    real_value = (data.astype(np.float32) - input_zero_point) * input_scale
    quantized_value = np.round(real_value / output_scale) + output_zero_point

    if out_dtype == "int8":
        dtype_min, dtype_max = -128, 127
        out_np_dtype = np.int8
    elif out_dtype == "uint8":
        dtype_min, dtype_max = 0, 255
        out_np_dtype = np.uint8
    elif out_dtype == "int16":
        dtype_min, dtype_max = -32768, 32767
        out_np_dtype = np.int16
    elif out_dtype == "uint16":
        dtype_min, dtype_max = 0, 65535
        out_np_dtype = np.uint16
    else:
        raise ValueError(f"Unsupported out_dtype: {out_dtype}")

    output = np.clip(quantized_value, dtype_min, dtype_max).astype(out_np_dtype)
    return output


def run_on_cpu(mod: tvm.IRModule, x: np.ndarray) -> np.ndarray:
    """Build and run the Relax module on CPU and return numpy output."""

    target = tvm.target.Target("llvm")
    rt_mod = relax.build(mod, target=target)

    dev = tvm.device("llvm", 0)
    vm = relax.VirtualMachine(rt_mod, dev)

    out = vm["main"](x)
    return out.numpy() if hasattr(out, "numpy") else out.asnumpy()


@pytest.mark.parametrize(
    "N, C, H, W, in_scales, in_zero_points, out_scale, out_zero_point, axis, inp_dtype, out_dtype",
    [
        (64, 2, 256, 32, [0.1039, 0.2075], [10, 5], 0.2599, 0, 1, "int8", "uint8"),
        (32, 3, 512, 16, [0.0512, 0.3099, 0.0975], [0, 12, 15], 0.1068, -3, 1, "int8", "int8"),
        (8, 1, 64, 128, [0.2031], [0], 0.1073, 10, 1, "uint8", "int8"),
        (1, 1, 1, 1, 0.2579, 0, 0.5035, 0, 0, "int8", "int8"),
        (32, 32, 128, 64, 0.0010, -10, 0.0579, 5, 1, "int8", "uint8"),
        (1, 256, 256, 3, [0.0137, 0.2574, 0.0010], [7, -13, 25], 0.0314, 3, -1, "uint8", "uint8"),
        (2, 8, 128, 128, 0.5001, 0, 0.0057, 0, 1, "int8", "int8"),
        (8, 8, 128, 64, 0.1012, -33, 0.2035, 57, 1, "int8", "int8"),
        (4, 8, 128, 128, [0.1235] * 8, [0] * 8, 0.1235, 0, 1, "int8", "uint8"),
        (
            8,
            4,
            64,
            64,
            [0.1079, 0.2046, 0.3032, 0.4099],
            [0, 5, 10, 15],
            0.2600,
            -5,
            1,
            "int8",
            "int8",
        ),
        (
            2,
            3,
            5,
            128,
            [0.0532, 0.1077, 0.2057, 0.4012, 0.8100],
            [0, 1, 2, 3, 4],
            0.1035,
            0,
            2,
            "int8",
            "int8",
        ),
    ],
)
def test_qnn_concatenate(
    N, C, H, W, in_scales, in_zero_points, out_scale, out_zero_point, axis, inp_dtype, out_dtype
):

    # 4D input: (N, C, H, W)
    x = np.random.randint(0, 255, (N, C, H, W)).astype(inp_dtype)
    mod = build_module(
        N,
        C,
        H,
        W,
        in_scales=in_scales,
        in_zero_points=in_zero_points,
        out_scale=out_scale,
        out_zero_point=out_zero_point,
        inp_dtype=inp_dtype,
        out_dtype=out_dtype,
        axis=axis,
    )
    out = run_on_cpu(mod, x)
    ref_out = requantize_reference(
        x, in_scales, in_zero_points, out_scale, out_zero_point, out_dtype, axis
    )
    np.testing.assert_allclose(out, ref_out, atol=1)


if __name__ == "__main__":
    tvm.testing.main()
