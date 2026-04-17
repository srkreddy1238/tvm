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
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.relax import TensorStructInfo


def get_conv2d_mod(
    data_shape,
    weight_shape,
    data_layout,
    weight_layout,
    out_layout,
    strides,
    padding,
    dilation,
    groups,
    dtype,
):
    data = relax.Var("data", TensorStructInfo(shape=data_shape, dtype=dtype))
    weight = relax.Var("weight", TensorStructInfo(shape=weight_shape, dtype=dtype))

    bb = relax.BlockBuilder()
    with bb.function("main", [data, weight]):
        with bb.dataflow():
            out = bb.emit(
                relax.op.nn.conv2d(
                    data,
                    weight,
                    strides=strides,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    data_layout=data_layout,
                    kernel_layout=weight_layout,
                    out_layout=out_layout,
                )
            )
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.finalize()


def get_conv2d_transpose_mod(
    data_shape,
    weight_shape,
    data_layout,
    weight_layout,
    out_layout,
    strides,
    padding,
    output_padding,
    dilation,
    groups,
    dtype,
):
    data = relax.Var("data", TensorStructInfo(shape=data_shape, dtype=dtype))
    weight = relax.Var("weight", TensorStructInfo(shape=weight_shape, dtype=dtype))

    bb = relax.BlockBuilder()
    with bb.function("main", [data, weight]):
        with bb.dataflow():
            out = bb.emit(
                relax.op.nn.conv2d_transpose(
                    data,
                    weight,
                    strides=strides,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    groups=groups,
                    data_layout=data_layout,
                    kernel_layout=weight_layout,
                    out_layout=out_layout,
                )
            )
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.finalize()


def run_cpu(ex, inputs):
    dev = tvm.cpu()
    inputs = [tvm.runtime.tensor(inp, dev) for inp in inputs]
    vm = relax.VirtualMachine(ex, dev)
    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    outputs = vm.get_outputs("main")
    if not isinstance(outputs, list):
        outputs = [outputs]
    return [out.numpy() for out in outputs]


@pytest.mark.parametrize(
    "data_shape,weight_shape,data_layout,weight_layout,out_layout,strides,padding,dilation,groups",
    [
        (
            (1, 3, 224, 224),
            (32, 3, 3, 3),
            "NCHW",
            "OIHW",
            "NCHW",
            (1, 1),
            (1, 1, 1, 1),
            (1, 1),
            1,
        ),
        (
            (1, 16, 64, 64),
            (32, 16, 3, 3),
            "NCHW",
            "OIHW",
            "NCHW",
            (2, 2),
            (0, 0, 0, 0),
            (1, 1),
            1,
        ),
        (
            (1, 8, 32, 32),
            (16, 8, 3, 3),
            "NCHW",
            "OIHW",
            "NCHW",
            (1, 1),
            (2, 2, 2, 2),
            (2, 2),
            1,
        ),
        (
            (1, 16, 32, 32),
            (16, 1, 3, 3),
            "NCHW",
            "OIHW",
            "NCHW",
            (1, 1),
            (1, 1, 1, 1),
            (1, 1),
            16,
        ),
        (
            (1, 224, 224, 3),
            (32, 3, 3, 3),
            "NHWC",
            "OHWI",
            "NHWC",
            (1, 1),
            (1, 1, 1, 1),
            (1, 1),
            1,
        ),
        (
            (1, 64, 64, 16),
            (32, 3, 3, 16),
            "NHWC",
            "OHWI",
            "NHWC",
            (2, 2),
            (0, 0, 0, 0),
            (1, 1),
            1,
        ),
        (
            (1, 32, 32, 8),
            (16, 3, 3, 8),
            "NHWC",
            "OHWI",
            "NHWC",
            (1, 1),
            (2, 2, 2, 2),
            (2, 2),
            1,
        ),
        (
            (1, 32, 32, 16),
            (16, 3, 3, 1),
            "NHWC",
            "OHWI",
            "NHWC",
            (1, 1),
            (1, 1, 1, 1),
            (1, 1),
            16,
        ),
    ],
)
def test_conv2d_layout(
    data_shape,
    weight_shape,
    data_layout,
    weight_layout,
    out_layout,
    strides,
    padding,
    dilation,
    groups,
):
    dtype = "float32"

    mod = get_conv2d_mod(
        data_shape,
        weight_shape,
        data_layout,
        weight_layout,
        out_layout,
        strides,
        padding,
        dilation,
        groups,
        dtype,
    )

    data_np = np.random.uniform(-1, 1, size=data_shape).astype(dtype)
    weight_np = np.random.uniform(-1, 1, size=weight_shape).astype(dtype)

    ex = relax.build(mod, "llvm")
    (result,) = run_cpu(ex, [data_np, weight_np])

    in_h = data_shape[data_layout.index("H")]
    in_w = data_shape[data_layout.index("W")]
    kh = weight_shape[weight_layout.index("H")]
    kw = weight_shape[weight_layout.index("W")]
    pad_h = padding[0] + padding[2]
    pad_w = padding[1] + padding[3]
    exp_h = (in_h + pad_h - dilation[0] * (kh - 1) - 1) // strides[0] + 1
    exp_w = (in_w + pad_w - dilation[1] * (kw - 1) - 1) // strides[1] + 1
    exp_c = weight_shape[weight_layout.index("O")]

    assert result.shape[out_layout.index("H")] == exp_h, (
        f"H mismatch: expected {exp_h}, got {result.shape[out_layout.index('H')]}"
    )
    assert result.shape[out_layout.index("W")] == exp_w, (
        f"W mismatch: expected {exp_w}, got {result.shape[out_layout.index('W')]}"
    )
    assert result.shape[out_layout.index("C")] == exp_c, (
        f"C mismatch: expected {exp_c}, got {result.shape[out_layout.index('C')]}"
    )
    assert result.dtype == dtype


@pytest.mark.parametrize(
    "data_shape,weight_shape,data_layout,weight_layout,out_layout,"
    "strides,padding,output_padding,dilation,groups",
    [
        # 1. NCHW stride=2, same padding, output_padding=1 — standard upsampling
        (
            (1, 32, 112, 112),
            (32, 16, 3, 3),
            "NCHW",
            "IOHW",
            "NCHW",
            (2, 2),
            (1, 1, 1, 1),
            (1, 1),
            (1, 1),
            1,
        ),
        # 2. NCHW stride=1, no padding — trivial transpose (output == input spatial)
        (
            (1, 8, 16, 16),
            (8, 4, 3, 3),
            "NCHW",
            "IOHW",
            "NCHW",
            (1, 1),
            (0, 0, 0, 0),
            (0, 0),
            (1, 1),
            1,
        ),
        # 3. NCHW stride=2, no padding — full upsampling without padding removal
        (
            (1, 16, 28, 28),
            (16, 8, 4, 4),
            "NCHW",
            "IOHW",
            "NCHW",
            (2, 2),
            (0, 0, 0, 0),
            (0, 0),
            (1, 1),
            1,
        ),
    ],
)
def test_conv2d_transpose_layout(
    data_shape,
    weight_shape,
    data_layout,
    weight_layout,
    out_layout,
    strides,
    padding,
    output_padding,
    dilation,
    groups,
):
    dtype = "float32"

    mod = get_conv2d_transpose_mod(
        data_shape,
        weight_shape,
        data_layout,
        weight_layout,
        out_layout,
        strides,
        padding,
        output_padding,
        dilation,
        groups,
        dtype,
    )

    data_np = np.random.uniform(-1, 1, size=data_shape).astype(dtype)
    weight_np = np.random.uniform(-1, 1, size=weight_shape).astype(dtype)

    ex = relax.build(mod, "llvm")
    (result,) = run_cpu(ex, [data_np, weight_np])

    # Compute expected spatial dims independently
    in_h = data_shape[data_layout.index("H")]
    in_w = data_shape[data_layout.index("W")]
    kh = weight_shape[weight_layout.index("H")]
    kw = weight_shape[weight_layout.index("W")]
    pad_h = padding[0] + padding[2]
    pad_w = padding[1] + padding[3]
    exp_h = (in_h - 1) * strides[0] + dilation[0] * (kh - 1) + 1 - pad_h + output_padding[0]
    exp_w = (in_w - 1) * strides[1] + dilation[1] * (kw - 1) + 1 - pad_w + output_padding[1]

    assert result.shape[out_layout.index("H")] == exp_h, (
        f"H mismatch: expected {exp_h}, got {result.shape[out_layout.index('H')]}"
    )
    assert result.shape[out_layout.index("W")] == exp_w, (
        f"W mismatch: expected {exp_w}, got {result.shape[out_layout.index('W')]}"
    )
    assert result.dtype == dtype


if __name__ == "__main__":
    tvm.testing.main()
