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
from tvm.relax import ShapeExpr, TensorStructInfo


# ---------------------------------------------------------------------------
# Reference implementation: manual dequant + conv2d + scale multiply
# ---------------------------------------------------------------------------
def get_ref_impl(
    data,
    weight,
    data_zero_point,
    weight_zero_point,
    data_scale,
    weight_scale,
    data_layout,
    weight_layout,
    groups,
    strides,
    padding,
    dilation,
    dtype,
    zp_dtype,
    scale_dtype,
    out_dtype,
    out_layout,
):
    data, weight = (
        relax.Var("data", TensorStructInfo(shape=data, dtype=dtype)),
        relax.Var("weight", TensorStructInfo(shape=weight, dtype=dtype)),
    )
    data_zero_point, weight_zero_point = (
        relax.const(data_zero_point, dtype=zp_dtype),
        relax.const(weight_zero_point, dtype=zp_dtype),
    )

    assert (data_scale is None and weight_scale is None) or (
        data_scale is not None and weight_scale is not None
    ), "Both must be None or a Constant"
    has_scale = data_scale is not None

    if has_scale:
        if isinstance(weight_scale, float):
            scale_reshape = [1, 1, 1, 1]
        else:
            out_idx = tvm.s_tir.layout(out_layout).index_of("C")
            scale_reshape = [1, 1, 1]
            scale_reshape.insert(out_idx, weight_scale.shape[0])
        scale_reshape = ShapeExpr(scale_reshape)
        data_scale, weight_scale = (
            relax.const(data_scale, dtype=scale_dtype),
            relax.const(weight_scale, dtype=scale_dtype),
        )

    bb = relax.BlockBuilder()
    with bb.function("main", [data, weight]):
        with bb.dataflow():
            data_cast = bb.emit(relax.op.datatype.astype(data, zp_dtype))
            weight_cast = bb.emit(relax.op.datatype.astype(weight, zp_dtype))
            data_dq = bb.emit(relax.op.subtract(data_cast, data_zero_point))
            weight_dq = bb.emit(relax.op.subtract(weight_cast, weight_zero_point))
            conv_op = relax.op.nn.conv2d(
                data_dq,
                weight_dq,
                strides,
                padding,
                dilation,
                groups,
                data_layout,
                weight_layout,
                out_layout,
                out_dtype,
            )
            if has_scale:
                scale = bb.emit(relax.op.multiply(data_scale, weight_scale))
                scale = bb.emit(relax.op.reshape(scale, scale_reshape))
                conv = bb.emit(conv_op)
                out = bb.emit_output(relax.op.multiply(conv, scale))
            else:
                out = bb.emit_output(conv_op)
        bb.emit_func_output(out)
    return bb.finalize()


# ---------------------------------------------------------------------------
# QNN implementation: conv2d (no scales) + requantize
# ---------------------------------------------------------------------------
def get_qnn_impl(
    data,
    weight,
    data_zero_point,
    weight_zero_point,
    data_scale,
    weight_scale,
    data_layout,
    weight_layout,
    groups,
    strides,
    padding,
    dilation,
    dtype,
    zp_dtype,
    scale_dtype,
    out_dtype,
    out_layout,
    output_scale=None,
    output_zero_point=0,
    requant_out_dtype="int8",
):
    data, weight = (
        relax.Var("data", TensorStructInfo(shape=data, dtype=dtype)),
        relax.Var("weight", TensorStructInfo(shape=weight, dtype=dtype)),
    )
    data_zero_point, weight_zero_point = (
        relax.const(data_zero_point, dtype=zp_dtype),
        relax.const(weight_zero_point, dtype=zp_dtype),
    )

    assert (data_scale is None and weight_scale is None) or (
        data_scale is not None and weight_scale is not None
    ), "Both must be None or a Constant"
    has_scale = data_scale is not None
    raw_data_scale = data_scale
    raw_weight_scale = weight_scale

    bb = relax.BlockBuilder()
    with bb.function("main", [data, weight]):
        with bb.dataflow():
            conv_out = bb.emit(
                relax.qnn.op.conv2d(
                    data,
                    weight,
                    data_zero_point,
                    weight_zero_point,
                    None,
                    None,
                    strides,
                    padding,
                    dilation,
                    groups,
                    data_layout,
                    weight_layout,
                    out_layout,
                    zp_dtype,
                )
            )

            if has_scale:
                assert output_scale is not None, (
                    "output_scale must be provided when data_scale is set"
                )
                if isinstance(raw_weight_scale, np.ndarray):
                    new_input_scale_val = raw_data_scale * raw_weight_scale
                else:
                    new_input_scale_val = float(raw_data_scale) * float(raw_weight_scale)

                oc_axis = tvm.s_tir.layout(out_layout).index_of("C")
                out = bb.emit_output(
                    relax.qnn.op.requantize(
                        conv_out,
                        input_scale=relax.const(new_input_scale_val, dtype=scale_dtype),
                        input_zero_point=relax.const(0, dtype=zp_dtype),
                        output_scale=relax.const(output_scale, dtype=scale_dtype),
                        output_zero_point=relax.const(output_zero_point, dtype=zp_dtype),
                        axis=oc_axis,
                        out_dtype=requant_out_dtype,
                    )
                )
            else:
                out = bb.emit_output(conv_out)

        bb.emit_func_output(out)
    return bb.finalize()


# ---------------------------------------------------------------------------
# CPU runner
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helper: build both mods, run, compare
# ---------------------------------------------------------------------------
def _run_and_compare(common_kwargs, output_s, output_zp, rng, data_shape, weight_shape):
    has_scale = common_kwargs["data_scale"] is not None

    ref_mod = get_ref_impl(**common_kwargs)
    qnn_mod = get_qnn_impl(
        **common_kwargs,
        output_scale=output_s,
        output_zero_point=output_zp,
        requant_out_dtype="int8",
    )

    data_np = rng.integers(-128, 127, size=data_shape, dtype=np.int8)
    weight_np = rng.integers(-128, 127, size=weight_shape, dtype=np.int8)
    inputs = [data_np, weight_np]

    ref_ex = relax.build(ref_mod, "llvm")
    qnn_ex = relax.build(qnn_mod, "llvm")

    ref_outputs = run_cpu(ref_ex, inputs)
    qnn_outputs = run_cpu(qnn_ex, inputs)

    assert len(ref_outputs) == len(qnn_outputs)
    for ref, res in zip(ref_outputs, qnn_outputs):
        if has_scale:
            ref_int8 = np.clip(
                np.round(ref / output_s).astype(np.int32) + output_zp,
                -128,
                127,
            ).astype(np.int8)
            np.testing.assert_array_equal(
                ref_int8,
                res,
                err_msg=(
                    f"Mismatch: data={data_shape}, weight={weight_shape}, output_zp={output_zp}"
                ),
            )
        else:
            np.testing.assert_array_equal(
                ref,
                res,
                err_msg=f"Mismatch: data={data_shape}, weight={weight_shape}",
            )


# ---------------------------------------------------------------------------
# Parametrize: layouts, shapes, scale type
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "data_shape, data_layout, weight_shape, weight_layout, out_layout",
    [
        # NCHW — standard
        ((1, 64, 28, 28), "NCHW", (32, 64, 3, 3), "OIHW", "NCHW"),
        # NCHW — grouped conv (groups=4)
        ((1, 64, 28, 28), "NCHW", (32, 16, 3, 3), "OIHW", "NCHW"),
        # NCHW — depthwise (groups = in_channels, out = in)
        ((1, 32, 28, 28), "NCHW", (32, 1, 3, 3), "OIHW", "NCHW"),
        # NHWC — standard
        ((1, 28, 28, 64), "NHWC", (32, 64, 3, 3), "OIHW", "NHWC"),
        # NHWC — grouped conv
        ((1, 28, 28, 64), "NHWC", (32, 16, 3, 3), "OIHW", "NHWC"),
        # NHWC — depthwise
        ((1, 28, 28, 32), "NHWC", (32, 1, 3, 3), "OIHW", "NHWC"),
        # 1x1 kernel
        ((1, 64, 28, 28), "NCHW", (32, 64, 1, 1), "OIHW", "NCHW"),
        # non-square input
        ((1, 64, 14, 28), "NCHW", (32, 64, 3, 3), "OIHW", "NCHW"),
    ],
)
@pytest.mark.parametrize(
    "strides, padding, dilation",
    [
        ((1, 1), (0, 0, 0, 0), (1, 1)),  # no padding, no stride, no dilation
        ((2, 2), (1, 1, 1, 1), (1, 1)),  # stride=2, same padding
        ((1, 1), (1, 1, 1, 1), (2, 2)),  # dilation=2
        ((1, 1), (0, 0, 0, 0), (1, 1)),  # valid padding
        ((2, 2), (0, 0, 0, 0), (1, 1)),  # stride=2, no padding
    ],
)
@pytest.mark.parametrize("has_scale", [True, False])
@pytest.mark.parametrize("is_weight_scalar", [True, False])
@pytest.mark.parametrize(
    "data_zp_val, weight_zp_val",
    [
        (0, 0),  # both zero points = 0
        (128, 0),  # data zp non-zero, weight zp = 0
        (0, 128),  # data zp = 0, weight zp non-zero
        (64, 32),  # both non-zero
        (-128, 127),  # boundary values
    ],
)
def test_qnn_conv2d(
    data_shape,
    data_layout,
    weight_shape,
    weight_layout,
    out_layout,
    strides,
    padding,
    dilation,
    has_scale,
    is_weight_scalar,
    data_zp_val,
    weight_zp_val,
):
    dtype, zp_dtype, scale_dtype = "int8", "int32", "float32"
    rng = np.random.default_rng(seed=42)

    # derive out_channel from weight shape (always OIHW)
    out_channel = weight_shape[0]
    input_channel = data_shape[data_layout.index("C")]
    kernel_channel = weight_shape[weight_layout.index("I")]

    if input_channel % kernel_channel != 0:
        pytest.skip(
            f"input_channel={input_channel} not divisible by kernel_channel={kernel_channel}"
        )
    groups = input_channel // kernel_channel

    if has_scale:
        data_scale = float(rng.uniform(0.01, 1.0))
        weight_scale = (
            float(rng.uniform(0.01, 1.0))
            if is_weight_scalar
            else rng.uniform(0.01, 1.0, size=(out_channel,)).astype(np.float32)
        )
        output_s = float(rng.uniform(0.001, 0.1))
        output_zp = int(rng.integers(-10, 10))
        out_dtype = scale_dtype
    else:
        data_scale = weight_scale = None
        output_s = None
        output_zp = 0
        out_dtype = zp_dtype

    common_kwargs = dict(
        data=data_shape,
        weight=weight_shape,
        data_zero_point=data_zp_val,
        weight_zero_point=weight_zp_val,
        data_scale=data_scale,
        weight_scale=weight_scale,
        data_layout=data_layout,
        weight_layout=weight_layout,
        groups=groups,
        strides=strides,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        zp_dtype=zp_dtype,
        scale_dtype=scale_dtype,
        out_dtype=out_dtype,
        out_layout=out_layout,
    )

    _run_and_compare(common_kwargs, output_s, output_zp, rng, data_shape, weight_shape)


# ---------------------------------------------------------------------------
# Edge case: zero-valued input tensor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("data_zp_val", [0, 64])
def test_qnn_conv2d_zero_input(data_zp_val):
    dtype, zp_dtype, scale_dtype = "int8", "int32", "float32"
    rng = np.random.default_rng(seed=0)

    data_shape, weight_shape = (1, 16, 8, 8), (8, 16, 3, 3)
    data_layout, weight_layout, out_layout = "NCHW", "OIHW", "NCHW"
    strides, padding, dilation = (1, 1), (0, 0, 0, 0), (1, 1)
    groups = 1

    data_scale = float(rng.uniform(0.01, 1.0))
    weight_scale = float(rng.uniform(0.01, 1.0))
    output_s = float(rng.uniform(0.001, 0.1))
    output_zp = 0

    common_kwargs = dict(
        data=data_shape,
        weight=weight_shape,
        data_zero_point=data_zp_val,
        weight_zero_point=0,
        data_scale=data_scale,
        weight_scale=weight_scale,
        data_layout=data_layout,
        weight_layout=weight_layout,
        groups=groups,
        strides=strides,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        zp_dtype=zp_dtype,
        scale_dtype=scale_dtype,
        out_dtype=scale_dtype,
        out_layout=out_layout,
    )

    ref_mod = get_ref_impl(**common_kwargs)
    qnn_mod = get_qnn_impl(
        **common_kwargs,
        output_scale=output_s,
        output_zero_point=output_zp,
        requant_out_dtype="int8",
    )

    # all-zero input
    data_np = np.zeros(data_shape, dtype=np.int8)
    weight_np = rng.integers(-128, 127, size=weight_shape, dtype=np.int8)

    ref_ex = relax.build(ref_mod, "llvm")
    qnn_ex = relax.build(qnn_mod, "llvm")
    ref_out = run_cpu(ref_ex, [data_np, weight_np])[0]
    qnn_out = run_cpu(qnn_ex, [data_np, weight_np])[0]

    ref_int8 = np.clip(np.round(ref_out / output_s).astype(np.int32) + output_zp, -128, 127).astype(
        np.int8
    )
    np.testing.assert_array_equal(ref_int8, qnn_out)


# ---------------------------------------------------------------------------
# Edge case: output zero point at int8 boundaries (-128 and 127)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("output_zp", [-128, 0, 127])
def test_qnn_conv2d_output_zp_boundary(output_zp):
    dtype, zp_dtype, scale_dtype = "int8", "int32", "float32"
    rng = np.random.default_rng(seed=7)

    data_shape, weight_shape = (1, 16, 8, 8), (8, 16, 3, 3)
    data_layout, weight_layout, out_layout = "NCHW", "OIHW", "NCHW"
    strides, padding, dilation = (1, 1), (0, 0, 0, 0), (1, 1)
    groups = 1

    data_scale = float(rng.uniform(0.01, 1.0))
    weight_scale = float(rng.uniform(0.01, 1.0))
    output_s = float(rng.uniform(0.001, 0.1))

    common_kwargs = dict(
        data=data_shape,
        weight=weight_shape,
        data_zero_point=0,
        weight_zero_point=0,
        data_scale=data_scale,
        weight_scale=weight_scale,
        data_layout=data_layout,
        weight_layout=weight_layout,
        groups=groups,
        strides=strides,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        zp_dtype=zp_dtype,
        scale_dtype=scale_dtype,
        out_dtype=scale_dtype,
        out_layout=out_layout,
    )

    _run_and_compare(common_kwargs, output_s, output_zp, rng, data_shape, weight_shape)


# ---------------------------------------------------------------------------
# Edge case: batch size > 1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch_size", [2, 4])
def test_qnn_conv2d_batch(batch_size):
    dtype, zp_dtype, scale_dtype = "int8", "int32", "float32"
    rng = np.random.default_rng(seed=99)

    data_shape = (batch_size, 16, 8, 8)
    weight_shape = (8, 16, 3, 3)
    data_layout, weight_layout, out_layout = "NCHW", "OIHW", "NCHW"
    strides, padding, dilation = (1, 1), (1, 1, 1, 1), (1, 1)
    groups = 1

    data_scale = float(rng.uniform(0.01, 1.0))
    weight_scale = float(rng.uniform(0.01, 1.0))
    output_s = float(rng.uniform(0.001, 0.1))
    output_zp = 0

    common_kwargs = dict(
        data=data_shape,
        weight=weight_shape,
        data_zero_point=0,
        weight_zero_point=0,
        data_scale=data_scale,
        weight_scale=weight_scale,
        data_layout=data_layout,
        weight_layout=weight_layout,
        groups=groups,
        strides=strides,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        zp_dtype=zp_dtype,
        scale_dtype=scale_dtype,
        out_dtype=scale_dtype,
        out_layout=out_layout,
    )

    _run_and_compare(common_kwargs, output_s, output_zp, rng, data_shape, weight_shape)


if __name__ == "__main__":
    tvm.testing.main()
