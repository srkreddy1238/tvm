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
from tvm import relax
from tvm.relax import TensorStructInfo


def _get_ref_impl(
    data_shape,
    weight_shape,
    data_zero_point,
    weight_zero_point,
    data_scale,
    weight_scale,
    data_layout,
    weight_layout,
    groups,
    strides,
    padding,
    output_padding,
    dilation,
    dtype,
    zp_dtype,
    scale_dtype,
    out_dtype,
    out_layout,
):
    """
    Reference implementation.
    """
    need_transpose = data_layout == "NHWC"

    if need_transpose:
        # data   : NHWC  (N, H, W, C)  →  NCHW  (N, C, H, W)
        internal_data_layout = "NCHW"
        internal_weight_layout = "IOHW"
        internal_out_layout = "NCHW"
    else:
        internal_data_layout = data_layout
        internal_weight_layout = weight_layout
        internal_out_layout = out_layout

    data = relax.Var("data", TensorStructInfo(shape=data_shape, dtype=dtype))
    weight = relax.Var("weight", TensorStructInfo(shape=weight_shape, dtype=dtype))

    data_zp = relax.const(data_zero_point, dtype=zp_dtype)
    weight_zp = relax.const(weight_zero_point, dtype=zp_dtype)

    has_scale = data_scale is not None
    assert has_scale == (weight_scale is not None)
    if has_scale:
        data_s = relax.const(data_scale, dtype=scale_dtype)
        weight_s = relax.const(weight_scale, dtype=scale_dtype)

    bb = relax.BlockBuilder()
    with bb.function("main", [data, weight]):
        with bb.dataflow():
            data_cast = bb.emit(relax.op.datatype.astype(data, zp_dtype))
            weight_cast = bb.emit(relax.op.datatype.astype(weight, zp_dtype))
            data_dq = bb.emit(relax.op.subtract(data_cast, data_zp))
            weight_dq = bb.emit(relax.op.subtract(weight_cast, weight_zp))

            if need_transpose:
                # NHWC (N,H,W,C) → NCHW (N,C,H,W)  axes: 0,3,1,2
                data_dq = bb.emit(relax.op.permute_dims(data_dq, axes=[0, 3, 1, 2]))
                # OHWI (O,H,W,I) → IOHW (I,O,H,W)  axes: 3,0,1,2
                weight_dq = bb.emit(relax.op.permute_dims(weight_dq, axes=[3, 0, 1, 2]))

            conv_out = bb.emit(
                relax.op.nn.conv2d_transpose(
                    data_dq,
                    weight_dq,
                    strides,
                    padding,
                    output_padding,
                    dilation,
                    groups,
                    internal_data_layout,
                    internal_weight_layout,
                    internal_out_layout,
                    out_dtype,
                )
            )

            if need_transpose:
                # NCHW (N,C,H,W) → NHWC (N,H,W,C)  axes: 0,2,3,1
                conv_out = bb.emit(relax.op.permute_dims(conv_out, axes=[0, 2, 3, 1]))

            if has_scale:
                scale = bb.emit(relax.op.multiply(data_s, weight_s))
                out = bb.emit_output(relax.op.multiply(conv_out, scale))
            else:
                out = bb.emit_output(conv_out)

        bb.emit_func_output(out)

    return bb.finalize()


def _get_qnn_impl(
    data_shape,
    weight_shape,
    data_zero_point,
    weight_zero_point,
    data_scale,
    weight_scale,
    data_layout,
    weight_layout,
    groups,
    strides,
    padding,
    output_padding,
    dilation,
    dtype,
    zp_dtype,
    scale_dtype,
    out_dtype,
    out_layout,
):
    """
    QNN implementation: single relax.op.qnn.conv2d_transpose call.
    """
    data = relax.Var("data", TensorStructInfo(shape=data_shape, dtype=dtype))
    weight = relax.Var("weight", TensorStructInfo(shape=weight_shape, dtype=dtype))

    data_zp = relax.const(data_zero_point, dtype=zp_dtype)
    weight_zp = relax.const(weight_zero_point, dtype=zp_dtype)

    has_scale = data_scale is not None
    assert has_scale == (weight_scale is not None), (
        "data_scale and weight_scale must both be None or both be set"
    )

    if has_scale:
        data_s = relax.const(data_scale, dtype=scale_dtype)
        weight_s = relax.const(weight_scale, dtype=scale_dtype)
    else:
        data_s = weight_s = None

    bb = relax.BlockBuilder()
    with bb.function("main", [data, weight]):
        with bb.dataflow():
            out = relax.qnn.op.conv2d_transpose(
                data,
                weight,
                data_zp,
                weight_zp,
                data_s,
                weight_s,
                strides,
                padding,
                output_padding,
                dilation,
                groups,
                data_layout,
                weight_layout,
                out_layout,
                out_dtype,
            )
        bb.emit_func_output(out)

    return bb.finalize()


def _run_cpu(executable, inputs):
    dev = tvm.cpu()
    rt_inputs = [tvm.runtime.tensor(inp, dev) for inp in inputs]
    vm = relax.VirtualMachine(executable, dev)
    vm.set_input("main", *rt_inputs)
    vm.invoke_stateful("main")
    outputs = vm.get_outputs("main")
    if not isinstance(outputs, list):
        outputs = [outputs]
    return [o.numpy() for o in outputs]


@pytest.mark.parametrize(
    "data_shape,data_layout,weight_shape,weight_layout, strides,padding,"
    "output_padding,dilation,out_layout",
    [
        (
            (1, 32, 224, 224),
            "NCHW",
            (32, 16, 3, 3),
            "IOHW",
            (2, 2),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            "NCHW",
        ),
        (
            (1, 16, 160, 320),
            "NCHW",
            (16, 16, 3, 3),
            "IOHW",
            (2, 2),
            (0, 1, 0, 1),
            (1, 1),
            (1, 1),
            "NCHW",
        ),
        (
            (1, 48, 56, 56),
            "NCHW",
            (48, 16, 3, 3),
            "IOHW",
            (1, 1),
            (0, 0, 0, 1),
            (0, 0),
            (1, 1),
            "NCHW",
        ),
        (
            (1, 16, 56, 56),
            "NCHW",
            (16, 16, 3, 3),
            "IOHW",
            (2, 2),
            (1, 1, 1, 1),
            (1, 1),
            (1, 1),
            "NCHW",
        ),
        (
            (1, 56, 56, 32),
            "NHWC",
            (16, 3, 3, 32),
            "OHWI",
            (2, 2),
            (0, 1, 0, 0),
            (1, 1),
            (1, 1),
            "NHWC",
        ),
        (
            (1, 56, 56, 16),
            "NHWC",
            (16, 3, 3, 16),
            "OHWI",
            (2, 2),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            "NHWC",
        ),
        (
            (1, 56, 56, 48),
            "NHWC",
            (16, 3, 3, 48),
            "OHWI",
            (1, 1),
            (0, 1, 1, 0),
            (0, 0),
            (1, 1),
            "NHWC",
        ),
        (
            (1, 56, 56, 16),
            "NHWC",
            (16, 3, 3, 16),
            "OHWI",
            (2, 2),
            (1, 1, 1, 1),
            (1, 1),
            (1, 1),
            "NHWC",
        ),
    ],
)
@pytest.mark.parametrize("has_scale", [False, True])
def test_qnn_conv2d_transpose(
    data_shape,
    data_layout,
    weight_shape,
    weight_layout,
    strides,
    padding,
    output_padding,
    dilation,
    out_layout,
    has_scale,
):
    dtype, zp_dtype, scale_dtype = "int8", "int32", "float32"
    C_in_data = data_shape[data_layout.index("C")]
    C_in_kernel = weight_shape[weight_layout.index("I")]

    groups = C_in_data // C_in_kernel

    rng = np.random.default_rng(seed=42)
    data_zp = int(rng.integers(0, 64))
    weight_zp = int(rng.integers(0, 64))

    if has_scale:
        data_s = float(rng.uniform(0.01, 1.0))
        weight_s = float(rng.uniform(0.01, 1.0))
        out_dtype = scale_dtype
    else:
        data_s = weight_s = None
        out_dtype = zp_dtype

    common_kwargs = dict(
        data_shape=data_shape,
        weight_shape=weight_shape,
        data_zero_point=data_zp,
        weight_zero_point=weight_zp,
        data_scale=data_s,
        weight_scale=weight_s,
        data_layout=data_layout,
        weight_layout=weight_layout,
        groups=groups,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        dtype=dtype,
        zp_dtype=zp_dtype,
        scale_dtype=scale_dtype,
        out_dtype=out_dtype,
        out_layout=out_layout,
    )

    ref_mod = _get_ref_impl(**common_kwargs)
    qnn_mod = _get_qnn_impl(**common_kwargs)
    data_np = rng.integers(0, 64, size=data_shape, dtype="int8")
    weight_np = rng.integers(0, 64, size=weight_shape, dtype="int8")
    inputs = [data_np, weight_np]

    ref_ex = relax.build(ref_mod, "llvm")
    qnn_ex = relax.build(qnn_mod, "llvm")
    ref_outputs = _run_cpu(ref_ex, inputs)
    qnn_outputs = _run_cpu(qnn_ex, inputs)

    assert len(ref_outputs) == len(qnn_outputs)
    for ref, res in zip(ref_outputs, qnn_outputs):
        np.testing.assert_allclose(
            ref,
            res,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Mismatch for has_scale={has_scale}, data={data_shape}, weight={weight_shape}",
        )


if __name__ == "__main__":
    tvm.testing.main()
