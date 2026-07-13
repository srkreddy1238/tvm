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
    # --- requantize parameters (mirrors OperatorConverter pattern) -------
    output_scale=None,
    output_zero_point=0,
    requant_out_dtype="int8",
):
    """
    QNN implementation that mirrors the TFLite OperatorConverter pattern:

      1. conv2d_transpose  with NO scales  →  raw int32 accumulator
      2. requantize with:
           input_scale  = data_scale * weight_scale  (new_input_scale_val)
           input_zp     = 0                          (new_input_zero_point)
           output_scale = output tensor scale
           output_zp    = output tensor zero point
           out_dtype    = output tensor dtype (e.g. int8)

    When data_scale is None (no quantization params), requantize is
    skipped and the raw int32 conv output is returned.
    """
    data = relax.Var("data", TensorStructInfo(shape=data_shape, dtype=dtype))
    weight = relax.Var("weight", TensorStructInfo(shape=weight_shape, dtype=dtype))

    data_zp = relax.const(data_zero_point, dtype=zp_dtype)
    weight_zp = relax.const(weight_zero_point, dtype=zp_dtype)

    has_scale = data_scale is not None
    assert has_scale == (weight_scale is not None), (
        "data_scale and weight_scale must both be None or both be set"
    )

    bb = relax.BlockBuilder()
    with bb.function("main", [data, weight]):
        with bb.dataflow():
            # ------------------------------------------------------------------
            # Step 1 — conv2d_transpose with NO scales (same as converter)
            # Scales are intentionally omitted here so the TOPI does NOT
            # apply them internally. The raw int32 accumulator is returned.
            # ------------------------------------------------------------------
            conv_out = bb.emit(
                relax.qnn.op.conv2d_transpose(
                    data,
                    weight,
                    data_zp,
                    weight_zp,
                    None,  # input_scale  ← None, mirrors converter
                    None,  # kernel_scale ← None, mirrors converter
                    strides,
                    padding,
                    output_padding,
                    dilation,
                    groups,
                    data_layout,
                    weight_layout,
                    out_layout,
                    zp_dtype,  # out_dtype = int32 (raw accumulator)
                )
            )

            # ------------------------------------------------------------------
            # Step 2 — requantize (same as converter)
            # new_input_scale_val = data_scale * weight_scale
            # new_input_zero_point = 0
            # output_scale / output_zero_point from the output tensor qnn_params
            # ------------------------------------------------------------------
            if has_scale:
                assert output_scale is not None, (
                    "output_scale must be provided when data_scale is set"
                )
                # mirrors: new_input_scale_val = data_scale_val * weight_scale_val
                new_input_scale = relax.const(data_scale * weight_scale, dtype=scale_dtype)
                new_input_zero_point = relax.const(0, dtype=zp_dtype)

                out = bb.emit_output(
                    relax.qnn.op.requantize(
                        conv_out,
                        input_scale=new_input_scale,
                        input_zero_point=new_input_zero_point,
                        output_scale=relax.const(output_scale, dtype=scale_dtype),
                        output_zero_point=relax.const(output_zero_point, dtype=zp_dtype),
                        out_dtype=requant_out_dtype,
                    )
                )
            else:
                # No quantization params — return raw int32 conv output
                out = bb.emit_output(conv_out)

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
        # output tensor qnn_params (mirrors output_tensor.qnn_params in converter)
        output_s = float(rng.uniform(0.001, 0.1))
        output_zp = int(rng.integers(-10, 10))
        out_dtype = scale_dtype  # ref produces float32
    else:
        data_s = weight_s = None
        output_s = None
        output_zp = 0
        out_dtype = zp_dtype  # both ref and qnn produce int32

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
    qnn_mod = _get_qnn_impl(
        **common_kwargs,
        output_scale=output_s,
        output_zero_point=output_zp,
        requant_out_dtype="int8",
    )
    data_np = rng.integers(0, 64, size=data_shape, dtype="int8")
    weight_np = rng.integers(0, 64, size=weight_shape, dtype="int8")
    inputs = [data_np, weight_np]

    ref_ex = relax.build(ref_mod, "llvm")
    qnn_ex = relax.build(qnn_mod, "llvm")
    ref_outputs = _run_cpu(ref_ex, inputs)
    qnn_outputs = _run_cpu(qnn_ex, inputs)

    assert len(ref_outputs) == len(qnn_outputs)
    for ref, res in zip(ref_outputs, qnn_outputs):
        if has_scale:
            # ref is float32: (data - data_zp) * (weight - weight_zp) * data_s * weight_s
            # qnn is int8:    requantize(int32_accum, new_input_scale, 0, output_s, output_zp)
            # Convert ref float32 → int8 using the same requantize formula:
            #   out = clip(round(ref / output_s) + output_zp, -128, 127)
            ref_int8 = np.clip(
                np.round(ref / output_s).astype(np.int32) + output_zp,
                -128,
                127,
            ).astype(np.int8)
            np.testing.assert_array_equal(
                ref_int8,
                res,
                err_msg=(
                    f"Mismatch for has_scale={has_scale}, data={data_shape}, weight={weight_shape}"
                ),
            )
        else:
            # Both int32 — exact match
            np.testing.assert_array_equal(
                ref,
                res,
                err_msg=(
                    f"Mismatch for has_scale={has_scale}, data={data_shape}, weight={weight_shape}"
                ),
            )


if __name__ == "__main__":
    tvm.testing.main()
