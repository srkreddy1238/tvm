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

from tvm import relax
from tvm.relax import TensorStructInfo


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
    data, weight = relax.Var("data", TensorStructInfo(shape=data, dtype=dtype)), relax.Var(
        "weight", TensorStructInfo(shape=weight, dtype=dtype)
    )
    data_zero_point, weight_zero_point = relax.const(data_zero_point, dtype=zp_dtype), relax.const(
        weight_zero_point, dtype=zp_dtype
    )

    assert (data_scale is None and weight_scale is None) or (
        not (data_scale is None) and not (weight_scale is None)
    ), "Both must be None or a Constant"
    has_scale = not (data_scale is None)

    if has_scale:
        data_scale, weight_scale = relax.const(data_scale, dtype=scale_dtype), relax.const(
            weight_scale, dtype=scale_dtype
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
                conv = bb.emit(conv_op)
                out = bb.emit_output(relax.op.multiply(conv, scale))
            else:
                out = bb.emit_output(conv_op)
        bb.emit_func_output(out)
    mod = bb.finalize()

    return mod


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
):
    data, weight = relax.Var("data", TensorStructInfo(shape=data, dtype=dtype)), relax.Var(
        "weight", TensorStructInfo(shape=weight, dtype=dtype)
    )
    data_zero_point, weight_zero_point = relax.const(data_zero_point, dtype=zp_dtype), relax.const(
        weight_zero_point, dtype=zp_dtype
    )

    assert (data_scale is None and weight_scale is None) or (
        not (data_scale is None) and not (weight_scale is None)
    ), "Both must be None or a Constant"
    has_scale = not (data_scale is None)

    if has_scale:
        data_scale, weight_scale = relax.const(data_scale, dtype=scale_dtype), relax.const(
            weight_scale, dtype=scale_dtype
        )

    bb = relax.BlockBuilder()
    with bb.function("main", [data, weight]):
        with bb.dataflow():
            out = relax.op.qnn.conv2d(
                data,
                weight,
                data_zero_point,
                weight_zero_point,
                data_scale,
                weight_scale,
                strides,
                padding,
                dilation,
                groups,
                data_layout,
                weight_layout,
                out_layout,
                out_dtype,
            )
            bb.emit_output(out)
        bb.emit_func_output(out)
    mod = bb.finalize()

    return mod


def run_cpu(ex, inputs):
    dev = tvm.cpu()
    inputs = [tvm.runtime.tensor(inp, dev) for inp in inputs]
    vm = relax.VirtualMachine(ex, dev)
    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    outputs = vm.get_outputs("main")

    if not isinstance(outputs, list):
        outputs = [outputs]

    outputs = [out.numpy() for out in outputs]
    return outputs


@pytest.mark.parametrize("data_shape,data_layout", [((1, 64, 224, 224), "NCHW")])
@pytest.mark.parametrize(
    "weight_shape,weight_layout", [((32, 16, 3, 3), "OIHW"), ((32, 64, 3, 3), "OIHW")]
)
@pytest.mark.parametrize("has_scale", [True, False])
def test_qnn_conv2d(data_shape, weight_shape, has_scale, data_layout, weight_layout):
    dtype, zp_dtype, scale_dtype = "int8", "int32", "float32"
    strides, padding, dilation = (2, 2), (2, 2, 2, 2), (2, 2)
    out_layout = "NCHW"

    data_zp, weight_zp = (
        np.random.randint(low=0, high=128, size=1)[0],
        np.random.randint(low=0, high=128, size=1)[0],
    )
    if has_scale:
        data_scale, weight_scale = np.random.uniform(-255, 255), np.random.uniform(-255, 255)
        out_dtype = scale_dtype
    else:
        data_scale, weight_scale = None, None
        out_dtype = zp_dtype

    input_channel, kernel_channel = (
        data_shape[data_layout.index("C")],
        weight_shape[weight_layout.index("I")],
    )
    assert input_channel % kernel_channel == 0

    groups = input_channel // kernel_channel
    ref_mod = get_ref_impl(
        data_shape,
        weight_shape,
        data_zp,
        weight_zp,
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
    )
    qnn_mod = get_qnn_impl(
        data_shape,
        weight_shape,
        data_zp,
        weight_zp,
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
    )

    data, weight = np.random.randint(
        low=0, high=128, size=data_shape, dtype="int8"
    ), np.random.randint(low=0, high=128, size=weight_shape, dtype="int8")
    inputs = [data, weight]
    ref_ex = relax.build(ref_mod, "llvm")
    qnn_ex = relax.build(qnn_mod, "llvm")

    ref_outputs = run_cpu(ref_ex, inputs)
    qnn_outputs = run_cpu(qnn_ex, inputs)

    assert len(ref_outputs) == len(qnn_outputs)
    for ref, res in zip(ref_outputs, qnn_outputs):
        np.testing.assert_allclose(ref, res)


if __name__ == "__main__":
    tvm.testing.main()
