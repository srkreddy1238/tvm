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
import os
import tvm
import numpy as np
import tempfile
import pytest
import tvm.testing

from tvm import relax
from tvm.contrib import ndk
from tvm.target import Target
from tvm.relax import TensorStructInfo

# Helper function to generate the reference module using dequantize -> float_op -> quantize pattern
def create_ref_module(
    op_name,
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    out_scale,
    out_zero_point,
):
    """
    Create a Relax module for reference using dequantize -> float_op -> quantize pattern.

    """
    bb = relax.BlockBuilder()

    # getattr the qnn op from relax.op.qnn
    try:
        float_op = getattr(relax.op, op_name)
    except AttributeError:
        raise ValueError(
            f"Unsupported operation: {op_name}. Must be one of ['add', 'multiply', 'subtract']"
        )

    with bb.function("main", [lhs, rhs]):
        with bb.dataflow():

            lhs_float = relax.op.dequantize(lhs, lhs_scale, lhs_zero_point)
            rhs_float = relax.op.dequantize(rhs, rhs_scale, rhs_zero_point)

            result_float = float_op(lhs_float, rhs_float)

            quatized_result = relax.op.quantize(result_float, out_scale, out_zero_point)

            gv = bb.emit_output(quatized_result)
        bb.emit_func_output(gv)

    mod = bb.finalize()
    return mod


def create_qnn_module(
    op_name,
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    out_scale,
    out_zero_point,
):
    """
    Create a Relax module for a specified QNN operation.
    As of now add, subrtract, multiply are supported.

    Returns:
        Relax module containing the QNN operation
    """
    bb = relax.BlockBuilder()

    try:
        qnn_op = getattr(relax.qnn.op, op_name)
    except AttributeError:
        raise ValueError(
            f"Unsupported operation: {op_name}. Must be one of ['add', 'multiply', 'subtract']"
        )

    with bb.function("main", [lhs, rhs]):
        with bb.dataflow():
            out = qnn_op(
                lhs,
                rhs,
                lhs_scale,
                lhs_zero_point,
                rhs_scale,
                rhs_zero_point,
                out_scale,
                out_zero_point,
            )
        bb.emit_func_output(out)

    mod = bb.finalize()
    return mod


def create_qnn_inputs(
    lhs_shape,
    rhs_shape,
    lhs_scale,
    rhs_scale,
    lhs_zero_point,
    rhs_zero_point,
    out_scale,
    out_zero_point,
    dtype="int8",
    zp_dtype="int8",
    scale_dtype="float32",
):
    """
    Create input variables and constants for QNN operations.

    """
    lhs = relax.Var("lhs", TensorStructInfo(shape=lhs_shape, dtype=dtype))
    rhs = relax.Var("rhs", TensorStructInfo(shape=rhs_shape, dtype=dtype))

    lhs_scale_const = relax.const(lhs_scale, dtype=scale_dtype)
    rhs_scale_const = relax.const(rhs_scale, dtype=scale_dtype)

    lhs_zero_point_const = relax.const(lhs_zero_point, dtype=zp_dtype)
    rhs_zero_point_const = relax.const(rhs_zero_point, dtype=zp_dtype)

    out_scale_const = relax.const(out_scale, dtype=scale_dtype)
    out_zero_point_const = relax.const(out_zero_point, dtype=zp_dtype)

    return (
        lhs,
        rhs,
        lhs_scale_const,
        lhs_zero_point_const,
        rhs_scale_const,
        rhs_zero_point_const,
        out_scale_const,
        out_zero_point_const,
    )


def build_and_run(mod, input_arrays, target="llvm"):
    """
    Build and run a Relax module.
    """
    tgt = tvm.target.Target(target, host=target)

    relax_pipeline = relax.pipeline.get_default_pipeline(tgt)
    tir_pipeline = tvm.tir.get_default_tir_pipeline(tgt)
    rt_mod = tvm.compile(mod, tgt, relax_pipeline=relax_pipeline, tir_pipeline=tir_pipeline)

    # Execute on device
    dev = tvm.cpu()
    vm = relax.VirtualMachine(rt_mod.mod, dev, profile=True)
    dev_inputs = [tvm.runtime.tensor(inp, dev) for inp in input_arrays]
    vm.set_input("main", *dev_inputs)
    vm.invoke_stateful("main")

    return vm.get_outputs("main").numpy()


def _test_qnn_operation(
    op_name,
    lhs_shape,
    rhs_shape,
    lhs_scale,
    rhs_scale,
    lhs_zero_point,
    rhs_zero_point,
    out_scale,
    out_zero_point,
):
    """
    Generic test function for QNN operations.
    """
    lhs_tensor = np.random.randint(
        max(-128, lhs_zero_point - 10), min(127, lhs_zero_point + 10), size=lhs_shape
    ).astype("int8")

    rhs_tensor = np.random.randint(
        max(-128, rhs_zero_point - 10), min(127, rhs_zero_point + 10), size=rhs_shape
    ).astype("int8")

    inputs = create_qnn_inputs(
        lhs_shape,
        rhs_shape,
        lhs_scale,
        rhs_scale,
        lhs_zero_point,
        rhs_zero_point,
        out_scale,
        out_zero_point,
    )

    mod = create_qnn_module(op_name, *inputs)
    ref_mod = create_ref_module(op_name, *inputs)

    inputs = [lhs_tensor, rhs_tensor]
    output = build_and_run(mod, inputs)
    ref_output = build_and_run(ref_mod, inputs)

    np.testing.assert_allclose(output, ref_output, atol=1)

    return output


QNN_TEST_PARAMS = [
    ("add", (1, 64, 224, 224), (1, 1, 1, 224), 0.3289, 0.4156, 90, 5, 1.5, 2),
    ("add", (1, 64, 224, 224), (1, 1, 1, 224), 0.3289, 0.4156, 90, 5, 1.5, 2),
    ("add", (1, 128, 112, 112), (1, 1, 1, 112), 0.2567, 0.3678, 80, 10, 1.2, 5),
    ("subtract", (1, 64, 224, 224), (1, 1, 1, 224), 0.3289, 0.4156, 90, 5, 1.5, 2),
    ("subtract", (1, 64, 224, 224), (1, 1, 1, 224), 0.3289, 0.4156, 90, 5, 1.5, 2),
    ("subtract", (1, 128, 112, 112), (1, 1, 1, 112), 0.2567, 0.3678, 80, 10, 1.2, 5),
    ("multiply", (1, 64, 224, 224), (1, 1, 1, 224), 0.3289, 0.4156, 90, 5, 1.5, 2),
    ("multiply", (1, 64, 224, 224), (1, 1, 1, 224), 0.3289, 0.4156, 90, 5, 1.5, 2),
    ("multiply", (1, 128, 112, 112), (1, 1, 1, 112), 0.2567, 0.3678, 80, 10, 1.2, 5),
]


######################### Unit Tests #########################

# Test QNN Binary Operations
@pytest.mark.parametrize(
    "op_name, lhs_shape, rhs_shape, lhs_scale, rhs_scale, lhs_zero_point, rhs_zero_point, out_scale, out_zero_point",
    QNN_TEST_PARAMS,
)
def test_qnn_binary_op(
    op_name,
    lhs_shape,
    rhs_shape,
    lhs_scale,
    rhs_scale,
    lhs_zero_point,
    rhs_zero_point,
    out_scale,
    out_zero_point,
):
    """Test QNN binary operations."""
    output = _test_qnn_operation(
        op_name,
        lhs_shape,
        rhs_shape,
        lhs_scale,
        rhs_scale,
        lhs_zero_point,
        rhs_zero_point,
        out_scale,
        out_zero_point,
    )
    assert output is not None


if __name__ == "__main__":
    tvm.testing.main()
