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
    vm = relax.VirtualMachine(rt_mod.mod, dev)
    dev_inputs = [tvm.runtime.tensor(inp, dev) for inp in input_arrays]
    vm.set_input("main", *dev_inputs)
    vm.invoke_stateful("main")

    outputs = vm.get_outputs("main")

    return outputs.numpy()


def _test_qnn_dense(
    data_shape,
    weight_shape,
    data_scale,
    data_zero_point,
    weight_scale,
    weight_zero_point,
    out_scale,
    out_zero_point,
    out_dtype="int32",
):
    """
    Generic test function for QNN dense operation.
    Compares a hand-written reference implementation against relax.qnn.op.dense.
    """
    scale_dtype = "float32"
    zp_dtype = "int8"

    # Random test inputs
    data_np = np.random.randint(low=20, high=100, size=data_shape, dtype="int8")
    weight_np = np.random.randint(low=20, high=100, size=weight_shape, dtype="int8")

    data = relax.Var("data", TensorStructInfo(shape=data_shape, dtype="int8"))
    weight = relax.Var("weight", TensorStructInfo(shape=weight_shape, dtype="int8"))

    data_scale_const = relax.const(data_scale, scale_dtype)
    weight_scale_const = relax.const(weight_scale, scale_dtype)
    data_zp_const = relax.const(data_zero_point, zp_dtype)
    weight_zp_const = relax.const(weight_zero_point, zp_dtype)
    out_scale_const = relax.const(out_scale, scale_dtype)
    out_zp_const = relax.const(out_zero_point, zp_dtype)

    # Reference implementation: dequantize -> float matmul -> quantize
    bb_ref = relax.BlockBuilder()
    with bb_ref.function("main", [data, weight]):
        with bb_ref.dataflow():
            data_deq = relax.op.dequantize(data, data_scale_const, data_zp_const)
            weight_deq = relax.op.dequantize(weight, weight_scale_const, weight_zp_const)
            ref_out = relax.op.linear_algebra.matmul(
                data_deq,
                weight_deq,
                out_dtype="float32",
            )
            q_out = relax.op.quantize(ref_out, out_scale_const, out_zp_const, out_dtype="int8")
            gv = bb_ref.emit_output(q_out)
        bb_ref.emit_func_output(gv)
    ref_mod = bb_ref.finalize()

    # QNN implementation: qnn.dense -> requantize
    bb_qnn = relax.BlockBuilder()
    with bb_qnn.function("main", [data, weight]):
        with bb_qnn.dataflow():
            mm_int32 = relax.qnn.op.dense(
                data,
                weight,
                data_scale_const,
                data_zp_const,
                weight_scale_const,
                weight_zp_const,
                out_dtype="int32",
            )
            in_s = relax.const(float(data_scale * weight_scale), dtype=scale_dtype)
            in_z = relax.const(0, dtype=zp_dtype)
            qnn_out = relax.qnn.op.requantize(
                mm_int32,
                in_s,
                in_z,
                out_scale_const,
                out_zp_const,
                out_dtype="int8",
            )
            gv_qnn = bb_qnn.emit_output(qnn_out)
        bb_qnn.emit_func_output(gv_qnn)
    qnn_mod = bb_qnn.finalize()

    inputs = [data_np, weight_np]
    ref_output = build_and_run(ref_mod, inputs)
    qnn_output = build_and_run(qnn_mod, inputs)
    np.testing.assert_allclose(qnn_output, ref_output, rtol=1e-5, atol=1e-5)


# Test parameters for QNN Dense
QNN_DENSE_TEST_PARAMS = [
    ((1, 16), (16, 32), 0.1, 25, 0.2, 34, 0.3, 20, "int32"),
    ((1, 16), (16, 32), 0.9, 6, 0.6, 7, 0.8, 15, "int32"),
    ((1, 64), (64, 128), 0.05, 10, 0.15, 20, 0.25, 30, "int32"),
    ((2, 32), (32, 64), 0.08, 15, 0.12, 25, 0.2, 20, "int32"),
    ((1, 128), (128, 256), 0.03, 5, 0.07, 15, 0.1, 10, "int32"),
    ((5, 32), (32, 16), 0.2, 20, 0.3, 40, 0.4, 30, "int32"),
]


# Unit Tests
@pytest.mark.parametrize(
    "data_shape, weight_shape, data_scale, data_zero_point, weight_scale,"
    "weight_zero_point, out_scale, out_zero_point, out_dtype",
    QNN_DENSE_TEST_PARAMS,
)
def test_qnn_dense(
    data_shape,
    weight_shape,
    data_scale,
    data_zero_point,
    weight_scale,
    weight_zero_point,
    out_scale,
    out_zero_point,
    out_dtype,
):
    """Test QNN dense operation."""
    _test_qnn_dense(
        data_shape,
        weight_shape,
        data_scale,
        data_zero_point,
        weight_scale,
        weight_zero_point,
        out_scale,
        out_zero_point,
        out_dtype,
    )


if __name__ == "__main__":
    tvm.testing.main()
