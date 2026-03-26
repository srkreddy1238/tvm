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
import os

import pytest
from utils import requires_adreno_vulkan, verify_results

import tvm
import tvm.testing
from tvm import relax, te, tir
from tvm.relax.struct_info import TensorStructInfo

TARGET_SUPPORTS_EXTENSION = os.getenv("ADRENO_TARGET_COOP", "").strip().lower() == "yes"
vk_target = tvm.target.Target(
    {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_float16": 1,
        "supports_16bit_buffer": 1,
        "supports_int64": 1,
        "supports_int8": 1,
        "supports_8bit_buffer": 1,
        "supports_storage_buffer_storage_class": 1,
        "supports_khr_cooperative_matrix": 1,
        "max_shared_memory_per_block": 32768,
    }
)
vk_target_qcom = tvm.target.Target(
    {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_float16": 1,
        "supports_16bit_buffer": 1,
        "supports_int64": 1,
        "supports_int8": 1,
        "supports_8bit_buffer": 1,
        "supports_storage_buffer_storage_class": 1,
        "supports_khr_cooperative_matrix": 1,
        "supports_qcom_cooperative_matrix_conversion": 1,
    }
)
ref_target = tvm.target.Target("opencl")


def check_codegen_pipeline(mod, target):
    relax_pipeline = relax.get_default_pipeline(target)
    tir_pipeline = tir.get_default_tir_pipeline(target)
    ex = tvm.compile(mod, target, tir_pipeline=tir_pipeline, relax_pipeline=relax_pipeline)
    source_str = ex.mod.imports[0].imports[0].inspect_source()
    assert "OpCooperativeMatrixMulAddKHR" in source_str


@requires_adreno_vulkan
@pytest.mark.skipif(not TARGET_SUPPORTS_EXTENSION, reason="Device not supported.")
@pytest.mark.parametrize("use_qcom", [False, True])
@pytest.mark.parametrize(
    "in_dtype,out_dtype", [("float16", "float16"), ("float32", "float32"), ("int8", "int32")]
)
@pytest.mark.parametrize("a_trans,b_trans", [(False, False), (True, True)])
@pytest.mark.parametrize("M,N,K", [(64, 64, 32), (1023, 1023, 1152)])
@pytest.mark.parametrize("B", [1, 16])
def test_matmul(
    B, M, N, K, a_trans: bool, b_trans: bool, in_dtype: str, out_dtype: str, use_qcom: bool
):
    A_shape = (B, M, K) if not a_trans else (B, K, M)
    B_shape = (B, K, N) if not b_trans else (B, N, K)
    C_shape = (B, M, N)

    if use_qcom:
        target = vk_target_qcom
    else:
        target = vk_target

    def te_batch_matmul(A, B):
        k = te.reduce_axis((0, K), name="k")

        def compute(b, i, j):
            a_indices = (b, i, k) if not a_trans else (b, k, i)
            b_indices = (b, k, j) if not b_trans else (b, j, k)

            return te.sum(
                te.multiply(A[a_indices].astype(out_dtype), B[b_indices].astype(out_dtype)),
                axis=[k],
            )

        return te.compute(C_shape, compute, name="matmul")

    bb = relax.BlockBuilder()
    A = relax.Var("A", TensorStructInfo(A_shape, in_dtype))
    B = relax.Var("B", TensorStructInfo(B_shape, in_dtype))
    with bb.function("main", [A, B]):
        with bb.dataflow():
            C = bb.call_te(te_batch_matmul, A, B)
            bb.emit_output(C)
        bb.emit_func_output(C)
    Matmul = bb.finalize()

    check_codegen_pipeline(Matmul, target)
    verify_results(Matmul, target, ref_target, atol=0.0, rtol=1e-2)


if __name__ == "__main__":
    tvm.testing.main()
