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

import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import te, s_tir
from tvm.script import tir as T, ir as I
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I_builder
from tvm.script.ir_builder import tir as T_builder
from tvm.s_tir import TensorIntrin
from tvm.tir import IntImm, Cast
from tvm.s_tir import Schedule
from tvm.s_tir.tensor_intrin.cuda import (
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
    WMMA_FILL_16x16x16_F32_INTRIN,
    WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f16_INTRIN,
    WMMA_FILL_16x16x16_F16_INTRIN,
    WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN,
)
from tvm.s_tir.tensor_intrin.adreno import get_adreno_wmma_intrin_group
from tvm.target import Target
from utils import requires_adreno_vulkan


@requires_adreno_vulkan
def test_coopmat_vec_qcom_extn():
    M, N, K = 512, 512, 256
    M_TILE, N_TILE, K_TILE = 64, 64, 16
    in_dtype, out_dtype = "float16", "float16"

    def get_matmul(m: int, n: int, k: int, in_dtype: str, out_dtype: str):
        A = te.placeholder((m, k), name="A", dtype=in_dtype)
        B = te.placeholder((n, k), name="B", dtype=in_dtype)
        r_k = te.reduce_axis((0, k), name="k")

        C = te.compute(
            (m, n),
            lambda i, j: te.sum(
                A[i, r_k].astype(out_dtype) * B[j, r_k].astype(out_dtype),
                axis=r_k,
            ),
            name="compute",
        )

        return te.create_prim_func([A, B, C])

    def schedule_fn(sch: s_tir.Schedule):
        compute_blk = sch.get_sblock("compute")
        i, j, k = sch.get_loops(compute_blk)

        io, ii = sch.split(i, [None, M_TILE])
        jo, ji = sch.split(j, [None, N_TILE])
        ko, ki = sch.split(k, [None, K_TILE])

        sch.reorder(io, jo, ko, ii, ji, ki)

        bx = sch.fuse(io, jo)
        sch.bind(bx, "blockIdx.x")

        A_wmma = sch.cache_read(compute_blk, 0, "wmma.matrix_a")
        A_local = sch.cache_read(A_wmma, 0, "local")
        sch.compute_at(A_wmma, ko)
        sch.compute_at(A_local, ko)

        B_wmma = sch.cache_read(compute_blk, 1, "wmma.matrix_b")
        B_local = sch.cache_read(B_wmma, 0, "local")
        sch.compute_at(B_wmma, ko)
        sch.compute_at(B_local, ko)

        C_wmma = sch.cache_write(compute_blk, 0, "wmma.accumulator")
        C_local = sch.cache_write(C_wmma, 0, "local")
        C_init = sch.decompose_reduction(compute_blk, ko)
        sch.reverse_compute_at(C_wmma, bx)
        sch.reverse_compute_at(C_local, bx)

        sch.bind(sch.get_loops(A_local)[-2], "threadIdx.x")
        sch.bind(sch.get_loops(A_wmma)[-2], "threadIdx.x")
        sch.bind(sch.get_loops(B_local)[-2], "threadIdx.x")
        sch.bind(sch.get_loops(B_wmma)[-2], "threadIdx.x")
        sch.bind(sch.get_loops(C_local)[-2], "threadIdx.x")
        sch.bind(sch.get_loops(C_wmma)[-2], "threadIdx.x")

        INTRIN = get_adreno_wmma_intrin_group(
            M_TILE,
            N_TILE,
            K_TILE,
            ("local", "local"),
            "local",
            False,
            True,
            in_dtype,
            out_dtype,
        )
        sch.tensorize(sch.get_loops(C_init)[-2], INTRIN["init"])
        sch.tensorize(sch.get_loops(A_wmma)[-1], INTRIN["load_a"])
        sch.tensorize(sch.get_loops(B_wmma)[-1], INTRIN["load_b"])
        sch.tensorize(sch.get_loops(compute_blk)[-3], INTRIN["compute"])
        sch.tensorize(sch.get_loops(C_wmma)[-1], INTRIN["store"])

    mod = get_matmul(M, N, K, in_dtype, out_dtype)
    sch = s_tir.Schedule(mod)
    schedule_fn(sch)

    target = Target(
        {
            "kind": "vulkan",
            "device": "adreno",
            "supports_int8": 1,
            "supports_8bit_buffer": 1,
            "supports_storage_buffer_storage_class": 1,
            "supports_float16": 1,
            "supports_16bit_buffer": 1,
            "supports_khr_cooperative_matrix": 1,
            "supports_qcom_cooperative_matrix_conversion": 1,
            "supports_storage_buffer_storage_class": 1,
        }
    )

    # Use SPIR-V validation during compile to validate
    tvm.compile(sch.mod, target)


if __name__ == "__main__":
    tvm.testing.main()
