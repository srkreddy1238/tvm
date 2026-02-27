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
import tvm
import tvm.testing

from tvm import te, tir
from utils import requires_adreno_opencl

target = {"kind": "opencl", "device": "adreno"}


@requires_adreno_opencl
def test_dp4a_codegen():
    from tvm.s_tir import tensor_intrin

    def reduction():
        n = 512
        A = te.placeholder((n, 4), "int8")
        B = te.placeholder((n, 4), "int8")
        j = te.reduce_axis((0, 4))
        C = te.compute(
            (n,),
            lambda i: te.sum(
                tvm.te.multiply(tvm.tir.Cast("int32", A[i, j]), tvm.tir.Cast("int32", B[i, j])),
                axis=j,
            ),
            name="C",
        )

        func = te.create_prim_func([A, B, C])
        sch = tvm.s_tir.Schedule(func)
        blk = sch.get_sblock("C")

        (i, j) = sch.get_loops(sch.get_sblock("C"))
        bx, tx = sch.split(i, [None, 256])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

        A_local = sch.cache_read(blk, 0, "local")
        B_local = sch.cache_read(blk, 1, "local")
        C_local = sch.cache_write(blk, 0, "local")
        sch.compute_at(A_local, tx)
        sch.compute_at(B_local, tx)
        sch.reverse_compute_at(C_local, tx)

        init_blk = sch.decompose_reduction(blk, j)
        sch.tensorize(j, tensor_intrin.adreno.ADRENO_DP4A_i8i8i32_INTRIN)

        ex = tvm.tir.build(sch.mod, target)
        assembly = ex.imports[0].inspect_source()

        pattern = "qcom_dot8_acc"
        assert assembly.count(pattern)

    def matmul():
        M, N, K = 256, 256, 512

        A = te.placeholder((M, K), "int8")
        B = te.placeholder((K, N), "int8")
        k = te.reduce_axis((0, K))
        C = te.compute(
            (M, N),
            lambda i, j: te.sum(
                te.multiply(tir.Cast("int32", A[i, k]), tir.Cast("int32", B[k, j])), axis=k
            ),
            name="C",
        )
        func = te.create_prim_func([A, B, C])
        sch = tvm.s_tir.Schedule(func)
        blk = sch.get_sblock("C")

        (i, j, k) = sch.get_loops(sch.get_sblock("C"))
        sch.bind(i, "blockIdx.x")
        sch.bind(j, "threadIdx.x")

        A_local = sch.cache_read(blk, 0, "local")
        B_local = sch.cache_read(blk, 1, "local")
        C_local = sch.cache_write(blk, 0, "local")
        ko, ki = sch.split(k, [None, 4])
        sch.compute_at(A_local, ko)
        sch.compute_at(B_local, ko)
        sch.reverse_compute_at(C_local, j)

        B_reindex = sch.transform_layout(B_local, ("write", 0), lambda i, j: (j, i))

        init_blk = sch.decompose_reduction(blk, ko)
        sch.tensorize(ki, tensor_intrin.adreno.ADRENO_DP4A_i8i8i32_INTRIN)

        ex = tvm.tir.build(sch.mod, target)
        assembly = ex.imports[0].inspect_source()

        pattern = "qcom_dot8_acc"
        assert assembly.count(pattern)

    def convolution():
        N, C, iH, iW, c4 = 1, 256, 56, 56, 4  # NCHW4c
        O, I, oH, oW, i4 = 64, 32, 3, 3, 4  # OIHW4i

        data = te.placeholder((N, C, iH, iW, c4), dtype="int8")
        kernel = te.placeholder((O, I, oH, oW, i4), dtype="int8")

        OH = (iH - oH) + 1
        OW = (iW - oW) + 1
        O_shape = (N, O, OH, OW)

        groups = C // I

        icc = te.reduce_axis((0, I), name="ic")
        icb = te.reduce_axis((0, 4), name="ic")
        kh = te.reduce_axis((0, oH), name="kh")
        kw = te.reduce_axis((0, oH), name="kw")

        conv = te.compute(
            O_shape,
            lambda n, c, h, w: te.sum(
                te.multiply(
                    tir.Cast(
                        "int32",
                        data[n, (c // (O // groups)) * (I // groups) + icc, h + kh, w + kw, icb],
                    ),
                    tir.Cast("int32", kernel[c, icc, kh, kw, icb]),
                ),
                axis=[icc, kh, kw, icb],
            ),
            name="conv2d_NCHW",
        )
        func = te.create_prim_func([data, kernel, conv])
        sch = tvm.s_tir.Schedule(func)
        blk = sch.get_sblock("conv2d_NCHW")

        (n, c, h, w, icc, kh, kw, icb) = sch.get_loops(blk)
        sch.bind(c, "blockIdx.x")
        sch.bind(h, "blockIdx.y")
        sch.bind(w, "threadIdx.x")

        A_local = sch.cache_read(blk, 0, "local")
        B_local = sch.cache_read(blk, 1, "local")
        C_local = sch.cache_write(blk, 0, "local")
        sch.compute_at(A_local, kw)
        sch.compute_at(B_local, kw)
        sch.reverse_compute_at(C_local, w)

        init_blk = sch.decompose_reduction(blk, w)
        sch.tensorize(icb, tensor_intrin.adreno.ADRENO_DP4A_i8i8i32_INTRIN)

        ex = tvm.tir.build(sch.mod, target)
        assembly = ex.imports[0].inspect_source()

        pattern = "qcom_dot8_acc"
        assert assembly.count(pattern)

    reduction()
    matmul()
    convolution()


if __name__ == "__main__":
    tvm.testing.main()
