# ruff: noqa: E501
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
"""Tests for _merge_state_inplace_cpu and _merge_state_inplace (GPU)."""

import tvm
import tvm.testing
from tvm.relax.frontend.nn.llm.kv_cache import (
    _merge_state_inplace,
    _merge_state_inplace_cpu,
)
from tvm.script import tir as T

# ---------------------------------------------------------------------------
# _merge_state_inplace_cpu
# ---------------------------------------------------------------------------


def test_merge_state_inplace_cpu_f16():
    fn = _merge_state_inplace_cpu("float16")

    # fmt: off
    @T.prim_func
    def merge_state_inplace_cpu(v: T.handle, s: T.handle, v_other: T.handle, s_other: T.handle):
        T.func_attr({"tir.is_scheduled": True})
        N, H, D = T.int32(is_size_var=True), T.int32(is_size_var=True), T.int32(is_size_var=True)
        V = T.match_buffer(v, (N, H, D), "float16")
        S = T.match_buffer(s, (N, H))
        V_other = T.match_buffer(v_other, (N, H, D), "float16")
        S_other = T.match_buffer(s_other, (N, H))
        for n, h in T.grid(N, H):
            with T.sblock("merge"):
                T.reads(S[n, h], S_other[n, h], V[n, h, 0:D], V_other[n, h, 0:D])
                T.writes(V[n, h, 0:D], S[n, h])
                s_val = T.alloc_buffer((1,))
                s_other_val = T.alloc_buffer((1,))
                s_max = T.alloc_buffer((1,))
                scale = T.alloc_buffer((1,))
                other_scale = T.alloc_buffer((1,))
                s_val[0] = S[n, h]
                s_other_val[0] = S_other[n, h]
                s_max[0] = T.max(s_val[0], s_other_val[0])
                s_val[0] = T.exp2(s_val[0] - s_max[0])
                s_other_val[0] = T.exp2(s_other_val[0] - s_max[0])
                scale[0] = s_val[0] / (s_val[0] + s_other_val[0])
                other_scale[0] = s_other_val[0] / (s_val[0] + s_other_val[0])
                for d in range(D):
                    V[n, h, d] = T.Cast("float16", T.Cast("float32", V[n, h, d]) * scale[0] + T.Cast("float32", V_other[n, h, d]) * other_scale[0])
                S[n, h] = T.log2(s_val[0] + s_other_val[0]) + s_max[0]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, merge_state_inplace_cpu)


def test_merge_state_inplace_cpu_f32():
    fn = _merge_state_inplace_cpu("float32")

    # fmt: off
    @T.prim_func
    def merge_state_inplace_cpu(v: T.handle, s: T.handle, v_other: T.handle, s_other: T.handle):
        T.func_attr({"tir.is_scheduled": True})
        N, H, D = T.int32(is_size_var=True), T.int32(is_size_var=True), T.int32(is_size_var=True)
        V = T.match_buffer(v, (N, H, D))
        S = T.match_buffer(s, (N, H))
        V_other = T.match_buffer(v_other, (N, H, D))
        S_other = T.match_buffer(s_other, (N, H))
        for n, h in T.grid(N, H):
            with T.sblock("merge"):
                T.reads(S[n, h], S_other[n, h], V[n, h, 0:D], V_other[n, h, 0:D])
                T.writes(V[n, h, 0:D], S[n, h])
                s_val = T.alloc_buffer((1,))
                s_other_val = T.alloc_buffer((1,))
                s_max = T.alloc_buffer((1,))
                scale = T.alloc_buffer((1,))
                other_scale = T.alloc_buffer((1,))
                s_val[0] = S[n, h]
                s_other_val[0] = S_other[n, h]
                s_max[0] = T.max(s_val[0], s_other_val[0])
                s_val[0] = T.exp2(s_val[0] - s_max[0])
                s_other_val[0] = T.exp2(s_other_val[0] - s_max[0])
                scale[0] = s_val[0] / (s_val[0] + s_other_val[0])
                other_scale[0] = s_other_val[0] / (s_val[0] + s_other_val[0])
                for d in range(D):
                    V[n, h, d] = V[n, h, d] * scale[0] + V_other[n, h, d] * other_scale[0]
                S[n, h] = T.log2(s_val[0] + s_other_val[0]) + s_max[0]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, merge_state_inplace_cpu)


def test_merge_state_inplace_gpu_h32_d128_f16():
    target = tvm.target.Target("cuda")
    fn = _merge_state_inplace(32, 128, "float16", target, "tir_merge_state_32_128_float16")

    # fmt: off
    @T.prim_func
    def merge_state_inplace_cpu(v: T.handle, s: T.handle, v_other: T.handle, s_other: T.handle):
        T.func_attr({"global_symbol": "tir_merge_state_32_128_float16", "tir.is_scheduled": True})
        N, H, D = T.int32(is_size_var=True), T.int32(is_size_var=True), T.int32(is_size_var=True)
        V = T.match_buffer(v, (N, H, D), "float16")
        S = T.match_buffer(s, (N, H))
        V_other = T.match_buffer(v_other, (N, H, D), "float16")
        S_other = T.match_buffer(s_other, (N, H))
        for bx in T.thread_binding(N, thread="blockIdx.x"):
            for by in T.thread_binding(1, thread="blockIdx.y"):
                for ty in T.thread_binding(32, thread="threadIdx.y"):
                    for tx in T.thread_binding(32, thread="threadIdx.x"):
                        with T.sblock("merge"):
                            T.reads(S[bx, ty + by * 32], S_other[bx, ty + by * 32], V[bx, ty + by * 32, tx * 4:tx * 4 + 4], V_other[bx, ty + by * 32, tx * 4:tx * 4 + 4])
                            T.writes(V[bx, ty + by * 32, tx * 4:tx * 4 + 4], S[bx, ty + by * 32])
                            s_val = T.alloc_buffer((1,), scope="local")
                            s_other_val = T.alloc_buffer((1,), scope="local")
                            s_max = T.alloc_buffer((1,), scope="local")
                            scale = T.alloc_buffer((1,), scope="local")
                            other_scale = T.alloc_buffer((1,), scope="local")
                            v_vec = T.alloc_buffer((4,), "float16", scope="local")
                            v_other_vec = T.alloc_buffer((4,), "float16", scope="local")
                            s_val[0] = S[bx, ty + by * 32]
                            s_other_val[0] = S_other[bx, ty + by * 32]
                            s_max[0] = T.max(s_val[0], s_other_val[0])
                            s_val[0] = T.exp2(s_val[0] - s_max[0])
                            s_other_val[0] = T.exp2(s_other_val[0] - s_max[0])
                            scale[0] = s_val[0] / (s_val[0] + s_other_val[0])
                            other_scale[0] = s_other_val[0] / (s_val[0] + s_other_val[0])
                            for vec in T.vectorized(4):
                                v_vec[vec] = V[bx, ty + by * 32, tx * 4 + vec]
                            for vec in T.vectorized(4):
                                v_other_vec[vec] = V_other[bx, ty + by * 32, tx * 4 + vec]
                            for vec in range(4):
                                v_vec[vec] = T.Cast("float16", T.Cast("float32", v_vec[vec]) * scale[0] + T.Cast("float32", v_other_vec[vec]) * other_scale[0])
                            for vec in T.vectorized(4):
                                V[bx, ty + by * 32, tx * 4 + vec] = v_vec[vec]
                            S[bx, ty + by * 32] = T.log2(s_val[0] + s_other_val[0]) + s_max[0]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, merge_state_inplace_cpu)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_merge_state_inplace_gpu_h4_d64_f32():
    target = tvm.target.Target("cuda")
    fn = _merge_state_inplace(4, 64, "float32", target, "tir_merge_state_4_64_float32")

    # fmt: off
    @T.prim_func
    def merge_state_inplace_cpu(v: T.handle, s: T.handle, v_other: T.handle, s_other: T.handle):
        T.func_attr({"global_symbol": "tir_merge_state_4_64_float32", "tir.is_scheduled": True})
        N, H, D = T.int32(is_size_var=True), T.int32(is_size_var=True), T.int32(is_size_var=True)
        V = T.match_buffer(v, (N, H, D))
        S = T.match_buffer(s, (N, H))
        V_other = T.match_buffer(v_other, (N, H, D))
        S_other = T.match_buffer(s_other, (N, H))
        for bx in T.thread_binding(N, thread="blockIdx.x"):
            for by in T.thread_binding(1, thread="blockIdx.y"):
                for ty in T.thread_binding(4, thread="threadIdx.y"):
                    for tx in T.thread_binding(16, thread="threadIdx.x"):
                        with T.sblock("merge"):
                            T.reads(S[bx, ty + by * 4], S_other[bx, ty + by * 4], V[bx, ty + by * 4, tx * 4:tx * 4 + 4], V_other[bx, ty + by * 4, tx * 4:tx * 4 + 4])
                            T.writes(V[bx, ty + by * 4, tx * 4:tx * 4 + 4], S[bx, ty + by * 4])
                            s_val = T.alloc_buffer((1,), scope="local")
                            s_other_val = T.alloc_buffer((1,), scope="local")
                            s_max = T.alloc_buffer((1,), scope="local")
                            scale = T.alloc_buffer((1,), scope="local")
                            other_scale = T.alloc_buffer((1,), scope="local")
                            v_vec = T.alloc_buffer((4,), scope="local")
                            v_other_vec = T.alloc_buffer((4,), scope="local")
                            s_val[0] = S[bx, ty + by * 4]
                            s_other_val[0] = S_other[bx, ty + by * 4]
                            s_max[0] = T.max(s_val[0], s_other_val[0])
                            s_val[0] = T.exp2(s_val[0] - s_max[0])
                            s_other_val[0] = T.exp2(s_other_val[0] - s_max[0])
                            scale[0] = s_val[0] / (s_val[0] + s_other_val[0])
                            other_scale[0] = s_other_val[0] / (s_val[0] + s_other_val[0])
                            for vec in T.vectorized(4):
                                v_vec[vec] = V[bx, ty + by * 4, tx * 4 + vec]
                            for vec in T.vectorized(4):
                                v_other_vec[vec] = V_other[bx, ty + by * 4, tx * 4 + vec]
                            for vec in range(4):
                                v_vec[vec] = v_vec[vec] * scale[0] + v_other_vec[vec] * other_scale[0]
                            for vec in T.vectorized(4):
                                V[bx, ty + by * 4, tx * 4 + vec] = v_vec[vec]
                            S[bx, ty + by * 4] = T.log2(s_val[0] + s_other_val[0]) + s_max[0]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, merge_state_inplace_cpu)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_merge_state_inplace_gpu_h8_d128_f16():
    target = tvm.target.Target("cuda")
    fn = _merge_state_inplace(8, 128, "float16", target, "tir_merge_state_8_128_float16")

    # fmt: off
    @T.prim_func
    def merge_state_inplace_cpu(v: T.handle, s: T.handle, v_other: T.handle, s_other: T.handle):
        T.func_attr({"global_symbol": "tir_merge_state_8_128_float16", "tir.is_scheduled": True})
        N, H, D = T.int32(is_size_var=True), T.int32(is_size_var=True), T.int32(is_size_var=True)
        V = T.match_buffer(v, (N, H, D), "float16")
        S = T.match_buffer(s, (N, H))
        V_other = T.match_buffer(v_other, (N, H, D), "float16")
        S_other = T.match_buffer(s_other, (N, H))
        for bx in T.thread_binding(N, thread="blockIdx.x"):
            for by in T.thread_binding(1, thread="blockIdx.y"):
                for ty in T.thread_binding(8, thread="threadIdx.y"):
                    for tx in T.thread_binding(32, thread="threadIdx.x"):
                        with T.sblock("merge"):
                            T.reads(S[bx, ty + by * 8], S_other[bx, ty + by * 8], V[bx, ty + by * 8, tx * 4:tx * 4 + 4], V_other[bx, ty + by * 8, tx * 4:tx * 4 + 4])
                            T.writes(V[bx, ty + by * 8, tx * 4:tx * 4 + 4], S[bx, ty + by * 8])
                            s_val = T.alloc_buffer((1,), scope="local")
                            s_other_val = T.alloc_buffer((1,), scope="local")
                            s_max = T.alloc_buffer((1,), scope="local")
                            scale = T.alloc_buffer((1,), scope="local")
                            other_scale = T.alloc_buffer((1,), scope="local")
                            v_vec = T.alloc_buffer((4,), "float16", scope="local")
                            v_other_vec = T.alloc_buffer((4,), "float16", scope="local")
                            s_val[0] = S[bx, ty + by * 8]
                            s_other_val[0] = S_other[bx, ty + by * 8]
                            s_max[0] = T.max(s_val[0], s_other_val[0])
                            s_val[0] = T.exp2(s_val[0] - s_max[0])
                            s_other_val[0] = T.exp2(s_other_val[0] - s_max[0])
                            scale[0] = s_val[0] / (s_val[0] + s_other_val[0])
                            other_scale[0] = s_other_val[0] / (s_val[0] + s_other_val[0])
                            for vec in T.vectorized(4):
                                v_vec[vec] = V[bx, ty + by * 8, tx * 4 + vec]
                            for vec in T.vectorized(4):
                                v_other_vec[vec] = V_other[bx, ty + by * 8, tx * 4 + vec]
                            for vec in range(4):
                                v_vec[vec] = T.Cast("float16", T.Cast("float32", v_vec[vec]) * scale[0] + T.Cast("float32", v_other_vec[vec]) * other_scale[0])
                            for vec in T.vectorized(4):
                                V[bx, ty + by * 8, tx * 4 + vec] = v_vec[vec]
                            S[bx, ty + by * 8] = T.log2(s_val[0] + s_other_val[0]) + s_max[0]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, merge_state_inplace_cpu)


if __name__ == "__main__":
    tvm.testing.main()
