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
"""Tests for _compact_kv_copy_cpu and _compact_kv_copy (GPU)."""

import tvm
import tvm.testing
from tvm.relax.frontend.nn.llm.kv_cache import (
    _compact_kv_copy,
    _compact_kv_copy_cpu,
)
from tvm.script import tir as T

# ---------------------------------------------------------------------------
# _compact_kv_copy_cpu
# ---------------------------------------------------------------------------


def test_compact_kv_copy_cpu_h4_d128_f16():
    fn = _compact_kv_copy_cpu(4, 128, "float16")

    # fmt: off
    @T.prim_func
    def compact_kv_copy_cpu(var_pages: T.handle, var_copy_length_indptr: T.handle, var_copy_src_dst_pos: T.handle, batch_size: T.int32):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, 4, 16, 128), "float16")
        copy_length_indptr = T.match_buffer(var_copy_length_indptr, (batch_size + 1,), "int32", offset_factor=1)
        total_copy_length = T.int32()
        copy_src_dst_pos = T.match_buffer(var_copy_src_dst_pos, (2, total_copy_length), "int32", offset_factor=1)
        with T.sblock("root"):
            T.reads()
            T.writes()
            for bhd_o, bhd_i in T.grid(batch_size * 64, 8):
                b: T.int32 = (bhd_o * 8 + bhd_i) // 512
                h: T.int32 = (bhd_o * 8 + bhd_i) // 128 % 4
                d: T.int32 = (bhd_o * 8 + bhd_i) % 128
                if bhd_o * 8 + bhd_i < batch_size * 4 * 128:
                    for i in range(copy_length_indptr[b + 1] - copy_length_indptr[b]):
                        src_pos: T.int32 = copy_src_dst_pos[0, copy_length_indptr[b] + i]
                        dst_pos: T.int32 = copy_src_dst_pos[1, copy_length_indptr[b] + i]
                        pages[dst_pos // 16, 0, h, dst_pos % 16, d] = pages[src_pos // 16, 0, h, src_pos % 16, d]
                        pages[dst_pos // 16, 1, h, dst_pos % 16, d] = pages[src_pos // 16, 1, h, src_pos % 16, d]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, compact_kv_copy_cpu, map_free_vars=True)


def test_compact_kv_copy_cpu_h8_d64_f32():
    fn = _compact_kv_copy_cpu(8, 64, "float32")

    # fmt: off
    @T.prim_func
    def compact_kv_copy_cpu(var_pages: T.handle, var_copy_length_indptr: T.handle, var_copy_src_dst_pos: T.handle, batch_size: T.int32):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, 8, 16, 64))
        copy_length_indptr = T.match_buffer(var_copy_length_indptr, (batch_size + 1,), "int32", offset_factor=1)
        total_copy_length = T.int32()
        copy_src_dst_pos = T.match_buffer(var_copy_src_dst_pos, (2, total_copy_length), "int32", offset_factor=1)
        with T.sblock("root"):
            T.reads()
            T.writes()
            for bhd_o, bhd_i in T.grid(batch_size * 64, 8):
                b: T.int32 = (bhd_o * 8 + bhd_i) // 512
                h: T.int32 = (bhd_o * 8 + bhd_i) // 64 % 8
                d: T.int32 = (bhd_o * 8 + bhd_i) % 64
                if bhd_o * 8 + bhd_i < batch_size * 8 * 64:
                    for i in range(copy_length_indptr[b + 1] - copy_length_indptr[b]):
                        src_pos: T.int32 = copy_src_dst_pos[0, copy_length_indptr[b] + i]
                        dst_pos: T.int32 = copy_src_dst_pos[1, copy_length_indptr[b] + i]
                        pages[dst_pos // 16, 0, h, dst_pos % 16, d] = pages[src_pos // 16, 0, h, src_pos % 16, d]
                        pages[dst_pos // 16, 1, h, dst_pos % 16, d] = pages[src_pos // 16, 1, h, src_pos % 16, d]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, compact_kv_copy_cpu, map_free_vars=True)


def test_compact_kv_copy_cpu_h1_d128_f16():
    fn = _compact_kv_copy_cpu(1, 128, "float16")

    # fmt: off
    @T.prim_func
    def compact_kv_copy_cpu(var_pages: T.handle, var_copy_length_indptr: T.handle, var_copy_src_dst_pos: T.handle, batch_size: T.int32):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, 1, 16, 128), "float16")
        copy_length_indptr = T.match_buffer(var_copy_length_indptr, (batch_size + 1,), "int32", offset_factor=1)
        total_copy_length = T.int32()
        copy_src_dst_pos = T.match_buffer(var_copy_src_dst_pos, (2, total_copy_length), "int32", offset_factor=1)
        with T.sblock("root"):
            T.reads()
            T.writes()
            for bhd_o, bhd_i in T.grid(batch_size * 16, 8):
                b: T.int32 = (bhd_o * 8 + bhd_i) // 128
                h: T.int32 = 0
                d: T.int32 = (bhd_o * 8 + bhd_i) % 128
                if bhd_o * 8 + bhd_i < batch_size * 128:
                    for i in range(copy_length_indptr[b + 1] - copy_length_indptr[b]):
                        src_pos: T.int32 = copy_src_dst_pos[0, copy_length_indptr[b] + i]
                        dst_pos: T.int32 = copy_src_dst_pos[1, copy_length_indptr[b] + i]
                        pages[dst_pos // 16, 0, h, dst_pos % 16, d] = pages[src_pos // 16, 0, h, src_pos % 16, d]
                        pages[dst_pos // 16, 1, h, dst_pos % 16, d] = pages[src_pos // 16, 1, h, src_pos % 16, d]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, compact_kv_copy_cpu, map_free_vars=True)


def test_compact_kv_copy_gpu_h4_d128_f16():
    target = tvm.target.Target("cuda")
    fn = _compact_kv_copy(4, 128, "float16", target)

    # fmt: off
    @T.prim_func
    def compact_kv_copy(var_pages: T.handle, var_copy_length_indptr: T.handle, var_copy_src_dst_pos: T.handle, batch_size: T.int32):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, 4, 16, 128), "float16", elem_offset=pages_elem_offset)
        copy_length_indptr_elem_offset = T.int32()
        copy_length_indptr = T.match_buffer(var_copy_length_indptr, (batch_size + 1,), "int32", elem_offset=copy_length_indptr_elem_offset)
        total_copy_length = T.int32()
        copy_src_dst_pos_elem_offset = T.int32()
        copy_src_dst_pos = T.match_buffer(var_copy_src_dst_pos, (2, total_copy_length), "int32", elem_offset=copy_src_dst_pos_elem_offset)
        with T.sblock("root"):
            T.reads()
            T.writes()
            for bhd_o in T.thread_binding((batch_size * 512 + 1023) // 1024, thread="blockIdx.x"):
                for bhd_i in T.thread_binding(1024, thread="threadIdx.x"):
                    b: T.int32 = (bhd_o * 1024 + bhd_i) // 512
                    h: T.int32 = (bhd_o * 1024 + bhd_i) // 128 % 4
                    d: T.int32 = (bhd_o * 1024 + bhd_i) % 128
                    if bhd_o * 1024 + bhd_i < batch_size * 4 * 128:
                        for i in range(copy_length_indptr[b + 1] - copy_length_indptr[b]):
                            src_pos: T.int32 = copy_src_dst_pos[0, copy_length_indptr[b] + i]
                            dst_pos: T.int32 = copy_src_dst_pos[1, copy_length_indptr[b] + i]
                            pages[dst_pos // 16, 0, h, dst_pos % 16, d] = pages[src_pos // 16, 0, h, src_pos % 16, d]
                            pages[dst_pos // 16, 1, h, dst_pos % 16, d] = pages[src_pos // 16, 1, h, src_pos % 16, d]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, compact_kv_copy, map_free_vars=True)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_compact_kv_copy_gpu_h8_d64_f32():
    target = tvm.target.Target("cuda")
    fn = _compact_kv_copy(8, 64, "float32", target)

    # fmt: off
    @T.prim_func
    def compact_kv_copy(var_pages: T.handle, var_copy_length_indptr: T.handle, var_copy_src_dst_pos: T.handle, batch_size: T.int32):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, 8, 16, 64), elem_offset=pages_elem_offset)
        copy_length_indptr_elem_offset = T.int32()
        copy_length_indptr = T.match_buffer(var_copy_length_indptr, (batch_size + 1,), "int32", elem_offset=copy_length_indptr_elem_offset)
        total_copy_length = T.int32()
        copy_src_dst_pos_elem_offset = T.int32()
        copy_src_dst_pos = T.match_buffer(var_copy_src_dst_pos, (2, total_copy_length), "int32", elem_offset=copy_src_dst_pos_elem_offset)
        with T.sblock("root"):
            T.reads()
            T.writes()
            for bhd_o in T.thread_binding((batch_size * 512 + 1023) // 1024, thread="blockIdx.x"):
                for bhd_i in T.thread_binding(1024, thread="threadIdx.x"):
                    b: T.int32 = (bhd_o * 1024 + bhd_i) // 512
                    h: T.int32 = (bhd_o * 1024 + bhd_i) // 64 % 8
                    d: T.int32 = (bhd_o * 1024 + bhd_i) % 64
                    if bhd_o * 1024 + bhd_i < batch_size * 8 * 64:
                        for i in range(copy_length_indptr[b + 1] - copy_length_indptr[b]):
                            src_pos: T.int32 = copy_src_dst_pos[0, copy_length_indptr[b] + i]
                            dst_pos: T.int32 = copy_src_dst_pos[1, copy_length_indptr[b] + i]
                            pages[dst_pos // 16, 0, h, dst_pos % 16, d] = pages[src_pos // 16, 0, h, src_pos % 16, d]
                            pages[dst_pos // 16, 1, h, dst_pos % 16, d] = pages[src_pos // 16, 1, h, src_pos % 16, d]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, compact_kv_copy, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
