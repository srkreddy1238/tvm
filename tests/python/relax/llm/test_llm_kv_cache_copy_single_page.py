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
"""Tests for _copy_single_page_cpu, _copy_single_page (GPU), and _copy_single_page_mla (GPU)."""

import tvm
import tvm.testing
from tvm.relax.frontend.nn.llm.kv_cache import (
    _copy_single_page,
    _copy_single_page_cpu,
    _copy_single_page_mla,
)
from tvm.script import tir as T

# ---------------------------------------------------------------------------
# _copy_single_page_cpu
# ---------------------------------------------------------------------------


def test_copy_single_page_cpu_h4_ps16_d128_f16():
    fn = _copy_single_page_cpu(4, 16, 128, "float16")

    # fmt: off
    @T.prim_func
    def copy_single_page_cpu(var_pages: T.handle, src_page_id: T.int64, tgt_page_id: T.int64, copy_length: T.int64):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, 4, 16, 128), "float16")
        for b, t in T.grid(copy_length * T.int64(512), 1):
            with T.sblock("copy"):
                vh = T.axis.spatial(4, T.Cast("int32", (b + T.Cast("int64", t)) // (copy_length * T.int64(128))))
                vp = T.axis.spatial(copy_length, (b + T.Cast("int64", t)) % (copy_length * T.int64(128)) // T.int64(128))
                vd = T.axis.spatial(128, T.Cast("int32", (b + T.Cast("int64", t)) % T.int64(128)))
                T.where(b + T.Cast("int64", t) < copy_length * T.int64(4) * T.int64(128))
                T.reads(pages[src_page_id, 0:2, vh, vp, vd])
                T.writes(pages[tgt_page_id, 0:2, vh, vp, vd])
                pages[tgt_page_id, 0, vh, vp, vd] = pages[src_page_id, 0, vh, vp, vd]
                pages[tgt_page_id, 1, vh, vp, vd] = pages[src_page_id, 1, vh, vp, vd]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, copy_single_page_cpu)


def test_copy_single_page_cpu_h8_ps16_d64_f32():
    fn = _copy_single_page_cpu(8, 16, 64, "float32")

    # fmt: off
    @T.prim_func
    def copy_single_page_cpu(var_pages: T.handle, src_page_id: T.int64, tgt_page_id: T.int64, copy_length: T.int64):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, 8, 16, 64))
        for b, t in T.grid(copy_length * T.int64(512), 1):
            with T.sblock("copy"):
                vh = T.axis.spatial(8, T.Cast("int32", (b + T.Cast("int64", t)) // (copy_length * T.int64(64))))
                vp = T.axis.spatial(copy_length, (b + T.Cast("int64", t)) % (copy_length * T.int64(64)) // T.int64(64))
                vd = T.axis.spatial(64, T.Cast("int32", (b + T.Cast("int64", t)) % T.int64(64)))
                T.where(b + T.Cast("int64", t) < copy_length * T.int64(8) * T.int64(64))
                T.reads(pages[src_page_id, 0:2, vh, vp, vd])
                T.writes(pages[tgt_page_id, 0:2, vh, vp, vd])
                pages[tgt_page_id, 0, vh, vp, vd] = pages[src_page_id, 0, vh, vp, vd]
                pages[tgt_page_id, 1, vh, vp, vd] = pages[src_page_id, 1, vh, vp, vd]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, copy_single_page_cpu)


def test_copy_single_page_cpu_h1_ps16_d128_f16():
    """MQA-like: single KV head."""
    fn = _copy_single_page_cpu(1, 16, 128, "float16")

    # fmt: off
    @T.prim_func
    def copy_single_page_cpu(var_pages: T.handle, src_page_id: T.int64, tgt_page_id: T.int64, copy_length: T.int64):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, 1, 16, 128), "float16")
        for b, t in T.grid(copy_length * T.int64(128), 1):
            with T.sblock("copy"):
                vh = T.axis.spatial(1, T.Cast("int32", (b + T.Cast("int64", t)) // (copy_length * T.int64(128))))
                vp = T.axis.spatial(copy_length, (b + T.Cast("int64", t)) % (copy_length * T.int64(128)) // T.int64(128))
                vd = T.axis.spatial(128, T.Cast("int32", (b + T.Cast("int64", t)) % T.int64(128)))
                T.where(b + T.Cast("int64", t) < copy_length * T.int64(128))
                T.reads(pages[src_page_id, 0:2, vh, vp, vd])
                T.writes(pages[tgt_page_id, 0:2, vh, vp, vd])
                pages[tgt_page_id, 0, vh, vp, vd] = pages[src_page_id, 0, vh, vp, vd]
                pages[tgt_page_id, 1, vh, vp, vd] = pages[src_page_id, 1, vh, vp, vd]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, copy_single_page_cpu)


def test_copy_single_page_gpu_h4_ps16_d128_f16():
    target = tvm.target.Target("cuda")
    fn = _copy_single_page(4, 16, 128, "float16", target)

    # fmt: off
    @T.prim_func
    def copy_single_page(var_pages: T.handle, src_page_id: T.int64, tgt_page_id: T.int64, copy_length: T.int64):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, 4, 16, 128), "float16", elem_offset=pages_elem_offset)
        for b in T.thread_binding((copy_length * T.int64(512) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for t in T.thread_binding(1024, thread="threadIdx.x"):
                with T.sblock("copy"):
                    vh = T.axis.spatial(4, T.Cast("int32", (b * T.int64(1024) + T.Cast("int64", t)) // (copy_length * T.int64(128))))
                    vp = T.axis.spatial(copy_length, (b * T.int64(1024) + T.Cast("int64", t)) % (copy_length * T.int64(128)) // T.int64(128))
                    vd = T.axis.spatial(128, T.Cast("int32", (b * T.int64(1024) + T.Cast("int64", t)) % T.int64(128)))
                    T.where(b * T.int64(1024) + T.Cast("int64", t) < copy_length * T.int64(4) * T.int64(128))
                    T.reads(pages[src_page_id, 0:2, vh, vp, vd])
                    T.writes(pages[tgt_page_id, 0:2, vh, vp, vd])
                    pages[tgt_page_id, 0, vh, vp, vd] = pages[src_page_id, 0, vh, vp, vd]
                    pages[tgt_page_id, 1, vh, vp, vd] = pages[src_page_id, 1, vh, vp, vd]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, copy_single_page, map_free_vars=True)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_copy_single_page_gpu_h8_ps16_d64_f32():
    target = tvm.target.Target("cuda")
    fn = _copy_single_page(8, 16, 64, "float32", target)

    # fmt: off
    @T.prim_func
    def copy_single_page(var_pages: T.handle, src_page_id: T.int64, tgt_page_id: T.int64, copy_length: T.int64):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, 8, 16, 64), elem_offset=pages_elem_offset)
        for b in T.thread_binding((copy_length * T.int64(512) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for t in T.thread_binding(1024, thread="threadIdx.x"):
                with T.sblock("copy"):
                    vh = T.axis.spatial(8, T.Cast("int32", (b * T.int64(1024) + T.Cast("int64", t)) // (copy_length * T.int64(64))))
                    vp = T.axis.spatial(copy_length, (b * T.int64(1024) + T.Cast("int64", t)) % (copy_length * T.int64(64)) // T.int64(64))
                    vd = T.axis.spatial(64, T.Cast("int32", (b * T.int64(1024) + T.Cast("int64", t)) % T.int64(64)))
                    T.where(b * T.int64(1024) + T.Cast("int64", t) < copy_length * T.int64(8) * T.int64(64))
                    T.reads(pages[src_page_id, 0:2, vh, vp, vd])
                    T.writes(pages[tgt_page_id, 0:2, vh, vp, vd])
                    pages[tgt_page_id, 0, vh, vp, vd] = pages[src_page_id, 0, vh, vp, vd]
                    pages[tgt_page_id, 1, vh, vp, vd] = pages[src_page_id, 1, vh, vp, vd]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, copy_single_page, map_free_vars=True)


# ---------------------------------------------------------------------------
# _copy_single_page_mla  (GPU)
# ---------------------------------------------------------------------------


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_copy_single_page_mla_gpu_ps16_d512_f16():
    target = tvm.target.Target("cuda")
    fn = _copy_single_page_mla(16, 512, "float16", target)

    # fmt: off
    @T.prim_func
    def copy_single_page_mla(var_pages: T.handle, src_page_id: T.int64, tgt_page_id: T.int64, copy_length: T.int64):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 16, 512), "float16", elem_offset=pages_elem_offset)
        for b in T.thread_binding((copy_length * T.int64(512) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for t in T.thread_binding(1024, thread="threadIdx.x"):
                with T.sblock("copy"):
                    vp = T.axis.spatial(copy_length, (b * T.int64(1024) + T.Cast("int64", t)) // T.int64(512))
                    vd = T.axis.spatial(512, T.Cast("int32", (b * T.int64(1024) + T.Cast("int64", t)) % T.int64(512)))
                    T.where(b * T.int64(1024) + T.Cast("int64", t) < copy_length * T.int64(512))
                    T.reads(pages[src_page_id, vp, vd])
                    T.writes(pages[tgt_page_id, vp, vd])
                    pages[tgt_page_id, vp, vd] = pages[src_page_id, vp, vd]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, copy_single_page_mla, map_free_vars=True)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_copy_single_page_mla_gpu_ps16_d256_f32():
    target = tvm.target.Target("cuda")
    fn = _copy_single_page_mla(16, 256, "float32", target)

    # fmt: off
    @T.prim_func
    def copy_single_page_mla(var_pages: T.handle, src_page_id: T.int64, tgt_page_id: T.int64, copy_length: T.int64):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 16, 256), elem_offset=pages_elem_offset)
        for b in T.thread_binding((copy_length * T.int64(256) + T.int64(1023)) // T.int64(1024), thread="blockIdx.x"):
            for t in T.thread_binding(1024, thread="threadIdx.x"):
                with T.sblock("copy"):
                    vp = T.axis.spatial(copy_length, (b * T.int64(1024) + T.Cast("int64", t)) // T.int64(256))
                    vd = T.axis.spatial(256, T.Cast("int32", (b * T.int64(1024) + T.Cast("int64", t)) % T.int64(256)))
                    T.where(b * T.int64(1024) + T.Cast("int64", t) < copy_length * T.int64(256))
                    T.reads(pages[src_page_id, vp, vd])
                    T.writes(pages[tgt_page_id, vp, vd])
                    pages[tgt_page_id, vp, vd] = pages[src_page_id, vp, vd]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, copy_single_page_mla, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
