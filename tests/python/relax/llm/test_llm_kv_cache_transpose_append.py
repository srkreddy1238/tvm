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
"""Tests for _kv_cache_transpose_append and _kv_cache_transpose_append_mla.

Each test calls the generator function and asserts structural equality
against the expected TVMScript captured from the IR printer (Pass 1).
"""

import tvm
import tvm.testing
from tvm.relax.frontend.nn.llm.kv_cache import (
    _kv_cache_transpose_append,
    _kv_cache_transpose_append_mla,
)
from tvm.script import tir as T

# ---------------------------------------------------------------------------
# _kv_cache_transpose_append
# ---------------------------------------------------------------------------


def test_kv_cache_transpose_append_h4_d128_f16():
    fn = _kv_cache_transpose_append(4, 128, "float16")

    # fmt: off
    @T.prim_func
    def tir_kv_cache_transpose_append(var_pages: T.handle, var_k_data: T.handle, var_v_data: T.handle, var_position_map: T.handle):
        T.func_attr({"tir.noalias": True})
        ntoken = T.SizeVar("num_tokens_excluding_cache", "int64")
        num_pages = T.int64()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, 4, 16, 128), "float16", elem_offset=pages_elem_offset)
        k_data = T.match_buffer(var_k_data, (ntoken, 4, 128), "float16")
        v_data = T.match_buffer(var_v_data, (ntoken, 4, 128), "float16")
        position_map_elem_offset = T.int32()
        position_map = T.match_buffer(var_position_map, (ntoken,), "int32", elem_offset=position_map_elem_offset)
        for global_pos, h, f in T.grid(ntoken, 4, 128):
            if position_map[global_pos] != T.int32(-1):
                with T.sblock("k_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                    T.writes(pages[position_map[vgpos] // 16, 0, vh, position_map[vgpos] % 16, vf])
                    position: T.int32 = position_map[vgpos]
                    pages[position // 16, 0, vh, position % 16, vf] = k_data[vgpos, vh, vf]
                with T.sblock("v_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], v_data[vgpos, vh, vf])
                    T.writes(pages[position_map[vgpos] // 16, 1, vh, position_map[vgpos] % 16, vf])
                    position: T.int32 = position_map[vgpos]
                    pages[position // 16, 1, vh, position % 16, vf] = v_data[vgpos, vh, vf]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, tir_kv_cache_transpose_append)


def test_kv_cache_transpose_append_h8_d64_f32():
    fn = _kv_cache_transpose_append(8, 64, "float32")

    # fmt: off
    @T.prim_func
    def tir_kv_cache_transpose_append(var_pages: T.handle, var_k_data: T.handle, var_v_data: T.handle, var_position_map: T.handle):
        T.func_attr({"tir.noalias": True})
        ntoken = T.SizeVar("num_tokens_excluding_cache", "int64")
        num_pages = T.int64()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, 8, 16, 64), elem_offset=pages_elem_offset)
        k_data = T.match_buffer(var_k_data, (ntoken, 8, 64))
        v_data = T.match_buffer(var_v_data, (ntoken, 8, 64))
        position_map_elem_offset = T.int32()
        position_map = T.match_buffer(var_position_map, (ntoken,), "int32", elem_offset=position_map_elem_offset)
        for global_pos, h, f in T.grid(ntoken, 8, 64):
            if position_map[global_pos] != T.int32(-1):
                with T.sblock("k_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                    T.writes(pages[position_map[vgpos] // 16, 0, vh, position_map[vgpos] % 16, vf])
                    position: T.int32 = position_map[vgpos]
                    pages[position // 16, 0, vh, position % 16, vf] = k_data[vgpos, vh, vf]
                with T.sblock("v_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], v_data[vgpos, vh, vf])
                    T.writes(pages[position_map[vgpos] // 16, 1, vh, position_map[vgpos] % 16, vf])
                    position: T.int32 = position_map[vgpos]
                    pages[position // 16, 1, vh, position % 16, vf] = v_data[vgpos, vh, vf]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, tir_kv_cache_transpose_append)


def test_kv_cache_transpose_append_h1_d128_f16():
    """MQA-like: single KV head."""
    fn = _kv_cache_transpose_append(1, 128, "float16")

    # fmt: off
    @T.prim_func
    def tir_kv_cache_transpose_append(var_pages: T.handle, var_k_data: T.handle, var_v_data: T.handle, var_position_map: T.handle):
        T.func_attr({"tir.noalias": True})
        ntoken = T.SizeVar("num_tokens_excluding_cache", "int64")
        num_pages = T.int64()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, 1, 16, 128), "float16", elem_offset=pages_elem_offset)
        k_data = T.match_buffer(var_k_data, (ntoken, 1, 128), "float16")
        v_data = T.match_buffer(var_v_data, (ntoken, 1, 128), "float16")
        position_map_elem_offset = T.int32()
        position_map = T.match_buffer(var_position_map, (ntoken,), "int32", elem_offset=position_map_elem_offset)
        for global_pos, h, f in T.grid(ntoken, 1, 128):
            if position_map[global_pos] != T.int32(-1):
                with T.sblock("k_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                    T.writes(pages[position_map[vgpos] // 16, 0, vh, position_map[vgpos] % 16, vf])
                    position: T.int32 = position_map[vgpos]
                    pages[position // 16, 0, vh, position % 16, vf] = k_data[vgpos, vh, vf]
                with T.sblock("v_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], v_data[vgpos, vh, vf])
                    T.writes(pages[position_map[vgpos] // 16, 1, vh, position_map[vgpos] % 16, vf])
                    position: T.int32 = position_map[vgpos]
                    pages[position // 16, 1, vh, position % 16, vf] = v_data[vgpos, vh, vf]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, tir_kv_cache_transpose_append)


def test_kv_cache_transpose_append_mla_d512_f16():
    fn = _kv_cache_transpose_append_mla(512, "float16")

    # fmt: off
    @T.prim_func
    def tir_kv_cache_transpose_append_mla(var_pages: T.handle, var_kv_data: T.handle, var_position_map: T.handle):
        T.func_attr({"tir.noalias": True})
        ntoken = T.SizeVar("num_tokens_excluding_cache", "int64")
        num_pages = T.int64()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 16, 512), "float16", elem_offset=pages_elem_offset)
        kv_data = T.match_buffer(var_kv_data, (ntoken, 512), "float16")
        position_map_elem_offset = T.int32()
        position_map = T.match_buffer(var_position_map, (ntoken,), "int32", elem_offset=position_map_elem_offset)
        for global_pos, f in T.grid(ntoken, 512):
            if position_map[global_pos] != T.int32(-1):
                with T.sblock("k_transpose_append"):
                    vgpos, vf = T.axis.remap("SS", [global_pos, f])
                    T.reads(position_map[vgpos], kv_data[vgpos, vf])
                    T.writes(pages[position_map[vgpos] // 16, position_map[vgpos] % 16, vf])
                    position: T.int32 = position_map[vgpos]
                    pages[position // 16, position % 16, vf] = kv_data[vgpos, vf]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, tir_kv_cache_transpose_append_mla)


def test_kv_cache_transpose_append_mla_d256_f32():
    fn = _kv_cache_transpose_append_mla(256, "float32")

    # fmt: off
    @T.prim_func
    def tir_kv_cache_transpose_append_mla(var_pages: T.handle, var_kv_data: T.handle, var_position_map: T.handle):
        T.func_attr({"tir.noalias": True})
        ntoken = T.SizeVar("num_tokens_excluding_cache", "int64")
        num_pages = T.int64()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 16, 256), elem_offset=pages_elem_offset)
        kv_data = T.match_buffer(var_kv_data, (ntoken, 256))
        position_map_elem_offset = T.int32()
        position_map = T.match_buffer(var_position_map, (ntoken,), "int32", elem_offset=position_map_elem_offset)
        for global_pos, f in T.grid(ntoken, 256):
            if position_map[global_pos] != T.int32(-1):
                with T.sblock("k_transpose_append"):
                    vgpos, vf = T.axis.remap("SS", [global_pos, f])
                    T.reads(position_map[vgpos], kv_data[vgpos, vf])
                    T.writes(pages[position_map[vgpos] // 16, position_map[vgpos] % 16, vf])
                    position: T.int32 = position_map[vgpos]
                    pages[position // 16, position % 16, vf] = kv_data[vgpos, vf]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, tir_kv_cache_transpose_append_mla)


if __name__ == "__main__":
    tvm.testing.main()
