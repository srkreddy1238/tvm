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
"""Tests for _kv_cache_debug_get_kv and _kv_cache_debug_get_kv_mla."""

import tvm
import tvm.testing
from tvm.relax.frontend.nn.llm.kv_cache import (
    _kv_cache_debug_get_kv,
    _kv_cache_debug_get_kv_mla,
)
from tvm.script import tir as T

# ---------------------------------------------------------------------------
# _kv_cache_debug_get_kv
# ---------------------------------------------------------------------------


def test_kv_cache_debug_get_kv_L4_h4_d128_f16():
    fn = _kv_cache_debug_get_kv(4, 4, 128, "float16")

    # fmt: off
    @T.prim_func
    def tir_kv_cache_debug_get_kv(var_pages: T.handle, var_position_map: T.handle, var_k_data: T.handle, var_v_data: T.handle, layer_id: T.int64):
        T.func_attr({"tir.noalias": True})
        num_pages, page_size = T.int64(), T.int64(is_size_var=True)
        pages = T.match_buffer(var_pages, (num_pages, 2, 4, page_size, 128), "float16", offset_factor=1)
        seqlen = T.int64(is_size_var=True)
        position_map = T.match_buffer(var_position_map, (seqlen,), "int32", offset_factor=1)
        k_data = T.match_buffer(var_k_data, (4, seqlen, 4, 128), "float16")
        v_data = T.match_buffer(var_v_data, (4, seqlen, 4, 128), "float16")
        for p, h, d in T.grid(seqlen, 4, 128):
            with T.sblock("copy0"):
                vp, vh, vd = T.axis.remap("SSS", [p, h, d])
                T.reads(position_map[vp], pages[T.Cast("int64", position_map[vp]) // page_size, 0:2, vh, T.Cast("int64", position_map[vp]) % page_size, vd])
                T.writes(k_data[layer_id, vp, vh, vd], v_data[layer_id, vp, vh, vd])
                position: T.int32 = position_map[vp]
                k_data[layer_id, vp, vh, vd] = pages[T.Cast("int64", position) // page_size, 0, vh, T.Cast("int64", position) % page_size, vd]
                v_data[layer_id, vp, vh, vd] = pages[T.Cast("int64", position) // page_size, 1, vh, T.Cast("int64", position) % page_size, vd]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, tir_kv_cache_debug_get_kv)


def test_kv_cache_debug_get_kv_L2_h8_d64_f32():
    fn = _kv_cache_debug_get_kv(2, 8, 64, "float32")

    # fmt: off
    @T.prim_func
    def tir_kv_cache_debug_get_kv(var_pages: T.handle, var_position_map: T.handle, var_k_data: T.handle, var_v_data: T.handle, layer_id: T.int64):
        T.func_attr({"tir.noalias": True})
        num_pages, page_size = T.int64(), T.int64(is_size_var=True)
        pages = T.match_buffer(var_pages, (num_pages, 2, 8, page_size, 64), offset_factor=1)
        seqlen = T.int64(is_size_var=True)
        position_map = T.match_buffer(var_position_map, (seqlen,), "int32", offset_factor=1)
        k_data = T.match_buffer(var_k_data, (2, seqlen, 8, 64))
        v_data = T.match_buffer(var_v_data, (2, seqlen, 8, 64))
        for p, h, d in T.grid(seqlen, 8, 64):
            with T.sblock("copy0"):
                vp, vh, vd = T.axis.remap("SSS", [p, h, d])
                T.reads(position_map[vp], pages[T.Cast("int64", position_map[vp]) // page_size, 0:2, vh, T.Cast("int64", position_map[vp]) % page_size, vd])
                T.writes(k_data[layer_id, vp, vh, vd], v_data[layer_id, vp, vh, vd])
                position: T.int32 = position_map[vp]
                k_data[layer_id, vp, vh, vd] = pages[T.Cast("int64", position) // page_size, 0, vh, T.Cast("int64", position) % page_size, vd]
                v_data[layer_id, vp, vh, vd] = pages[T.Cast("int64", position) // page_size, 1, vh, T.Cast("int64", position) % page_size, vd]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, tir_kv_cache_debug_get_kv)


def test_kv_cache_debug_get_kv_L1_h1_d128_f16():
    fn = _kv_cache_debug_get_kv(1, 1, 128, "float16")

    # fmt: off
    @T.prim_func
    def tir_kv_cache_debug_get_kv(var_pages: T.handle, var_position_map: T.handle, var_k_data: T.handle, var_v_data: T.handle, layer_id: T.int64):
        T.func_attr({"tir.noalias": True})
        num_pages, page_size = T.int64(), T.int64(is_size_var=True)
        pages = T.match_buffer(var_pages, (num_pages, 2, 1, page_size, 128), "float16", offset_factor=1)
        seqlen = T.int64(is_size_var=True)
        position_map = T.match_buffer(var_position_map, (seqlen,), "int32", offset_factor=1)
        k_data = T.match_buffer(var_k_data, (1, seqlen, 1, 128), "float16")
        v_data = T.match_buffer(var_v_data, (1, seqlen, 1, 128), "float16")
        for p, h, d in T.grid(seqlen, 1, 128):
            with T.sblock("copy0"):
                vp, vh, vd = T.axis.remap("SSS", [p, h, d])
                T.reads(position_map[vp], pages[T.Cast("int64", position_map[vp]) // page_size, 0:2, vh, T.Cast("int64", position_map[vp]) % page_size, vd])
                T.writes(k_data[layer_id, vp, vh, vd], v_data[layer_id, vp, vh, vd])
                position: T.int32 = position_map[vp]
                k_data[layer_id, vp, vh, vd] = pages[T.Cast("int64", position) // page_size, 0, vh, T.Cast("int64", position) % page_size, vd]
                v_data[layer_id, vp, vh, vd] = pages[T.Cast("int64", position) // page_size, 1, vh, T.Cast("int64", position) % page_size, vd]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, tir_kv_cache_debug_get_kv)


def test_kv_cache_debug_get_kv_mla_L4_d512_f16():
    fn = _kv_cache_debug_get_kv_mla(4, 512, "float16")

    # fmt: off
    @T.prim_func
    def tir_kv_cache_debug_get_kv_mla(var_pages: T.handle, var_position_map: T.handle, var_compressed_kv_with_k_pe_data: T.handle, layer_id: T.int64):
        T.func_attr({"tir.noalias": True})
        num_pages, page_size = T.int64(), T.int64(is_size_var=True)
        pages = T.match_buffer(var_pages, (num_pages, page_size, 512), "float16", offset_factor=1)
        seqlen = T.int64(is_size_var=True)
        position_map = T.match_buffer(var_position_map, (seqlen,), "int32", offset_factor=1)
        compressed_kv_with_k_pe_data = T.match_buffer(var_compressed_kv_with_k_pe_data, (4, seqlen, 512), "float16")
        for p, d in T.grid(seqlen, 512):
            with T.sblock("copy0"):
                vp, vd = T.axis.remap("SS", [p, d])
                T.reads(position_map[vp], pages[T.Cast("int64", position_map[vp]) // page_size, T.Cast("int64", position_map[vp]) % page_size, vd])
                T.writes(compressed_kv_with_k_pe_data[layer_id, vp, vd])
                position: T.int32 = position_map[vp]
                compressed_kv_with_k_pe_data[layer_id, vp, vd] = pages[T.Cast("int64", position) // page_size, T.Cast("int64", position) % page_size, vd]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, tir_kv_cache_debug_get_kv_mla)


def test_kv_cache_debug_get_kv_mla_L2_d256_f32():
    fn = _kv_cache_debug_get_kv_mla(2, 256, "float32")

    # fmt: off
    @T.prim_func
    def tir_kv_cache_debug_get_kv_mla(var_pages: T.handle, var_position_map: T.handle, var_compressed_kv_with_k_pe_data: T.handle, layer_id: T.int64):
        T.func_attr({"tir.noalias": True})
        num_pages, page_size = T.int64(), T.int64(is_size_var=True)
        pages = T.match_buffer(var_pages, (num_pages, page_size, 256), offset_factor=1)
        seqlen = T.int64(is_size_var=True)
        position_map = T.match_buffer(var_position_map, (seqlen,), "int32", offset_factor=1)
        compressed_kv_with_k_pe_data = T.match_buffer(var_compressed_kv_with_k_pe_data, (2, seqlen, 256))
        for p, d in T.grid(seqlen, 256):
            with T.sblock("copy0"):
                vp, vd = T.axis.remap("SS", [p, d])
                T.reads(position_map[vp], pages[T.Cast("int64", position_map[vp]) // page_size, T.Cast("int64", position_map[vp]) % page_size, vd])
                T.writes(compressed_kv_with_k_pe_data[layer_id, vp, vd])
                position: T.int32 = position_map[vp]
                compressed_kv_with_k_pe_data[layer_id, vp, vd] = pages[T.Cast("int64", position) // page_size, T.Cast("int64", position) % page_size, vd]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, tir_kv_cache_debug_get_kv_mla)


if __name__ == "__main__":
    tvm.testing.main()
