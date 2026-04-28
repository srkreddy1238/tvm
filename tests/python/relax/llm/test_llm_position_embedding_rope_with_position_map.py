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
"""Tests for llama_rope_with_position_map.

Each test calls ``llama_rope_with_position_map`` directly (it returns a TIR
PrimFunc) and compares the result against a hand-written expected PrimFunc
in TVMScript using structural equality.

Config used across all tests
-----------------------------
* num_q_heads = 2, num_kv_heads = 2  (MHA)
* head_dim = 8, dtype = "float16"
* theta = 10000.0, scale = 1.0
"""

import tvm
import tvm.testing
from tvm.relax.frontend.nn.llm.position_embedding import llama_rope_with_position_map
from tvm.script import tir as T

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
NUM_Q = 2
NUM_KV = 2
HEAD_DIM = 8
FUSED = NUM_Q + NUM_KV * 2  # 6
DTYPE = "float16"
THETA = 10000.0
SCALE = 1.0

ROPE_SCALING_NONE: dict = {}
ROPE_SCALING_GPTJ = {"rope_type": "gptj"}
ROPE_SCALING_LLAMA3 = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}
ROPE_SCALING_LONGROPE = {
    "rope_type": "longrope",
    "max_position_embeddings": 131072,
    "original_max_position_embeddings": 4096,
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build(rope_scaling, rotary_dim=None):
    return llama_rope_with_position_map(
        theta=THETA,
        scale=SCALE,
        head_dim=HEAD_DIM,
        num_q_heads=NUM_Q,
        num_kv_heads=NUM_KV,
        dtype=DTYPE,
        rope_scaling=rope_scaling,
        rotary_dim=rotary_dim,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rope_with_position_map_none_scaling():
    """rope_scaling={} - structural equality with expected TVMScript."""
    fn = _build(ROPE_SCALING_NONE)

    # fmt: off
    @T.prim_func
    def fused_rope(var_qkv: T.handle, var_position_map: T.handle, var_q: T.handle, var_k: T.handle, var_v: T.handle, apply_rope: T.int32):
        T.func_attr({"op_pattern": 8, "tir.noalias": True})
        seq_len = T.int32()
        position_map_elem_offset = T.int32()
        qkv = T.match_buffer(var_qkv, (seq_len, 6, 8), "float16")
        position_map = T.match_buffer(var_position_map, (seq_len,), "int32", elem_offset=position_map_elem_offset)
        q = T.match_buffer(var_q, (seq_len, 2, 8), "float16")
        k = T.match_buffer(var_k, (seq_len, 2, 8), "float16")
        v = T.match_buffer(var_v, (seq_len, 2, 8), "float16")
        for iters_0, iters_1, iters_2 in T.grid(seq_len, 6, 8):
            with T.sblock("llama_fused_rope"):
                s, h, d = T.axis.remap("SSS", [iters_0, iters_1, iters_2])
                T.reads(position_map[s], qkv[s, h, d - 4:d - 4 + 9])
                T.writes(q[s, h, d], k[s, h - 2, d], v[s, h - 4, d])
                if h < 2:
                    freq = T.float32()
                    q[s, h, d] = T.if_then_else(apply_rope > 0 and d < 8, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", qkv[s, h, d]) + T.sin(freq) * T.Cast("float32", T.if_then_else(d < 4, qkv[s, h, d + 4] * T.float16(-1.0), qkv[s, h, d - 4]))), where={freq: T.Cast("float32", position_map[s]) / T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 8) / T.float32(8.0))}), qkv[s, h, d])
                else:
                    if h < 4:
                        freq = T.float32()
                        k[s, h - 2, d] = T.if_then_else(apply_rope > 0 and d < 8, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", qkv[s, h, d]) + T.sin(freq) * T.Cast("float32", T.if_then_else(d < 4, qkv[s, h, d + 4] * T.float16(-1.0), qkv[s, h, d - 4]))), where={freq: T.Cast("float32", position_map[s]) / T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 8) / T.float32(8.0))}), qkv[s, h, d])
                    else:
                        v[s, h - 4, d] = qkv[s, h, d]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, fused_rope)


def test_rope_with_position_map_gptj_scaling():
    """rope_type='gptj' - structural equality with expected TVMScript."""
    fn = _build(ROPE_SCALING_GPTJ)

    # fmt: off
    @T.prim_func
    def fused_rope(var_qkv: T.handle, var_position_map: T.handle, var_q: T.handle, var_k: T.handle, var_v: T.handle, apply_rope: T.int32):
        T.func_attr({"op_pattern": 8, "tir.noalias": True})
        seq_len = T.int32()
        position_map_elem_offset = T.int32()
        qkv = T.match_buffer(var_qkv, (seq_len, 6, 8), "float16")
        position_map = T.match_buffer(var_position_map, (seq_len,), "int32", elem_offset=position_map_elem_offset)
        q = T.match_buffer(var_q, (seq_len, 2, 8), "float16")
        k = T.match_buffer(var_k, (seq_len, 2, 8), "float16")
        v = T.match_buffer(var_v, (seq_len, 2, 8), "float16")
        for iters_0, iters_1, iters_2 in T.grid(seq_len, 6, 8):
            with T.sblock("llama_fused_rope"):
                s, h, d = T.axis.remap("SSS", [iters_0, iters_1, iters_2])
                T.reads(position_map[s], qkv[s, h, d - 1:d - 1 + 3])
                T.writes(q[s, h, d], k[s, h - 2, d], v[s, h - 4, d])
                if h < 2:
                    freq = T.float32()
                    q[s, h, d] = T.if_then_else(apply_rope > 0 and d < 8, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", qkv[s, h, d]) + T.sin(freq) * T.Cast("float32", T.if_then_else(d % 2 == 0, qkv[s, h, d + 1] * T.float16(-1.0), qkv[s, h, d - 1]))), where={freq: T.Cast("float32", position_map[s]) / T.pow(T.float32(10000.0), T.Cast("float32", 2 * (d // 2) % 8) / T.float32(8.0))}), qkv[s, h, d])
                else:
                    if h < 4:
                        freq = T.float32()
                        k[s, h - 2, d] = T.if_then_else(apply_rope > 0 and d < 8, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", qkv[s, h, d]) + T.sin(freq) * T.Cast("float32", T.if_then_else(d % 2 == 0, qkv[s, h, d + 1] * T.float16(-1.0), qkv[s, h, d - 1]))), where={freq: T.Cast("float32", position_map[s]) / T.pow(T.float32(10000.0), T.Cast("float32", 2 * (d // 2) % 8) / T.float32(8.0))}), qkv[s, h, d])
                    else:
                        v[s, h - 4, d] = qkv[s, h, d]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, fused_rope)


def test_rope_with_position_map_llama3_scaling():
    """rope_type='llama3' - structural equality with expected TVMScript."""
    fn = _build(ROPE_SCALING_LLAMA3)

    # fmt: off
    @T.prim_func
    def fused_rope(var_qkv: T.handle, var_position_map: T.handle, var_q: T.handle, var_k: T.handle, var_v: T.handle, apply_rope: T.int32):
        T.func_attr({"op_pattern": 8, "tir.noalias": True})
        seq_len = T.int32()
        position_map_elem_offset = T.int32()
        qkv = T.match_buffer(var_qkv, (seq_len, 6, 8), "float16")
        position_map = T.match_buffer(var_position_map, (seq_len,), "int32", elem_offset=position_map_elem_offset)
        q = T.match_buffer(var_q, (seq_len, 2, 8), "float16")
        k = T.match_buffer(var_k, (seq_len, 2, 8), "float16")
        v = T.match_buffer(var_v, (seq_len, 2, 8), "float16")
        for iters_0, iters_1, iters_2 in T.grid(seq_len, 6, 8):
            with T.sblock("llama_fused_rope"):
                s, h, d = T.axis.remap("SSS", [iters_0, iters_1, iters_2])
                T.reads(position_map[s], qkv[s, h, d - 4:d - 4 + 9])
                T.writes(q[s, h, d], k[s, h - 2, d], v[s, h - 4, d])
                if h < 2:
                    orig_freq = T.float32()
                    smoothed_freq = T.float32()
                    q[s, h, d] = T.if_then_else(apply_rope > 0 and d < 8, T.Let(T.Let(T.Cast("float16", T.cos(smoothed_freq) * T.Cast("float32", qkv[s, h, d]) + T.sin(smoothed_freq) * T.Cast("float32", T.if_then_else(d < 4, qkv[s, h, d + 4] * T.float16(-1.0), qkv[s, h, d - 4]))), where={smoothed_freq: T.Cast("float32", position_map[s]) * ((T.float32(1.0) - T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331)))) * orig_freq * T.float32(0.125) + T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331))) * orig_freq)}), where={orig_freq: T.float32(1.0) / T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 8) / T.float32(8.0))}), qkv[s, h, d])
                else:
                    if h < 4:
                        orig_freq = T.float32()
                        smoothed_freq = T.float32()
                        k[s, h - 2, d] = T.if_then_else(apply_rope > 0 and d < 8, T.Let(T.Let(T.Cast("float16", T.cos(smoothed_freq) * T.Cast("float32", qkv[s, h, d]) + T.sin(smoothed_freq) * T.Cast("float32", T.if_then_else(d < 4, qkv[s, h, d + 4] * T.float16(-1.0), qkv[s, h, d - 4]))), where={smoothed_freq: T.Cast("float32", position_map[s]) * ((T.float32(1.0) - T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331)))) * orig_freq * T.float32(0.125) + T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331))) * orig_freq)}), where={orig_freq: T.float32(1.0) / T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 8) / T.float32(8.0))}), qkv[s, h, d])
                    else:
                        v[s, h - 4, d] = qkv[s, h, d]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, fused_rope)


def test_rope_with_position_map_longrope_scaling():
    """rope_type='longrope' - structural equality with expected TVMScript."""
    fn = _build(ROPE_SCALING_LONGROPE)

    # fmt: off
    @T.prim_func
    def fused_rope_longrope_scaling(var_qkv: T.handle, var_position_map: T.handle, var_q: T.handle, var_k: T.handle, var_v: T.handle, ext_factors: T.Buffer((8,), "float32")):
        T.func_attr({"op_pattern": 8, "tir.noalias": True})
        seq_len = T.int64()
        position_map_elem_offset = T.int64()
        qkv = T.match_buffer(var_qkv, (seq_len, 6, 8), "float16")
        position_map = T.match_buffer(var_position_map, (seq_len,), "int32", elem_offset=position_map_elem_offset)
        q = T.match_buffer(var_q, (seq_len, 2, 8), "float16")
        k = T.match_buffer(var_k, (seq_len, 2, 8), "float16")
        v = T.match_buffer(var_v, (seq_len, 2, 8), "float16")
        if seq_len > T.int64(4096):
            for iters_0, iters_1, iters_2 in T.grid(seq_len, 6, 8):
                with T.sblock("llama_fused_rope"):
                    s, h, d = T.axis.remap("SSS", [iters_0, iters_1, iters_2])
                    long_factors = T.Buffer((4,), data=ext_factors.data)
                    T.reads(position_map[s], long_factors[d % 4], qkv[s, h, d - 4:d - 4 + 9])
                    T.writes(q[s, h, d], k[s, h - 2, d], v[s, h - 4, d])
                    if h < 2:
                        freq = T.float32()
                        q[s, h, d] = T.if_then_else(d < 8, T.Let(T.Cast("float16", T.cos(freq) * T.float32(1.1902380714238083) * T.Cast("float32", qkv[s, h, d]) + T.sin(freq) * T.float32(1.1902380714238083) * T.Cast("float32", T.if_then_else(d < 4, qkv[s, h, d + 4] * T.float16(-1.0), qkv[s, h, d - 4]))), where={freq: T.Cast("float32", position_map[s]) / (long_factors[d % 4] * T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 8) / T.float32(8.0)))}), qkv[s, h, d])
                    else:
                        if h < 4:
                            freq = T.float32()
                            k[s, h - 2, d] = T.if_then_else(d < 8, T.Let(T.Cast("float16", T.cos(freq) * T.float32(1.1902380714238083) * T.Cast("float32", qkv[s, h, d]) + T.sin(freq) * T.float32(1.1902380714238083) * T.Cast("float32", T.if_then_else(d < 4, qkv[s, h, d + 4] * T.float16(-1.0), qkv[s, h, d - 4]))), where={freq: T.Cast("float32", position_map[s]) / (long_factors[d % 4] * T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 8) / T.float32(8.0)))}), qkv[s, h, d])
                        else:
                            v[s, h - 4, d] = qkv[s, h, d]
        else:
            for iters_0, iters_1, iters_2 in T.grid(seq_len, 6, 8):
                with T.sblock("llama_fused_rope"):
                    s, h, d = T.axis.remap("SSS", [iters_0, iters_1, iters_2])
                    short_factors = T.Buffer((4,), data=ext_factors.data, elem_offset=4)
                    T.reads(position_map[s], short_factors[d % 4], qkv[s, h, d - 4:d - 4 + 9])
                    T.writes(q[s, h, d], k[s, h - 2, d], v[s, h - 4, d])
                    if h < 2:
                        freq = T.float32()
                        q[s, h, d] = T.if_then_else(d < 8, T.Let(T.Cast("float16", T.cos(freq) * T.float32(1.1902380714238083) * T.Cast("float32", qkv[s, h, d]) + T.sin(freq) * T.float32(1.1902380714238083) * T.Cast("float32", T.if_then_else(d < 4, qkv[s, h, d + 4] * T.float16(-1.0), qkv[s, h, d - 4]))), where={freq: T.Cast("float32", position_map[s]) / (short_factors[d % 4] * T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 8) / T.float32(8.0)))}), qkv[s, h, d])
                    else:
                        if h < 4:
                            freq = T.float32()
                            k[s, h - 2, d] = T.if_then_else(d < 8, T.Let(T.Cast("float16", T.cos(freq) * T.float32(1.1902380714238083) * T.Cast("float32", qkv[s, h, d]) + T.sin(freq) * T.float32(1.1902380714238083) * T.Cast("float32", T.if_then_else(d < 4, qkv[s, h, d + 4] * T.float16(-1.0), qkv[s, h, d - 4]))), where={freq: T.Cast("float32", position_map[s]) / (short_factors[d % 4] * T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 8) / T.float32(8.0)))}), qkv[s, h, d])
                        else:
                            v[s, h - 4, d] = qkv[s, h, d]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, fused_rope_longrope_scaling)


def test_rope_with_position_map_partial_rotary_dim():
    """Partial rotary_dim = HEAD_DIM // 2 = 4 - structural equality with expected TVMScript."""
    fn = _build(ROPE_SCALING_NONE, rotary_dim=HEAD_DIM // 2)

    # fmt: off
    @T.prim_func
    def fused_rope(var_qkv: T.handle, var_position_map: T.handle, var_q: T.handle, var_k: T.handle, var_v: T.handle, apply_rope: T.int32):
        T.func_attr({"op_pattern": 8, "tir.noalias": True})
        seq_len = T.int32()
        position_map_elem_offset = T.int32()
        qkv = T.match_buffer(var_qkv, (seq_len, 6, 8), "float16")
        position_map = T.match_buffer(var_position_map, (seq_len,), "int32", elem_offset=position_map_elem_offset)
        q = T.match_buffer(var_q, (seq_len, 2, 8), "float16")
        k = T.match_buffer(var_k, (seq_len, 2, 8), "float16")
        v = T.match_buffer(var_v, (seq_len, 2, 8), "float16")
        for iters_0, iters_1, iters_2 in T.grid(seq_len, 6, 8):
            with T.sblock("llama_fused_rope"):
                s, h, d = T.axis.remap("SSS", [iters_0, iters_1, iters_2])
                T.reads(position_map[s], qkv[s, h, d - 2:d - 2 + 5])
                T.writes(q[s, h, d], k[s, h - 2, d], v[s, h - 4, d])
                if h < 2:
                    freq = T.float32()
                    q[s, h, d] = T.if_then_else(apply_rope > 0 and d < 4, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", qkv[s, h, d]) + T.sin(freq) * T.Cast("float32", T.if_then_else(d < 2, qkv[s, h, d + 2] * T.float16(-1.0), qkv[s, h, d - 2]))), where={freq: T.Cast("float32", position_map[s]) / T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 4) / T.float32(4.0))}), qkv[s, h, d])
                else:
                    if h < 4:
                        freq = T.float32()
                        k[s, h - 2, d] = T.if_then_else(apply_rope > 0 and d < 4, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", qkv[s, h, d]) + T.sin(freq) * T.Cast("float32", T.if_then_else(d < 2, qkv[s, h, d + 2] * T.float16(-1.0), qkv[s, h, d - 2]))), where={freq: T.Cast("float32", position_map[s]) / T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 4) / T.float32(4.0))}), qkv[s, h, d])
                    else:
                        v[s, h - 4, d] = qkv[s, h, d]
    # fmt: on

    tvm.ir.assert_structural_equal(fn, fused_rope)


if __name__ == "__main__":
    tvm.testing.main()
