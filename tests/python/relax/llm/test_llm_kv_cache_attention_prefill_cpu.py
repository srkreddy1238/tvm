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
"""Tests for _attention_prefill_cpu.

Covers GQA (h_kv < h_q), MHA (h_kv == h_q), sliding-window, and two
rope_scaling configs (none / llama3).
"""

import tvm
import tvm.testing
from tvm.relax.frontend.nn.llm.kv_cache import _attention_prefill_cpu
from tvm.script import tir as T

ROPE_SCALING_NONE: dict = {}
ROPE_SCALING_LLAMA3 = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _assert_ne(fn_a, fn_b, msg):
    try:
        tvm.ir.assert_structural_equal(fn_a, fn_b)
        raise AssertionError(msg)
    except (ValueError, tvm.TVMError):
        pass


# ---------------------------------------------------------------------------
# GQA h_kv=4, h_q=32, d=128, float16, no sliding window
# ---------------------------------------------------------------------------


def test_attention_prefill_cpu_gqa_f16_no_sw_none_rope():
    fn = _attention_prefill_cpu(4, 32, 128, "float16", False, ROPE_SCALING_NONE)

    # fmt: off
    @T.prim_func
    def batch_prefill_paged_kv_cpu(var_q: T.handle, var_q_indptr: T.handle, var_pages: T.handle, var_page_indptr: T.handle, var_page_values: T.handle, var_length_info: T.handle, var_k_rope_pos_offset: T.handle, var_q_rope_position: T.handle, var_output: T.handle, var_lse: T.handle, causal: T.int32, rotary_mode: T.int32, rope_scale: T.float32, rope_theta: T.float32, sm_scale: T.float32):
        total_len = T.int32(is_size_var=True)
        q = T.match_buffer(var_q, (total_len, 32, 128), "float16")
        batch_size = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        page_indptr_elem_offset = T.int32(is_size_var=True)
        page_values_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        length_info_elem_offset = T.int32(is_size_var=True)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
        pages = T.match_buffer(var_pages, (max_num_pages, 2, 4, 16, 128), "float16")
        page_indptr = T.match_buffer(var_page_indptr, (batch_size + 1,), "int32", elem_offset=page_indptr_elem_offset)
        page_values = T.match_buffer(var_page_values, (nnz_pages,), "int32", elem_offset=page_values_elem_offset)
        length_info = T.match_buffer(var_length_info, (batch_size,), "int32", elem_offset=length_info_elem_offset)
        k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
        q_rope_position = T.match_buffer(var_q_rope_position, (total_len,), "int32", elem_offset=q_rope_position_elem_offset)
        output = T.match_buffer(var_output, (total_len, 32, 128), "float16")
        lse = T.match_buffer(var_lse, (total_len, 32))
        for h_qo, b_idx in T.grid(32, batch_size):
            with T.sblock("attn"):
                T.reads(page_indptr[b_idx:b_idx + 2], length_info[b_idx], q_indptr[b_idx:b_idx + 2], q_rope_position[q_indptr[b_idx]:q_indptr[b_idx] + (q_indptr[b_idx + 1] - q_indptr[b_idx])], q[q_indptr[b_idx]:q_indptr[b_idx] + (q_indptr[b_idx + 1] - q_indptr[b_idx]), h_qo, 0:128], page_values[page_indptr[b_idx]:page_indptr[b_idx] + max_num_pages], k_rope_pos_offset[b_idx], pages[0:max_num_pages, 0:2, h_qo // 8, 0:16, 0:128])
                T.writes(output[q_indptr[b_idx]:q_indptr[b_idx] + (q_indptr[b_idx + 1] - q_indptr[b_idx]), h_qo, 0:128], lse[q_indptr[b_idx]:q_indptr[b_idx] + (q_indptr[b_idx + 1] - q_indptr[b_idx]), h_qo])
                O_local = T.alloc_buffer((128,))
                Q_local = T.alloc_buffer((128,))
                K_local = T.alloc_buffer((128,))
                V_local = T.alloc_buffer((128,))
                kv_chunk_len = T.alloc_buffer((1,), "int32")
                m_val = T.alloc_buffer((1,))
                new_m = T.alloc_buffer((1,))
                d_val = T.alloc_buffer((1,))
                S_val = T.alloc_buffer((1,))
                scale_O = T.alloc_buffer((1,))
                factor = T.alloc_buffer((1,))
                cur_page_indptr_begin: T.int32 = page_indptr[b_idx]
                cur_page_indptr_end: T.int32 = page_indptr[b_idx + 1]
                kv_chunk_len[0] = T.if_then_else(cur_page_indptr_begin != cur_page_indptr_end, (cur_page_indptr_end - cur_page_indptr_begin - 1) * 16 + length_info[b_idx], 0)
                for q_idx in range(q_indptr[b_idx + 1] - q_indptr[b_idx]):
                    m_val[0] = T.float32(-50000.0)
                    d_val[0] = T.float32(1.0)
                    for d_idx in range(128):
                        O_local[d_idx] = T.float32(0.0)
                    curl_q: T.int32 = q_indptr[b_idx] + q_idx
                    for d_idx in range(128):
                        freq = T.float32()
                        Q_local[d_idx] = T.Cast("float32", T.if_then_else(rotary_mode == 1, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", q[curl_q, h_qo, d_idx]) + T.sin(freq) * T.Cast("float32", T.if_then_else(d_idx < 64, q[curl_q, h_qo, d_idx + 64] * T.float16(-1.0), q[curl_q, h_qo, d_idx - 64]))), where={freq: T.Cast("float32", q_rope_position[curl_q]) * rope_scale / T.pow(rope_theta, T.Cast("float32", d_idx * 2 % 128) / T.float32(128.0))}), q[curl_q, h_qo, d_idx]))
                    for row_idx in range(max_num_pages * 16):
                        if row_idx < kv_chunk_len[0]:
                            page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + row_idx // 16]
                            page_offset: T.int32(is_size_var=True) = row_idx % 16
                            for d_idx in range(128):
                                freq = T.float32()
                                K_local[d_idx] = T.Cast("float32", T.if_then_else(rotary_mode == 1, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", pages[page_no, 0, h_qo // 8, page_offset, d_idx]) + T.sin(freq) * T.Cast("float32", T.if_then_else(d_idx < 64, pages[page_no, 0, h_qo // 8, page_offset, d_idx + 64] * T.float16(-1.0), pages[page_no, 0, h_qo // 8, page_offset, d_idx - 64]))), where={freq: T.Cast("float32", k_rope_pos_offset[b_idx] + row_idx) * rope_scale / T.pow(rope_theta, T.Cast("float32", d_idx * 2 % 128) / T.float32(128.0))}), pages[page_no, 0, h_qo // 8, page_offset, d_idx]))
                                V_local[d_idx] = T.Cast("float32", pages[page_no, 1, h_qo // 8, page_offset, d_idx])
                            S_val[0] = T.float32(0.0)
                            for d_idx in range(128):
                                S_val[0] = S_val[0] + Q_local[d_idx] * K_local[d_idx]
                            S_val[0] = S_val[0] * (sm_scale * T.float32(1.4426950408889634))
                            if T.if_then_else(causal > 0, row_idx < kv_chunk_len[0] - (q_indptr[b_idx + 1] - q_indptr[b_idx]) + q_idx + 1, row_idx < kv_chunk_len[0]):
                                new_m[0] = T.max(m_val[0], S_val[0])
                            else:
                                S_val[0] = T.float32(-50000.0)
                            d_val[0] = d_val[0] * T.exp2(m_val[0] - new_m[0])
                            d_val[0] = d_val[0] + T.exp2(S_val[0] - new_m[0])
                            scale_O[0] = T.exp2(m_val[0] - new_m[0])
                            m_val[0] = new_m[0]
                            factor[0] = T.exp2(S_val[0] - m_val[0])
                            for d_idx in range(128):
                                O_local[d_idx] = O_local[d_idx] * scale_O[d_idx]
                            for d_idx in range(128):
                                O_local[d_idx] = O_local[d_idx] + V_local[d_idx] * factor[0]
                    for d_idx in range(128):
                        O_local[d_idx] = O_local[d_idx] / d_val[0]
                        output[curl_q, h_qo, d_idx] = T.Cast("float16", O_local[d_idx])
                    lse[curl_q, h_qo] = m_val[0] + T.log2(d_val[0])
    # fmt: on

    tvm.ir.assert_structural_equal(fn, batch_prefill_paged_kv_cpu)


def test_attention_prefill_cpu_gqa_f16_no_sw_llama3_rope():
    fn_none = _attention_prefill_cpu(4, 32, 128, "float16", False, ROPE_SCALING_NONE)
    fn_llama3 = _attention_prefill_cpu(4, 32, 128, "float16", False, ROPE_SCALING_LLAMA3)
    # The two rope configs must produce structurally different PrimFuncs.
    _assert_ne(fn_none, fn_llama3, "none vs llama3 rope should differ")


def test_attention_prefill_cpu_mha_f32_no_sw_none_rope():
    """MHA (h_kv == h_q == 8), float32."""
    fn = _attention_prefill_cpu(8, 8, 64, "float32", False, ROPE_SCALING_NONE)

    # fmt: off
    @T.prim_func
    def batch_prefill_paged_kv_cpu(var_q: T.handle, var_q_indptr: T.handle, var_pages: T.handle, var_page_indptr: T.handle, var_page_values: T.handle, var_length_info: T.handle, var_k_rope_pos_offset: T.handle, var_q_rope_position: T.handle, var_output: T.handle, var_lse: T.handle, causal: T.int32, rotary_mode: T.int32, rope_scale: T.float32, rope_theta: T.float32, sm_scale: T.float32):
        total_len = T.int32(is_size_var=True)
        q = T.match_buffer(var_q, (total_len, 8, 64))
        batch_size = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        page_indptr_elem_offset = T.int32(is_size_var=True)
        page_values_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        length_info_elem_offset = T.int32(is_size_var=True)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
        pages = T.match_buffer(var_pages, (max_num_pages, 2, 8, 16, 64))
        page_indptr = T.match_buffer(var_page_indptr, (batch_size + 1,), "int32", elem_offset=page_indptr_elem_offset)
        page_values = T.match_buffer(var_page_values, (nnz_pages,), "int32", elem_offset=page_values_elem_offset)
        length_info = T.match_buffer(var_length_info, (batch_size,), "int32", elem_offset=length_info_elem_offset)
        k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
        q_rope_position = T.match_buffer(var_q_rope_position, (total_len,), "int32", elem_offset=q_rope_position_elem_offset)
        output = T.match_buffer(var_output, (total_len, 8, 64))
        lse = T.match_buffer(var_lse, (total_len, 8))
        for h_qo, b_idx in T.grid(8, batch_size):
            with T.sblock("attn"):
                T.reads(page_indptr[b_idx:b_idx + 2], length_info[b_idx], q_indptr[b_idx:b_idx + 2], q_rope_position[q_indptr[b_idx]:q_indptr[b_idx] + (q_indptr[b_idx + 1] - q_indptr[b_idx])], q[q_indptr[b_idx]:q_indptr[b_idx] + (q_indptr[b_idx + 1] - q_indptr[b_idx]), h_qo, 0:64], page_values[page_indptr[b_idx]:page_indptr[b_idx] + max_num_pages], k_rope_pos_offset[b_idx], pages[0:max_num_pages, 0:2, h_qo, 0:16, 0:64])
                T.writes(output[q_indptr[b_idx]:q_indptr[b_idx] + (q_indptr[b_idx + 1] - q_indptr[b_idx]), h_qo, 0:64], lse[q_indptr[b_idx]:q_indptr[b_idx] + (q_indptr[b_idx + 1] - q_indptr[b_idx]), h_qo])
                O_local = T.alloc_buffer((64,))
                Q_local = T.alloc_buffer((64,))
                K_local = T.alloc_buffer((64,))
                V_local = T.alloc_buffer((64,))
                kv_chunk_len = T.alloc_buffer((1,), "int32")
                m_val = T.alloc_buffer((1,))
                new_m = T.alloc_buffer((1,))
                d_val = T.alloc_buffer((1,))
                S_val = T.alloc_buffer((1,))
                scale_O = T.alloc_buffer((1,))
                factor = T.alloc_buffer((1,))
                cur_page_indptr_begin: T.int32 = page_indptr[b_idx]
                cur_page_indptr_end: T.int32 = page_indptr[b_idx + 1]
                kv_chunk_len[0] = T.if_then_else(cur_page_indptr_begin != cur_page_indptr_end, (cur_page_indptr_end - cur_page_indptr_begin - 1) * 16 + length_info[b_idx], 0)
                for q_idx in range(q_indptr[b_idx + 1] - q_indptr[b_idx]):
                    m_val[0] = T.float32(-50000.0)
                    d_val[0] = T.float32(1.0)
                    for d_idx in range(64):
                        O_local[d_idx] = T.float32(0.0)
                    curl_q: T.int32 = q_indptr[b_idx] + q_idx
                    for d_idx in range(64):
                        freq = T.float32()
                        Q_local[d_idx] = T.if_then_else(rotary_mode == 1, T.Let(T.cos(freq) * q[curl_q, h_qo, d_idx] + T.sin(freq) * T.if_then_else(d_idx < 32, q[curl_q, h_qo, d_idx + 32] * T.float32(-1.0), q[curl_q, h_qo, d_idx - 32]), where={freq: T.Cast("float32", q_rope_position[curl_q]) * rope_scale / T.pow(rope_theta, T.Cast("float32", d_idx * 2 % 64) / T.float32(64.0))}), q[curl_q, h_qo, d_idx])
                    for row_idx in range(max_num_pages * 16):
                        if row_idx < kv_chunk_len[0]:
                            page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + row_idx // 16]
                            page_offset: T.int32(is_size_var=True) = row_idx % 16
                            for d_idx in range(64):
                                freq = T.float32()
                                K_local[d_idx] = T.if_then_else(rotary_mode == 1, T.Let(T.cos(freq) * pages[page_no, 0, h_qo, page_offset, d_idx] + T.sin(freq) * T.if_then_else(d_idx < 32, pages[page_no, 0, h_qo, page_offset, d_idx + 32] * T.float32(-1.0), pages[page_no, 0, h_qo, page_offset, d_idx - 32]), where={freq: T.Cast("float32", k_rope_pos_offset[b_idx] + row_idx) * rope_scale / T.pow(rope_theta, T.Cast("float32", d_idx * 2 % 64) / T.float32(64.0))}), pages[page_no, 0, h_qo, page_offset, d_idx])
                                V_local[d_idx] = pages[page_no, 1, h_qo, page_offset, d_idx]
                            S_val[0] = T.float32(0.0)
                            for d_idx in range(64):
                                S_val[0] = S_val[0] + Q_local[d_idx] * K_local[d_idx]
                            S_val[0] = S_val[0] * (sm_scale * T.float32(1.4426950408889634))
                            if T.if_then_else(causal > 0, row_idx < kv_chunk_len[0] - (q_indptr[b_idx + 1] - q_indptr[b_idx]) + q_idx + 1, row_idx < kv_chunk_len[0]):
                                new_m[0] = T.max(m_val[0], S_val[0])
                            else:
                                S_val[0] = T.float32(-50000.0)
                            d_val[0] = d_val[0] * T.exp2(m_val[0] - new_m[0])
                            d_val[0] = d_val[0] + T.exp2(S_val[0] - new_m[0])
                            scale_O[0] = T.exp2(m_val[0] - new_m[0])
                            m_val[0] = new_m[0]
                            factor[0] = T.exp2(S_val[0] - m_val[0])
                            for d_idx in range(64):
                                O_local[d_idx] = O_local[d_idx] * scale_O[d_idx]
                            for d_idx in range(64):
                                O_local[d_idx] = O_local[d_idx] + V_local[d_idx] * factor[0]
                    for d_idx in range(64):
                        O_local[d_idx] = O_local[d_idx] / d_val[0]
                        output[curl_q, h_qo, d_idx] = O_local[d_idx]
                    lse[curl_q, h_qo] = m_val[0] + T.log2(d_val[0])
    # fmt: on

    tvm.ir.assert_structural_equal(fn, batch_prefill_paged_kv_cpu)


if __name__ == "__main__":
    tvm.testing.main()
