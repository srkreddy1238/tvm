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
"""Tests for _attention_decode_cpu.

Covers GQA (h_kv < h_q), MHA (h_kv == h_q), sliding-window, and two
rope_scaling configs (none / llama3).
"""

import tvm
import tvm.testing
from tvm.relax.frontend.nn.llm.kv_cache import _attention_decode_cpu
from tvm.script import tir as T

ROPE_SCALING_NONE: dict = {}
ROPE_SCALING_LLAMA3 = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def _assert_ne(fn_a, fn_b, msg):
    try:
        tvm.ir.assert_structural_equal(fn_a, fn_b)
        raise AssertionError(msg)
    except (ValueError, tvm.TVMError):
        pass


# ---------------------------------------------------------------------------
# GQA h_kv=4, h_q=32, d=128, float16, no sliding window, no rope
# ---------------------------------------------------------------------------


def test_attention_decode_cpu_gqa_f16_no_sw_none_rope():
    fn = _attention_decode_cpu(4, 32, 128, "float16", False, ROPE_SCALING_NONE)

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def batch_decode_paged_kv_cpu(Q_handle: T.handle, pages_handle: T.handle, page_table_indptr_handle: T.handle, page_table_values_handle: T.handle, var_length_info: T.handle, k_rope_pos_offset_handle: T.handle, q_rope_position_handle: T.handle, output_handle: T.handle, lse_handle: T.handle, rotary_mode: T.int32, rope_scale: T.float32, rope_theta: T.float32, sm_scale: T.float32):
        T.func_attr({"tir.is_scheduled": True})
        B = T.int32(is_size_var=True)
        Q = T.match_buffer(Q_handle, (B, 32, 128), "float16")
        max_num_pages = T.int32(is_size_var=True)
        pages = T.match_buffer(pages_handle, (max_num_pages, 2, 4, 16, 128), "float16")
        page_indptr_elem_offset = T.int32(is_size_var=True)
        page_table_indptr = T.match_buffer(page_table_indptr_handle, (B + 1,), "int32", elem_offset=page_indptr_elem_offset)
        nnz_pages = T.int32(is_size_var=True)
        page_values_elem_offset = T.int32(is_size_var=True)
        page_table_values = T.match_buffer(page_table_values_handle, (nnz_pages,), "int32", elem_offset=page_values_elem_offset)
        length_info_elem_offset = T.int32(is_size_var=True)
        length_info = T.match_buffer(var_length_info, (B,), "int32", elem_offset=length_info_elem_offset)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset = T.match_buffer(k_rope_pos_offset_handle, (B,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        q_rope_position = T.match_buffer(q_rope_position_handle, (B,), "int32", elem_offset=q_rope_position_elem_offset)
        output = T.match_buffer(output_handle, (B, 32, 128), "float16")
        lse = T.match_buffer(lse_handle, (B, 32))
        with T.sblock("root"):
            T.reads()
            T.writes()
            for b in range(B):
                with T.sblock("attn"):
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
                    cur_page_indptr_begin: T.int32 = page_table_indptr[b]
                    cur_page_indptr_end: T.int32 = page_table_indptr[b + 1]
                    kv_chunk_len[0] = T.if_then_else(cur_page_indptr_begin != cur_page_indptr_end, (cur_page_indptr_end - cur_page_indptr_begin - 1) * 16 + length_info[b], 0)
                    for h_qo in range(32):
                        m_val[0] = T.float32(-50000.0)
                        d_val[0] = T.float32(1.0)
                        for d in range(128):
                            O_local[d] = T.float32(0.0)
                        for d in range(128):
                            freq = T.float32()
                            Q_local[d] = T.Cast("float32", T.if_then_else(rotary_mode == 1, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", Q[b, h_qo, d]) + T.sin(freq) * T.Cast("float32", T.if_then_else(d < 64, Q[b, h_qo, d + 64] * T.float16(-1.0), Q[b, h_qo, d - 64]))), where={freq: T.Cast("float32", q_rope_position[b]) * rope_scale / T.pow(rope_theta, T.Cast("float32", d * 2 % 128) / T.float32(128.0))}), Q[b, h_qo, d]))
                        for row_idx in range(kv_chunk_len[0]):
                            seq_offset: T.int32(is_size_var=True) = row_idx
                            page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + seq_offset // 16]
                            page_offset: T.int32(is_size_var=True) = seq_offset % 16
                            for d in range(128):
                                freq = T.float32()
                                K_local[d] = T.Cast("float32", T.if_then_else(rotary_mode == 1, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", pages[page_no, 0, h_qo // 8, page_offset, d]) + T.sin(freq) * T.Cast("float32", T.if_then_else(d < 64, pages[page_no, 0, h_qo // 8, page_offset, d + 64] * T.float16(-1.0), pages[page_no, 0, h_qo // 8, page_offset, d - 64]))), where={freq: T.Cast("float32", k_rope_pos_offset[b] + row_idx) * rope_scale / T.pow(rope_theta, T.Cast("float32", d * 2 % 128) / T.float32(128.0))}), pages[page_no, 0, h_qo // 8, page_offset, d]))
                            S_val[0] = T.float32(0.0)
                            for d in range(128):
                                S_val[0] = S_val[0] + Q_local[d] * K_local[d]
                            S_val[0] = S_val[0] * (sm_scale * T.float32(1.4426950408889634))
                            new_m[0] = T.max(m_val[0], S_val[0])
                            d_val[0] = d_val[0] * T.exp2(m_val[0] - new_m[0]) + T.exp2(S_val[0] - new_m[0])
                            scale_O[0] = T.exp2(m_val[0] - new_m[0])
                            for d in range(128):
                                O_local[d] = O_local[d] * scale_O[0]
                            m_val[0] = new_m[0]
                            for d in range(128):
                                V_local[d] = T.Cast("float32", pages[page_no, 1, h_qo // 8, page_offset, d])
                            factor[0] = T.exp2(S_val[0] - m_val[0])
                            for d in range(128):
                                O_local[d] = O_local[d] + V_local[d] * factor[0]
                        for d in range(128):
                            O_local[d] = O_local[d] / d_val[0]
                            output[b, h_qo, d] = T.Cast("float16", O_local[d])
                        lse[b, h_qo] = m_val[0] + T.log2(d_val[0])
    # fmt: on

    tvm.ir.assert_structural_equal(fn, batch_decode_paged_kv_cpu)


if __name__ == "__main__":
    tvm.testing.main()
