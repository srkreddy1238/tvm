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
"""Tests for tree_attn_with_paged_kv_cache_cpu.

Each test calls ``tree_attn_with_paged_kv_cache_cpu`` directly (it returns a
TIR PrimFunc) and compares the result against an expected PrimFunc written in
TVMScript – exactly the same pattern used throughout
``tests/python/relax/test_frontend_nn_op.py``.

The expected IR was captured from the log produced by::

    python tests/python/relax/print_ir_tree_attn.py 2>&1 \\
        | tee ir_log_tree_attn.txt

Config used across all tests
-----------------------------
* h_kv = 2, h_q = 4, d = 32, dtype = "float16"
"""

import tvm
import tvm.testing
from tvm.relax.frontend.nn.llm.tree_attn import tree_attn_with_paged_kv_cache_cpu
from tvm.script import tir as T

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
H_KV = 2
H_Q = 4
D = 32
DTYPE = "float16"

ROPE_SCALING_NONE: dict = {}
ROPE_SCALING_LLAMA3 = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tree_attn_with_paged_kv_cache_cpu_none_scaling():
    """rope_scaling={} – structural equality with expected TVMScript."""
    fn = tree_attn_with_paged_kv_cache_cpu(H_KV, H_Q, D, DTYPE, ROPE_SCALING_NONE)

    # fmt: off
    @T.prim_func
    def tree_attn_paged_kv_cpu(var_q: T.handle, var_q_indptr: T.handle, var_pages: T.handle, var_page_indptr: T.handle, var_page_values: T.handle, var_length_info: T.handle, var_k_rope_pos_offset: T.handle, var_q_rope_position: T.handle, var_output: T.handle, var_lse: T.handle, rotary_mode: T.int32, rope_scale: T.float32, rope_theta: T.float32, sm_scale: T.float32, tree_order_indptr_handle: T.handle, tree_order_handle: T.handle):
        T.func_attr({"global_symbol": "tree_attn_paged_kv_cpu"})
        batch_size = T.int32(is_size_var=True)
        total_len = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        page_indptr_elem_offset = T.int32(is_size_var=True)
        page_values_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        length_info_elem_offset = T.int32(is_size_var=True)
        tree_order_elem_offset = T.int32(is_size_var=True)
        tree_order_indptr_elem_offset = T.int32(is_size_var=True)
        q = T.match_buffer(var_q, (total_len, 4, 32), "float16")
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
        pages = T.match_buffer(var_pages, (max_num_pages, 2, 2, 16, 32), "float16")
        page_indptr = T.match_buffer(var_page_indptr, (batch_size + 1,), "int32", elem_offset=page_indptr_elem_offset)
        page_values = T.match_buffer(var_page_values, (nnz_pages,), "int32", elem_offset=page_values_elem_offset)
        length_info = T.match_buffer(var_length_info, (batch_size,), "int32", elem_offset=length_info_elem_offset)
        k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
        q_rope_position = T.match_buffer(var_q_rope_position, (total_len,), "int32", elem_offset=q_rope_position_elem_offset)
        output = T.match_buffer(var_output, (total_len, 4, 32), "float16")
        lse = T.match_buffer(var_lse, (total_len, 4))
        tree_order_indptr = T.match_buffer(tree_order_indptr_handle, (batch_size + 1,), "int32", elem_offset=tree_order_indptr_elem_offset)
        total_tree_order_len = T.int32(is_size_var=True)
        tree_order = T.match_buffer(tree_order_handle, (total_tree_order_len, 2), "int32", elem_offset=tree_order_elem_offset)
        assert rotary_mode == 0, "Inline rotary mode is not supported in tree attention."
        for h_qo in range(4):
            for b_idx in range(batch_size):
                with T.sblock("attn"):
                    T.reads()
                    T.writes()
                    O_local = T.alloc_buffer((32,))
                    Q_local = T.alloc_buffer((32,))
                    K_local = T.alloc_buffer((32,))
                    V_local = T.alloc_buffer((32,))
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
                        for d_idx in range(32):
                            O_local[d_idx] = T.float32(0.0)
                        curl_q: T.int32 = q_indptr[b_idx] + q_idx
                        for d_idx in range(32):
                            freq = T.float32()
                            Q_local[d_idx] = T.Cast("float32", T.if_then_else(rotary_mode == 1, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", q[curl_q, h_qo, d_idx]) + T.sin(freq) * T.Cast("float32", T.if_then_else(d_idx < 16, q[curl_q, h_qo, d_idx + 16] * T.float16(-1.0), q[curl_q, h_qo, d_idx - 16]))), where={freq: T.Cast("float32", q_rope_position[curl_q]) * rope_scale / T.pow(rope_theta, T.Cast("float32", d_idx * 2 % 32) / T.float32(32.0))}), q[curl_q, h_qo, d_idx]))
                        for row_idx in range(max_num_pages * 16):
                            if row_idx < kv_chunk_len[0]:
                                page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + row_idx // 16]
                                page_offset: T.int32(is_size_var=True) = row_idx % 16
                                for d_idx in range(32):
                                    freq = T.float32()
                                    K_local[d_idx] = T.Cast("float32", T.if_then_else(rotary_mode == 1, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", pages[page_no, 0, h_qo // 2, page_offset, d_idx]) + T.sin(freq) * T.Cast("float32", T.if_then_else(d_idx < 16, pages[page_no, 0, h_qo // 2, page_offset, d_idx + 16] * T.float16(-1.0), pages[page_no, 0, h_qo // 2, page_offset, d_idx - 16]))), where={freq: T.Cast("float32", k_rope_pos_offset[b_idx] + row_idx) * rope_scale / T.pow(rope_theta, T.Cast("float32", d_idx * 2 % 32) / T.float32(32.0))}), pages[page_no, 0, h_qo // 2, page_offset, d_idx]))
                                    V_local[d_idx] = T.Cast("float32", pages[page_no, 1, h_qo // 2, page_offset, d_idx])
                                S_val[0] = T.float32(0.0)
                                for d_idx in range(32):
                                    S_val[0] = S_val[0] + Q_local[d_idx] * K_local[d_idx]
                                S_val[0] = S_val[0] * (sm_scale * T.float32(1.4426950408889634))
                                if row_idx < kv_chunk_len[0] and (row_idx < kv_chunk_len[0] - (tree_order_indptr[b_idx + 1] - tree_order_indptr[b_idx]) or tree_order[tree_order_indptr[b_idx] + (q_idx + (tree_order_indptr[b_idx + 1] - tree_order_indptr[b_idx]) - (q_indptr[b_idx + 1] - q_indptr[b_idx])), 0] >= tree_order[tree_order_indptr[b_idx] + (row_idx - (kv_chunk_len[0] - (tree_order_indptr[b_idx + 1] - tree_order_indptr[b_idx]))), 0] and tree_order[tree_order_indptr[b_idx] + (q_idx + (tree_order_indptr[b_idx + 1] - tree_order_indptr[b_idx]) - (q_indptr[b_idx + 1] - q_indptr[b_idx])), 0] < tree_order[tree_order_indptr[b_idx] + (row_idx - (kv_chunk_len[0] - (tree_order_indptr[b_idx + 1] - tree_order_indptr[b_idx]))), 1]):
                                    new_m[0] = T.max(m_val[0], S_val[0])
                                else:
                                    S_val[0] = T.float32(-50000.0)
                                d_val[0] = d_val[0] * T.exp2(m_val[0] - new_m[0])
                                d_val[0] = d_val[0] + T.exp2(S_val[0] - new_m[0])
                                scale_O[0] = T.exp2(m_val[0] - new_m[0])
                                m_val[0] = new_m[0]
                                factor[0] = T.exp2(S_val[0] - m_val[0])
                                for d_idx in range(32):
                                    O_local[d_idx] = O_local[d_idx] * scale_O[d_idx]
                                for d_idx in range(32):
                                    O_local[d_idx] = O_local[d_idx] + V_local[d_idx] * factor[0]
                        for d_idx in range(32):
                            O_local[d_idx] = O_local[d_idx] / d_val[0]
                            output[curl_q, h_qo, d_idx] = T.Cast("float16", O_local[d_idx])
                        lse[curl_q, h_qo] = m_val[0] + T.log2(d_val[0])
    # fmt: on

    tvm.ir.assert_structural_equal(fn, tree_attn_paged_kv_cpu)


def test_tree_attn_with_paged_kv_cache_cpu_llama3_scaling():
    """rope_scaling=llama3 – structural equality with expected TVMScript."""
    fn = tree_attn_with_paged_kv_cache_cpu(H_KV, H_Q, D, DTYPE, ROPE_SCALING_LLAMA3)

    # fmt: off
    @T.prim_func
    def tree_attn_paged_kv_cpu(var_q: T.handle, var_q_indptr: T.handle, var_pages: T.handle, var_page_indptr: T.handle, var_page_values: T.handle, var_length_info: T.handle, var_k_rope_pos_offset: T.handle, var_q_rope_position: T.handle, var_output: T.handle, var_lse: T.handle, rotary_mode: T.int32, rope_scale: T.float32, rope_theta: T.float32, sm_scale: T.float32, tree_order_indptr_handle: T.handle, tree_order_handle: T.handle):
        T.func_attr({"global_symbol": "tree_attn_paged_kv_cpu"})
        batch_size = T.int32(is_size_var=True)
        total_len = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        page_indptr_elem_offset = T.int32(is_size_var=True)
        page_values_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        length_info_elem_offset = T.int32(is_size_var=True)
        tree_order_elem_offset = T.int32(is_size_var=True)
        tree_order_indptr_elem_offset = T.int32(is_size_var=True)
        q = T.match_buffer(var_q, (total_len, 4, 32), "float16")
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
        pages = T.match_buffer(var_pages, (max_num_pages, 2, 2, 16, 32), "float16")
        page_indptr = T.match_buffer(var_page_indptr, (batch_size + 1,), "int32", elem_offset=page_indptr_elem_offset)
        page_values = T.match_buffer(var_page_values, (nnz_pages,), "int32", elem_offset=page_values_elem_offset)
        length_info = T.match_buffer(var_length_info, (batch_size,), "int32", elem_offset=length_info_elem_offset)
        k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
        q_rope_position = T.match_buffer(var_q_rope_position, (total_len,), "int32", elem_offset=q_rope_position_elem_offset)
        output = T.match_buffer(var_output, (total_len, 4, 32), "float16")
        lse = T.match_buffer(var_lse, (total_len, 4))
        tree_order_indptr = T.match_buffer(tree_order_indptr_handle, (batch_size + 1,), "int32", elem_offset=tree_order_indptr_elem_offset)
        total_tree_order_len = T.int32(is_size_var=True)
        tree_order = T.match_buffer(tree_order_handle, (total_tree_order_len, 2), "int32", elem_offset=tree_order_elem_offset)
        assert rotary_mode == 0, "Inline rotary mode is not supported in tree attention."
        for h_qo in range(4):
            for b_idx in range(batch_size):
                with T.sblock("attn"):
                    T.reads()
                    T.writes()
                    O_local = T.alloc_buffer((32,))
                    Q_local = T.alloc_buffer((32,))
                    K_local = T.alloc_buffer((32,))
                    V_local = T.alloc_buffer((32,))
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
                        for d_idx in range(32):
                            O_local[d_idx] = T.float32(0.0)
                        curl_q: T.int32 = q_indptr[b_idx] + q_idx
                        for d_idx in range(32):
                            orig_freq = T.float32()
                            smoothed_freq = T.float32()
                            Q_local[d_idx] = T.Cast("float32", T.if_then_else(rotary_mode == 1, T.Let(T.Let(T.Cast("float16", T.cos(smoothed_freq) * T.Cast("float32", q[curl_q, h_qo, d_idx]) + T.sin(smoothed_freq) * T.Cast("float32", T.if_then_else(d_idx < 16, q[curl_q, h_qo, d_idx + 16] * T.float16(-1.0), q[curl_q, h_qo, d_idx - 16]))), where={smoothed_freq: T.Cast("float32", q_rope_position[curl_q]) * rope_scale * ((T.float32(1.0) - T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331)))) * orig_freq * T.float32(0.125) + T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331))) * orig_freq)}), where={orig_freq: T.float32(1.0) / T.pow(rope_theta, T.Cast("float32", d_idx * 2 % 32) / T.float32(32.0))}), q[curl_q, h_qo, d_idx]))
                        for row_idx in range(max_num_pages * 16):
                            if row_idx < kv_chunk_len[0]:
                                page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + row_idx // 16]
                                page_offset: T.int32(is_size_var=True) = row_idx % 16
                                for d_idx in range(32):
                                    orig_freq = T.float32()
                                    smoothed_freq = T.float32()
                                    K_local[d_idx] = T.Cast("float32", T.if_then_else(rotary_mode == 1, T.Let(T.Let(T.Cast("float16", T.cos(smoothed_freq) * T.Cast("float32", pages[page_no, 0, h_qo // 2, page_offset, d_idx]) + T.sin(smoothed_freq) * T.Cast("float32", T.if_then_else(d_idx < 16, pages[page_no, 0, h_qo // 2, page_offset, d_idx + 16] * T.float16(-1.0), pages[page_no, 0, h_qo // 2, page_offset, d_idx - 16]))), where={smoothed_freq: T.Cast("float32", k_rope_pos_offset[b_idx] + row_idx) * rope_scale * ((T.float32(1.0) - T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331)))) * orig_freq * T.float32(0.125) + T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331))) * orig_freq)}), where={orig_freq: T.float32(1.0) / T.pow(rope_theta, T.Cast("float32", d_idx * 2 % 32) / T.float32(32.0))}), pages[page_no, 0, h_qo // 2, page_offset, d_idx]))
                                    V_local[d_idx] = T.Cast("float32", pages[page_no, 1, h_qo // 2, page_offset, d_idx])
                                S_val[0] = T.float32(0.0)
                                for d_idx in range(32):
                                    S_val[0] = S_val[0] + Q_local[d_idx] * K_local[d_idx]
                                S_val[0] = S_val[0] * (sm_scale * T.float32(1.4426950408889634))
                                if row_idx < kv_chunk_len[0] and (row_idx < kv_chunk_len[0] - (tree_order_indptr[b_idx + 1] - tree_order_indptr[b_idx]) or tree_order[tree_order_indptr[b_idx] + (q_idx + (tree_order_indptr[b_idx + 1] - tree_order_indptr[b_idx]) - (q_indptr[b_idx + 1] - q_indptr[b_idx])), 0] >= tree_order[tree_order_indptr[b_idx] + (row_idx - (kv_chunk_len[0] - (tree_order_indptr[b_idx + 1] - tree_order_indptr[b_idx]))), 0] and tree_order[tree_order_indptr[b_idx] + (q_idx + (tree_order_indptr[b_idx + 1] - tree_order_indptr[b_idx]) - (q_indptr[b_idx + 1] - q_indptr[b_idx])), 0] < tree_order[tree_order_indptr[b_idx] + (row_idx - (kv_chunk_len[0] - (tree_order_indptr[b_idx + 1] - tree_order_indptr[b_idx]))), 1]):
                                    new_m[0] = T.max(m_val[0], S_val[0])
                                else:
                                    S_val[0] = T.float32(-50000.0)
                                d_val[0] = d_val[0] * T.exp2(m_val[0] - new_m[0])
                                d_val[0] = d_val[0] + T.exp2(S_val[0] - new_m[0])
                                scale_O[0] = T.exp2(m_val[0] - new_m[0])
                                m_val[0] = new_m[0]
                                factor[0] = T.exp2(S_val[0] - m_val[0])
                                for d_idx in range(32):
                                    O_local[d_idx] = O_local[d_idx] * scale_O[d_idx]
                                for d_idx in range(32):
                                    O_local[d_idx] = O_local[d_idx] + V_local[d_idx] * factor[0]
                        for d_idx in range(32):
                            O_local[d_idx] = O_local[d_idx] / d_val[0]
                            output[curl_q, h_qo, d_idx] = T.Cast("float16", O_local[d_idx])
                        lse[curl_q, h_qo] = m_val[0] + T.log2(d_val[0])
    # fmt: on

    tvm.ir.assert_structural_equal(fn, tree_attn_paged_kv_cpu)


def test_tree_attn_with_paged_kv_cache_cpu_none_vs_llama3_differ():
    """Different rope_scaling configs must produce structurally different PrimFuncs."""
    fn_none = tree_attn_with_paged_kv_cache_cpu(H_KV, H_Q, D, DTYPE, ROPE_SCALING_NONE)
    fn_llama3 = tree_attn_with_paged_kv_cache_cpu(H_KV, H_Q, D, DTYPE, ROPE_SCALING_LLAMA3)
    try:
        tvm.ir.assert_structural_equal(fn_none, fn_llama3)
        raise AssertionError("Expected structural inequality between none and llama3 scaling")
    except (ValueError, tvm.TVMError):
        pass  # expected


if __name__ == "__main__":
    tvm.testing.main()
