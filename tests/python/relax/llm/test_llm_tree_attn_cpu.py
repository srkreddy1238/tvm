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
"""Tests for tree_attn_cpu.

Each test calls ``tree_attn_cpu`` directly (it returns a TIR PrimFunc) and
compares the result against an expected PrimFunc written in TVMScript –
exactly the same pattern used throughout
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
from tvm.relax.frontend.nn.llm.tree_attn import tree_attn_cpu
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


def test_tree_attn_cpu_none_scaling():
    """rope_scaling={} – structural equality with expected TVMScript."""
    fn = tree_attn_cpu(H_KV, H_Q, D, DTYPE, ROPE_SCALING_NONE)

    # fmt: off
    @T.prim_func
    def batch_tree_attn(var_q: T.handle, var_q_indptr: T.handle, var_k: T.handle, var_v: T.handle, var_kv_indptr: T.handle, var_q_rope_position: T.handle, var_mn_indptr: T.handle, var_mask: T.handle, var_output: T.handle, var_lse: T.handle, rotary_mode: T.int32, rope_scale: T.float32, rope_theta: T.float32, sm_scale: T.float32):
        T.func_attr({"global_symbol": "batch_tree_attn"})
        qo_len = T.int32(is_size_var=True)
        kv_len = T.int32(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        kv_indptr_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        mn_indptr_elem_offset = T.int32(is_size_var=True)
        mask_elem_offset = T.int32(is_size_var=True)
        tree_size = T.int32(is_size_var=True)
        batch_size_plus_1 = T.int32(is_size_var=True)
        q = T.match_buffer(var_q, (qo_len, 4, 32), "float16")
        q_indptr = T.match_buffer(var_q_indptr, (batch_size_plus_1,), "int32", elem_offset=q_indptr_elem_offset)
        k = T.match_buffer(var_k, (kv_len, 2, 32), "float16")
        v = T.match_buffer(var_v, (kv_len, 2, 32), "float16")
        kv_indptr = T.match_buffer(var_kv_indptr, (batch_size_plus_1,), "int32", elem_offset=kv_indptr_elem_offset)
        q_rope_position = T.match_buffer(var_q_rope_position, (qo_len,), "int32", elem_offset=q_rope_position_elem_offset)
        mn_indptr = T.match_buffer(var_mn_indptr, (batch_size_plus_1,), "int32", elem_offset=mn_indptr_elem_offset)
        mask = T.match_buffer(var_mask, (tree_size, 2), "int32", elem_offset=mask_elem_offset)
        output = T.match_buffer(var_output, (qo_len, 4, 32), "float16")
        lse = T.match_buffer(var_lse, (qo_len, 4))
        for b in range(batch_size_plus_1 - 1):
            with T.sblock("attn"):
                T.reads(q_indptr[b:b + 2], kv_indptr[b:b + 2], mn_indptr[b:b + 2], mask[T.min(mn_indptr[b + 1] + q_indptr[b] - q_indptr[b + 1], kv_indptr[b] + mn_indptr[b + 1] - kv_indptr[b + 1]):T.min(mn_indptr[b + 1] + q_indptr[b] - q_indptr[b + 1], kv_indptr[b] + mn_indptr[b + 1] - kv_indptr[b + 1]) + (mn_indptr[b + 1] - T.min(mn_indptr[b + 1] + q_indptr[b] - q_indptr[b + 1], kv_indptr[b] + mn_indptr[b + 1] - kv_indptr[b + 1])), 0:2], q_rope_position[T.min(q_indptr[b], kv_indptr[b]):T.min(q_indptr[b], kv_indptr[b]) + (T.max(q_indptr[b + 1], kv_indptr[b + 1]) - T.min(q_indptr[b], kv_indptr[b]))], q[q_indptr[b]:q_indptr[b] + (q_indptr[b + 1] - q_indptr[b]), 0:4, 0:32], k[kv_indptr[b]:kv_indptr[b] + (kv_indptr[b + 1] - kv_indptr[b]), 0:2, 0:32], v[kv_indptr[b]:kv_indptr[b] + (kv_indptr[b + 1] - kv_indptr[b]), 0:2, 0:32])
                T.writes(output[q_indptr[b]:q_indptr[b] + (q_indptr[b + 1] - q_indptr[b]), 0:4, 0:32], lse[q_indptr[b]:q_indptr[b] + (q_indptr[b + 1] - q_indptr[b]), 0:4])
                softmax_sum = T.alloc_buffer((4,))
                m_prev = T.alloc_buffer((4,))
                m_prev_1 = T.alloc_buffer((4,))
                d_prev = T.alloc_buffer((4,))
                d_prev_1 = T.alloc_buffer((4,))
                p_sum = T.alloc_buffer((32,))
                max_score = T.alloc_buffer((4,))
                attention_scores = T.alloc_buffer((kv_len, 4))
                exp_scores = T.alloc_buffer((kv_len, 4))
                attention_score = T.alloc_buffer((1,))
                query_val = T.alloc_buffer((1,))
                key_val = T.alloc_buffer((1,))
                result = T.alloc_buffer((1,))
                for q_idx in range(q_indptr[b + 1] - q_indptr[b]):
                    for i in range(4):
                        max_score[i] = T.float32(-50000.0)
                        m_prev[i] = T.float32(-50000.0)
                        d_prev[i] = T.float32(1.0)
                    for k_idx in range(kv_indptr[b + 1] - kv_indptr[b]):
                        for h in range(4):
                            h_kv_idx: T.int32 = h // 2
                            if k_idx < kv_indptr[b + 1] - kv_indptr[b] and (k_idx < kv_indptr[b + 1] - kv_indptr[b] - (mn_indptr[b + 1] - mn_indptr[b]) or mask[mn_indptr[b] + (q_idx + (mn_indptr[b + 1] - mn_indptr[b]) - (q_indptr[b + 1] - q_indptr[b])), 0] >= mask[mn_indptr[b] + (k_idx - (kv_indptr[b + 1] - kv_indptr[b] - (mn_indptr[b + 1] - mn_indptr[b]))), 0] and mask[mn_indptr[b] + (q_idx + (mn_indptr[b + 1] - mn_indptr[b]) - (q_indptr[b + 1] - q_indptr[b])), 0] < mask[mn_indptr[b] + (k_idx - (kv_indptr[b + 1] - kv_indptr[b] - (mn_indptr[b + 1] - mn_indptr[b]))), 1]):
                                result[0] = T.float32(0.0)
                                for d_idx in range(32):
                                    freq = T.float32()
                                    query_val[0] = T.Cast("float32", T.if_then_else(rotary_mode == 1, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", q[q_indptr[b] + q_idx, h, d_idx]) + T.sin(freq) * T.Cast("float32", T.if_then_else(d_idx < 16, q[q_indptr[b] + q_idx, h, d_idx + 16] * T.float16(-1.0), q[q_indptr[b] + q_idx, h, d_idx - 16]))), where={freq: T.Cast("float32", q_rope_position[q_indptr[b] + q_idx]) * rope_scale / T.pow(rope_theta, T.Cast("float32", d_idx * 2 % 32) / T.float32(32.0))}), q[q_indptr[b] + q_idx, h, d_idx]))
                                    freq_1 = T.float32()
                                    key_val[0] = T.Cast("float32", T.if_then_else(rotary_mode == 1, T.Let(T.Cast("float16", T.cos(freq_1) * T.Cast("float32", k[kv_indptr[b] + k_idx, h_kv_idx, d_idx]) + T.sin(freq_1) * T.Cast("float32", T.if_then_else(d_idx < 16, k[kv_indptr[b] + k_idx, h_kv_idx, d_idx + 16] * T.float16(-1.0), k[kv_indptr[b] + k_idx, h_kv_idx, d_idx - 16]))), where={freq_1: T.Cast("float32", q_rope_position[kv_indptr[b] + k_idx]) * rope_scale / T.pow(rope_theta, T.Cast("float32", d_idx * 2 % 32) / T.float32(32.0))}), k[kv_indptr[b] + k_idx, h_kv_idx, d_idx]))
                                    result[0] = result[0] + query_val[0] * key_val[0]
                                attention_score[0] = result[0] * T.float32(1.4426950408889634) * sm_scale
                            else:
                                attention_score[0] = T.float32(-72134.752044448163) * sm_scale
                            attention_scores[k_idx, h] = attention_score[0]
                            max_score[h] = T.max(max_score[h], attention_score[0])
                            m_prev_1[h] = T.max(m_prev[h], max_score[h])
                    for h in range(4):
                        d_prev_1[h] = d_prev[h] * T.exp2(m_prev[h] - m_prev_1[h])
                    for h in range(4):
                        softmax_sum[h] = T.float32(0.0)
                        for k_idx in range(kv_indptr[b + 1] - kv_indptr[b]):
                            exp_scores[k_idx, h] = T.exp2(attention_scores[k_idx, h] - m_prev_1[h])
                            softmax_sum[h] = softmax_sum[h] + exp_scores[k_idx, h]
                        d_prev_1[h] = d_prev_1[h] + softmax_sum[h]
                    for h in range(4):
                        h_kv_idx: T.int32 = h // 2
                        for i in range(32):
                            p_sum[i] = T.float32(0.0)
                        for v_idx in range(kv_indptr[b + 1] - kv_indptr[b]):
                            weight: T.float32 = exp_scores[v_idx, h] / d_prev_1[h]
                            for i in range(32):
                                p_sum[i] = p_sum[i] + T.Cast("float32", v[kv_indptr[b] + v_idx, h_kv_idx, i]) * weight
                        for i in range(32):
                            output[q_indptr[b] + q_idx, h, i] = T.Cast("float16", p_sum[i])
                        lse[q_indptr[b] + q_idx, h] = m_prev_1[h] + T.log2(d_prev_1[h])
    # fmt: on

    tvm.ir.assert_structural_equal(fn, batch_tree_attn)


def test_tree_attn_cpu_llama3_scaling():
    """rope_scaling=llama3 – structural equality with expected TVMScript."""
    fn = tree_attn_cpu(H_KV, H_Q, D, DTYPE, ROPE_SCALING_LLAMA3)

    # fmt: off
    @T.prim_func
    def batch_tree_attn(var_q: T.handle, var_q_indptr: T.handle, var_k: T.handle, var_v: T.handle, var_kv_indptr: T.handle, var_q_rope_position: T.handle, var_mn_indptr: T.handle, var_mask: T.handle, var_output: T.handle, var_lse: T.handle, rotary_mode: T.int32, rope_scale: T.float32, rope_theta: T.float32, sm_scale: T.float32):
        T.func_attr({"global_symbol": "batch_tree_attn"})
        qo_len = T.int32(is_size_var=True)
        kv_len = T.int32(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        kv_indptr_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        mn_indptr_elem_offset = T.int32(is_size_var=True)
        mask_elem_offset = T.int32(is_size_var=True)
        tree_size = T.int32(is_size_var=True)
        batch_size_plus_1 = T.int32(is_size_var=True)
        q = T.match_buffer(var_q, (qo_len, 4, 32), "float16")
        q_indptr = T.match_buffer(var_q_indptr, (batch_size_plus_1,), "int32", elem_offset=q_indptr_elem_offset)
        k = T.match_buffer(var_k, (kv_len, 2, 32), "float16")
        v = T.match_buffer(var_v, (kv_len, 2, 32), "float16")
        kv_indptr = T.match_buffer(var_kv_indptr, (batch_size_plus_1,), "int32", elem_offset=kv_indptr_elem_offset)
        q_rope_position = T.match_buffer(var_q_rope_position, (qo_len,), "int32", elem_offset=q_rope_position_elem_offset)
        mn_indptr = T.match_buffer(var_mn_indptr, (batch_size_plus_1,), "int32", elem_offset=mn_indptr_elem_offset)
        mask = T.match_buffer(var_mask, (tree_size, 2), "int32", elem_offset=mask_elem_offset)
        output = T.match_buffer(var_output, (qo_len, 4, 32), "float16")
        lse = T.match_buffer(var_lse, (qo_len, 4))
        for b in range(batch_size_plus_1 - 1):
            with T.sblock("attn"):
                T.reads(q_indptr[b:b + 2], kv_indptr[b:b + 2], mn_indptr[b:b + 2], mask[T.min(mn_indptr[b + 1] + q_indptr[b] - q_indptr[b + 1], kv_indptr[b] + mn_indptr[b + 1] - kv_indptr[b + 1]):T.min(mn_indptr[b + 1] + q_indptr[b] - q_indptr[b + 1], kv_indptr[b] + mn_indptr[b + 1] - kv_indptr[b + 1]) + (mn_indptr[b + 1] - T.min(mn_indptr[b + 1] + q_indptr[b] - q_indptr[b + 1], kv_indptr[b] + mn_indptr[b + 1] - kv_indptr[b + 1])), 0:2], q_rope_position[T.min(q_indptr[b], kv_indptr[b]):T.min(q_indptr[b], kv_indptr[b]) + (T.max(q_indptr[b + 1], kv_indptr[b + 1]) - T.min(q_indptr[b], kv_indptr[b]))], q[q_indptr[b]:q_indptr[b] + (q_indptr[b + 1] - q_indptr[b]), 0:4, 0:32], k[kv_indptr[b]:kv_indptr[b] + (kv_indptr[b + 1] - kv_indptr[b]), 0:2, 0:32], v[kv_indptr[b]:kv_indptr[b] + (kv_indptr[b + 1] - kv_indptr[b]), 0:2, 0:32])
                T.writes(output[q_indptr[b]:q_indptr[b] + (q_indptr[b + 1] - q_indptr[b]), 0:4, 0:32], lse[q_indptr[b]:q_indptr[b] + (q_indptr[b + 1] - q_indptr[b]), 0:4])
                softmax_sum = T.alloc_buffer((4,))
                m_prev = T.alloc_buffer((4,))
                m_prev_1 = T.alloc_buffer((4,))
                d_prev = T.alloc_buffer((4,))
                d_prev_1 = T.alloc_buffer((4,))
                p_sum = T.alloc_buffer((32,))
                max_score = T.alloc_buffer((4,))
                attention_scores = T.alloc_buffer((kv_len, 4))
                exp_scores = T.alloc_buffer((kv_len, 4))
                attention_score = T.alloc_buffer((1,))
                query_val = T.alloc_buffer((1,))
                key_val = T.alloc_buffer((1,))
                result = T.alloc_buffer((1,))
                for q_idx in range(q_indptr[b + 1] - q_indptr[b]):
                    for i in range(4):
                        max_score[i] = T.float32(-50000.0)
                        m_prev[i] = T.float32(-50000.0)
                        d_prev[i] = T.float32(1.0)
                    for k_idx in range(kv_indptr[b + 1] - kv_indptr[b]):
                        for h in range(4):
                            h_kv_idx: T.int32 = h // 2
                            if k_idx < kv_indptr[b + 1] - kv_indptr[b] and (k_idx < kv_indptr[b + 1] - kv_indptr[b] - (mn_indptr[b + 1] - mn_indptr[b]) or mask[mn_indptr[b] + (q_idx + (mn_indptr[b + 1] - mn_indptr[b]) - (q_indptr[b + 1] - q_indptr[b])), 0] >= mask[mn_indptr[b] + (k_idx - (kv_indptr[b + 1] - kv_indptr[b] - (mn_indptr[b + 1] - mn_indptr[b]))), 0] and mask[mn_indptr[b] + (q_idx + (mn_indptr[b + 1] - mn_indptr[b]) - (q_indptr[b + 1] - q_indptr[b])), 0] < mask[mn_indptr[b] + (k_idx - (kv_indptr[b + 1] - kv_indptr[b] - (mn_indptr[b + 1] - mn_indptr[b]))), 1]):
                                result[0] = T.float32(0.0)
                                for d_idx in range(32):
                                    orig_freq = T.float32()
                                    smoothed_freq = T.float32()
                                    query_val[0] = T.Cast("float32", T.if_then_else(rotary_mode == 1, T.Let(T.Let(T.Cast("float16", T.cos(smoothed_freq) * T.Cast("float32", q[q_indptr[b] + q_idx, h, d_idx]) + T.sin(smoothed_freq) * T.Cast("float32", T.if_then_else(d_idx < 16, q[q_indptr[b] + q_idx, h, d_idx + 16] * T.float16(-1.0), q[q_indptr[b] + q_idx, h, d_idx - 16]))), where={smoothed_freq: T.Cast("float32", q_rope_position[q_indptr[b] + q_idx]) * rope_scale * ((T.float32(1.0) - T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331)))) * orig_freq * T.float32(0.125) + T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331))) * orig_freq)}), where={orig_freq: T.float32(1.0) / T.pow(rope_theta, T.Cast("float32", d_idx * 2 % 32) / T.float32(32.0))}), q[q_indptr[b] + q_idx, h, d_idx]))
                                    orig_freq_1 = T.float32()
                                    smoothed_freq_1 = T.float32()
                                    key_val[0] = T.Cast("float32", T.if_then_else(rotary_mode == 1, T.Let(T.Let(T.Cast("float16", T.cos(smoothed_freq_1) * T.Cast("float32", k[kv_indptr[b] + k_idx, h_kv_idx, d_idx]) + T.sin(smoothed_freq_1) * T.Cast("float32", T.if_then_else(d_idx < 16, k[kv_indptr[b] + k_idx, h_kv_idx, d_idx + 16] * T.float16(-1.0), k[kv_indptr[b] + k_idx, h_kv_idx, d_idx - 16]))), where={smoothed_freq_1: T.Cast("float32", q_rope_position[kv_indptr[b] + k_idx]) * rope_scale * ((T.float32(1.0) - T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq_1 - T.float32(0.33333333333333331)))) * orig_freq_1 * T.float32(0.125) + T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq_1 - T.float32(0.33333333333333331))) * orig_freq_1)}), where={orig_freq_1: T.float32(1.0) / T.pow(rope_theta, T.Cast("float32", d_idx * 2 % 32) / T.float32(32.0))}), k[kv_indptr[b] + k_idx, h_kv_idx, d_idx]))
                                    result[0] = result[0] + query_val[0] * key_val[0]
                                attention_score[0] = result[0] * T.float32(1.4426950408889634) * sm_scale
                            else:
                                attention_score[0] = T.float32(-72134.752044448163) * sm_scale
                            attention_scores[k_idx, h] = attention_score[0]
                            max_score[h] = T.max(max_score[h], attention_score[0])
                            m_prev_1[h] = T.max(m_prev[h], max_score[h])
                    for h in range(4):
                        d_prev_1[h] = d_prev[h] * T.exp2(m_prev[h] - m_prev_1[h])
                    for h in range(4):
                        softmax_sum[h] = T.float32(0.0)
                        for k_idx in range(kv_indptr[b + 1] - kv_indptr[b]):
                            exp_scores[k_idx, h] = T.exp2(attention_scores[k_idx, h] - m_prev_1[h])
                            softmax_sum[h] = softmax_sum[h] + exp_scores[k_idx, h]
                        d_prev_1[h] = d_prev_1[h] + softmax_sum[h]
                    for h in range(4):
                        h_kv_idx: T.int32 = h // 2
                        for i in range(32):
                            p_sum[i] = T.float32(0.0)
                        for v_idx in range(kv_indptr[b + 1] - kv_indptr[b]):
                            weight: T.float32 = exp_scores[v_idx, h] / d_prev_1[h]
                            for i in range(32):
                                p_sum[i] = p_sum[i] + T.Cast("float32", v[kv_indptr[b] + v_idx, h_kv_idx, i]) * weight
                        for i in range(32):
                            output[q_indptr[b] + q_idx, h, i] = T.Cast("float16", p_sum[i])
                        lse[q_indptr[b] + q_idx, h] = m_prev_1[h] + T.log2(d_prev_1[h])
    # fmt: on

    tvm.ir.assert_structural_equal(fn, batch_tree_attn)


def test_tree_attn_cpu_none_vs_llama3_differ():
    """Different rope_scaling configs must produce structurally different PrimFuncs."""
    fn_none = tree_attn_cpu(H_KV, H_Q, D, DTYPE, ROPE_SCALING_NONE)
    fn_llama3 = tree_attn_cpu(H_KV, H_Q, D, DTYPE, ROPE_SCALING_LLAMA3)
    try:
        tvm.ir.assert_structural_equal(fn_none, fn_llama3)
        raise AssertionError("Expected structural inequality between none and llama3 scaling")
    except (ValueError, tvm.TVMError):
        pass  # expected


if __name__ == "__main__":
    tvm.testing.main()
