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
"""Test for _attention_prefill - GQA h_kv=4, h_q=32, d=128, float16, no sliding window, no rope."""

import pytest

import tvm
import tvm.testing
from tvm.relax.frontend.nn.llm.kv_cache import _attention_prefill
from tvm.script import tir as T

ROPE_SCALING_NONE: dict = {}


class TestAttentionPrefillGPU_GQA_F16_NoSW_NoneRope:
    @pytest.fixture(autouse=True)
    def _target(self):
        self.target = tvm.target.Target("cuda")

    def test_gqa_f16_no_sw_none_rope(self):
        fn = _attention_prefill(4, 32, 128, "float16", False, ROPE_SCALING_NONE, self.target)

        # fmt: off
        @T.prim_func
        def batch_prefill_paged_kv(var_q: T.handle, var_q_indptr: T.handle, var_pages: T.handle, var_page_indptr: T.handle, var_page_values: T.handle, var_length_info: T.handle, var_k_rope_pos_offset: T.handle, var_q_rope_position: T.handle, var_output: T.handle, var_lse: T.handle, causal: T.int32, rotary_mode: T.int32, rope_scale: T.float32, rope_theta: T.float32, sm_scale: T.float32):
            T.func_attr({"tir.is_scheduled": True, "global_symbol": "batch_prefill_paged_kv"})
            batch_size = T.int32(is_size_var=True)
            total_len = T.int32(is_size_var=True)
            nnz_pages = T.int32(is_size_var=True)
            max_num_pages = T.int32(is_size_var=True)
            pages_elem_offset = T.int64(is_size_var=True)
            q_indptr_elem_offset = T.int32(is_size_var=True)
            page_indptr_elem_offset = T.int32(is_size_var=True)
            page_values_elem_offset = T.int32(is_size_var=True)
            k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)
            q_rope_position_elem_offset = T.int32(is_size_var=True)
            length_info_elem_offset = T.int32(is_size_var=True)
            q = T.match_buffer(var_q, (total_len, 32, 128), "float16")
            q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
            pages = T.match_buffer(var_pages, (max_num_pages, 2, 4, 16, 128), "float16", elem_offset=pages_elem_offset)
            page_indptr = T.match_buffer(var_page_indptr, (batch_size + 1,), "int32", elem_offset=page_indptr_elem_offset)
            page_values = T.match_buffer(var_page_values, (nnz_pages,), "int32", elem_offset=page_values_elem_offset)
            length_info = T.match_buffer(var_length_info, (batch_size,), "int32", elem_offset=length_info_elem_offset)
            k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
            q_rope_position = T.match_buffer(var_q_rope_position, (total_len,), "int32", elem_offset=q_rope_position_elem_offset)
            output = T.match_buffer(var_output, (total_len, 32, 128), "float16")
            lse = T.match_buffer(var_lse, (total_len, 32))
            # with T.sblock("root"):
            for lbx in T.thread_binding(16, thread="blockIdx.x"):
                for lby in T.thread_binding(4, thread="blockIdx.y"):
                    for lty in T.thread_binding(4, thread="threadIdx.y"):
                        for ltx in T.thread_binding(32, thread="threadIdx.x"):
                            with T.sblock("attn"):
                                bx, by, ty, tx = T.axis.remap("SSSS", [lbx, lby, lty, ltx])
                                T.reads()
                                T.writes()
                                tile_id = T.alloc_buffer((1,), "int32", scope="local")
                                batch_idx = T.alloc_buffer((1,), "int32", scope="local")
                                batch_tiles = T.alloc_buffer((1,), "int32", scope="local")
                                batch_rows = T.alloc_buffer((1,), "int32", scope="local")
                                iterator = T.alloc_buffer((1,), "int32", scope="local")  # noqa: F841
                                kv_chunk_len = T.alloc_buffer((1,), "int32", scope="local")
                                Q_smem = T.alloc_buffer((32, 128), "float16", scope="shared")
                                K_smem = T.alloc_buffer((32, 128), "float16", scope="shared")
                                V_smem = T.alloc_buffer((32, 128), "float16", scope="shared")
                                S_smem = T.alloc_buffer((32, 32), scope="shared")
                                S_local = T.alloc_buffer((32, 32), scope="local")
                                O_local = T.alloc_buffer((32, 128), scope="local")
                                m_smem = T.alloc_buffer((32,), scope="shared")
                                m_prev_smem = T.alloc_buffer((32,), scope="shared")
                                d_smem = T.alloc_buffer((32,), scope="shared")
                                m_new = T.alloc_buffer((1,), scope="local")
                                m_prev = T.alloc_buffer((1,), scope="local")
                                d_new = T.alloc_buffer((1,), scope="local")
                                tile_id[0] = bx
                                batch_idx[0] = 0
                                batch_rows[0] = (q_indptr[1] - q_indptr[0]) * 8
                                batch_tiles[0] = (batch_rows[0] + 32 - 1) // 32
                                while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                    while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size:
                                        tile_id[0] = tile_id[0] - batch_tiles[0]
                                        batch_idx[0] = batch_idx[0] + 1
                                        if batch_idx[0] < batch_size:
                                            b_idx: T.int32 = batch_idx[0]
                                            batch_rows[0] = (q_indptr[b_idx + 1] - q_indptr[b_idx]) * 8
                                            batch_tiles[0] = (batch_rows[0] + 32 - 1) // 32
                                    if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                        b_idx: T.int32 = batch_idx[0]
                                        LH_start: T.int32 = tile_id[0] * 32
                                        q_indptr_val: T.int32 = q_indptr[b_idx]
                                        cur_page_indptr_begin: T.int32 = page_indptr[b_idx]
                                        cur_page_indptr_end: T.int32 = page_indptr[b_idx + 1]
                                        kv_chunk_len[0] = T.if_then_else(cur_page_indptr_begin != cur_page_indptr_end, (cur_page_indptr_end - cur_page_indptr_begin - 1) * 16 + length_info[b_idx], 0)
                                        T.tvm_storage_sync("shared")
                                        for i in range(1):
                                            row: T.int32 = i * 32 * 4 + ty * 32 + tx
                                            if row < 32:
                                                m_smem[row] = T.float32(-50000.0)
                                                d_smem[row] = T.float32(1.0)
                                        for li_0_lj_0_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                            for li_0_lj_0_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                                for li_1 in range(4):
                                                    for lj_1_0 in T.unroll(2):
                                                        for lj_1_1 in T.vectorized(4):
                                                            with T.sblock("O_init"):
                                                                i = T.axis.spatial(32, (li_0_lj_0_fused_0 * 32 + li_0_lj_0_fused_1) // 16 * 4 + li_1)
                                                                j = T.axis.spatial(128, (li_0_lj_0_fused_0 * 32 + li_0_lj_0_fused_1) % 16 * 8 + lj_1_0 * 4 + lj_1_1)
                                                                T.reads()
                                                                T.writes(O_local[i, j])
                                                                O_local[i, j] = T.float32(0.0)
                                        T.tvm_storage_sync("shared")
                                        for li_1_lj_0_1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                            for li_1_lj_0_1_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                                for li_0, lj_0_0 in T.grid(2, 4):
                                                    for lj_1 in T.vectorized(4):
                                                        with T.sblock("Q_load"):
                                                            i = T.axis.spatial(32, li_0 * 16 + (li_1_lj_0_1_fused_0 * 32 + li_1_lj_0_1_fused_1) // 8)
                                                            j = T.axis.spatial(128, lj_0_0 * 32 + (li_1_lj_0_1_fused_0 * 32 + li_1_lj_0_1_fused_1) % 8 * 4 + lj_1)
                                                            T.reads()
                                                            T.writes()
                                                            cur_L: T.int32 = q_indptr_val + (LH_start + i) // 8
                                                            cur_H_qo: T.int32 = by * 8 + (LH_start + i) % 8
                                                            if cur_L < q_indptr[b_idx + 1]:
                                                                freq = T.float32()
                                                                Q_smem[i, j] = T.if_then_else(rotary_mode == 1, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", q[cur_L, cur_H_qo, j]) + T.sin(freq) * T.Cast("float32", T.if_then_else(j < 64, q[cur_L, cur_H_qo, j + 64] * T.float16(-1.0), q[cur_L, cur_H_qo, j - 64]))), where={freq: T.Cast("float32", q_rope_position[cur_L]) * rope_scale / T.pow(rope_theta, T.Cast("float32", j * 2 % 128) / T.float32(128.0))}), q[cur_L, cur_H_qo, j])
                                                            else:
                                                                Q_smem[i, j] = T.float16(0.0)
                                        T.tvm_storage_sync("shared")
                                        for iterator_1 in range((kv_chunk_len[0] + 31) // 32):
                                            L_kv_start: T.int32 = iterator_1 * 32
                                            for lz_1_ly_0_1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                                for lz_1_ly_0_1_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                                    for lz_0, ly_0_0 in T.grid(2, 4):
                                                        for ly_1 in T.vectorized(4):
                                                            with T.sblock("K_load"):
                                                                i = T.axis.spatial(32, lz_0 * 16 + (lz_1_ly_0_1_fused_0 * 32 + lz_1_ly_0_1_fused_1) // 8)
                                                                j = T.axis.spatial(128, ly_0_0 * 32 + (lz_1_ly_0_1_fused_0 * 32 + lz_1_ly_0_1_fused_1) % 8 * 4 + ly_1)
                                                                T.reads()
                                                                T.writes()
                                                                cur_L: T.int32 = L_kv_start + i
                                                                if cur_L < kv_chunk_len[0]:
                                                                    seq_offset: T.int32(is_size_var=True) = cur_L
                                                                    page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + seq_offset // 16]
                                                                    page_offset: T.int32(is_size_var=True) = seq_offset % 16
                                                                    freq = T.float32()
                                                                    K_smem[i, j] = T.if_then_else(rotary_mode == 1, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", pages[page_no, 0, by, page_offset, j]) + T.sin(freq) * T.Cast("float32", T.if_then_else(j < 64, pages[page_no, 0, by, page_offset, j + 64] * T.float16(-1.0), pages[page_no, 0, by, page_offset, j - 64]))), where={freq: T.Cast("float32", k_rope_pos_offset[b_idx] + cur_L) * rope_scale / T.pow(rope_theta, T.Cast("float32", j * 2 % 128) / T.float32(128.0))}), pages[page_no, 0, by, page_offset, j])
                                                                else:
                                                                    K_smem[i, j] = T.float16(0.0)
                                            T.tvm_storage_sync("shared")
                                            for lz_1_ly_0_1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                                for lz_1_ly_0_1_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                                    for lz_0, ly_0_0 in T.grid(2, 4):
                                                        for ly_1 in T.vectorized(4):
                                                            with T.sblock("V_load"):
                                                                i = T.axis.spatial(32, lz_0 * 16 + (lz_1_ly_0_1_fused_0 * 32 + lz_1_ly_0_1_fused_1) // 8)
                                                                j = T.axis.spatial(128, ly_0_0 * 32 + (lz_1_ly_0_1_fused_0 * 32 + lz_1_ly_0_1_fused_1) % 8 * 4 + ly_1)
                                                                T.reads()
                                                                T.writes()
                                                                cur_L: T.int32 = L_kv_start + i
                                                                if cur_L < kv_chunk_len[0]:
                                                                    seq_offset: T.int32(is_size_var=True) = cur_L
                                                                    page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + seq_offset // 16]
                                                                    page_offset: T.int32(is_size_var=True) = seq_offset % 16
                                                                    V_smem[i, j] = pages[page_no, 1, by, page_offset, j]
                                                                else:
                                                                    V_smem[i, j] = T.float16(0.0)
                                            T.tvm_storage_sync("shared")
                                            with T.sblock(""):
                                                T.reads(Q_smem[0:32, 0:128], K_smem[0:32, 0:128])
                                                T.writes(S_local[0:32, 0:32])
                                                for li_0_lj_0_fused_0_init in T.thread_binding(4, thread="threadIdx.y"):
                                                    for li_0_lj_0_fused_1_init in T.thread_binding(32, thread="threadIdx.x"):
                                                        for li_1_init in T.unroll(2):
                                                            for lj_1_0_init in T.unroll(1):
                                                                for lj_1_1_init in T.vectorized(4):
                                                                    with T.sblock("S_gemm_init"):
                                                                        i = T.axis.spatial(32, (li_0_lj_0_fused_0_init * 32 + li_0_lj_0_fused_1_init) // 8 * 2 + li_1_init)
                                                                        j = T.axis.spatial(32, (li_0_lj_0_fused_0_init * 32 + li_0_lj_0_fused_1_init) % 8 * 4 + lj_1_0_init * 4 + lj_1_1_init)
                                                                        T.reads()
                                                                        T.writes(S_local[i, j])
                                                                        S_local[i, j] = T.float32(0.0)
                                                for li_0_lj_0_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                                    for li_0_lj_0_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                                        for lk_0 in range(8):
                                                            for li_1 in T.unroll(2):
                                                                for lj_1_0 in T.unroll(1):
                                                                    for lj_1_1 in T.vectorized(4):
                                                                        for lk_1 in range(16):
                                                                            with T.sblock("S_gemm_update"):
                                                                                i = T.axis.spatial(32, (li_0_lj_0_fused_0 * 32 + li_0_lj_0_fused_1) // 8 * 2 + li_1)
                                                                                j = T.axis.spatial(32, (li_0_lj_0_fused_0 * 32 + li_0_lj_0_fused_1) % 8 * 4 + lj_1_0 * 4 + lj_1_1)
                                                                                k = T.axis.reduce(128, lk_0 * 16 + lk_1)
                                                                                T.reads(S_local[i, j], Q_smem[i, k], K_smem[j, k])
                                                                                T.writes(S_local[i, j])
                                                                                S_local[i, j] = S_local[i, j] + T.Cast("float32", Q_smem[i, k]) * T.Cast("float32", K_smem[j, k]) * sm_scale * T.float32(1.4426950408889634)
                                            T.tvm_storage_sync("shared")
                                            for li_0_lj_0_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                                for li_0_lj_0_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                                    for li_1 in range(2):
                                                        for lj_1_0 in T.unroll(1):
                                                            for lj_1_1 in T.vectorized(4):
                                                                with T.sblock("S_store"):
                                                                    i = T.axis.spatial(32, (li_0_lj_0_fused_0 * 32 + li_0_lj_0_fused_1) // 8 * 2 + li_1)
                                                                    j = T.axis.spatial(32, (li_0_lj_0_fused_0 * 32 + li_0_lj_0_fused_1) % 8 * 4 + lj_1_0 * 4 + lj_1_1)
                                                                    T.reads(S_local[i, j])
                                                                    T.writes(S_smem[i, j])
                                                                    S_smem[i, j] = S_local[i, j]
                                            T.tvm_storage_sync("shared")
                                            for i in range(1):
                                                row: T.int32 = i * 32 * 4 + ty * 32 + tx
                                                if row < 32:
                                                    with T.sblock("update1"):
                                                        T.reads(m_smem[row], kv_chunk_len[0], q_indptr[b_idx:b_idx + 2], m_new[i], S_smem[row, 0:32], d_smem[row], m_prev[i])
                                                        T.writes(m_prev[i], m_new[i], d_new[i])
                                                        m_prev[i] = m_smem[row]
                                                        m_new[i] = m_smem[row]
                                                        row_: T.int32 = (LH_start + row) // 8
                                                        for j in range(32):
                                                            if T.if_then_else(causal > 0, L_kv_start + j < kv_chunk_len[0] - (q_indptr[b_idx + 1] - q_indptr[b_idx]) + row_ + 1, L_kv_start + j < kv_chunk_len[0]):
                                                                m_new[i] = T.max(m_new[i], S_smem[row, j])
                                                        d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])
                                            for i in range(1):
                                                row: T.int32 = i * 32 * 4 + ty * 32 + tx
                                                with T.sblock("update"):
                                                    T.reads(kv_chunk_len[0], q_indptr[b_idx:b_idx + 2], S_smem[row, 0:32], m_new[i])
                                                    T.writes(S_smem[row, 0:32])
                                                    for j in range(32):
                                                        if row < 32:
                                                            row_: T.int32 = (LH_start + row) // 8
                                                            if T.if_then_else(causal > 0, L_kv_start + j < kv_chunk_len[0] - (q_indptr[b_idx + 1] - q_indptr[b_idx]) + row_ + 1, L_kv_start + j < kv_chunk_len[0]):
                                                                S_smem[row, j] = T.exp2(S_smem[row, j] - m_new[i])
                                                            else:
                                                                S_smem[row, j] = T.exp2(T.float32(-50000.0) - m_new[i])
                                            for i in range(1):
                                                row: T.int32 = i * 32 * 4 + ty * 32 + tx
                                                if row < 32:
                                                    with T.sblock("update"):
                                                        T.reads(d_new[i], S_smem[row, 0:32], m_new[i], m_prev[i])
                                                        T.writes(d_new[i], m_smem[row], d_smem[row], m_prev_smem[row])
                                                        for j in range(32):
                                                            d_new[i] = d_new[i] + S_smem[row, j]
                                                        m_smem[row] = m_new[i]
                                                        d_smem[row] = d_new[i]
                                                        m_prev_smem[row] = m_prev[i]
                                            T.tvm_storage_sync("shared")
                                            with T.sblock(""):
                                                T.reads(m_prev_smem[0:32], m_smem[0:32], S_smem[0:32, 0:32], V_smem[0:32, 0:128])
                                                T.writes(O_local[0:32, 0:128])
                                                for li_0_lj_0_fused_0_init in T.thread_binding(4, thread="threadIdx.y"):
                                                    for li_0_lj_0_fused_1_init in T.thread_binding(32, thread="threadIdx.x"):
                                                        for li_1_init in T.unroll(4):
                                                            for lj_1_0_init in T.unroll(2):
                                                                for lj_1_1_init in T.vectorized(4):
                                                                    with T.sblock("O_gemm_init"):
                                                                        i = T.axis.spatial(32, (li_0_lj_0_fused_0_init * 32 + li_0_lj_0_fused_1_init) // 16 * 4 + li_1_init)
                                                                        j = T.axis.spatial(128, (li_0_lj_0_fused_0_init * 32 + li_0_lj_0_fused_1_init) % 16 * 8 + lj_1_0_init * 4 + lj_1_1_init)
                                                                        T.reads()
                                                                        T.writes(O_local[i, j])
                                                                        O_local[i, j] = O_local[i, j] * T.exp2(m_prev_smem[i] - m_smem[i])
                                                for li_0_lj_0_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                                    for li_0_lj_0_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                                        for lk_0, lk_1 in T.grid(2, 16):
                                                            for li_1 in T.unroll(4):
                                                                for lj_1_0 in T.unroll(2):
                                                                    for lj_1_1 in T.vectorized(4):
                                                                        with T.sblock("O_gemm_update"):
                                                                            i = T.axis.spatial(32, (li_0_lj_0_fused_0 * 32 + li_0_lj_0_fused_1) // 16 * 4 + li_1)
                                                                            j = T.axis.spatial(128, (li_0_lj_0_fused_0 * 32 + li_0_lj_0_fused_1) % 16 * 8 + lj_1_0 * 4 + lj_1_1)
                                                                            k = T.axis.reduce(32, lk_0 * 16 + lk_1)
                                                                            T.reads(O_local[i, j], m_prev_smem[i], m_smem[i], S_smem[i, k], V_smem[k, j])
                                                                            T.writes(O_local[i, j])
                                                                            O_local[i, j] = O_local[i, j] + S_smem[i, k] * T.Cast("float32", V_smem[k, j])
                                        for li_0_lj_0_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                            for li_0_lj_0_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                                for li_1 in range(4):
                                                    for lj_1_0 in T.unroll(2):
                                                        for lj_1_1 in T.vectorized(4):
                                                            with T.sblock("O_store"):
                                                                i = T.axis.spatial(32, (li_0_lj_0_fused_0 * 32 + li_0_lj_0_fused_1) // 16 * 4 + li_1)
                                                                j = T.axis.spatial(128, (li_0_lj_0_fused_0 * 32 + li_0_lj_0_fused_1) % 16 * 8 + lj_1_0 * 4 + lj_1_1)
                                                                T.reads(q_indptr[b_idx:b_idx + 2], O_local[i, j], d_smem[i])
                                                                T.writes(output[q_indptr[b_idx] + (LH_start + i) // 8, by * 8 + (LH_start + i) % 8, j])
                                                                cur_L: T.int32 = q_indptr[b_idx] + (LH_start + i) // 8
                                                                cur_H_qo: T.int32 = by * 8 + (LH_start + i) % 8
                                                                if cur_L < q_indptr[b_idx + 1]:
                                                                    output[cur_L, cur_H_qo, j] = T.Cast("float16", O_local[i, j] / d_smem[i])
                                        for li_0 in range(1):
                                            for li_1 in T.thread_binding(4, thread="threadIdx.y"):
                                                for li_2 in T.thread_binding(32, thread="threadIdx.x"):
                                                    with T.sblock("lse_store"):
                                                        i = T.axis.spatial(32, li_0 * 128 + li_1 * 32 + li_2)
                                                        T.where((li_0 * 4 + li_1) * 32 + li_2 < 32)
                                                        T.reads(q_indptr[b_idx:b_idx + 2], m_smem[i], d_smem[i])
                                                        T.writes(lse[q_indptr[b_idx] + (LH_start + i) // 8, by * 8 + (LH_start + i) % 8])
                                                        cur_L: T.int32 = q_indptr[b_idx] + (LH_start + i) // 8
                                                        cur_H_qo: T.int32 = by * 8 + (LH_start + i) % 8
                                                        if cur_L < q_indptr[b_idx + 1]:
                                                            lse[cur_L, cur_H_qo] = m_smem[i] + T.log2(d_smem[i])
                                        tile_id[0] = tile_id[0] + 16
        # fmt: on

        tvm.ir.assert_structural_equal(fn, batch_prefill_paged_kv)


if __name__ == "__main__":
    tvm.testing.main()
