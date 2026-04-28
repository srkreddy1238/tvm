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
"""Test for _attention_decode - sliding window h_kv=4, h_q=32, d=128, float16, no rope."""

import pytest

import tvm
import tvm.testing
from tvm.relax.frontend.nn.llm.kv_cache import _attention_decode
from tvm.script import tir as T

ROPE_SCALING_NONE: dict = {}


class TestAttentionDecodeGPU_SlidingWindow_NoneRope:
    @pytest.fixture(autouse=True)
    def _target(self):
        self.target = tvm.target.Target("cuda")

    def test_sliding_window_none_rope(self):
        fn = _attention_decode(4, 32, 128, "float16", True, ROPE_SCALING_NONE, self.target)

        # fmt: off
        # from tvm.script import tir as T

        @T.prim_func
        def batch_decode_paged_kv_sliding_window(Q_handle: T.handle, pages_handle: T.handle, page_table_indptr_handle: T.handle, page_table_values_handle: T.handle, var_length_info: T.handle, k_rope_pos_offset_handle: T.handle, q_rope_position_handle: T.handle, output_handle: T.handle, lse_handle: T.handle, rotary_mode: T.int32, rope_scale: T.float32, rope_theta: T.float32, sm_scale: T.float32):
            T.func_attr({"tir.is_scheduled": True, "global_symbol": "batch_decode_paged_kv_sliding_window"})
            B = T.int32(is_size_var=True)
            nnz_pages = T.int32(is_size_var=True)
            max_num_pages = T.int32(is_size_var=True)
            pages_elem_offset = T.int64(is_size_var=True)
            page_indptr_elem_offset = T.int32(is_size_var=True)
            page_values_elem_offset = T.int32(is_size_var=True)
            k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)
            q_rope_position_elem_offset = T.int32(is_size_var=True)
            length_info_elem_offset = T.int32(is_size_var=True)
            Q = T.match_buffer(Q_handle, (B, 32, 128), "float16")
            pages = T.match_buffer(pages_handle, (max_num_pages, 2, 4, 16, 128), "float16", elem_offset=pages_elem_offset)
            page_table_indptr = T.match_buffer(page_table_indptr_handle, (B + 1,), "int32", elem_offset=page_indptr_elem_offset)
            page_table_values = T.match_buffer(page_table_values_handle, (nnz_pages,), "int32", elem_offset=page_values_elem_offset)
            length_info = T.match_buffer(var_length_info, (3, B), "int32", elem_offset=length_info_elem_offset)
            k_rope_pos_offset = T.match_buffer(k_rope_pos_offset_handle, (B,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
            q_rope_position = T.match_buffer(q_rope_position_handle, (B,), "int32", elem_offset=q_rope_position_elem_offset)
            output = T.match_buffer(output_handle, (B, 32, 128), "float16")
            lse = T.match_buffer(lse_handle, (B, 32))
            # with T.sblock("root"):
            for bx in T.thread_binding(B, thread="blockIdx.x"):
                for fused_by_bz in T.thread_binding(4, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        for tx in T.thread_binding(32, thread="threadIdx.x"):
                            for tz in T.thread_binding(2, thread="threadIdx.z"):
                                with T.sblock("attn"):
                                    T.reads(page_table_indptr[bx:bx + 2], length_info[0:3, bx], q_rope_position[bx], Q[bx, fused_by_bz // 4 * 8 + fused_by_bz % 4 * 8 + ty, tx * 4 - 64:tx * 4 - 64 + 132])
                                    T.writes(output[bx, fused_by_bz % 4 * 8 + fused_by_bz // 4 * 8 + ty, tx * 4:tx * 4 + 4], lse[bx, fused_by_bz % 4 * 8 + fused_by_bz // 4 * 8 + ty])
                                    Q_local = T.alloc_buffer((4,), "float16", scope="local")
                                    kv_chunk_len = T.alloc_buffer((1,), "int32", scope="local")
                                    K_smem = T.alloc_buffer((16, 128), "float16", scope="shared")
                                    V_smem = T.alloc_buffer((16, 128), "float16", scope="shared")
                                    O_allreduce = T.alloc_buffer((2, 8, 128), scope="shared")
                                    md_allreduce = T.alloc_buffer((2, 8, 2), scope="shared")
                                    S_reduce_local = T.alloc_buffer((1,), scope="local")
                                    t0 = T.alloc_buffer((1,), scope="local")
                                    S_local = T.alloc_buffer((8,), scope="local")
                                    QK_local = T.alloc_buffer((4,), scope="local")
                                    V_local = T.alloc_buffer((4,), "float16", scope="local")
                                    m_prev = T.alloc_buffer((1,), scope="local")
                                    d_prev = T.alloc_buffer((1,), scope="local")
                                    other_m = T.alloc_buffer((1,), scope="local")
                                    other_d = T.alloc_buffer((1,), scope="local")
                                    exp_mprev = T.alloc_buffer((1,), scope="local")
                                    exp_otherm = T.alloc_buffer((1,), scope="local")
                                    other_o = T.alloc_buffer((4,), scope="local")
                                    st_m = T.alloc_buffer((1,), scope="local")
                                    st_d = T.alloc_buffer((1,), scope="local")
                                    O_local = T.alloc_buffer((4,), scope="local")
                                    by: T.int32 = fused_by_bz % 4
                                    bz: T.int32 = fused_by_bz // 4
                                    batch_idx: T.int32 = bx
                                    cur_page_indptr_begin: T.int32 = page_table_indptr[batch_idx]
                                    cur_page_indptr_end: T.int32 = page_table_indptr[batch_idx + 1]
                                    kv_chunk_len[0] = T.if_then_else(cur_page_indptr_begin != cur_page_indptr_end, (cur_page_indptr_end - cur_page_indptr_begin - 1) * 16 + length_info[0, batch_idx] - length_info[1, batch_idx] + length_info[2, batch_idx], 0)
                                    st_m[0] = T.float32(-50000.0)
                                    st_d[0] = T.float32(1.0)
                                    for vec in T.vectorized(4):
                                        O_local[vec] = T.float32(0.0)
                                    for vec in T.vectorized(4):
                                        freq = T.float32()
                                        Q_local[vec] = T.if_then_else(rotary_mode == 1, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", Q[bx, by * 8 + bz * 8 + ty, tx * 4 + vec]) + T.sin(freq) * T.Cast("float32", T.if_then_else(tx * 4 + vec < 64, Q[bx, by * 8 + bz * 8 + ty, tx * 4 + vec + 64] * T.float16(-1.0), Q[bx, by * 8 + bz * 8 + ty, tx * 4 + vec - 64]))), where={freq: T.Cast("float32", q_rope_position[batch_idx]) * rope_scale / T.pow(rope_theta, T.Cast("float32", (tx * 4 + vec) * 2 % 128) / T.float32(128.0))}), Q[bx, by * 8 + bz * 8 + ty, tx * 4 + vec])
                                    for iterator in range((kv_chunk_len[0] + 15) // 16):
                                        tile_start_s: T.int32(is_size_var=True) = tz * 8 + ty
                                        tile_start_g: T.int32(is_size_var=True) = (iterator * 2 + tz) * 8 + ty
                                        for j in range(1):
                                            with T.sblock("KV_load"):
                                                T.reads()
                                                T.writes()
                                                row_g: T.int32(is_size_var=True) = tile_start_g + j
                                                if row_g < kv_chunk_len[0]:
                                                    seq_offset: T.int32(is_size_var=True) = T.if_then_else(row_g < length_info[2, batch_idx], row_g, row_g - length_info[2, batch_idx] + length_info[1, batch_idx])
                                                    page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + seq_offset // 16]
                                                    page_offset: T.int32(is_size_var=True) = seq_offset % 16
                                                    for vec in T.vectorized(4):
                                                        freq = T.float32()
                                                        K_smem[tile_start_s + j, tx * 4 + vec] = T.if_then_else(rotary_mode == 1, T.Let(T.Cast("float16", T.cos(freq) * T.Cast("float32", pages[page_no, 0, by, page_offset, tx * 4 + vec]) + T.sin(freq) * T.Cast("float32", T.if_then_else(tx * 4 + vec < 64, pages[page_no, 0, by, page_offset, tx * 4 + vec + 64] * T.float16(-1.0), pages[page_no, 0, by, page_offset, tx * 4 + vec - 64]))), where={freq: T.Cast("float32", k_rope_pos_offset[batch_idx] + row_g) * rope_scale / T.pow(rope_theta, T.Cast("float32", (tx * 4 + vec) * 2 % 128) / T.float32(128.0))}), pages[page_no, 0, by, page_offset, tx * 4 + vec])
                                                        V_smem[tile_start_s + j, tx * 4 + vec] = pages[page_no, 1, by, page_offset, tx * 4 + vec]
                                                else:
                                                    for vec in T.vectorized(4):
                                                        K_smem[tile_start_s + j, tx * 4 + vec] = T.float16(0.0)
                                                        V_smem[tile_start_s + j, tx * 4 + vec] = T.float16(0.0)
                                        T.tvm_storage_sync("shared")
                                        m_prev[0] = st_m[0]
                                        for j in range(8):
                                            for vec in T.vectorized(4):
                                                QK_local[vec] = T.Cast("float32", Q_local[vec]) * T.Cast("float32", K_smem[tz * 8 + j, tx * 4 + vec]) * sm_scale * T.float32(1.4426950408889634)
                                            S_reduce_local[0] = T.float32(0.0)
                                            for vec in T.unroll(4):
                                                S_reduce_local[0] = S_reduce_local[0] + QK_local[vec]
                                            with T.sblock("block_cross_thread"):
                                                T.reads(S_reduce_local[0])
                                                T.writes(t0[0])
                                                T.attr(T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0.0)]), "reduce_scope", T.reinterpret("handle", T.uint64(0)))
                                                T.tvm_thread_allreduce(T.uint32(1), S_reduce_local[0], T.bool(True), t0[0], tx)
                                            S_local[j] = T.float32(-50000.0)
                                            if (iterator * 2 + tz) * 8 + j < kv_chunk_len[0]:
                                                S_local[j] = t0[0]
                                            st_m[0] = T.max(st_m[0], S_local[j])
                                        o_scale: T.float32 = T.exp2(m_prev[0] - st_m[0])
                                        st_d[0] = st_d[0] * o_scale
                                        for j in range(8):
                                            S_local[j] = T.exp2(S_local[j] - st_m[0])
                                            st_d[0] = st_d[0] + S_local[j]
                                        for j in T.vectorized(4):
                                            O_local[j] = O_local[j] * o_scale
                                        for j in range(8):
                                            for vec in T.vectorized(4):
                                                V_local[vec] = V_smem[tz * 8 + j, tx * 4 + vec]
                                            for vec in T.vectorized(4):
                                                O_local[vec] = O_local[vec] + T.Cast("float32", V_local[vec]) * S_local[j]
                                    for vec in T.vectorized(4):
                                        O_allreduce[tz, ty, tx * 4 + vec] = O_local[vec]
                                    md_allreduce[tz, ty, 0] = st_m[0]
                                    md_allreduce[tz, ty, 1] = st_d[0]
                                    T.tvm_storage_sync("shared")
                                    st_m[0] = T.float32(-50000.0)
                                    st_d[0] = T.float32(1.0)
                                    for vec in T.vectorized(4):
                                        O_local[vec] = T.float32(0.0)
                                    for j in range(2):
                                        m_prev[0] = st_m[0]
                                        d_prev[0] = st_d[0]
                                        other_m[0] = md_allreduce[j, ty, 0]
                                        other_d[0] = md_allreduce[j, ty, 1]
                                        for vec in T.vectorized(4):
                                            other_o[vec] = O_allreduce[j, ty, tx * 4 + vec]
                                        st_m[0] = T.max(st_m[0], other_m[0])
                                        st_d[0] = d_prev[0] * T.exp2(m_prev[0] - st_m[0]) + other_d[0] * T.exp2(other_m[0] - st_m[0])
                                        exp_mprev[0] = T.exp2(m_prev[0] - st_m[0])
                                        exp_otherm[0] = T.exp2(other_m[0] - st_m[0])
                                        for vec in T.vectorized(4):
                                            O_local[vec] = O_local[vec] * exp_mprev[0] + other_o[vec] * exp_otherm[0]
                                    for vec in T.vectorized(4):
                                        O_local[vec] = O_local[vec] / st_d[0]
                                    for vec in T.vectorized(4):
                                        output[batch_idx, by * 8 + bz * 8 + ty, tx * 4 + vec] = T.Cast("float16", O_local[vec])
                                    lse[batch_idx, by * 8 + bz * 8 + ty] = st_m[0] + T.log2(st_d[0])
        # fmt: on
        tvm.ir.assert_structural_equal(fn, batch_decode_paged_kv_sliding_window)


if __name__ == "__main__":
    tvm.testing.main()
