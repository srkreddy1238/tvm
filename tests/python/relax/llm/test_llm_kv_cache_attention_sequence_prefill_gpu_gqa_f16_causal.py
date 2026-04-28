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
"""Test for _attention_sequence_prefill - GQA h_kv=4, h_q=32, d=128, float16, causal."""

import pytest

import tvm
import tvm.testing
from tvm.relax.frontend.nn.llm.kv_cache import _attention_sequence_prefill
from tvm.script import tir as T


class TestAttentionSequencePrefillGPU_GQA_F16_Causal:
    @pytest.fixture(autouse=True)
    def _target(self):
        self.target = tvm.target.Target("cuda")

    def test_gqa_f16_causal(self):
        fn = _attention_sequence_prefill(4, 32, 128, "float16", self.target, causal=1)

        # fmt: off
        # from tvm.script import tir as T

        @T.prim_func
        def batch_sequence_prefill_kv(var_q: T.handle, var_k: T.handle, var_v: T.handle, var_output: T.handle, var_lse: T.handle):
            T.func_attr({"tir.is_scheduled": True})
            batch_size, qo_len = T.int32(is_size_var=True), T.int32(is_size_var=True)
            q = T.match_buffer(var_q, (batch_size, qo_len, 32, 128), "float16")
            kv_len = T.int32(is_size_var=True)
            k = T.match_buffer(var_k, (batch_size, kv_len, 4, 128), "float16")
            v = T.match_buffer(var_v, (batch_size, kv_len, 4, 128), "float16")
            output = T.match_buffer(var_output, (batch_size, qo_len, 32, 128), "float16")
            lse = T.match_buffer(var_lse, (batch_size, qo_len, 32), "float16")
            # with T.sblock("root"):
            batch_tiles: T.int32 = (qo_len * 8 + 32 - 1) // 32
            for lbx in T.thread_binding(batch_size * batch_tiles, thread="blockIdx.x"):
                for lby in T.thread_binding(4, thread="blockIdx.y"):
                    for lty in T.thread_binding(4, thread="threadIdx.y"):
                        for ltx in T.thread_binding(32, thread="threadIdx.x"):
                            with T.sblock("attn"):
                                vbx, by, ty, tx = T.axis.remap("SSSS", [lbx, lby, lty, ltx])
                                T.reads()
                                T.writes()
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
                                b_idx: T.int32 = vbx // batch_tiles
                                tile_id: T.int32 = vbx % batch_tiles
                                LH_start: T.int32 = tile_id * 32
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
                                                    cur_L: T.int32 = (LH_start + i) // 8
                                                    cur_H_qo: T.int32 = by * 8 + (LH_start + i) % 8
                                                    if cur_L < qo_len:
                                                        Q_smem[i, j] = q[b_idx, cur_L, cur_H_qo, j]
                                                    else:
                                                        Q_smem[i, j] = T.float16(0.0)
                                T.tvm_storage_sync("shared")
                                for iterator in range((kv_len + 31) // 32):
                                    L_kv_start: T.int32 = iterator * 32
                                    L_kv_base: T.int32 = 0
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
                                                        if cur_L < kv_len:
                                                            K_smem[i, j] = k[b_idx, L_kv_base + cur_L, by, j]
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
                                                        if cur_L < kv_len:
                                                            V_smem[i, j] = v[b_idx, L_kv_base + cur_L, by, j]
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
                                                                        k_1 = T.axis.reduce(128, lk_0 * 16 + lk_1)
                                                                        T.reads(S_local[i, j], Q_smem[i, k_1], K_smem[j, k_1])
                                                                        T.writes(S_local[i, j])
                                                                        S_local[i, j] = S_local[i, j] + T.Cast("float32", Q_smem[i, k_1]) * T.Cast("float32", K_smem[j, k_1]) * T.float32(1.4426950408889634)
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
                                                T.reads(m_smem[row], m_new[i], S_smem[row, 0:32], d_smem[row], m_prev[i])
                                                T.writes(m_prev[i], m_new[i], d_new[i])
                                                m_prev[i] = m_smem[row]
                                                m_new[i] = m_smem[row]
                                                row_ = (LH_start + row) // 8
                                                for j in range(32):
                                                    if L_kv_start + j < kv_len - qo_len + row_ + 1:
                                                        m_new[i] = T.max(m_new[i], S_smem[row, j])
                                                d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])
                                    for i in range(1):
                                        row: T.int32 = i * 32 * 4 + ty * 32 + tx
                                        with T.sblock("update"):
                                            T.reads(S_smem[row, 0:32], m_new[i])
                                            T.writes(S_smem[row, 0:32])
                                            for j in range(32):
                                                if row < 32:
                                                    row_ = (LH_start + row) // 8
                                                    if L_kv_start + j < kv_len - qo_len + row_ + 1:
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
                                                                    k_1 = T.axis.reduce(32, lk_0 * 16 + lk_1)
                                                                    T.reads(O_local[i, j], m_prev_smem[i], m_smem[i], S_smem[i, k_1], V_smem[k_1, j])
                                                                    T.writes(O_local[i, j])
                                                                    O_local[i, j] = O_local[i, j] + S_smem[i, k_1] * T.Cast("float32", V_smem[k_1, j])
                                for li_0_lj_0_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                    for li_0_lj_0_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                        for li_1 in range(4):
                                            for lj_1_0 in T.unroll(2):
                                                for lj_1_1 in T.vectorized(4):
                                                    with T.sblock("O_store"):
                                                        i = T.axis.spatial(32, (li_0_lj_0_fused_0 * 32 + li_0_lj_0_fused_1) // 16 * 4 + li_1)
                                                        j = T.axis.spatial(128, (li_0_lj_0_fused_0 * 32 + li_0_lj_0_fused_1) % 16 * 8 + lj_1_0 * 4 + lj_1_1)
                                                        T.reads(O_local[i, j], d_smem[i])
                                                        T.writes(output[b_idx, (LH_start + i) // 8, by * 8 + (LH_start + i) % 8, j])
                                                        cur_L: T.int32 = (LH_start + i) // 8
                                                        cur_H_qo: T.int32 = by * 8 + (LH_start + i) % 8
                                                        if cur_L < qo_len:
                                                            output[b_idx, cur_L, cur_H_qo, j] = T.Cast("float16", O_local[i, j] / d_smem[i])
                                for li_0 in range(1):
                                    for li_1 in T.thread_binding(4, thread="threadIdx.y"):
                                        for li_2 in T.thread_binding(32, thread="threadIdx.x"):
                                            with T.sblock("lse_store"):
                                                i = T.axis.spatial(32, li_0 * 128 + li_1 * 32 + li_2)
                                                T.where((li_0 * 4 + li_1) * 32 + li_2 < 32)
                                                T.reads(m_smem[i], d_smem[i])
                                                T.writes(lse[b_idx, (LH_start + i) // 8, by * 8 + (LH_start + i) % 8])
                                                cur_L: T.int32 = (LH_start + i) // 8
                                                cur_H_qo: T.int32 = by * 8 + (LH_start + i) % 8
                                                if cur_L < qo_len:
                                                    lse[b_idx, cur_L, cur_H_qo] = T.Cast("float16", m_smem[i] + T.log2(d_smem[i]))
        # fmt: on
        tvm.ir.assert_structural_equal(fn, batch_sequence_prefill_kv)


if __name__ == "__main__":
    tvm.testing.main()
