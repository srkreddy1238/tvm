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

import math
from typing import Any

from tvm import s_tir
from tvm.script import tir as T


def _var(dtype):
    return T.alloc_buffer((1,), dtype, scope="local")


def _causal_mask(causal, row, col, kv_len, qo_len):
    return T.if_then_else(
        causal > 0,
        col < kv_len - qo_len + row + 1,
        col < kv_len,
    )


def _declare_length_info(var_length_info, batch_size, sliding_window, elem_offset):
    return (
        T.match_buffer(var_length_info, (3, batch_size), "int32", elem_offset=elem_offset)
        if sliding_window
        else T.match_buffer(var_length_info, (batch_size,), "int32", elem_offset=elem_offset)
    )


def _get_kv_chunk_len(num_pages, page_size, seq_id, length_info, sliding_window):
    if not sliding_window:
        return (num_pages - 1) * page_size + length_info[seq_id]
    # ((num_pages - 1) * page_size + last_page_len) - sliding_window_offset + sink_size
    return (
        (num_pages - 1) * page_size
        + length_info[0, seq_id]
        - length_info[1, seq_id]
        + length_info[2, seq_id]
    )


def attention_prefill_ragged_adreno(h_kv, h_q, d_qk, d_v, dtype, rope_scaling: dict[str, Any]):
    tile_x = 256
    NUM_BLKS = 16
    group_size = h_q // h_kv

    @T.prim_func(private=True)
    def batch_prefill_ragged_kv(
        var_q: T.handle,
        var_q_indptr: T.handle,
        var_k: T.handle,
        var_v: T.handle,
        var_kv_indptr: T.handle,
        var_q_rope_position: T.handle,
        var_k_rope_pos_offset: T.handle,
        var_output: T.handle,
        var_lse: T.handle,  # [total_len, h_q]
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):
        T.func_attr({"tir.noalias": True, "tir.is_scheduled": 1})

        batch_size = T.int32(is_size_var=True)
        qo_len = T.int32(is_size_var=True)
        kv_len = T.int32(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        kv_indptr_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)

        q = T.match_buffer(var_q, (qo_len, h_q, d_qk), "float16")
        q_indptr = T.match_buffer(
            var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset
        )
        k = T.match_buffer(var_k, (kv_len, h_kv, d_qk), "float16")
        v = T.match_buffer(var_v, (kv_len, h_kv, d_v), "float16")
        kv_indptr = T.match_buffer(
            var_kv_indptr, (batch_size + 1,), "int32", elem_offset=kv_indptr_elem_offset
        )
        k_rope_pos_offset = T.match_buffer(  # noqa: F841
            var_k_rope_pos_offset,
            (batch_size,),
            "int32",
            elem_offset=k_rope_pos_offset_elem_offset,
        )
        output = T.match_buffer(var_output, (qo_len, h_q, d_v), "float16")
        lse = T.match_buffer(var_lse, (qo_len, h_q), "float32")  # pylint: disable=unused-variable
        with T.sblock("root"):
            Q_mem_pad = T.alloc_buffer(
                (
                    T.int64(h_q),
                    T.int64(((qo_len + tile_x - 1) // tile_x) * tile_x),
                    T.int64(d_qk),
                ),
                "float16",
                scope="global",
            )
            K_mem_pad = T.alloc_buffer(
                (T.int64(h_kv), T.int64(((kv_len + 63) // 64) * 64), T.int64(d_qk)),
                "float16",
                scope="global",
            )
            V_mem_pad = T.alloc_buffer(
                (T.int64(h_kv), T.int64(((kv_len + 63) // 64) * 64), T.int64(d_v)),
                "float16",
                scope="global",
            )
            for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
                for lby in T.thread_binding(h_q, thread="blockIdx.y"):
                    for lbz in T.thread_binding(d_qk // 16, thread="blockIdx.z"):
                        with T.sblock("Q_load_root"):
                            bx, by, bz = T.axis.remap("SSS", [lbx, lby, lbz])
                            T.reads()
                            T.writes()
                            tile_id = T.alloc_buffer((1,), "int32", scope="local")
                            batch_idx = T.alloc_buffer((1,), "int32", scope="local")
                            batch_tiles = T.alloc_buffer((1,), "int32", scope="local")
                            batch_rows = T.alloc_buffer((1,), "int32", scope="local")
                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = q_indptr[1] - q_indptr[0]
                            batch_tiles[0] = (batch_rows[0] + tile_x - 1) // tile_x
                            while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size:
                                    tile_id[0] = tile_id[0] - batch_tiles[0]
                                    batch_idx[0] = batch_idx[0] + 1
                                    if batch_idx[0] < batch_size:
                                        b_idx: T.int32 = batch_idx[0]
                                        batch_rows[0] = q_indptr[b_idx + 1] - q_indptr[b_idx]
                                        batch_tiles[0] = (batch_rows[0] + tile_x - 1) // tile_x
                                if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                    b_idx: T.int32 = batch_idx[0]
                                    q_indptr_val: T.int32 = q_indptr[b_idx]
                                    LH_start: T.int32 = tile_id[0] * tile_x
                                    for ty in T.thread_binding(4, thread="threadIdx.y"):
                                        for tx in T.thread_binding(64, thread="threadIdx.x"):
                                            for lu in T.unroll(tile_x // 64):
                                                for lv in T.vectorized(4):
                                                    with T.sblock("Q_load"):
                                                        i = T.axis.spatial(tile_x, lu * 64 + tx)
                                                        j = T.axis.spatial(
                                                            d_qk, bz * 16 + ty * 4 + lv
                                                        )
                                                        T.reads()
                                                        T.writes()
                                                        cur_L: T.int32 = q_indptr_val + (
                                                            LH_start + i
                                                        )
                                                        cur_H_qo: T.int32 = by
                                                        Q_mem_pad[cur_H_qo, cur_L, j] = (
                                                            T.if_then_else(
                                                                cur_L < q_indptr[b_idx + 1],
                                                                q[cur_L, cur_H_qo, j],
                                                                T.float16(0.0),
                                                            )
                                                        )
                                tile_id[0] += NUM_BLKS

            for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
                for lby in T.thread_binding(h_kv, thread="blockIdx.y"):
                    for lbz in T.thread_binding(d_qk // 16, thread="blockIdx.z"):
                        with T.sblock("K_load_root"):
                            bx, by, bz = T.axis.remap("SSS", [lbx, lby, lbz])
                            T.reads()
                            T.writes()
                            tile_id = T.alloc_buffer((1,), "int32", scope="local")
                            batch_idx = T.alloc_buffer((1,), "int32", scope="local")
                            batch_tiles = T.alloc_buffer((1,), "int32", scope="local")
                            batch_rows = T.alloc_buffer((1,), "int32", scope="local")
                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = kv_indptr[1] - kv_indptr[0]
                            batch_tiles[0] = (batch_rows[0] + 64 - 1) // 64
                            while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size:
                                    tile_id[0] = tile_id[0] - batch_tiles[0]
                                    batch_idx[0] = batch_idx[0] + 1
                                    if batch_idx[0] < batch_size:
                                        b_idx: T.int32 = batch_idx[0]
                                        batch_rows[0] = kv_indptr[b_idx + 1] - kv_indptr[b_idx]
                                        batch_tiles[0] = (batch_rows[0] + 64 - 1) // 64
                                if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                    b_idx: T.int32 = batch_idx[0]
                                    kv_indptr_val: T.int32 = kv_indptr[b_idx]
                                    LH_start: T.int32 = tile_id[0] * 64
                                    for ty in T.thread_binding(4, thread="threadIdx.y"):
                                        for tx in T.thread_binding(64, thread="threadIdx.x"):
                                            for lv in T.vectorized(4):
                                                with T.sblock("K_load"):
                                                    i = T.axis.spatial(64, tx)
                                                    j = T.axis.spatial(d_qk, bz * 16 + ty * 4 + lv)
                                                    T.reads()
                                                    T.writes()
                                                    cur_L: T.int32 = LH_start + i
                                                    cur_H_qo: T.int32 = by
                                                    K_mem_pad[cur_H_qo, cur_L, j] = T.if_then_else(
                                                        cur_L < kv_indptr[b_idx + 1],
                                                        k[cur_L, cur_H_qo, j],
                                                        T.float16(0.0),
                                                    )
                                tile_id[0] += NUM_BLKS

            for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
                for lby in T.thread_binding(h_kv, thread="blockIdx.y"):
                    for lbz in T.thread_binding(d_v // 16, thread="blockIdx.z"):
                        with T.sblock("V_load_root"):
                            bx, by, bz = T.axis.remap("SSS", [lbx, lby, lbz])
                            T.reads()
                            T.writes()
                            tile_id = T.alloc_buffer((1,), "int32", scope="local")
                            batch_idx = T.alloc_buffer((1,), "int32", scope="local")
                            batch_tiles = T.alloc_buffer((1,), "int32", scope="local")
                            batch_rows = T.alloc_buffer((1,), "int32", scope="local")
                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = kv_indptr[1] - kv_indptr[0]
                            batch_tiles[0] = (batch_rows[0] + 64 - 1) // 64
                            while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size:
                                    tile_id[0] = tile_id[0] - batch_tiles[0]
                                    batch_idx[0] = batch_idx[0] + 1
                                    if batch_idx[0] < batch_size:
                                        b_idx: T.int32 = batch_idx[0]
                                        batch_rows[0] = kv_indptr[b_idx + 1] - kv_indptr[b_idx]
                                        batch_tiles[0] = (batch_rows[0] + 64 - 1) // 64
                                if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                    b_idx: T.int32 = batch_idx[0]
                                    kv_indptr_val: T.int32 = kv_indptr[b_idx]
                                    LH_start: T.int32 = tile_id[0] * 64
                                    for ty in T.thread_binding(4, thread="threadIdx.y"):
                                        for tx in T.thread_binding(64, thread="threadIdx.x"):
                                            for lv in T.vectorized(4):
                                                with T.sblock("V_load"):
                                                    i = T.axis.spatial(64, tx)
                                                    j = T.axis.spatial(d_v, bz * 16 + ty * 4 + lv)
                                                    T.reads()
                                                    T.writes()
                                                    cur_L: T.int32 = kv_indptr_val + (LH_start + i)
                                                    cur_H_qo: T.int32 = by
                                                    V_mem_pad[cur_H_qo, cur_L, j] = T.if_then_else(
                                                        cur_L < kv_indptr[b_idx + 1],
                                                        v[cur_L, cur_H_qo, j],
                                                        T.float16(0.0),
                                                    )
                                tile_id[0] += NUM_BLKS

            for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
                for lby in T.thread_binding(h_q, thread="blockIdx.y"):
                    for lbz in T.thread_binding(d_v // 64, thread="blockIdx.z"):
                        for ltz in T.thread_binding(tile_x // 64, thread="threadIdx.z"):
                            for lty in T.thread_binding(1, thread="threadIdx.y"):
                                for ltx in T.thread_binding(64, thread="threadIdx.x"):
                                    with T.sblock("attn"):
                                        bx, by, bz, tz, ty, tx = T.axis.remap(
                                            "SSSSSS", [lbx, lby, lbz, ltz, lty, ltx]
                                        )
                                        T.reads()
                                        T.writes()
                                        kv_chunk_len = T.alloc_buffer((1,), "int32", scope="local")
                                        Q_frag = T.alloc_buffer(
                                            (tile_x, 16),
                                            "float16",
                                            scope="wmma.matrix_a",
                                        )
                                        K_frag = T.alloc_buffer(
                                            (64, 16), "float16", scope="wmma.matrix_b"
                                        )
                                        S_frag = T.alloc_buffer(
                                            (tile_x, 64),
                                            "float16",
                                            scope="wmma.accumulator",
                                        )
                                        Smem_frag = T.alloc_buffer(
                                            (64, 16), "float16", scope="wmma.matrix_a"
                                        )
                                        S_local = T.alloc_buffer(
                                            (tile_x, 64), "float16", scope="local"
                                        )
                                        S_local_1 = T.alloc_buffer(
                                            (64, 64), "float16", scope="local"
                                        )
                                        O_local = T.alloc_buffer(
                                            (tile_x, d_v), "float16", scope="local"
                                        )
                                        O_frag = T.alloc_buffer(
                                            (64, d_v),
                                            "float16",
                                            scope="wmma.accumulator",
                                        )
                                        V_frag = T.alloc_buffer(
                                            (16, d_v), "float16", scope="wmma.matrix_b"
                                        )
                                        m_smem = T.alloc_buffer((tile_x,), scope="local")
                                        m_prev_smem = T.alloc_buffer((tile_x,), scope="local")
                                        d_smem = T.alloc_buffer((tile_x,), scope="local")
                                        m_new = T.alloc_buffer((1,), scope="local")
                                        m_prev = T.alloc_buffer((1,), scope="local")
                                        d_new = T.alloc_buffer((1,), scope="local")
                                        tile_id = T.alloc_buffer((1,), "int32", scope="local")
                                        batch_idx = T.alloc_buffer((1,), "int32", scope="local")
                                        batch_tiles = T.alloc_buffer((1,), "int32", scope="local")
                                        batch_rows = T.alloc_buffer((1,), "int32", scope="local")
                                        tile_id[0] = bx
                                        batch_idx[0] = 0
                                        batch_rows[0] = q_indptr[1] - q_indptr[0]
                                        batch_tiles[0] = (batch_rows[0] + tile_x - 1) // tile_x
                                        while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                            while (
                                                tile_id[0] >= batch_tiles[0]
                                                and batch_idx[0] < batch_size
                                            ):
                                                tile_id[0] = tile_id[0] - batch_tiles[0]
                                                batch_idx[0] = batch_idx[0] + 1
                                                if batch_idx[0] < batch_size:
                                                    b_idx: T.int32 = batch_idx[0]
                                                    batch_rows[0] = (
                                                        q_indptr[b_idx + 1] - q_indptr[b_idx]
                                                    )
                                                    batch_tiles[0] = (
                                                        batch_rows[0] + tile_x - 1
                                                    ) // tile_x
                                            if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                                b_idx: T.int32 = batch_idx[0]
                                                q_indptr_val: T.int32 = q_indptr[b_idx]
                                                LH_start: T.int32 = tile_id[0] * tile_x
                                                kv_chunk_len[0] = (
                                                    kv_indptr[b_idx + 1] - kv_indptr[b_idx]
                                                )
                                                for i in range(1):
                                                    row: T.int32 = i * tile_x + tz * 64 + tx
                                                    if row < tile_x:
                                                        m_smem[row] = T.float32(-50000.0)
                                                        d_smem[row] = T.float32(1.0)
                                                with T.sblock("O_frag_init"):
                                                    T.reads()
                                                    T.writes(O_frag[0:64, bz * 64 : bz * 64 + 64])
                                                    C = T.match_buffer(
                                                        O_frag[0:64, bz * 64 : bz * 64 + 64],
                                                        (64, 64),
                                                        "float16",
                                                        strides=("C_s0", "C_s1"),
                                                        scope="wmma.accumulator",
                                                        offset_factor=64,
                                                    )
                                                    # zero the accumulator fragment
                                                    T.tvm_fill_fragment(
                                                        C.data,
                                                        64,
                                                        64,
                                                        16,
                                                        C.elem_offset
                                                        // C.strides[0]
                                                        // 64
                                                        * (C.strides[0] // 64)
                                                        + C.elem_offset % C.strides[0] // 64,
                                                        T.float32(0.0),
                                                    )
                                                for iterator_1 in range(
                                                    (kv_chunk_len[0] + 63) // 64
                                                ):
                                                    L_kv_start: T.int32 = iterator_1 * 64
                                                    L_kv_base: T.int32 = kv_indptr[b_idx]
                                                    with T.sblock(""):
                                                        with T.sblock("S_frag_init"):
                                                            T.reads()
                                                            T.writes(
                                                                S_frag[
                                                                    tz * 64 : tz * 64 + 64,
                                                                    0:64,
                                                                ]
                                                            )
                                                            C = T.match_buffer(
                                                                S_frag[
                                                                    tz * 64 : tz * 64 + 64,
                                                                    0:64,
                                                                ],
                                                                (64, 64),
                                                                "float16",
                                                                strides=(
                                                                    "C_s0",
                                                                    "C_s1",
                                                                ),
                                                                scope="wmma.accumulator",
                                                                offset_factor=64,
                                                            )
                                                            # zero the accumulator fragment
                                                            T.tvm_fill_fragment(
                                                                C.data,
                                                                64,
                                                                64,
                                                                16,
                                                                C.elem_offset
                                                                // C.strides[0]
                                                                // 64
                                                                * (C.strides[0] // 64)
                                                                + C.elem_offset
                                                                % C.strides[0]
                                                                // 64,
                                                                T.float32(0.0),
                                                            )
                                                        for lk_11 in range(d_qk // 16):
                                                            with T.sblock("Q_wmma_a_load"):
                                                                T.reads()
                                                                T.writes()
                                                                A_src = T.match_buffer(
                                                                    Q_mem_pad[
                                                                        by,
                                                                        q_indptr_val
                                                                        + LH_start
                                                                        + tz * 64 : q_indptr_val
                                                                        + LH_start
                                                                        + tz * 64
                                                                        + 64,
                                                                        lk_11 * 16 : lk_11 * 16
                                                                        + 16,
                                                                    ],
                                                                    (64, 16),
                                                                    "float16",
                                                                    strides=(
                                                                        "A_s0",
                                                                        "A_s1",
                                                                    ),
                                                                    scope="global",
                                                                    offset_factor=16,
                                                                )
                                                                A_dst = T.match_buffer(
                                                                    Q_frag[
                                                                        tz * 64 : tz * 64 + 64,
                                                                        0:16,
                                                                    ],
                                                                    (64, 16),
                                                                    "float16",
                                                                    strides=(
                                                                        "C_s0",
                                                                        "C_s1",
                                                                    ),
                                                                    scope="wmma.matrix_a",
                                                                    offset_factor=16,
                                                                )
                                                                T.tvm_load_matrix_sync(
                                                                    A_dst.data,
                                                                    64,
                                                                    64,
                                                                    16,
                                                                    A_dst.elem_offset
                                                                    // A_dst.strides[0]
                                                                    // 64
                                                                    * (A_dst.strides[0] // 16)
                                                                    + A_dst.elem_offset
                                                                    % A_dst.strides[0]
                                                                    // 16,
                                                                    T.tvm_access_ptr(
                                                                        T.type_annotation(
                                                                            "float16"
                                                                        ),
                                                                        A_src.data,
                                                                        A_src.elem_offset,
                                                                        A_src.strides[0] * 64,
                                                                        1,
                                                                    ),
                                                                    A_src.strides[0],
                                                                    "row_major",
                                                                )
                                                            with T.sblock("K_wmma_b_load"):
                                                                T.reads()
                                                                T.writes()
                                                                B_src = T.match_buffer(
                                                                    K_mem_pad[
                                                                        by // group_size,
                                                                        L_kv_base
                                                                        + L_kv_start : L_kv_base
                                                                        + L_kv_start
                                                                        + 64,
                                                                        lk_11 * 16 : lk_11 * 16
                                                                        + 16,
                                                                    ],
                                                                    (64, 16),
                                                                    "float16",
                                                                    strides=(
                                                                        "B_s0",
                                                                        "B_s1",
                                                                    ),
                                                                    scope="global",
                                                                    offset_factor=16,
                                                                )
                                                                B_dst = T.match_buffer(
                                                                    K_frag[0:64, 0:16],
                                                                    (64, 16),
                                                                    "float16",
                                                                    strides=(
                                                                        "C_s0",
                                                                        "C_s1",
                                                                    ),
                                                                    scope="wmma.matrix_b",
                                                                    offset_factor=16,
                                                                )
                                                                T.tvm_load_matrix_sync(
                                                                    B_dst.data,
                                                                    64,
                                                                    64,
                                                                    16,
                                                                    B_dst.elem_offset
                                                                    // B_dst.strides[0]
                                                                    // 64
                                                                    * (B_dst.strides[0] // 16)
                                                                    + B_dst.elem_offset
                                                                    % B_dst.strides[0]
                                                                    // 16,
                                                                    T.tvm_access_ptr(
                                                                        T.type_annotation(
                                                                            "float16"
                                                                        ),
                                                                        B_src.data,
                                                                        B_src.elem_offset,
                                                                        B_src.strides[0] * 64,
                                                                        1,
                                                                    ),
                                                                    B_src.strides[0],
                                                                    "col_major",
                                                                )
                                                            with T.sblock(""):
                                                                with T.sblock("S_mma_sync"):
                                                                    T.reads()
                                                                    T.writes(
                                                                        S_frag[
                                                                            tz * 64 : tz * 64 + 64,
                                                                            0:64,
                                                                        ]
                                                                    )
                                                                    C = T.match_buffer(
                                                                        S_frag[
                                                                            tz * 64 : tz * 64 + 64,
                                                                            0:64,
                                                                        ],
                                                                        (64, 64),
                                                                        "float16",
                                                                        strides=(
                                                                            "C_s0",
                                                                            "C_s1",
                                                                        ),
                                                                        scope="wmma.accumulator",
                                                                        offset_factor=64,
                                                                    )
                                                                    A = T.match_buffer(
                                                                        Q_frag[
                                                                            tz * 64 : tz * 64 + 64,
                                                                            0:16,
                                                                        ],
                                                                        (64, 16),
                                                                        "float16",
                                                                        strides=(
                                                                            "A_s0",
                                                                            "A_s1",
                                                                        ),
                                                                        scope="wmma.matrix_a",
                                                                        offset_factor=16,
                                                                    )
                                                                    B = T.match_buffer(
                                                                        K_frag[0:64, 0:16],
                                                                        (64, 16),
                                                                        "float16",
                                                                        strides=(
                                                                            "B_s0",
                                                                            "B_s1",
                                                                        ),
                                                                        scope="wmma.matrix_b",
                                                                        offset_factor=16,
                                                                    )
                                                                    T.tvm_mma_sync(
                                                                        C.data,
                                                                        C.elem_offset
                                                                        // C.strides[0]
                                                                        // 64
                                                                        * (C.strides[0] // 64)
                                                                        + C.elem_offset
                                                                        % C.strides[0]
                                                                        // 64,
                                                                        A.data,
                                                                        A.elem_offset
                                                                        // A.strides[0]
                                                                        // 64
                                                                        * (A.strides[0] // 16)
                                                                        + A.elem_offset
                                                                        % A.strides[0]
                                                                        // 16,
                                                                        B.data,
                                                                        B.elem_offset
                                                                        // B.strides[0]
                                                                        // 64
                                                                        * (B.strides[0] // 16)
                                                                        + B.elem_offset
                                                                        % B.strides[0]
                                                                        // 16,
                                                                        C.data,
                                                                        C.elem_offset
                                                                        // C.strides[0]
                                                                        // 64
                                                                        * (C.strides[0] // 64)
                                                                        + C.elem_offset
                                                                        % C.strides[0]
                                                                        // 64,
                                                                    )
                                                    for li_0_lj_0_fused_1 in T.thread_binding(
                                                        64, thread="threadIdx.x"
                                                    ):
                                                        with T.sblock("S_frag_store_local"):
                                                            T.reads(
                                                                S_frag[
                                                                    tz * 64 : tz * 64 + 64,
                                                                    0:64,
                                                                ]
                                                            )
                                                            T.writes(
                                                                S_local[
                                                                    tz * 64 : tz * 64 + 64,
                                                                    0:64,
                                                                ]
                                                            )
                                                            i = T.axis.spatial(
                                                                64, li_0_lj_0_fused_1
                                                            )
                                                            C = T.match_buffer(
                                                                S_frag[tz * 64 + i, 0:64],
                                                                (64,),
                                                                "float16",
                                                                scope="wmma.accumulator",
                                                                offset_factor=64,
                                                            )
                                                            L = T.match_buffer(
                                                                S_local[tz * 64 + i, 0:64],
                                                                (64,),
                                                                "float16",
                                                                scope="local",
                                                                offset_factor=64,
                                                            )
                                                            T.tvm_deconstruct_coopmat_qcom(
                                                                C.data,
                                                                64,
                                                                64,
                                                                16,
                                                                L.data,
                                                            )
                                                    for li_0_lj_0_fused_1 in T.thread_binding(
                                                        64, thread="threadIdx.x"
                                                    ):
                                                        for lj_1_0 in T.unroll(16):
                                                            for lj_1_1 in T.vectorized(4):
                                                                with T.sblock("S_store"):
                                                                    i = T.axis.spatial(
                                                                        tile_x,
                                                                        (
                                                                            tz * 64
                                                                            + li_0_lj_0_fused_1
                                                                        ),
                                                                    )
                                                                    j = T.axis.spatial(
                                                                        64,
                                                                        lj_1_0 * 4 + lj_1_1,
                                                                    )
                                                                    T.reads(S_local[i, j])
                                                                    T.writes(S_local[i, j])
                                                                    S_local[i, j] = (
                                                                        T.Cast(
                                                                            "float32",
                                                                            S_local[i, j],
                                                                        )
                                                                        * sm_scale
                                                                        * math.log2(math.exp(1))
                                                                    )
                                                    for i in range(1):
                                                        row: T.int32 = i * tile_x + tz * 64 + tx
                                                        if row < tile_x:
                                                            with T.sblock("update1"):
                                                                T.reads(
                                                                    m_smem[row],
                                                                    kv_chunk_len[0],
                                                                    m_new[i],
                                                                    S_local[row, 0:64],
                                                                    d_smem[row],
                                                                    m_prev[i],
                                                                )
                                                                T.writes(
                                                                    m_prev[i],
                                                                    m_new[i],
                                                                    d_new[i],
                                                                )
                                                                m_prev[i] = m_smem[row]
                                                                m_new[i] = m_smem[row]
                                                                row_: T.int32 = LH_start + row
                                                                for j in range(64):
                                                                    if _causal_mask(
                                                                        causal,
                                                                        row=row_,
                                                                        col=L_kv_start + j,
                                                                        kv_len=kv_chunk_len[0],
                                                                        qo_len=q_indptr[b_idx + 1]
                                                                        - q_indptr[b_idx],
                                                                    ):
                                                                        m_new[i] = T.max(
                                                                            m_new[i],
                                                                            S_local[
                                                                                row,
                                                                                j,
                                                                            ],
                                                                        )
                                                                d_new[i] = d_smem[row] * T.exp2(
                                                                    m_prev[i] - m_new[i]
                                                                )
                                                    for i in range(1):
                                                        row: T.int32 = i * tile_x + tz * 64 + tx
                                                        with T.sblock("update"):
                                                            T.reads(
                                                                kv_chunk_len[0],
                                                                S_local[row, 0:64],
                                                                m_new[i],
                                                            )
                                                            T.writes(S_local[row, 0:64])
                                                            for j in range(64):
                                                                if row < tile_x:
                                                                    row_: T.int32 = LH_start + row
                                                                    if _causal_mask(
                                                                        causal,
                                                                        row=row_,
                                                                        col=L_kv_start + j,
                                                                        kv_len=kv_chunk_len[0],
                                                                        qo_len=q_indptr[b_idx + 1]
                                                                        - q_indptr[b_idx],
                                                                    ):
                                                                        S_local[row, j] = T.exp2(
                                                                            S_local[row, j]
                                                                            - m_new[i]
                                                                        )
                                                                    else:
                                                                        S_local[row, j] = T.exp2(
                                                                            T.float32(-50000.0)
                                                                            - m_new[i]
                                                                        )
                                                    for i in range(1):
                                                        row: T.int32 = i * tile_x + tz * 64 + tx
                                                        if row < tile_x:
                                                            with T.sblock("update"):
                                                                T.reads(
                                                                    d_new[i],
                                                                    S_local[row, 0:64],
                                                                    m_new[i],
                                                                    m_prev[i],
                                                                )
                                                                T.writes(
                                                                    d_new[i],
                                                                    m_smem[row],
                                                                    d_smem[row],
                                                                    m_prev_smem[row],
                                                                )
                                                                for j in range(64):
                                                                    d_new[i] = (
                                                                        d_new[i] + S_local[row, j]
                                                                    )
                                                                m_smem[row] = m_new[i]
                                                                d_smem[row] = d_new[i]
                                                                m_prev_smem[row] = m_prev[i]
                                                    with T.sblock(""):
                                                        T.reads(
                                                            m_prev_smem[0:tile_x],
                                                            m_smem[0:tile_x],
                                                            S_local[0:tile_x, 0:64],
                                                        )
                                                        T.writes(O_local[0:tile_x, 0:64])

                                                        for li_0_lj_0_fused_1 in T.thread_binding(
                                                            64, thread="threadIdx.x"
                                                        ):
                                                            with T.sblock("O_frag_store_local"):
                                                                T.reads(O_frag[0:64, 0:64])
                                                                T.writes(O_local[0:64, 0:64])
                                                                i = T.axis.spatial(
                                                                    tile_x,
                                                                    tz * 64 + li_0_lj_0_fused_1,
                                                                )
                                                                C = T.match_buffer(
                                                                    O_frag[
                                                                        i,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (64,),
                                                                    "float16",
                                                                    scope="wmma.accumulator",
                                                                    offset_factor=64,
                                                                )
                                                                L = T.match_buffer(
                                                                    O_local[
                                                                        i,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (64,),
                                                                    "float16",
                                                                    scope="local",
                                                                    offset_factor=64,
                                                                )
                                                                T.tvm_deconstruct_coopmat_qcom(
                                                                    C.data,
                                                                    64,
                                                                    64,
                                                                    16,
                                                                    L.data,
                                                                )
                                                        for li_0_1_init in T.thread_binding(
                                                            64, thread="threadIdx.x"
                                                        ):
                                                            for lj_1_0_init in T.unroll(16):
                                                                for lj_1_1_init in T.vectorized(4):
                                                                    with T.sblock("O_gemm_init"):
                                                                        i = T.axis.spatial(
                                                                            tile_x,
                                                                            (tz * 64 + li_0_1_init),
                                                                        )
                                                                        j = T.axis.spatial(
                                                                            d_v,
                                                                            bz * 64
                                                                            + ty * 64
                                                                            + lj_1_0_init * 4
                                                                            + lj_1_1_init,
                                                                        )
                                                                        T.reads()
                                                                        T.writes(O_local[i, j])
                                                                        O_local[i, j] = O_local[
                                                                            i, j
                                                                        ] * T.exp2(
                                                                            m_prev_smem[i]
                                                                            - m_smem[i]
                                                                        )
                                                        for li_0_lj_0_fused_1 in T.thread_binding(
                                                            64, thread="threadIdx.x"
                                                        ):
                                                            with T.sblock("O_frag_store_local"):
                                                                T.reads(O_frag[0:64, 0:64])
                                                                T.writes(O_local[0:64, 0:64])
                                                                i = T.axis.spatial(
                                                                    tile_x,
                                                                    tz * 64 + li_0_lj_0_fused_1,
                                                                )
                                                                C = T.match_buffer(
                                                                    O_frag[
                                                                        i,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (64,),
                                                                    "float16",
                                                                    scope="wmma.accumulator",
                                                                    offset_factor=64,
                                                                )
                                                                L = T.match_buffer(
                                                                    O_local[
                                                                        i,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (64,),
                                                                    "float16",
                                                                    scope="local",
                                                                    offset_factor=64,
                                                                )
                                                                T.tvm_construct_coopmat_qcom(
                                                                    C.data,
                                                                    64,
                                                                    64,
                                                                    64,
                                                                    L.data,
                                                                )
                                                        for lv_11 in range(4):
                                                            for (
                                                                li_0_lj_0_fused_1
                                                            ) in T.thread_binding(
                                                                64, thread="threadIdx.x"
                                                            ):
                                                                for i_01 in T.serial(16):
                                                                    with T.sblock("Smem_loal_load"):
                                                                        i = T.axis.spatial(
                                                                            tile_x,
                                                                            tz * 64
                                                                            + li_0_lj_0_fused_1,
                                                                        )
                                                                        j = T.axis.spatial(
                                                                            64,
                                                                            lv_11 * 16 + i_01,
                                                                        )
                                                                        jk = T.axis.spatial(
                                                                            16, i_01
                                                                        )
                                                                        T.reads()
                                                                        T.writes()
                                                                        S_local_1[i, jk] = S_local[
                                                                            i, j
                                                                        ]

                                                            for (
                                                                li_0_lj_0_fused_1
                                                            ) in T.thread_binding(
                                                                64, thread="threadIdx.x"
                                                            ):
                                                                with T.sblock("S_wmma_a_load"):
                                                                    i = T.axis.spatial(
                                                                        tile_x,
                                                                        tz * 64 + li_0_lj_0_fused_1,
                                                                    )
                                                                    T.reads(S_local_1[i, 0:16])
                                                                    T.writes()
                                                                    A_src = T.match_buffer(
                                                                        S_local_1[i, 0:16],
                                                                        (16,),
                                                                        "float16",
                                                                        scope="local",
                                                                        offset_factor=16,
                                                                    )
                                                                    A_dst = T.match_buffer(
                                                                        Smem_frag[i, 0:16],
                                                                        (16,),
                                                                        "float16",
                                                                        scope="wmma.matrix_a",
                                                                        offset_factor=16,
                                                                    )
                                                                    T.tvm_construct_coopmat_qcom(
                                                                        A_dst.data,
                                                                        64,
                                                                        64,
                                                                        16,
                                                                        A_src.data,
                                                                    )
                                                            with T.sblock("V_wmma_b_load"):
                                                                T.reads()
                                                                T.writes()
                                                                B_src = T.match_buffer(
                                                                    V_mem_pad[
                                                                        by // group_size,
                                                                        L_kv_base
                                                                        + L_kv_start
                                                                        + lv_11 * 16 : L_kv_base
                                                                        + L_kv_start
                                                                        + lv_11 * 16
                                                                        + 16,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (16, 64),
                                                                    "float16",
                                                                    strides=(
                                                                        "B_s0",
                                                                        "B_s1",
                                                                    ),
                                                                    scope="global",
                                                                    offset_factor=64,
                                                                )
                                                                B_dst = T.match_buffer(
                                                                    V_frag[
                                                                        0:16,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (16, 64),
                                                                    "float16",
                                                                    strides=(
                                                                        "C_s0",
                                                                        "C_s1",
                                                                    ),
                                                                    scope="wmma.matrix_b",
                                                                    offset_factor=64,
                                                                )
                                                                T.tvm_load_matrix_sync(
                                                                    B_dst.data,
                                                                    64,
                                                                    64,
                                                                    16,
                                                                    B_dst.elem_offset
                                                                    // B_dst.strides[0]
                                                                    // 16
                                                                    * (B_dst.strides[0] // 64)
                                                                    + B_dst.elem_offset
                                                                    % B_dst.strides[0]
                                                                    // 64,
                                                                    T.tvm_access_ptr(
                                                                        T.type_annotation(
                                                                            "float16"
                                                                        ),
                                                                        B_src.data,
                                                                        B_src.elem_offset,
                                                                        B_src.strides[0] * 16,
                                                                        1,
                                                                    ),
                                                                    B_src.strides[0],
                                                                    "row_major",
                                                                )

                                                            with T.sblock("O_mma_sync"):
                                                                T.reads()
                                                                T.writes(O_frag[0:64, 0:64])
                                                                C = T.match_buffer(
                                                                    O_frag[
                                                                        0:64,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (64, 64),
                                                                    "float16",
                                                                    strides=(
                                                                        "C_s0",
                                                                        "C_s1",
                                                                    ),
                                                                    scope="wmma.accumulator",
                                                                    offset_factor=64,
                                                                )
                                                                A = T.match_buffer(
                                                                    Smem_frag[0:64, 0:16],
                                                                    (64, 16),
                                                                    "float16",
                                                                    strides=(
                                                                        "A_s0",
                                                                        "A_s1",
                                                                    ),
                                                                    scope="wmma.matrix_a",
                                                                    offset_factor=16,
                                                                )
                                                                B = T.match_buffer(
                                                                    V_frag[
                                                                        0:16,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (16, 64),
                                                                    "float16",
                                                                    strides=(
                                                                        "B_s0",
                                                                        "B_s1",
                                                                    ),
                                                                    scope="wmma.matrix_b",
                                                                    offset_factor=64,
                                                                )
                                                                T.tvm_mma_sync(
                                                                    C.data,
                                                                    C.elem_offset
                                                                    // C.strides[0]
                                                                    // 64
                                                                    * (C.strides[0] // 64)
                                                                    + C.elem_offset
                                                                    % C.strides[0]
                                                                    // 64,
                                                                    A.data,
                                                                    A.elem_offset
                                                                    // A.strides[0]
                                                                    // 64
                                                                    * (A.strides[0] // 16)
                                                                    + A.elem_offset
                                                                    % A.strides[0]
                                                                    // 16,
                                                                    B.data,
                                                                    B.elem_offset
                                                                    // B.strides[0]
                                                                    // 16
                                                                    * (B.strides[0] // 64)
                                                                    + B.elem_offset
                                                                    % B.strides[0]
                                                                    // 64,
                                                                    C.data,
                                                                    C.elem_offset
                                                                    // C.strides[0]
                                                                    // 64
                                                                    * (C.strides[0] // 64)
                                                                    + C.elem_offset
                                                                    % C.strides[0]
                                                                    // 64,
                                                                )
                                                for li_0_lj_0_fused_1 in T.thread_binding(
                                                    64, thread="threadIdx.x"
                                                ):
                                                    with T.sblock("O_frag_store_local"):
                                                        T.reads(O_frag[0:64, 0:64])
                                                        T.writes(O_local[0:64, 0:64])
                                                        i = T.axis.spatial(
                                                            tile_x,
                                                            tz * 64 + li_0_lj_0_fused_1,
                                                        )
                                                        C = T.match_buffer(
                                                            O_frag[
                                                                i,
                                                                bz * 64 : bz * 64 + 64,
                                                            ],
                                                            (64,),
                                                            "float16",
                                                            scope="wmma.accumulator",
                                                            offset_factor=64,
                                                        )
                                                        L = T.match_buffer(
                                                            O_local[
                                                                i,
                                                                bz * 64 : bz * 64 + 64,
                                                            ],
                                                            (64,),
                                                            "float16",
                                                            scope="local",
                                                            offset_factor=64,
                                                        )
                                                        T.tvm_deconstruct_coopmat_qcom(
                                                            C.data, 64, 64, 16, L.data
                                                        )
                                                for li_0_lj_0_fused_1 in T.thread_binding(
                                                    64, thread="threadIdx.x"
                                                ):
                                                    for lj_1_0 in range(16):
                                                        for lj_1_1 in range(4):
                                                            with T.sblock("O_store"):
                                                                i = T.axis.spatial(
                                                                    tile_x,
                                                                    (tz * 64 + li_0_lj_0_fused_1),
                                                                )
                                                                j = T.axis.spatial(
                                                                    d_v,
                                                                    bz * 64
                                                                    + ty * 64
                                                                    + lj_1_0 * 4
                                                                    + lj_1_1,
                                                                )
                                                                T.reads(
                                                                    O_local[i, j],
                                                                    d_smem[i],
                                                                )
                                                                T.writes(
                                                                    output[
                                                                        (
                                                                            q_indptr_val
                                                                            + LH_start
                                                                            + i
                                                                        ),
                                                                        by,
                                                                        j,
                                                                    ]
                                                                )
                                                                cur_L: T.int32 = (
                                                                    q_indptr_val + LH_start + i
                                                                )
                                                                cur_H_qo: T.int32 = by
                                                                if cur_L < qo_len:
                                                                    output[
                                                                        cur_L,
                                                                        cur_H_qo,
                                                                        j,
                                                                    ] = T.Cast(
                                                                        "float16",
                                                                        O_local[i, j] / d_smem[i],
                                                                    )
                                                # Store LSE to gmem
                                                for li_0_lj_0_fused_1 in T.thread_binding(
                                                    64, thread="threadIdx.x"
                                                ):
                                                    with T.sblock("lse_store"):
                                                        i = T.axis.spatial(
                                                            tile_x,
                                                            (tz * 64 + li_0_lj_0_fused_1),
                                                        )
                                                        cur_L: T.int32 = q_indptr_val + LH_start + i
                                                        cur_H_qo: T.int32 = by
                                                        if cur_L < q_indptr[b_idx + 1]:
                                                            lse[cur_L, cur_H_qo] = m_smem[
                                                                i
                                                            ] + T.log2(d_smem[i])
                                            tile_id[0] += NUM_BLKS

    sch = s_tir.Schedule(batch_prefill_ragged_kv)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


def attention_prefill_paged_adreno(
    h_kv,
    h_q,
    d,
    dtype,
    rope_scaling: dict[str, Any],
    page_size: int = 64,
):
    tile_x = 256
    NUM_BLKS = 16
    group_size = h_q // h_kv

    # pylint: disable=line-too-long,too-many-branches
    @T.prim_func(private=True)
    def batch_prefill_paged_kv(
        var_q: T.handle,
        var_q_indptr: T.handle,
        var_pages: T.handle,
        var_page_indptr: T.handle,
        var_page_values: T.handle,
        var_length_info: T.handle,
        var_k_rope_pos_offset: T.handle,
        var_q_rope_position: T.handle,
        var_output: T.handle,
        var_lse: T.handle,
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):
        T.func_attr({"tir.noalias": True, "tir.is_scheduled": 1})
        total_len = T.int32(is_size_var=True)
        q = T.match_buffer(var_q, (total_len, h_q, d), "float16")
        batch_size = T.int32(is_size_var=True)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", offset_factor=0)
        max_num_pages = T.int32(is_size_var=True)
        pages = T.match_buffer(
            var_pages,
            (max_num_pages, 2, h_kv, page_size, d),
            "float16",
            offset_factor=0,
        )
        page_indptr = T.match_buffer(var_page_indptr, (batch_size + 1,), "int32", offset_factor=0)
        nnz_pages = T.int32(is_size_var=True)
        page_values = T.match_buffer(var_page_values, (nnz_pages,), "int32", offset_factor=0)
        length_info = T.match_buffer(var_length_info, (batch_size,), "int32", offset_factor=0)
        output = T.match_buffer(var_output, (total_len, h_q, d), "float16")
        lse = T.match_buffer(var_lse, (total_len, h_q))
        with T.sblock("root"):
            Q_mem_pad = T.alloc_buffer(
                (
                    T.int64(h_q),
                    T.int64(((total_len + tile_x - 1) // tile_x) * tile_x),
                    T.int64(d),
                ),
                "float16",
                scope="global",
            )
            for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
                for lby in T.thread_binding(h_q, thread="blockIdx.y"):
                    for lbz in T.thread_binding(d // 16, thread="blockIdx.z"):
                        with T.sblock("Q_load_root"):
                            bx, by, bz = T.axis.remap("SSS", [lbx, lby, lbz])
                            T.reads()
                            T.writes()
                            tile_id = T.alloc_buffer((1,), "int32", scope="local")
                            batch_idx = T.alloc_buffer((1,), "int32", scope="local")
                            batch_tiles = T.alloc_buffer((1,), "int32", scope="local")
                            batch_rows = T.alloc_buffer((1,), "int32", scope="local")
                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = q_indptr[1] - q_indptr[0]
                            batch_tiles[0] = (batch_rows[0] + tile_x - 1) // tile_x
                            while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size:
                                    tile_id[0] = tile_id[0] - batch_tiles[0]
                                    batch_idx[0] = batch_idx[0] + 1
                                    if batch_idx[0] < batch_size:
                                        b_idx: T.int32 = batch_idx[0]
                                        batch_rows[0] = q_indptr[b_idx + 1] - q_indptr[b_idx]
                                        batch_tiles[0] = (batch_rows[0] + tile_x - 1) // tile_x
                                if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                    b_idx: T.int32 = batch_idx[0]
                                    q_indptr_val: T.int32 = q_indptr[b_idx]
                                    LH_start: T.int32 = tile_id[0] * tile_x
                                    for ty in T.thread_binding(4, thread="threadIdx.y"):
                                        for tx in T.thread_binding(64, thread="threadIdx.x"):
                                            for lu in T.unroll(tile_x // 64):
                                                for lv in T.vectorized(4):
                                                    with T.sblock("Q_load"):
                                                        i = T.axis.spatial(tile_x, lu * 64 + tx)
                                                        j = T.axis.spatial(d, bz * 16 + ty * 4 + lv)
                                                        T.reads()
                                                        T.writes()
                                                        cur_L: T.int32 = q_indptr_val + (
                                                            LH_start + i
                                                        )
                                                        cur_H_qo: T.int32 = by
                                                        Q_mem_pad[cur_H_qo, cur_L, j] = (
                                                            T.if_then_else(
                                                                cur_L < q_indptr[b_idx + 1],
                                                                q[cur_L, cur_H_qo, j],
                                                                T.float16(0.0),
                                                            )
                                                        )
                                tile_id[0] += NUM_BLKS

            for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
                for lby in T.thread_binding(h_q, thread="blockIdx.y"):
                    for lbz in T.thread_binding(d // 64, thread="blockIdx.z"):
                        for ltz in T.thread_binding(tile_x // 64, thread="threadIdx.z"):
                            for lty in T.thread_binding(1, thread="threadIdx.y"):
                                for ltx in T.thread_binding(64, thread="threadIdx.x"):
                                    with T.sblock("attn"):
                                        bx, by, bz, tz, ty, tx = T.axis.remap(
                                            "SSSSSS", [lbx, lby, lbz, ltz, lty, ltx]
                                        )
                                        T.reads()
                                        T.writes()
                                        tile_id = T.alloc_buffer((1,), "int32", scope="local")
                                        batch_idx = T.alloc_buffer((1,), "int32", scope="local")
                                        batch_tiles = T.alloc_buffer((1,), "int32", scope="local")
                                        batch_rows = T.alloc_buffer((1,), "int32", scope="local")
                                        kv_chunk_len = T.alloc_buffer((1,), "int32", scope="local")
                                        Q_frag = T.alloc_buffer(
                                            (tile_x, 16),
                                            "float16",
                                            scope="wmma.matrix_a",
                                        )
                                        K_frag = T.alloc_buffer(
                                            (64, 16), "float16", scope="wmma.matrix_b"
                                        )
                                        S_frag = T.alloc_buffer(
                                            (tile_x, 64),
                                            "float16",
                                            scope="wmma.accumulator",
                                        )
                                        O_local = T.alloc_buffer(
                                            (tile_x, d), "float16", scope="local"
                                        )
                                        Smem_frag = T.alloc_buffer(
                                            (64, 16), "float16", scope="wmma.matrix_a"
                                        )
                                        S_local = T.alloc_buffer(
                                            (tile_x, 64), "float16", scope="local"
                                        )
                                        S_local_1 = T.alloc_buffer(
                                            (64, 64), "float16", scope="local"
                                        )
                                        O_frag = T.alloc_buffer(
                                            (64, d), "float16", scope="wmma.accumulator"
                                        )
                                        V_frag = T.alloc_buffer(
                                            (16, d), "float16", scope="wmma.matrix_b"
                                        )
                                        m_smem = T.alloc_buffer((tile_x,), scope="local")
                                        m_prev_smem = T.alloc_buffer((tile_x,), scope="local")
                                        d_smem = T.alloc_buffer((tile_x,), scope="local")
                                        m_new = T.alloc_buffer((1,), scope="local")
                                        m_prev = T.alloc_buffer((1,), scope="local")
                                        d_new = T.alloc_buffer((1,), scope="local")
                                        tile_id[0] = bx
                                        batch_idx[0] = 0
                                        batch_rows[0] = q_indptr[1] - q_indptr[0]
                                        batch_tiles[0] = (batch_rows[0] + tile_x - 1) // tile_x
                                        while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                            while (
                                                tile_id[0] >= batch_tiles[0]
                                                and batch_idx[0] < batch_size
                                            ):
                                                tile_id[0] = tile_id[0] - batch_tiles[0]
                                                batch_idx[0] = batch_idx[0] + 1
                                                if batch_idx[0] < batch_size:
                                                    b_idx: T.int32 = batch_idx[0]
                                                    batch_rows[0] = (
                                                        q_indptr[b_idx + 1] - q_indptr[b_idx]
                                                    )
                                                    batch_tiles[0] = (
                                                        batch_rows[0] + tile_x - 1
                                                    ) // tile_x
                                            if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                                b_idx: T.int32 = batch_idx[0]
                                                LH_start: T.int32 = tile_id[0] * tile_x
                                                q_indptr_val: T.int32 = q_indptr[b_idx]
                                                cur_page_indptr_begin: T.int32 = page_indptr[b_idx]
                                                cur_page_indptr_end: T.int32 = page_indptr[
                                                    b_idx + 1
                                                ]
                                                kv_chunk_len[0] = T.if_then_else(
                                                    cur_page_indptr_begin != cur_page_indptr_end,
                                                    (
                                                        cur_page_indptr_end
                                                        - cur_page_indptr_begin
                                                        - 1
                                                    )
                                                    * 64
                                                    + length_info[b_idx],
                                                    0,
                                                )
                                                for i in range(1):
                                                    row: T.int32 = i * tile_x + tz * 64 + tx
                                                    if row < tile_x:
                                                        m_smem[row] = T.float32(-50000.0)
                                                        d_smem[row] = T.float32(1.0)
                                                for li_0_lj_0_fused_1 in T.thread_binding(
                                                    64, thread="threadIdx.x"
                                                ):
                                                    for li_1 in range(4):
                                                        for lj_1_0 in T.unroll(4):
                                                            for lj_1_1 in T.vectorized(4):
                                                                with T.sblock("O_init"):
                                                                    i = T.axis.spatial(
                                                                        tile_x,
                                                                        tz * 64 + li_0_lj_0_fused_1,
                                                                    )
                                                                    j = T.axis.spatial(
                                                                        d,
                                                                        bz * 64
                                                                        + li_1 * 16
                                                                        + lj_1_0 * 4
                                                                        + lj_1_1,
                                                                    )
                                                                    T.reads()
                                                                    T.writes(O_local[i, j])
                                                                    O_local[i, j] = T.float32(0.0)
                                                for iterator_1 in range(
                                                    (kv_chunk_len[0] + 64 - 1) // 64
                                                ):
                                                    L_kv_start: T.int32 = iterator_1 * 64
                                                    page_no: T.int32 = page_values[
                                                        cur_page_indptr_begin + L_kv_start // 64
                                                    ]
                                                    with T.sblock(""):
                                                        T.reads()
                                                        T.writes(S_local[0:tile_x, 0:64])
                                                        with T.sblock("S_frag_init"):
                                                            T.reads()
                                                            T.writes(
                                                                S_frag[
                                                                    tz * 64 : tz * 64 + 64,
                                                                    0:64,
                                                                ]
                                                            )
                                                            C = T.match_buffer(
                                                                S_frag[
                                                                    tz * 64 : tz * 64 + 64,
                                                                    0:64,
                                                                ],
                                                                (64, 64),
                                                                "float16",
                                                                strides=(
                                                                    "C_s0",
                                                                    "C_s1",
                                                                ),
                                                                scope="wmma.accumulator",
                                                                offset_factor=64,
                                                            )
                                                            # zero the accumulator fragment
                                                            T.tvm_fill_fragment(
                                                                C.data,
                                                                64,
                                                                64,
                                                                16,
                                                                C.elem_offset
                                                                // C.strides[0]
                                                                // 64
                                                                * (C.strides[0] // 64)
                                                                + C.elem_offset
                                                                % C.strides[0]
                                                                // 64,
                                                                T.float32(0.0),
                                                            )
                                                        for lk_11 in range(d // 16):
                                                            with T.sblock("Q_wmma_a_load"):
                                                                T.reads()
                                                                T.writes()
                                                                A_src = T.match_buffer(
                                                                    Q_mem_pad[
                                                                        by,
                                                                        q_indptr_val
                                                                        + LH_start
                                                                        + tz * 64 : q_indptr_val
                                                                        + LH_start
                                                                        + tz * 64
                                                                        + 64,
                                                                        lk_11 * 16 : lk_11 * 16
                                                                        + 16,
                                                                    ],
                                                                    (64, 16),
                                                                    "float16",
                                                                    strides=(
                                                                        "A_s0",
                                                                        "A_s1",
                                                                    ),
                                                                    scope="global",
                                                                    offset_factor=16,
                                                                )
                                                                A_dst = T.match_buffer(
                                                                    Q_frag[
                                                                        tz * 64 : tz * 64 + 64,
                                                                        0:16,
                                                                    ],
                                                                    (64, 16),
                                                                    "float16",
                                                                    strides=(
                                                                        "C_s0",
                                                                        "C_s1",
                                                                    ),
                                                                    scope="wmma.matrix_a",
                                                                    offset_factor=16,
                                                                )
                                                                T.tvm_load_matrix_sync(
                                                                    A_dst.data,
                                                                    64,
                                                                    64,
                                                                    16,
                                                                    A_dst.elem_offset
                                                                    // A_dst.strides[0]
                                                                    // 64
                                                                    * (A_dst.strides[0] // 16)
                                                                    + A_dst.elem_offset
                                                                    % A_dst.strides[0]
                                                                    // 16,
                                                                    T.tvm_access_ptr(
                                                                        T.type_annotation(
                                                                            "float16"
                                                                        ),
                                                                        A_src.data,
                                                                        A_src.elem_offset,
                                                                        A_src.strides[0] * 64,
                                                                        1,
                                                                    ),
                                                                    A_src.strides[0],
                                                                    "row_major",
                                                                )
                                                            with T.sblock("K_wmma_b_load"):
                                                                T.reads()
                                                                T.writes()
                                                                B_src = T.match_buffer(
                                                                    pages[
                                                                        page_no,
                                                                        0,
                                                                        by // group_size,
                                                                        0:64,
                                                                        lk_11 * 16 : lk_11 * 16
                                                                        + 16,
                                                                    ],
                                                                    (64, 16),
                                                                    "float16",
                                                                    strides=(
                                                                        "B_s0",
                                                                        "B_s1",
                                                                    ),
                                                                    scope="global",
                                                                    offset_factor=16,
                                                                )
                                                                B_dst = T.match_buffer(
                                                                    K_frag[0:64, 0:16],
                                                                    (64, 16),
                                                                    "float16",
                                                                    strides=(
                                                                        "C_s0",
                                                                        "C_s1",
                                                                    ),
                                                                    scope="wmma.matrix_b",
                                                                    offset_factor=16,
                                                                )
                                                                T.tvm_load_matrix_sync(
                                                                    B_dst.data,
                                                                    64,
                                                                    64,
                                                                    16,
                                                                    B_dst.elem_offset
                                                                    // B_dst.strides[0]
                                                                    // 64
                                                                    * (B_dst.strides[0] // 16)
                                                                    + B_dst.elem_offset
                                                                    % B_dst.strides[0]
                                                                    // 16,
                                                                    T.tvm_access_ptr(
                                                                        T.type_annotation(
                                                                            "float16"
                                                                        ),
                                                                        B_src.data,
                                                                        B_src.elem_offset,
                                                                        B_src.strides[0] * 64,
                                                                        1,
                                                                    ),
                                                                    B_src.strides[0],
                                                                    "col_major",
                                                                )
                                                            with T.sblock(""):
                                                                with T.sblock("S_mma_sync"):
                                                                    T.reads()
                                                                    T.writes(
                                                                        S_frag[
                                                                            tz * 64 : tz * 64 + 64,
                                                                            0:64,
                                                                        ]
                                                                    )
                                                                    C = T.match_buffer(
                                                                        S_frag[
                                                                            tz * 64 : tz * 64 + 64,
                                                                            0:64,
                                                                        ],
                                                                        (64, 64),
                                                                        "float16",
                                                                        strides=(
                                                                            "C_s0",
                                                                            "C_s1",
                                                                        ),
                                                                        scope="wmma.accumulator",
                                                                        offset_factor=64,
                                                                    )
                                                                    A = T.match_buffer(
                                                                        Q_frag[
                                                                            tz * 64 : tz * 64 + 64,
                                                                            0:16,
                                                                        ],
                                                                        (64, 16),
                                                                        "float16",
                                                                        strides=(
                                                                            "A_s0",
                                                                            "A_s1",
                                                                        ),
                                                                        scope="wmma.matrix_a",
                                                                        offset_factor=16,
                                                                    )
                                                                    B = T.match_buffer(
                                                                        K_frag[0:64, 0:16],
                                                                        (64, 16),
                                                                        "float16",
                                                                        strides=(
                                                                            "B_s0",
                                                                            "B_s1",
                                                                        ),
                                                                        scope="wmma.matrix_b",
                                                                        offset_factor=16,
                                                                    )
                                                                    T.tvm_mma_sync(
                                                                        C.data,
                                                                        C.elem_offset
                                                                        // C.strides[0]
                                                                        // 64
                                                                        * (C.strides[0] // 64)
                                                                        + C.elem_offset
                                                                        % C.strides[0]
                                                                        // 64,
                                                                        A.data,
                                                                        A.elem_offset
                                                                        // A.strides[0]
                                                                        // 64
                                                                        * (A.strides[0] // 16)
                                                                        + A.elem_offset
                                                                        % A.strides[0]
                                                                        // 16,
                                                                        B.data,
                                                                        B.elem_offset
                                                                        // B.strides[0]
                                                                        // 64
                                                                        * (B.strides[0] // 16)
                                                                        + B.elem_offset
                                                                        % B.strides[0]
                                                                        // 16,
                                                                        C.data,
                                                                        C.elem_offset
                                                                        // C.strides[0]
                                                                        // 64
                                                                        * (C.strides[0] // 64)
                                                                        + C.elem_offset
                                                                        % C.strides[0]
                                                                        // 64,
                                                                    )
                                                    for li_0_lj_0_fused_1 in T.thread_binding(
                                                        64, thread="threadIdx.x"
                                                    ):
                                                        with T.sblock("S_frag_store_local"):
                                                            T.reads(
                                                                S_frag[
                                                                    tz * 64 : tz * 64 + 64,
                                                                    0:64,
                                                                ]
                                                            )
                                                            T.writes(
                                                                S_local[
                                                                    tz * 64 : tz * 64 + 64,
                                                                    0:64,
                                                                ]
                                                            )
                                                            i = T.axis.spatial(
                                                                64, li_0_lj_0_fused_1
                                                            )
                                                            C = T.match_buffer(
                                                                S_frag[tz * 64 + i, 0:64],
                                                                (64,),
                                                                "float16",
                                                                scope="wmma.accumulator",
                                                                offset_factor=64,
                                                            )
                                                            L = T.match_buffer(
                                                                S_local[tz * 64 + i, 0:64],
                                                                (64,),
                                                                "float16",
                                                                scope="local",
                                                                offset_factor=64,
                                                            )
                                                            T.tvm_deconstruct_coopmat_qcom(
                                                                C.data,
                                                                64,
                                                                64,
                                                                16,
                                                                L.data,
                                                            )
                                                    for li_0_lj_0_fused_0 in T.thread_binding(
                                                        1, thread="threadIdx.y"
                                                    ):
                                                        for li_0_lj_0_fused_1 in T.thread_binding(
                                                            64, thread="threadIdx.x"
                                                        ):
                                                            for li_1 in range(4):
                                                                for lj_1_0 in T.unroll(4):
                                                                    for lj_1_1 in T.vectorized(4):
                                                                        with T.sblock("S_store"):
                                                                            i = T.axis.spatial(
                                                                                tile_x,
                                                                                tz * 64
                                                                                + li_0_lj_0_fused_1,
                                                                            )
                                                                            j = T.axis.spatial(
                                                                                64,
                                                                                li_1 * 16
                                                                                + lj_1_0 * 4
                                                                                + lj_1_1,
                                                                            )
                                                                            T.reads(S_local[i, j])
                                                                            T.writes(S_local[i, j])
                                                                            S_local[i, j] = (
                                                                                S_local[i, j]
                                                                                * sm_scale
                                                                                * math.log2(
                                                                                    math.exp(1)
                                                                                )
                                                                            )
                                                    for i in range(1):
                                                        row: T.int32 = i * tile_x + tz * 64 + tx
                                                        if row < tile_x:
                                                            with T.sblock("update1"):
                                                                T.reads(
                                                                    m_smem[row],
                                                                    kv_chunk_len[0],
                                                                    q_indptr[b_idx : b_idx + 2],
                                                                    m_new[i],
                                                                    S_local[row, 0:64],
                                                                    d_smem[row],
                                                                    m_prev[i],
                                                                )
                                                                T.writes(
                                                                    m_prev[i],
                                                                    m_new[i],
                                                                    d_new[i],
                                                                )
                                                                m_prev[i] = m_smem[row]
                                                                m_new[i] = m_smem[row]
                                                                row_: T.int32 = LH_start + row
                                                                for j in range(64):
                                                                    if T.if_then_else(
                                                                        causal > 0,
                                                                        L_kv_start + j
                                                                        < kv_chunk_len[0]
                                                                        - (
                                                                            q_indptr[b_idx + 1]
                                                                            - q_indptr[b_idx]
                                                                        )
                                                                        + row_
                                                                        + 1,
                                                                        L_kv_start + j
                                                                        < kv_chunk_len[0],
                                                                    ):
                                                                        m_new[i] = T.max(
                                                                            m_new[i],
                                                                            S_local[
                                                                                row,
                                                                                j,
                                                                            ],
                                                                        )
                                                                d_new[i] = d_smem[row] * T.exp2(
                                                                    m_prev[i] - m_new[i]
                                                                )
                                                    for i in range(1):
                                                        row: T.int32 = i * tile_x + tz * 64 + tx
                                                        with T.sblock("update"):
                                                            T.reads(
                                                                kv_chunk_len[0],
                                                                q_indptr[b_idx : b_idx + 2],
                                                                S_local[row, 0:64],
                                                                m_new[i],
                                                            )
                                                            T.writes(S_local[row, 0:64])
                                                            for j in range(64):
                                                                if row < tile_x:
                                                                    row_: T.int32 = LH_start + row
                                                                    if T.if_then_else(
                                                                        causal > 0,
                                                                        L_kv_start + j
                                                                        < kv_chunk_len[0]
                                                                        - (
                                                                            q_indptr[b_idx + 1]
                                                                            - q_indptr[b_idx]
                                                                        )
                                                                        + row_
                                                                        + 1,
                                                                        L_kv_start + j
                                                                        < kv_chunk_len[0],
                                                                    ):
                                                                        S_local[row, j] = T.exp2(
                                                                            S_local[row, j]
                                                                            - m_new[i]
                                                                        )
                                                                    else:
                                                                        S_local[row, j] = T.exp2(
                                                                            T.float32(-50000.0)
                                                                            - m_new[i]
                                                                        )
                                                    for i in range(1):
                                                        row: T.int32 = i * tile_x + tz * 64 + tx
                                                        if row < tile_x:
                                                            with T.sblock("update"):
                                                                T.reads(
                                                                    d_new[i],
                                                                    S_local[row, 0:32],
                                                                    m_new[i],
                                                                    m_prev[i],
                                                                )
                                                                T.writes(
                                                                    d_new[i],
                                                                    m_smem[row],
                                                                    d_smem[row],
                                                                    m_prev_smem[row],
                                                                )
                                                                for j in range(64):
                                                                    d_new[i] = (
                                                                        d_new[i] + S_local[row, j]
                                                                    )
                                                                m_smem[row] = m_new[i]
                                                                d_smem[row] = d_new[i]
                                                                m_prev_smem[row] = m_prev[i]
                                                    with T.sblock(""):
                                                        T.reads()
                                                        T.writes(O_local[0:64, 0:64])
                                                        for li_0_lj_0_fused_1 in T.thread_binding(
                                                            64, thread="threadIdx.x"
                                                        ):
                                                            with T.sblock("O_frag_store_local"):
                                                                T.reads(O_frag[0:64, 0:64])
                                                                T.writes(O_local[0:64, 0:64])
                                                                i = T.axis.spatial(
                                                                    tile_x,
                                                                    tz * 64 + li_0_lj_0_fused_1,
                                                                )
                                                                C = T.match_buffer(
                                                                    O_frag[
                                                                        i,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (64,),
                                                                    "float16",
                                                                    scope="wmma.accumulator",
                                                                    offset_factor=64,
                                                                )
                                                                L = T.match_buffer(
                                                                    O_local[
                                                                        i,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (64,),
                                                                    "float16",
                                                                    scope="local",
                                                                    offset_factor=64,
                                                                )
                                                                T.tvm_deconstruct_coopmat_qcom(
                                                                    C.data,
                                                                    64,
                                                                    64,
                                                                    16,
                                                                    L.data,
                                                                )
                                                        for li_0_init in T.thread_binding(
                                                            64, thread="threadIdx.x"
                                                        ):
                                                            for lj_1_0_init in T.unroll(16):
                                                                for lj_1_1_init in T.vectorized(4):
                                                                    with T.sblock("O_gemm_init"):
                                                                        i = T.axis.spatial(
                                                                            tile_x,
                                                                            (tz * 64 + li_0_init),
                                                                        )
                                                                        j = T.axis.spatial(
                                                                            d,
                                                                            bz * 64
                                                                            + ty * 64
                                                                            + lj_1_0_init * 4
                                                                            + lj_1_1_init,
                                                                        )
                                                                        T.reads()
                                                                        T.writes(O_local[i, j])
                                                                        O_local[i, j] = O_local[
                                                                            i, j
                                                                        ] * T.exp2(
                                                                            m_prev_smem[i]
                                                                            - m_smem[i]
                                                                        )
                                                        for li_0_lj_0_fused_1 in T.thread_binding(
                                                            64, thread="threadIdx.x"
                                                        ):
                                                            with T.sblock("O_frag_store_local"):
                                                                T.reads(O_frag[0:64, 0:64])
                                                                T.writes(O_local[0:64, 0:64])
                                                                i = T.axis.spatial(
                                                                    tile_x,
                                                                    tz * 64 + li_0_lj_0_fused_1,
                                                                )
                                                                C = T.match_buffer(
                                                                    O_frag[
                                                                        i,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (64,),
                                                                    "float16",
                                                                    scope="wmma.accumulator",
                                                                    offset_factor=64,
                                                                )
                                                                L = T.match_buffer(
                                                                    O_local[
                                                                        i,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (64,),
                                                                    "float16",
                                                                    scope="local",
                                                                    offset_factor=64,
                                                                )
                                                                T.tvm_construct_coopmat_qcom(
                                                                    C.data,
                                                                    64,
                                                                    64,
                                                                    64,
                                                                    L.data,
                                                                )
                                                        for lv_11 in range(4):
                                                            for (
                                                                li_0_lj_0_fused_1
                                                            ) in T.thread_binding(
                                                                64, thread="threadIdx.x"
                                                            ):
                                                                for i_01 in T.serial(16):
                                                                    with T.sblock("Smem_loal_load"):
                                                                        i = T.axis.spatial(
                                                                            tile_x,
                                                                            tz * 64
                                                                            + li_0_lj_0_fused_1,
                                                                        )
                                                                        j = T.axis.spatial(
                                                                            64,
                                                                            lv_11 * 16 + i_01,
                                                                        )
                                                                        jk = T.axis.spatial(
                                                                            16, i_01
                                                                        )
                                                                        T.reads()
                                                                        T.writes()
                                                                        S_local_1[i, jk] = S_local[
                                                                            i, j
                                                                        ]

                                                            for (
                                                                li_0_lj_0_fused_1
                                                            ) in T.thread_binding(
                                                                64, thread="threadIdx.x"
                                                            ):
                                                                with T.sblock("S_wmma_a_load"):
                                                                    i = T.axis.spatial(
                                                                        tile_x,
                                                                        tz * 64 + li_0_lj_0_fused_1,
                                                                    )
                                                                    T.reads(S_local_1[i, 0:16])
                                                                    T.writes()
                                                                    A_src = T.match_buffer(
                                                                        S_local_1[i, 0:16],
                                                                        (16,),
                                                                        "float16",
                                                                        scope="local",
                                                                        offset_factor=16,
                                                                    )
                                                                    A_dst = T.match_buffer(
                                                                        Smem_frag[i, 0:16],
                                                                        (16,),
                                                                        "float16",
                                                                        scope="wmma.matrix_a",
                                                                        offset_factor=16,
                                                                    )
                                                                    T.tvm_construct_coopmat_qcom(
                                                                        A_dst.data,
                                                                        64,
                                                                        64,
                                                                        16,
                                                                        A_src.data,
                                                                    )
                                                            with T.sblock("V_wmma_b_load"):
                                                                T.reads()
                                                                T.writes()
                                                                B_src = T.match_buffer(
                                                                    pages[
                                                                        page_no,
                                                                        1,
                                                                        by // group_size,
                                                                        lv_11 * 16 : lv_11 * 16
                                                                        + 16,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (16, 64),
                                                                    "float16",
                                                                    strides=(
                                                                        "B_s0",
                                                                        "B_s1",
                                                                    ),
                                                                    scope="global",
                                                                    offset_factor=64,
                                                                )
                                                                B_dst = T.match_buffer(
                                                                    V_frag[
                                                                        0:16,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (16, 64),
                                                                    "float16",
                                                                    strides=(
                                                                        "C_s0",
                                                                        "C_s1",
                                                                    ),
                                                                    scope="wmma.matrix_b",
                                                                    offset_factor=64,
                                                                )
                                                                T.tvm_load_matrix_sync(
                                                                    B_dst.data,
                                                                    64,
                                                                    64,
                                                                    16,
                                                                    B_dst.elem_offset
                                                                    // B_dst.strides[0]
                                                                    // 16
                                                                    * (B_dst.strides[0] // 64)
                                                                    + B_dst.elem_offset
                                                                    % B_dst.strides[0]
                                                                    // 64,
                                                                    T.tvm_access_ptr(
                                                                        T.type_annotation(
                                                                            "float16"
                                                                        ),
                                                                        B_src.data,
                                                                        B_src.elem_offset,
                                                                        B_src.strides[0] * 16,
                                                                        1,
                                                                    ),
                                                                    B_src.strides[0],
                                                                    "row_major",
                                                                )

                                                            with T.sblock("O_mma_sync"):
                                                                T.reads()
                                                                T.writes(O_frag[0:64, 0:64])
                                                                C = T.match_buffer(
                                                                    O_frag[
                                                                        0:64,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (64, 64),
                                                                    "float16",
                                                                    strides=(
                                                                        "C_s0",
                                                                        "C_s1",
                                                                    ),
                                                                    scope="wmma.accumulator",
                                                                    offset_factor=64,
                                                                )
                                                                A = T.match_buffer(
                                                                    Smem_frag[0:64, 0:16],
                                                                    (64, 16),
                                                                    "float16",
                                                                    strides=(
                                                                        "A_s0",
                                                                        "A_s1",
                                                                    ),
                                                                    scope="wmma.matrix_a",
                                                                    offset_factor=16,
                                                                )
                                                                B = T.match_buffer(
                                                                    V_frag[
                                                                        0:16,
                                                                        bz * 64 : bz * 64 + 64,
                                                                    ],
                                                                    (16, 64),
                                                                    "float16",
                                                                    strides=(
                                                                        "B_s0",
                                                                        "B_s1",
                                                                    ),
                                                                    scope="wmma.matrix_b",
                                                                    offset_factor=64,
                                                                )
                                                                T.tvm_mma_sync(
                                                                    C.data,
                                                                    C.elem_offset
                                                                    // C.strides[0]
                                                                    // 64
                                                                    * (C.strides[0] // 64)
                                                                    + C.elem_offset
                                                                    % C.strides[0]
                                                                    // 64,
                                                                    A.data,
                                                                    A.elem_offset
                                                                    // A.strides[0]
                                                                    // 64
                                                                    * (A.strides[0] // 16)
                                                                    + A.elem_offset
                                                                    % A.strides[0]
                                                                    // 16,
                                                                    B.data,
                                                                    B.elem_offset
                                                                    // B.strides[0]
                                                                    // 16
                                                                    * (B.strides[0] // 64)
                                                                    + B.elem_offset
                                                                    % B.strides[0]
                                                                    // 64,
                                                                    C.data,
                                                                    C.elem_offset
                                                                    // C.strides[0]
                                                                    // 64
                                                                    * (C.strides[0] // 64)
                                                                    + C.elem_offset
                                                                    % C.strides[0]
                                                                    // 64,
                                                                )
                                                for li_0_lj_0_fused_1 in T.thread_binding(
                                                    64, thread="threadIdx.x"
                                                ):
                                                    with T.sblock("O_frag_store_local"):
                                                        T.reads(O_frag[0:64, 0:64])
                                                        T.writes(O_local[0:64, 0:64])
                                                        i = T.axis.spatial(
                                                            tile_x,
                                                            tz * 64 + li_0_lj_0_fused_1,
                                                        )
                                                        C = T.match_buffer(
                                                            O_frag[
                                                                i,
                                                                bz * 64 : bz * 64 + 64,
                                                            ],
                                                            (64,),
                                                            "float16",
                                                            scope="wmma.accumulator",
                                                            offset_factor=64,
                                                        )
                                                        L = T.match_buffer(
                                                            O_local[
                                                                i,
                                                                bz * 64 : bz * 64 + 64,
                                                            ],
                                                            (64,),
                                                            "float16",
                                                            scope="local",
                                                            offset_factor=64,
                                                        )
                                                        T.tvm_deconstruct_coopmat_qcom(
                                                            C.data, 64, 64, 16, L.data
                                                        )
                                                for li_0_lj_0_fused_1 in T.thread_binding(
                                                    64, thread="threadIdx.x"
                                                ):
                                                    for li_1 in range(4):
                                                        for lj_1_0 in range(4):
                                                            for lj_1_1 in range(4):
                                                                with T.sblock("O_store"):
                                                                    i = T.axis.spatial(
                                                                        tile_x,
                                                                        tz * 64 + li_0_lj_0_fused_1,
                                                                    )
                                                                    j = T.axis.spatial(
                                                                        d,
                                                                        bz * 64
                                                                        + li_1 * 16
                                                                        + lj_1_0 * 4
                                                                        + lj_1_1,
                                                                    )
                                                                    T.reads(
                                                                        q_indptr[b_idx : b_idx + 2],
                                                                        O_local[i, j],
                                                                        d_smem[i],
                                                                    )
                                                                    T.writes(
                                                                        output[
                                                                            q_indptr[b_idx]
                                                                            + (LH_start + i),
                                                                            by,
                                                                            j,
                                                                        ]
                                                                    )
                                                                    cur_L: T.int32 = q_indptr[
                                                                        b_idx
                                                                    ] + (LH_start + i)
                                                                    cur_H_qo: T.int32 = by
                                                                    if cur_L < q_indptr[b_idx + 1]:
                                                                        output[
                                                                            cur_L,
                                                                            cur_H_qo,
                                                                            j,
                                                                        ] = T.Cast(
                                                                            "float16",
                                                                            O_local[i, j]
                                                                            / d_smem[i],
                                                                        )
                                                for li_0 in range(1):
                                                    for li_2 in T.thread_binding(
                                                        64, thread="threadIdx.x"
                                                    ):
                                                        with T.sblock("lse_store"):
                                                            i = T.axis.spatial(
                                                                tile_x, tz * 64 + li_2
                                                            )
                                                            T.reads(
                                                                q_indptr[b_idx : b_idx + 2],
                                                                m_smem[i],
                                                                d_smem[i],
                                                            )
                                                            T.writes(
                                                                lse[
                                                                    q_indptr[b_idx]
                                                                    + (LH_start + i),
                                                                    by,
                                                                ]
                                                            )
                                                            cur_L: T.int32 = q_indptr[b_idx] + (
                                                                LH_start + i
                                                            )
                                                            cur_H_qo: T.int32 = by
                                                            if cur_L < q_indptr[b_idx + 1]:
                                                                lse[cur_L, cur_H_qo] = m_smem[
                                                                    i
                                                                ] + T.log2(d_smem[i])
                                                tile_id[0] = tile_id[0] + NUM_BLKS

    sch = s_tir.Schedule(batch_prefill_paged_kv)
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)
