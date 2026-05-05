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
# ruff: noqa: E501, F841
import tvm
import tvm.testing
from tvm import IRModule
from tvm.s_tir import dlight as dl
from tvm.script import tir as T
from tvm.target import Target


def test_dequant_matmul_khr():
    # fmt: off
    @T.prim_func
    def before(quant: T.Buffer((T.int64(512), T.int64(12288)), "uint32"), scale: T.Buffer((T.int64(128), T.int64(12288)), "float16"), p_rms_norm130: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12288),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(4096)), "float16")
        T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "float16")
        compute = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
        dequantize_intermediate_intermediate = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
        matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(12288)), "float16")
        for i0, i1 in T.grid(T.int64(4096), T.int64(12288)):
            with T.sblock("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(quant[v_i0 // T.int64(8), v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(quant[v_i0 // T.int64(8), v_i1], T.Cast("uint32", v_i0 % T.int64(8) * T.int64(4))), T.uint32(15)))
        for i0, i1 in T.grid(T.int64(4096), T.int64(12288)):
            with T.sblock("dequantize"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(compute[v_i0, v_i1], scale[v_i0 // T.int64(32), v_i1])
                T.writes(dequantize_intermediate_intermediate[v_i0, v_i1])
                dequantize_intermediate_intermediate[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * scale[v_i0 // T.int64(32), v_i1]
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(12288), T.int64(4096)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(rms_norm130[v_i0, v_i1, v_k], dequantize_intermediate_intermediate[v_k, v_i2])
                T.writes(matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                matmul_intermediate[v_i0, v_i1, v_i2] = matmul_intermediate[v_i0, v_i1, v_i2] + rms_norm130[v_i0, v_i1, v_k] * dequantize_intermediate_intermediate[v_k, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(12288)):
            with T.sblock("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(matmul_intermediate[v_ax0, v_ax1, v_ax2], transformer_h_0_attn_c_attn_bias3[v_ax2])
                T.writes(T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2] = matmul_intermediate[v_ax0, v_ax1, v_ax2] + transformer_h_0_attn_c_attn_bias3[v_ax2]

    @T.prim_func
    def expected(quant: T.Buffer((T.int64(512), T.int64(12288)), "uint32"), scale: T.Buffer((T.int64(128), T.int64(12288)), "float16"), p_rms_norm130: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12288),), "float16"), p_output0: T.handle):
        T.func_attr({"global_symbol": "before", "tir.is_scheduled": True, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(4096)), "float16")
        T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "float16")
        # with T.sblock("root"):
        dequantize_intermediate_intermediate_shared_wmma_matrix_b = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16", scope="wmma.matrix_b")
        quant_local = T.alloc_buffer((T.int64(12288), T.int64(512)), "uint32", scope="local")
        rms_norm130_pad = T.alloc_buffer((T.int64(1), (seq_len + T.int64(63)) // T.int64(64) * T.int64(64), T.int64(4096)), "float16")
        matmul_intermediate_pad_wmma_accumulator = T.alloc_buffer((T.int64(1), (seq_len + T.int64(63)) // T.int64(64) * T.int64(64), T.int64(12288)), "float16", scope="wmma.accumulator")
        rms_norm130_pad_global_wmma_matrix_a = T.alloc_buffer((T.int64(1), (seq_len + T.int64(63)) // T.int64(64) * T.int64(64), T.int64(4096)), "float16", scope="wmma.matrix_a")
        matmul_intermediate_pad_shared = T.alloc_buffer((T.int64(1), (seq_len + T.int64(63)) // T.int64(64) * T.int64(64), T.int64(12288)), "float16", scope="shared")
        dequantize_intermediate_intermediate_shared_shared = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16", scope="shared")
        for i0 in T.thread_binding(T.int64(1), thread="blockIdx.y"):
            for i1_i2_0_fused_0 in T.thread_binding((seq_len + T.int64(63)) // T.int64(64) * T.int64(512), thread="blockIdx.x"):
                for i1_i2_0_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                    for i2_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                        for i2_2 in T.vectorized(T.int64(4)):
                            with T.sblock("rms_norm130_pad"):
                                v0 = T.axis.spatial(T.int64(1), i0)
                                v1 = T.axis.spatial((seq_len + T.int64(63)) // T.int64(64) * T.int64(64), (i1_i2_0_fused_0 * T.int64(4) + i1_i2_0_fused_1) // T.int64(32))
                                v2 = T.axis.spatial(T.int64(4096), (i1_i2_0_fused_0 * T.int64(4) + i1_i2_0_fused_1) % T.int64(32) * T.int64(128) + i2_1 * T.int64(4) + i2_2)
                                T.reads(rms_norm130[v0, v1, v2])
                                T.writes(rms_norm130_pad[v0, v1, v2])
                                T.sblock_attr({"buffer_dim_align": [[0, 1, 4096, 64]]})
                                rms_norm130_pad[v0, v1, v2] = T.if_then_else(v1 < seq_len, rms_norm130[v0, v1, v2], T.float16(0.0))
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0 in T.thread_binding((seq_len + T.int64(63)) // T.int64(64), thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(T.int64(96), thread="blockIdx.y"):
                    for ax1_2 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(1), thread="vthread.y"):
                            for ax1_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax2_1 in T.thread_binding(T.int64(2), thread="threadIdx.z", annotations={"pragma_auto_unroll_max_step": 8, "pragma_unroll_explicit": 1}):
                                    with T.sblock("matmul_init_o"):
                                        v0_o = T.axis.spatial(T.int64(1), ax0)
                                        v1_o = T.axis.spatial((seq_len + T.int64(63)) // T.int64(64), ax1_0 + ax1_1 + ax1_2)
                                        v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                        T.reads()
                                        T.writes(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                        C = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                        T.tvm_fill_fragment(C.data, 64, 64, 16, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), T.float32(0.0))
                                    for ax3_0, ax3_1 in T.grid(T.int64(128), T.int64(2)):
                                        for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                            for ax1 in T.unroll(T.int64(2)):
                                                with T.sblock("quant_local"):
                                                    v0 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_1)
                                                    v1 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(4) + ax3_1 * T.int64(2) + ax1)
                                                    T.reads(quant[v1, v0])
                                                    T.writes(quant_local[v0, v1])
                                                    quant_local[v0, v1] = quant[v1, v0]
                                        for ax3_2 in range(T.int64(1)):
                                            with T.sblock("rms_norm130_pad_global_wmma.matrix_a_o"):
                                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1_o = T.axis.spatial((seq_len + T.int64(63)) // T.int64(64), ax1_0)
                                                v2_o = T.axis.spatial(T.int64(256), ax3_0 * T.int64(2) + ax3_1)
                                                T.reads(rms_norm130_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                T.writes(rms_norm130_pad_global_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                A = T.match_buffer(rms_norm130_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), offset_factor=16)
                                                C = T.match_buffer(rms_norm130_pad_global_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                                for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                    T.tvm_load_matrix_sync(C.data, 64, 64, 16, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(64), 1), A.strides[0], "row_major")
                                            for ax0_0 in range(T.int64(1)):
                                                for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                    for ax1_0_1 in T.unroll(T.int64(4)):
                                                        for ax1_1_1 in T.vectorized(T.int64(4)):
                                                            with T.sblock("dequantize"):
                                                                v0 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_0 * T.int64(64) + ax0_1)
                                                                v1 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(32) + ax3_1 * T.int64(16) + ax1_0_1 * T.int64(4) + ax1_1_1)
                                                                T.reads(quant_local[v0, v1 // T.int64(8)], scale[v1 // T.int64(32), v0])
                                                                T.writes(dequantize_intermediate_intermediate_shared_shared[v0, v1])
                                                                dequantize_intermediate_intermediate_shared_shared[v0, v1] = (T.Cast("float16", T.bitwise_and(T.shift_right(quant_local[v0, v1 // T.int64(8)], T.Cast("uint32", v1 % T.int64(8) * T.int64(4))), T.uint32(15))) - T.float16(7.0)) * scale[v1 // T.int64(32), v0]
                                            for ax0_0 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                                                with T.sblock("dequantize_intermediate_intermediate_shared_shared_o"):
                                                    v0_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax0_0)
                                                    v1_o = T.axis.spatial(T.int64(256), ax3_0 * T.int64(2) + ax3_1)
                                                    T.reads(dequantize_intermediate_intermediate_shared_shared[v0_o * T.int64(64):v0_o * T.int64(64) + T.int64(64), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                                    T.writes(dequantize_intermediate_intermediate_shared_wmma_matrix_b[v0_o * T.int64(64):v0_o * T.int64(64) + T.int64(64), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                                    A = T.match_buffer(dequantize_intermediate_intermediate_shared_shared[v0_o * T.int64(64):v0_o * T.int64(64) + T.int64(64), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared", offset_factor=16)
                                                    C = T.match_buffer(dequantize_intermediate_intermediate_shared_wmma_matrix_b[v0_o * T.int64(64):v0_o * T.int64(64) + T.int64(64), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                                    for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                        T.tvm_load_matrix_sync(C.data, 64, 64, 16, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(64), 1), A.strides[0], "col_major")
                                            with T.sblock("matmul_update_o"):
                                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                                v1_o = T.axis.spatial((seq_len + T.int64(63)) // T.int64(64), ax1_0 + ax1_1 + ax1_2)
                                                v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                                v3_o = T.axis.reduce(T.int64(256), ax3_0 * T.int64(2) + ax3_1 + ax3_2)
                                                T.reads(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], rms_norm130_pad_global_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], dequantize_intermediate_intermediate_shared_wmma_matrix_b[v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                                T.writes(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                                A = T.match_buffer(rms_norm130_pad_global_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                                B = T.match_buffer(dequantize_intermediate_intermediate_shared_wmma_matrix_b[v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                                C = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                                T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), A.data, A.elem_offset // A.strides[0] // T.int64(64) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(64) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64))
                                    for ax0_1 in range(T.int64(1)):
                                        with T.sblock("matmul_intermediate_pad_shared_o"):
                                            v0_o = T.axis.spatial(T.int64(1), ax0_1)
                                            v1_o = T.axis.spatial((seq_len + T.int64(63)) // T.int64(64), ax1_0 + ax1_1 + ax1_2)
                                            v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                            T.reads(matmul_intermediate_pad_wmma_accumulator[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                            T.writes(matmul_intermediate_pad_shared[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                            A = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=64)
                                            C = T.match_buffer(matmul_intermediate_pad_shared[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="shared", offset_factor=64)
                                            T.tvm_store_matrix_sync(A.data, 64, 64, 16, A.elem_offset // A.strides[0] // T.int64(64) * (A.strides[0] // T.int64(64)) + A.elem_offset % A.strides[0] // T.int64(64), T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(64), 2), C.strides[0], "row_major")
                                    for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                        for ax1_0_1 in T.unroll(T.int64(16)):
                                            for ax1_1_1 in range(T.int64(4)):
                                                with T.sblock("matmul_intermediate_pad"):
                                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                    v1 = T.axis.spatial(seq_len, ax1_0 * T.int64(64) + ax0_1)
                                                    v2 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax1_0_1 * T.int64(4) + ax1_1_1)
                                                    T.where((ax1_0 - (seq_len + T.int64(63)) // T.int64(64) < T.int64(0) or ax1_0 * T.int64(64) + ax0_1 == T.int64(0)) and ax1_0 * T.int64(64) + ax0_1 < seq_len)
                                                    T.reads(matmul_intermediate_pad_shared[v0, v1, v2], transformer_h_0_attn_c_attn_bias3[v2])
                                                    T.writes(T_add_intermediate_intermediate[v0, v1, v2])
                                                    T_add_intermediate_intermediate[v0, v1, v2] = matmul_intermediate_pad_shared[v0, v1, v2] + transformer_h_0_attn_c_attn_bias3[v2]
    # fmt: on

    mod = IRModule({"main": before})
    vk_target = {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_khr_cooperative_matrix": True,
    }
    with Target(vk_target):
        mod = dl.ApplyDefaultSchedule(dl.adreno.DequantMatmulTensorization()).transform_module(
            mod, False
        )
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_dequant_matmul_qcom():
    # fmt: off
    @T.prim_func
    def before(quant: T.Buffer((T.int64(512), T.int64(12288)), "uint32"), scale: T.Buffer((T.int64(128), T.int64(12288)), "float16"), p_rms_norm130: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12288),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(4096)), "float16")
        T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "float16")
        compute = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
        dequantize_intermediate_intermediate = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
        matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(12288)), "float16")
        for i0, i1 in T.grid(T.int64(4096), T.int64(12288)):
            with T.sblock("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])

                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(quant[v_i0 // T.int64(8), v_i1], T.Cast("uint32", v_i0 % T.int64(8) * T.int64(4))), T.uint32(15)))
        for i0, i1 in T.grid(T.int64(4096), T.int64(12288)):
            with T.sblock("dequantize"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(compute[v_i0, v_i1], scale[v_i0 // T.int64(32), v_i1])
                T.writes(dequantize_intermediate_intermediate[v_i0, v_i1])
                dequantize_intermediate_intermediate[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * scale[v_i0 // T.int64(32), v_i1]
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(12288), T.int64(4096)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(rms_norm130[v_i0, v_i1, v_k], dequantize_intermediate_intermediate[v_k, v_i2])
                T.writes(matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                matmul_intermediate[v_i0, v_i1, v_i2] = matmul_intermediate[v_i0, v_i1, v_i2] + rms_norm130[v_i0, v_i1, v_k] * dequantize_intermediate_intermediate[v_k, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(12288)):
            with T.sblock("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(matmul_intermediate[v_ax0, v_ax1, v_ax2], transformer_h_0_attn_c_attn_bias3[v_ax2])
                T.writes(T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2] = matmul_intermediate[v_ax0, v_ax1, v_ax2] + transformer_h_0_attn_c_attn_bias3[v_ax2]

    @T.prim_func
    def expected(quant: T.Buffer((T.int64(512), T.int64(12288)), "uint32"), scale: T.Buffer((T.int64(128), T.int64(12288)), "float16"), p_rms_norm130: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12288),), "float16"), p_output0: T.handle):
        T.func_attr({"global_symbol": "before", "tir.is_scheduled": True, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(4096)), "float16")
        T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "float16")
        # with T.sblock("root"):
        dequantize_intermediate_intermediate_local_wmma_matrix_b = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16", scope="wmma.matrix_b")
        quant_local = T.alloc_buffer((T.int64(12288), T.int64(512)), "uint32", scope="local")
        rms_norm130_pad = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(4096)), "float16")
        matmul_intermediate_pad_wmma_accumulator = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(12288)), "float16", scope="wmma.accumulator")
        rms_norm130_pad_global_wmma_matrix_a = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(4096)), "float16", scope="wmma.matrix_a")
        matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(12288)), "float16", scope="local")
        dequantize_intermediate_intermediate_local_local = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16", scope="local")
        for i0 in T.thread_binding(T.int64(1), thread="blockIdx.y"):
            for i1_i2_0_fused_0 in T.thread_binding((seq_len + T.int64(127)) // T.int64(128) * T.int64(1024), thread="blockIdx.x"):
                for i1_i2_0_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                    for i2_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                        for i2_2 in T.vectorized(T.int64(4)):
                            with T.sblock("rms_norm130_pad"):
                                v0 = T.axis.spatial(T.int64(1), i0)
                                v1 = T.axis.spatial((seq_len + T.int64(127)) // T.int64(128) * T.int64(128), (i1_i2_0_fused_0 * T.int64(4) + i1_i2_0_fused_1) // T.int64(32))
                                v2 = T.axis.spatial(T.int64(4096), (i1_i2_0_fused_0 * T.int64(4) + i1_i2_0_fused_1) % T.int64(32) * T.int64(128) + i2_1 * T.int64(4) + i2_2)
                                T.reads(rms_norm130[v0, v1, v2])
                                T.writes(rms_norm130_pad[v0, v1, v2])
                                T.sblock_attr({"buffer_dim_align": [[0, 1, 4096, 64]]})
                                rms_norm130_pad[v0, v1, v2] = T.if_then_else(v1 < seq_len, rms_norm130[v0, v1, v2], T.float16(0.0))
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0 in T.thread_binding((seq_len + T.int64(127)) // T.int64(128), thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(T.int64(96), thread="blockIdx.y"):
                    for ax1_2 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(1), thread="vthread.y"):
                            for ax1_1 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                                for ax2_1 in T.thread_binding(T.int64(2), thread="threadIdx.z", annotations={"pragma_auto_unroll_max_step": 8, "pragma_unroll_explicit": 1}):
                                    with T.sblock("matmul_init_o"):
                                        v0_o = T.axis.spatial(T.int64(1), ax0)
                                        v1_o = T.axis.spatial(T.int64(2) * ((seq_len + T.int64(127)) // T.int64(128)), ax1_0 * T.int64(2) + ax1_1 + ax1_2)
                                        v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                        T.reads()
                                        T.writes(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                        C = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                        T.tvm_fill_fragment(C.data, 64, 64, 16, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), T.float32(0.0))
                                    for ax3_0, ax3_1 in T.grid(T.int64(128), T.int64(2)):
                                        for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                            for ax1 in T.unroll(T.int64(2)):
                                                with T.sblock("quant_local"):
                                                    v0 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_1)
                                                    v1 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(4) + ax3_1 * T.int64(2) + ax1)
                                                    T.reads(quant[v1, v0])
                                                    T.writes(quant_local[v0, v1])
                                                    quant_local[v0, v1] = quant[v1, v0]
                                        for ax3_2 in range(T.int64(1)):
                                            with T.sblock("rms_norm130_pad_global_wmma.matrix_a_o"):
                                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1_o = T.axis.spatial(T.int64(2) * ((seq_len + T.int64(127)) // T.int64(128)), ax1_0 * T.int64(2) + ax1_1)
                                                v2_o = T.axis.spatial(T.int64(256), ax3_0 * T.int64(2) + ax3_1)
                                                T.reads(rms_norm130_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                T.writes(rms_norm130_pad_global_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                A = T.match_buffer(rms_norm130_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), offset_factor=16)
                                                C = T.match_buffer(rms_norm130_pad_global_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                                for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                    T.tvm_load_matrix_sync(C.data, 64, 64, 16, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(64), 1), A.strides[0], "row_major")
                                            for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                for ax1 in T.unroll(T.int64(16)):
                                                    with T.sblock("dequantize"):
                                                        v0 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_1)
                                                        v1 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(32) + ax3_1 * T.int64(16) + ax1)
                                                        T.reads(quant_local[v0, v1 // T.int64(8)], scale[v1 // T.int64(32), v0])
                                                        T.writes(dequantize_intermediate_intermediate_local_local[v0, v1])
                                                        dequantize_intermediate_intermediate_local_local[v0, v1] = (T.Cast("float16", T.bitwise_and(T.shift_right(quant_local[v0, v1 // T.int64(8)], T.Cast("uint32", v1 % T.int64(8) * T.int64(4))), T.uint32(15))) - T.float16(7.0)) * scale[v1 // T.int64(32), v0]
                                            for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                with T.sblock("dequantize_intermediate_intermediate_local_local_o"):
                                                    v0_o = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_1)
                                                    v1_o = T.axis.spatial(T.int64(256), ax3_0 * T.int64(2) + ax3_1)
                                                    T.reads(dequantize_intermediate_intermediate_local_local[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                                    T.writes(dequantize_intermediate_intermediate_local_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                                    A = T.match_buffer(dequantize_intermediate_intermediate_local_local[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16),), "float16", scope="local", offset_factor=16)
                                                    C = T.match_buffer(dequantize_intermediate_intermediate_local_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16),), "float16", scope="wmma.matrix_b", offset_factor=16)
                                                    T.tvm_construct_coopmat_qcom(C.data, 64, 64, 16, A.data)
                                            with T.sblock("matmul_update_o"):
                                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                                v1_o = T.axis.spatial(T.int64(2) * ((seq_len + T.int64(127)) // T.int64(128)), ax1_0 * T.int64(2) + ax1_1 + ax1_2)
                                                v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                                v3_o = T.axis.reduce(T.int64(256), ax3_0 * T.int64(2) + ax3_1 + ax3_2)
                                                T.reads(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], rms_norm130_pad_global_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], dequantize_intermediate_intermediate_local_wmma_matrix_b[v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                                T.writes(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                                A = T.match_buffer(rms_norm130_pad_global_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                                B = T.match_buffer(dequantize_intermediate_intermediate_local_wmma_matrix_b[v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                                C = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                                T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), A.data, A.elem_offset // A.strides[0] // T.int64(64) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(64) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64))
                                    for ax0_1 in range(T.int64(1)):
                                        for ax1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                            with T.sblock("matmul_intermediate_pad_local_o"):
                                                v0_o = T.axis.spatial(T.int64(1), ax0_1)
                                                v1_o = T.axis.spatial(T.int64(128) * ((seq_len + T.int64(127)) // T.int64(128)), ax1_0 * T.int64(128) + ax1_1 * T.int64(64) + ax1_2 * T.int64(64) + ax1)
                                                v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                                T.reads(matmul_intermediate_pad_wmma_accumulator[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                                T.writes(matmul_intermediate_pad_local[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                                A = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64),), "float16", scope="wmma.accumulator", offset_factor=64)
                                                C = T.match_buffer(matmul_intermediate_pad_local[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64),), "float16", scope="local", offset_factor=64)
                                                T.tvm_deconstruct_coopmat_qcom(A.data, 64, 64, 16, C.data)
                                    for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                        for ax1_0_1 in T.unroll(T.int64(16)):
                                            for ax1_1_1 in range(T.int64(4)):
                                                with T.sblock("matmul_intermediate_pad"):
                                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                    v1 = T.axis.spatial(seq_len, ax1_0 * T.int64(128) + ax1_1 * T.int64(64) + ax0_1)
                                                    v2 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax1_0_1 * T.int64(4) + ax1_1_1)
                                                    T.where((ax1_0 - (seq_len + T.int64(127)) // T.int64(128) < T.int64(0) or ax1_0 * T.int64(128) + ax1_1 * T.int64(64) + ax0_1 == T.int64(0)) and ax1_0 * T.int64(128) + ax1_1 * T.int64(64) + ax0_1 < seq_len)
                                                    T.reads(matmul_intermediate_pad_local[v0, v1, v2], transformer_h_0_attn_c_attn_bias3[v2])
                                                    T.writes(T_add_intermediate_intermediate[v0, v1, v2])
                                                    T_add_intermediate_intermediate[v0, v1, v2] = matmul_intermediate_pad_local[v0, v1, v2] + transformer_h_0_attn_c_attn_bias3[v2]
    # fmt: on

    mod = IRModule({"main": before})
    vk_target = {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_khr_cooperative_matrix": True,
        "supports_qcom_cooperative_matrix_conversion": True,
    }
    with Target(vk_target):
        mod = dl.ApplyDefaultSchedule(dl.adreno.DequantMatmulTensorization()).transform_module(
            mod, False
        )
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_dequant_matmul_const_qcom():
    # fmt: off
    @T.prim_func
    def before(quant: T.Buffer((T.int64(512), T.int64(12288)), "uint32"), scale: T.Buffer((T.int64(128), T.int64(12288)), "float16"), rms_norm130: T.Buffer((T.int64(1), T.int64(576), T.int64(4096)), "float16"), transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12288),), "float16"), T_add_intermediate_intermediate: T.Buffer((T.int64(1), T.int64(576), T.int64(12288)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        compute = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
        dequantize_intermediate_intermediate = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
        matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(576), T.int64(12288)), "float16")
        for i0, i1 in T.grid(T.int64(4096), T.int64(12288)):
            with T.sblock("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])

                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(quant[v_i0 // T.int64(8), v_i1], T.Cast("uint32", v_i0 % T.int64(8) * T.int64(4))), T.uint32(15)))
        for i0, i1 in T.grid(T.int64(4096), T.int64(12288)):
            with T.sblock("dequantize"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(compute[v_i0, v_i1], scale[v_i0 // T.int64(32), v_i1])
                T.writes(dequantize_intermediate_intermediate[v_i0, v_i1])
                dequantize_intermediate_intermediate[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * scale[v_i0 // T.int64(32), v_i1]
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(576), T.int64(12288), T.int64(4096)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(rms_norm130[v_i0, v_i1, v_k], dequantize_intermediate_intermediate[v_k, v_i2])
                T.writes(matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                matmul_intermediate[v_i0, v_i1, v_i2] = matmul_intermediate[v_i0, v_i1, v_i2] + rms_norm130[v_i0, v_i1, v_k] * dequantize_intermediate_intermediate[v_k, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(576), T.int64(12288)):
            with T.sblock("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(matmul_intermediate[v_ax0, v_ax1, v_ax2], transformer_h_0_attn_c_attn_bias3[v_ax2])
                T.writes(T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2] = matmul_intermediate[v_ax0, v_ax1, v_ax2] + transformer_h_0_attn_c_attn_bias3[v_ax2]

    @T.prim_func
    def expected(quant_handle: T.handle, scale_handle: T.handle, rms_norm130_handle: T.handle, transformer_h_0_attn_c_attn_bias3_handle: T.handle, T_add_intermediate_intermediate_handle: T.handle):
        T.func_attr({"global_symbol": "before", "tir.is_scheduled": True, "tir.noalias": T.bool(True)})
        quant = T.match_buffer(quant_handle, (T.int64(512), T.int64(12288)), "uint32")
        scale = T.match_buffer(scale_handle, (T.int64(128), T.int64(12288)), "float16")
        rms_norm130 = T.match_buffer(rms_norm130_handle, (T.int64(1), T.int64(576), T.int64(4096)), "float16")
        transformer_h_0_attn_c_attn_bias3 = T.match_buffer(transformer_h_0_attn_c_attn_bias3_handle, (T.int64(12288),), "float16")
        T_add_intermediate_intermediate = T.match_buffer(T_add_intermediate_intermediate_handle, (T.int64(1), T.int64(576), T.int64(12288)), "float16")
        with T.sblock("root"):
            T.reads()
            T.writes()
            dequantize_intermediate_intermediate_local_wmma_matrix_b = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16", scope="wmma.matrix_b")
            quant_local = T.alloc_buffer((T.int64(12288), T.int64(512)), "uint32", scope="local")
            rms_norm130_pad = T.alloc_buffer((T.int64(1), T.int64(640), T.int64(4096)), "float16")
            matmul_intermediate_pad_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(640), T.int64(12288)), "float16", scope="wmma.accumulator")
            rms_norm130_pad_global_wmma_matrix_a = T.alloc_buffer((T.int64(1), T.int64(640), T.int64(4096)), "float16", scope="wmma.matrix_a")
            matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), T.int64(640), T.int64(12288)), "float16", scope="local")
            dequantize_intermediate_intermediate_local_local = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16", scope="local")
            for i0 in T.thread_binding(T.int64(1), thread="blockIdx.y"):
                for i1_i2_0_fused_0 in T.thread_binding(T.int64(5120), thread="blockIdx.x"):
                    for i1_i2_0_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                        for i2_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            for i2_2 in T.vectorized(T.int64(4)):
                                with T.sblock("rms_norm130_pad"):
                                    v0 = T.axis.spatial(T.int64(1), i0)
                                    v1 = T.axis.spatial(T.int64(640), (i1_i2_0_fused_0 * T.int64(4) + i1_i2_0_fused_1) // T.int64(32))
                                    v2 = T.axis.spatial(T.int64(4096), (i1_i2_0_fused_0 * T.int64(4) + i1_i2_0_fused_1) % T.int64(32) * T.int64(128) + i2_1 * T.int64(4) + i2_2)
                                    T.reads(rms_norm130[v0, v1, v2])
                                    T.writes(rms_norm130_pad[v0, v1, v2])
                                    T.sblock_attr({"buffer_dim_align": [[0, 1, 4096, 64]]})
                                    rms_norm130_pad[v0, v1, v2] = T.if_then_else(v1 < T.int64(576), rms_norm130[v0, v1, v2], T.float16(0.0))
            for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
                for ax1_0 in T.thread_binding(T.int64(5), thread="blockIdx.x"):
                    for ax2_0 in T.thread_binding(T.int64(96), thread="blockIdx.y"):
                        for ax1_2 in T.thread_binding(T.int64(1), thread="vthread.x"):
                            for ax2_2 in T.thread_binding(T.int64(1), thread="vthread.y"):
                                for ax1_1 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                                    for ax2_1 in T.thread_binding(T.int64(2), thread="threadIdx.z", annotations={"pragma_auto_unroll_max_step": 8, "pragma_unroll_explicit": 1}):
                                        with T.sblock("matmul_init_o"):
                                            v0_o = T.axis.spatial(T.int64(1), ax0)
                                            v1_o = T.axis.spatial(T.int64(10), ax1_0 * T.int64(2) + ax1_1 + ax1_2)
                                            v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                            T.reads()
                                            T.writes(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                            C = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                            T.tvm_fill_fragment(C.data, 64, 64, 16, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), T.float32(0.0))
                                        for ax3_0 in range(T.int64(128)):
                                            for ax3_1 in range(T.int64(2)):
                                                for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                    for ax1 in T.unroll(T.int64(2)):
                                                        with T.sblock("quant_local"):
                                                            v0 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_1)
                                                            v1 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(4) + ax3_1 * T.int64(2) + ax1)
                                                            T.reads(quant[v1, v0])
                                                            T.writes(quant_local[v0, v1])
                                                            quant_local[v0, v1] = quant[v1, v0]
                                                for ax3_2 in range(T.int64(1)):
                                                    with T.sblock("rms_norm130_pad_global_wmma.matrix_a_o"):
                                                        v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1_o = T.axis.spatial(T.int64(10), ax1_0 * T.int64(2) + ax1_1)
                                                        v2_o = T.axis.spatial(T.int64(256), ax3_0 * T.int64(2) + ax3_1)
                                                        T.reads(rms_norm130_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                        T.writes(rms_norm130_pad_global_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                        A = T.match_buffer(rms_norm130_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), offset_factor=16)
                                                        C = T.match_buffer(rms_norm130_pad_global_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                                        for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                            T.tvm_load_matrix_sync(C.data, 64, 64, 16, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(64), 1), A.strides[0], "row_major")
                                                    for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                        for ax1 in T.unroll(T.int64(16)):
                                                            with T.sblock("dequantize"):
                                                                v0 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_1)
                                                                v1 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(32) + ax3_1 * T.int64(16) + ax1)
                                                                T.reads(quant_local[v0, v1 // T.int64(8)], scale[v1 // T.int64(32), v0])
                                                                T.writes(dequantize_intermediate_intermediate_local_local[v0, v1])
                                                                dequantize_intermediate_intermediate_local_local[v0, v1] = (T.Cast("float16", T.bitwise_and(T.shift_right(quant_local[v0, v1 // T.int64(8)], T.Cast("uint32", v1 % T.int64(8) * T.int64(4))), T.uint32(15))) - T.float16(7.0)) * scale[v1 // T.int64(32), v0]
                                                    for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                        with T.sblock("dequantize_intermediate_intermediate_local_local_o"):
                                                            v0_o = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_1)
                                                            v1_o = T.axis.spatial(T.int64(256), ax3_0 * T.int64(2) + ax3_1)
                                                            T.reads(dequantize_intermediate_intermediate_local_local[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                                            T.writes(dequantize_intermediate_intermediate_local_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                                            A = T.match_buffer(dequantize_intermediate_intermediate_local_local[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16),), "float16", scope="local", offset_factor=16)
                                                            C = T.match_buffer(dequantize_intermediate_intermediate_local_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16),), "float16", scope="wmma.matrix_b", offset_factor=16)
                                                            T.tvm_construct_coopmat_qcom(C.data, 64, 64, 16, A.data)
                                                    with T.sblock("matmul_update_o"):
                                                        v0_o = T.axis.spatial(T.int64(1), ax0)
                                                        v1_o = T.axis.spatial(T.int64(10), ax1_0 * T.int64(2) + ax1_1 + ax1_2)
                                                        v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                                        v3_o = T.axis.reduce(T.int64(256), ax3_0 * T.int64(2) + ax3_1 + ax3_2)
                                                        T.reads(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], rms_norm130_pad_global_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], dequantize_intermediate_intermediate_local_wmma_matrix_b[v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                                        T.writes(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                                        A = T.match_buffer(rms_norm130_pad_global_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                                        B = T.match_buffer(dequantize_intermediate_intermediate_local_wmma_matrix_b[v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                                        C = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                                        T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), A.data, A.elem_offset // A.strides[0] // T.int64(64) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(64) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64))
                                        for ax0_1 in range(T.int64(1)):
                                            for ax1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                with T.sblock("matmul_intermediate_pad_local_o"):
                                                    v0_o = T.axis.spatial(T.int64(1), ax0_1)
                                                    v1_o = T.axis.spatial(T.int64(640), ax1_0 * T.int64(128) + ax1_1 * T.int64(64) + ax1_2 * T.int64(64) + ax1)
                                                    v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                                    T.reads(matmul_intermediate_pad_wmma_accumulator[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                                    T.writes(matmul_intermediate_pad_local[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                                    A = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64),), "float16", scope="wmma.accumulator", offset_factor=64)
                                                    C = T.match_buffer(matmul_intermediate_pad_local[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64),), "float16", scope="local", offset_factor=64)
                                                    T.tvm_deconstruct_coopmat_qcom(A.data, 64, 64, 16, C.data)
                                        for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                            for ax1_0_1 in T.unroll(T.int64(16)):
                                                for ax1_1_1 in range(T.int64(4)):
                                                    with T.sblock("matmul_intermediate_pad"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(576), (ax1_0 * T.int64(2) + ax1_1) * T.int64(64) + ax0_1)
                                                        v2 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax1_0_1 * T.int64(4) + ax1_1_1)
                                                        T.where(ax1_0 * T.int64(2) + ax1_1 < T.int64(9))
                                                        T.reads(matmul_intermediate_pad_local[v0, v1, v2], transformer_h_0_attn_c_attn_bias3[v2])
                                                        T.writes(T_add_intermediate_intermediate[v0, v1, v2])
                                                        T_add_intermediate_intermediate[v0, v1, v2] = matmul_intermediate_pad_local[v0, v1, v2] + transformer_h_0_attn_c_attn_bias3[v2]
    # fmt: on

    mod = IRModule({"main": before})
    vk_target = {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_khr_cooperative_matrix": True,
        "supports_qcom_cooperative_matrix_conversion": True,
    }
    with Target(vk_target):
        mod = dl.ApplyDefaultSchedule(dl.adreno.DequantMatmulTensorization()).transform_module(
            mod, False
        )
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_dequant_matmul_trans_khr():
    # fmt: off
    @T.prim_func
    def before(quant: T.Buffer((T.int64(12288), T.int64(512)), "uint32"), scale: T.Buffer((T.int64(12288), T.int64(128)), "float16"), p_rms_norm130: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12288),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(4096)), "float16")
        T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "float16")
        compute = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16")
        dequantize_intermediate_intermediate = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16")
        matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(12288)), "float16")
        for i0, i1 in T.grid(T.int64(12288), T.int64(4096)):
            with T.sblock("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(quant[v_i0, v_i1 // T.int64(8)])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(quant[v_i0, v_i1 // T.int64(8)], T.Cast("uint32", v_i1 % T.int64(8) * T.int64(4)),), T.uint32(15),),)
        for i0, i1 in T.grid(T.int64(12288), T.int64(4096)):
            with T.sblock("dequantize"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(compute[v_i0, v_i1], scale[v_i0, v_i1 // T.int64(32)])
                T.writes(dequantize_intermediate_intermediate[v_i0, v_i1])
                dequantize_intermediate_intermediate[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * scale[v_i0, v_i1 // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(12288), T.int64(4096)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(
                    rms_norm130[v_i0, v_i1, v_k], dequantize_intermediate_intermediate[v_i2, v_k]
                )
                T.writes(matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                matmul_intermediate[v_i0, v_i1, v_i2] = (matmul_intermediate[v_i0, v_i1, v_i2] + rms_norm130[v_i0, v_i1, v_k] * dequantize_intermediate_intermediate[v_i2, v_k])
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(12288)):
            with T.sblock("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(
                    matmul_intermediate[v_ax0, v_ax1, v_ax2],
                    transformer_h_0_attn_c_attn_bias3[v_ax2],
                )
                T.writes(T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2] = (matmul_intermediate[v_ax0, v_ax1, v_ax2] + transformer_h_0_attn_c_attn_bias3[v_ax2])

    @T.prim_func
    def expected(quant: T.Buffer((T.int64(12288), T.int64(512)), "uint32"), scale: T.Buffer((T.int64(12288), T.int64(128)), "float16"), p_rms_norm130: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12288),), "float16"), p_output0: T.handle):
        T.func_attr({"global_symbol": "before", "tir.is_scheduled": True, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(4096)), "float16")
        T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "float16")
        # with T.sblock("root"):
        dequantize_intermediate_intermediate_shared_wmma_matrix_b = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16", scope="wmma.matrix_b")
        quant_local = T.alloc_buffer((T.int64(12288), T.int64(512)), "uint32", scope="local")
        rms_norm130_pad = T.alloc_buffer((T.int64(1), (seq_len + T.int64(63)) // T.int64(64) * T.int64(64), T.int64(4096)), "float16")
        matmul_intermediate_pad_wmma_accumulator = T.alloc_buffer((T.int64(1), (seq_len + T.int64(63)) // T.int64(64) * T.int64(64), T.int64(12288)), "float16", scope="wmma.accumulator")
        rms_norm130_pad_global_wmma_matrix_a = T.alloc_buffer((T.int64(1), (seq_len + T.int64(63)) // T.int64(64) * T.int64(64), T.int64(4096)), "float16", scope="wmma.matrix_a")
        matmul_intermediate_pad_shared = T.alloc_buffer((T.int64(1), (seq_len + T.int64(63)) // T.int64(64) * T.int64(64), T.int64(12288)), "float16", scope="shared")
        dequantize_intermediate_intermediate_shared_shared = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16", scope="shared")
        for i0 in T.thread_binding(T.int64(1), thread="blockIdx.y"):
            for i1_i2_0_fused_0 in T.thread_binding((seq_len + T.int64(63)) // T.int64(64) * T.int64(512), thread="blockIdx.x"):
                for i1_i2_0_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                    for i2_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                        for i2_2 in T.vectorized(T.int64(4)):
                            with T.sblock("rms_norm130_pad"):
                                v0 = T.axis.spatial(T.int64(1), i0)
                                v1 = T.axis.spatial((seq_len + T.int64(63)) // T.int64(64) * T.int64(64), (i1_i2_0_fused_0 * T.int64(4) + i1_i2_0_fused_1) // T.int64(32))
                                v2 = T.axis.spatial(T.int64(4096), (i1_i2_0_fused_0 * T.int64(4) + i1_i2_0_fused_1) % T.int64(32) * T.int64(128) + i2_1 * T.int64(4) + i2_2)
                                T.reads(rms_norm130[v0, v1, v2])
                                T.writes(rms_norm130_pad[v0, v1, v2])
                                T.sblock_attr({"buffer_dim_align": [[0, 1, 4096, 64]]})
                                rms_norm130_pad[v0, v1, v2] = T.if_then_else(v1 < seq_len, rms_norm130[v0, v1, v2], T.float16(0.0))
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0 in T.thread_binding((seq_len + T.int64(63)) // T.int64(64), thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(T.int64(96), thread="blockIdx.y"):
                    for ax1_2 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(1), thread="vthread.y"):
                            for ax1_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax2_1 in T.thread_binding(T.int64(2), thread="threadIdx.z", annotations={"pragma_auto_unroll_max_step": 8, "pragma_unroll_explicit": 1}):
                                    with T.sblock("matmul_init_o"):
                                        v0_o = T.axis.spatial(T.int64(1), ax0)
                                        v1_o = T.axis.spatial((seq_len + T.int64(63)) // T.int64(64), ax1_0 + ax1_1 + ax1_2)
                                        v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                        T.reads()
                                        T.writes(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                        C = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                        T.tvm_fill_fragment(C.data, 64, 64, 16, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), T.float32(0.0))
                                    for ax3_0, ax3_1 in T.grid(T.int64(128), T.int64(2)):
                                        for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                            for ax1 in T.unroll(T.int64(2)):
                                                with T.sblock("quant_local"):
                                                    v0 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_1)
                                                    v1 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(4) + ax3_1 * T.int64(2) + ax1)
                                                    T.reads(quant[v0, v1])
                                                    T.writes(quant_local[v0, v1])
                                                    quant_local[v0, v1] = quant[v0, v1]
                                        for ax3_2 in range(T.int64(1)):
                                            with T.sblock("rms_norm130_pad_global_wmma.matrix_a_o"):
                                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1_o = T.axis.spatial((seq_len + T.int64(63)) // T.int64(64), ax1_0)
                                                v2_o = T.axis.spatial(T.int64(256), ax3_0 * T.int64(2) + ax3_1)
                                                T.reads(rms_norm130_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                T.writes(rms_norm130_pad_global_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                A = T.match_buffer(rms_norm130_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), offset_factor=16)
                                                C = T.match_buffer(rms_norm130_pad_global_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                                for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                    T.tvm_load_matrix_sync(C.data, 64, 64, 16, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(64), 1), A.strides[0], "row_major")
                                            for ax0_0 in range(T.int64(1)):
                                                for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                    for ax1_0_1 in T.unroll(T.int64(4)):
                                                        for ax1_1_1 in T.vectorized(T.int64(4)):
                                                            with T.sblock("dequantize"):
                                                                v_i0 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_0 * T.int64(64) + ax0_1)
                                                                v_i1 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(32) + ax3_1 * T.int64(16) + ax1_0_1 * T.int64(4) + ax1_1_1)
                                                                T.reads(quant_local[v_i0, v_i1 // T.int64(8)], scale[v_i0, v_i1 // T.int64(32)])
                                                                T.writes(dequantize_intermediate_intermediate_shared_shared[v_i0, v_i1])
                                                                dequantize_intermediate_intermediate_shared_shared[v_i0, v_i1] = (T.Cast("float16", T.bitwise_and(T.shift_right(quant_local[v_i0, v_i1 // T.int64(8)], T.Cast("uint32", v_i1 % T.int64(8) * T.int64(4))), T.uint32(15))) - T.float16(7.0)) * scale[v_i0, v_i1 // T.int64(32)]
                                            for ax0_0 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                                                with T.sblock("dequantize_intermediate_intermediate_shared_shared_o"):
                                                    v0_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax0_0)
                                                    v1_o = T.axis.spatial(T.int64(256), ax3_0 * T.int64(2) + ax3_1)
                                                    T.reads(dequantize_intermediate_intermediate_shared_shared[v0_o * T.int64(64):v0_o * T.int64(64) + T.int64(64), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                                    T.writes(dequantize_intermediate_intermediate_shared_wmma_matrix_b[v0_o * T.int64(64):v0_o * T.int64(64) + T.int64(64), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                                    A = T.match_buffer(dequantize_intermediate_intermediate_shared_shared[v0_o * T.int64(64):v0_o * T.int64(64) + T.int64(64), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared", offset_factor=16)
                                                    C = T.match_buffer(dequantize_intermediate_intermediate_shared_wmma_matrix_b[v0_o * T.int64(64):v0_o * T.int64(64) + T.int64(64), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                                    for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                        T.tvm_load_matrix_sync(C.data, 64, 64, 16, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(64), 1), A.strides[0], "col_major")
                                            with T.sblock("matmul_update_o"):
                                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                                v1_o = T.axis.spatial((seq_len + T.int64(63)) // T.int64(64), ax1_0 + ax1_1 + ax1_2)
                                                v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                                v3_o = T.axis.reduce(T.int64(256), ax3_0 * T.int64(2) + ax3_1 + ax3_2)
                                                T.reads(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], rms_norm130_pad_global_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], dequantize_intermediate_intermediate_shared_wmma_matrix_b[v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                                T.writes(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                                A = T.match_buffer(rms_norm130_pad_global_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                                B = T.match_buffer(dequantize_intermediate_intermediate_shared_wmma_matrix_b[v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                                C = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                                T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), A.data, A.elem_offset // A.strides[0] // T.int64(64) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(64) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64))
                                    for ax0_1 in range(T.int64(1)):
                                        with T.sblock("matmul_intermediate_pad_shared_o"):
                                            v0_o = T.axis.spatial(T.int64(1), ax0_1)
                                            v1_o = T.axis.spatial((seq_len + T.int64(63)) // T.int64(64), ax1_0 + ax1_1 + ax1_2)
                                            v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                            T.reads(matmul_intermediate_pad_wmma_accumulator[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                            T.writes(matmul_intermediate_pad_shared[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                            A = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=64)
                                            C = T.match_buffer(matmul_intermediate_pad_shared[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="shared", offset_factor=64)
                                            T.tvm_store_matrix_sync(A.data, 64, 64, 16, A.elem_offset // A.strides[0] // T.int64(64) * (A.strides[0] // T.int64(64)) + A.elem_offset % A.strides[0] // T.int64(64), T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(64), 2), C.strides[0], "row_major")
                                    for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                        for ax1_0_1 in T.unroll(T.int64(16)):
                                            for ax1_1_1 in range(T.int64(4)):
                                                with T.sblock("matmul_intermediate_pad"):
                                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                    v1 = T.axis.spatial(seq_len, ax1_0 * T.int64(64) + ax0_1)
                                                    v2 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax1_0_1 * T.int64(4) + ax1_1_1)
                                                    T.where((ax1_0 - (seq_len + T.int64(63)) // T.int64(64) < T.int64(0) or ax1_0 * T.int64(64) + ax0_1 == T.int64(0)) and ax1_0 * T.int64(64) + ax0_1 < seq_len)
                                                    T.reads(matmul_intermediate_pad_shared[v0, v1, v2], transformer_h_0_attn_c_attn_bias3[v2])
                                                    T.writes(T_add_intermediate_intermediate[v0, v1, v2])
                                                    T_add_intermediate_intermediate[v0, v1, v2] = matmul_intermediate_pad_shared[v0, v1, v2] + transformer_h_0_attn_c_attn_bias3[v2]
    # fmt: on

    mod = IRModule({"main": before})
    vk_target = {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_khr_cooperative_matrix": True,
    }
    with Target(vk_target):
        mod = dl.ApplyDefaultSchedule(dl.adreno.DequantMatmulTensorization()).transform_module(
            mod, False
        )
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_dequant_matmul_trans_qcom():
    # fmt: off
    @T.prim_func
    def before(quant: T.Buffer((T.int64(12288), T.int64(512)), "uint32"), scale: T.Buffer((T.int64(12288), T.int64(128)), "float16"), p_rms_norm130: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12288),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(4096)), "float16")
        T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "float16")
        compute = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16")
        dequantize_intermediate_intermediate = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16")
        matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(12288)), "float16")
        for i0, i1 in T.grid(T.int64(12288), T.int64(4096)):
            with T.sblock("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(quant[v_i0, v_i1 // T.int64(8)])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(quant[v_i0, v_i1 // T.int64(8)], T.Cast("uint32", v_i1 % T.int64(8) * T.int64(4)),), T.uint32(15),),)
        for i0, i1 in T.grid(T.int64(12288), T.int64(4096)):
            with T.sblock("dequantize"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(compute[v_i0, v_i1], scale[v_i0, v_i1 // T.int64(32)])
                T.writes(dequantize_intermediate_intermediate[v_i0, v_i1])
                dequantize_intermediate_intermediate[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * scale[v_i0, v_i1 // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(12288), T.int64(4096)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(
                    rms_norm130[v_i0, v_i1, v_k], dequantize_intermediate_intermediate[v_i2, v_k]
                )
                T.writes(matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                matmul_intermediate[v_i0, v_i1, v_i2] = (matmul_intermediate[v_i0, v_i1, v_i2] + rms_norm130[v_i0, v_i1, v_k] * dequantize_intermediate_intermediate[v_i2, v_k])
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(12288)):
            with T.sblock("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(
                    matmul_intermediate[v_ax0, v_ax1, v_ax2],
                    transformer_h_0_attn_c_attn_bias3[v_ax2],
                )
                T.writes(T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2] = (matmul_intermediate[v_ax0, v_ax1, v_ax2] + transformer_h_0_attn_c_attn_bias3[v_ax2])

    @T.prim_func
    def expected(quant: T.Buffer((T.int64(12288), T.int64(512)), "uint32"), scale: T.Buffer((T.int64(12288), T.int64(128)), "float16"), p_rms_norm130: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12288),), "float16"), p_output0: T.handle):
        T.func_attr({"global_symbol": "before", "tir.is_scheduled": True, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(4096)), "float16")
        T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "float16")
        # with T.sblock("root"):
        dequantize_intermediate_intermediate_local_wmma_matrix_b = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16", scope="wmma.matrix_b")
        quant_local = T.alloc_buffer((T.int64(12288), T.int64(512)), "uint32", scope="local")
        rms_norm130_pad = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(4096)), "float16")
        matmul_intermediate_pad_wmma_accumulator = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(12288)), "float16", scope="wmma.accumulator")
        rms_norm130_pad_global_wmma_matrix_a = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(4096)), "float16", scope="wmma.matrix_a")
        matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(12288)), "float16", scope="local")
        dequantize_intermediate_intermediate_local_local = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16", scope="local")
        for i0 in T.thread_binding(T.int64(1), thread="blockIdx.y"):
            for i1_i2_0_fused_0 in T.thread_binding((seq_len + T.int64(127)) // T.int64(128) * T.int64(1024), thread="blockIdx.x"):
                for i1_i2_0_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                    for i2_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                        for i2_2 in T.vectorized(T.int64(4)):
                            with T.sblock("rms_norm130_pad"):
                                v0 = T.axis.spatial(T.int64(1), i0)
                                v1 = T.axis.spatial((seq_len + T.int64(127)) // T.int64(128) * T.int64(128), (i1_i2_0_fused_0 * T.int64(4) + i1_i2_0_fused_1) // T.int64(32))
                                v2 = T.axis.spatial(T.int64(4096), (i1_i2_0_fused_0 * T.int64(4) + i1_i2_0_fused_1) % T.int64(32) * T.int64(128) + i2_1 * T.int64(4) + i2_2)
                                T.reads(rms_norm130[v0, v1, v2])
                                T.writes(rms_norm130_pad[v0, v1, v2])
                                T.sblock_attr({"buffer_dim_align": [[0, 1, 4096, 64]]})
                                rms_norm130_pad[v0, v1, v2] = T.if_then_else(v1 < seq_len, rms_norm130[v0, v1, v2], T.float16(0.0))
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0 in T.thread_binding((seq_len + T.int64(127)) // T.int64(128), thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(T.int64(96), thread="blockIdx.y"):
                    for ax1_2 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(1), thread="vthread.y"):
                            for ax1_1 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                                for ax2_1 in T.thread_binding(T.int64(2), thread="threadIdx.z", annotations={"pragma_auto_unroll_max_step": 8, "pragma_unroll_explicit": 1}):
                                    with T.sblock("matmul_init_o"):
                                        v0_o = T.axis.spatial(T.int64(1), ax0)
                                        v1_o = T.axis.spatial(T.int64(2) * ((seq_len + T.int64(127)) // T.int64(128)), ax1_0 * T.int64(2) + ax1_1 + ax1_2)
                                        v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                        T.reads()
                                        T.writes(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                        C = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                        T.tvm_fill_fragment(C.data, 64, 64, 16, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), T.float32(0.0))
                                    for ax3_0, ax3_1 in T.grid(T.int64(128), T.int64(2)):
                                        for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                            for ax1 in T.unroll(T.int64(2)):
                                                with T.sblock("quant_local"):
                                                    v0 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_1)
                                                    v1 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(4) + ax3_1 * T.int64(2) + ax1)
                                                    T.reads(quant[v0, v1])
                                                    T.writes(quant_local[v0, v1])
                                                    quant_local[v0, v1] = quant[v0, v1]
                                        for ax3_2 in range(T.int64(1)):
                                            with T.sblock("rms_norm130_pad_global_wmma.matrix_a_o"):
                                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1_o = T.axis.spatial(T.int64(2) * ((seq_len + T.int64(127)) // T.int64(128)), ax1_0 * T.int64(2) + ax1_1)
                                                v2_o = T.axis.spatial(T.int64(256), ax3_0 * T.int64(2) + ax3_1)
                                                T.reads(rms_norm130_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                T.writes(rms_norm130_pad_global_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                A = T.match_buffer(rms_norm130_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), offset_factor=16)
                                                C = T.match_buffer(rms_norm130_pad_global_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                                for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                    T.tvm_load_matrix_sync(C.data, 64, 64, 16, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(64), 1), A.strides[0], "row_major")
                                            for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                for ax1 in T.unroll(T.int64(16)):
                                                    with T.sblock("dequantize"):
                                                        v_i0 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_1)
                                                        v_i1 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(32) + ax3_1 * T.int64(16) + ax1)
                                                        T.reads(quant_local[v_i0, v_i1 // T.int64(8)], scale[v_i0, v_i1 // T.int64(32)])
                                                        T.writes(dequantize_intermediate_intermediate_local_local[v_i0, v_i1])
                                                        dequantize_intermediate_intermediate_local_local[v_i0, v_i1] = (T.Cast("float16", T.bitwise_and(T.shift_right(quant_local[v_i0, v_i1 // T.int64(8)], T.Cast("uint32", v_i1 % T.int64(8) * T.int64(4))), T.uint32(15))) - T.float16(7.0)) * scale[v_i0, v_i1 // T.int64(32)]
                                            for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                with T.sblock("dequantize_intermediate_intermediate_local_local_o"):
                                                    v0_o = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_1)
                                                    v1_o = T.axis.spatial(T.int64(256), ax3_0 * T.int64(2) + ax3_1)
                                                    T.reads(dequantize_intermediate_intermediate_local_local[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                                    T.writes(dequantize_intermediate_intermediate_local_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                                    A = T.match_buffer(dequantize_intermediate_intermediate_local_local[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16),), "float16", scope="local", offset_factor=16)
                                                    C = T.match_buffer(dequantize_intermediate_intermediate_local_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16),), "float16", scope="wmma.matrix_b", offset_factor=16)
                                                    T.tvm_construct_coopmat_qcom(C.data, 64, 64, 16, A.data)
                                            with T.sblock("matmul_update_o"):
                                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                                v1_o = T.axis.spatial(T.int64(2) * ((seq_len + T.int64(127)) // T.int64(128)), ax1_0 * T.int64(2) + ax1_1 + ax1_2)
                                                v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                                v3_o = T.axis.reduce(T.int64(256), ax3_0 * T.int64(2) + ax3_1 + ax3_2)
                                                T.reads(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], rms_norm130_pad_global_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], dequantize_intermediate_intermediate_local_wmma_matrix_b[v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                                T.writes(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                                A = T.match_buffer(rms_norm130_pad_global_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                                B = T.match_buffer(dequantize_intermediate_intermediate_local_wmma_matrix_b[v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                                C = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                                T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), A.data, A.elem_offset // A.strides[0] // T.int64(64) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(64) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64))
                                    for ax0_1 in range(T.int64(1)):
                                        for ax1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                            with T.sblock("matmul_intermediate_pad_local_o"):
                                                v0_o = T.axis.spatial(T.int64(1), ax0_1)
                                                v1_o = T.axis.spatial(T.int64(128) * ((seq_len + T.int64(127)) // T.int64(128)), ax1_0 * T.int64(128) + ax1_1 * T.int64(64) + ax1_2 * T.int64(64) + ax1)
                                                v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                                T.reads(matmul_intermediate_pad_wmma_accumulator[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                                T.writes(matmul_intermediate_pad_local[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                                A = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64),), "float16", scope="wmma.accumulator", offset_factor=64)
                                                C = T.match_buffer(matmul_intermediate_pad_local[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64),), "float16", scope="local", offset_factor=64)
                                                T.tvm_deconstruct_coopmat_qcom(A.data, 64, 64, 16, C.data)
                                    for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                        for ax1_0_1 in T.unroll(T.int64(16)):
                                            for ax1_1_1 in range(T.int64(4)):
                                                with T.sblock("matmul_intermediate_pad"):
                                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                    v1 = T.axis.spatial(seq_len, ax1_0 * T.int64(128) + ax1_1 * T.int64(64) + ax0_1)
                                                    v2 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax1_0_1 * T.int64(4) + ax1_1_1)
                                                    T.where((ax1_0 - (seq_len + T.int64(127)) // T.int64(128) < T.int64(0) or ax1_0 * T.int64(128) + ax1_1 * T.int64(64) + ax0_1 == T.int64(0)) and ax1_0 * T.int64(128) + ax1_1 * T.int64(64) + ax0_1 < seq_len)
                                                    T.reads(matmul_intermediate_pad_local[v0, v1, v2], transformer_h_0_attn_c_attn_bias3[v2])
                                                    T.writes(T_add_intermediate_intermediate[v0, v1, v2])
                                                    T_add_intermediate_intermediate[v0, v1, v2] = matmul_intermediate_pad_local[v0, v1, v2] + transformer_h_0_attn_c_attn_bias3[v2]
    # fmt: on

    mod = IRModule({"main": before})
    vk_target = {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_khr_cooperative_matrix": True,
        "supports_qcom_cooperative_matrix_conversion": True,
    }
    with Target(vk_target):
        mod = dl.ApplyDefaultSchedule(dl.adreno.DequantMatmulTensorization()).transform_module(
            mod, False
        )
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_dequant_matmul_int8():
    # fmt: off
    @T.prim_func
    def before(quant: T.Buffer((T.int64(512), T.int64(12288)), "uint32"), scale: T.Buffer((T.int64(1), T.int64(12288)), "float16"), p_rms_norm130: T.handle, in_scale_var: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12277),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(512)), "int8")
        in_scale = T.match_buffer(in_scale_var, (T.int64(1), seq_len), "float16")
        T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "float16")
        # with T.block("root"):
        compute = T.alloc_buffer((T.int64(4096), T.int64(12288)), "int8")
        dequantize_intermediate_intermediate = T.alloc_buffer((T.int64(4096), T.int64(12288)), "int8")
        matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(12288)), "int32")
        for i0, i1 in T.grid(T.int64(4096), T.int64(12288)):
            with T.sblock("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(quant[v_i0 // T.int64(8), v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("int8", T.bitwise_and(T.shift_right(quant[v_i0 // T.int64(8), v_i1], T.Cast("uint32", v_i0 % T.int64(8) * T.int64(4))), T.uint32(15)))
        for i0, i1 in T.grid(T.int64(4096), T.int64(12288)):
            with T.sblock("dequantize"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(compute[v_i0, v_i1])
                T.writes(dequantize_intermediate_intermediate[v_i0, v_i1])
                dequantize_intermediate_intermediate[v_i0, v_i1] = (compute[v_i0, v_i1] - T.int8(7))
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(12288), T.int64(4096)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(rms_norm130[v_i0, v_i1, v_k], dequantize_intermediate_intermediate[v_k, v_i2])
                T.writes(matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    matmul_intermediate[v_i0, v_i1, v_i2] = T.int32(0)
                matmul_intermediate[v_i0, v_i1, v_i2] = matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("int32", rms_norm130[v_i0, v_i1, v_k]) * T.Cast("int32", dequantize_intermediate_intermediate[v_k, v_i2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(12288)):
            with T.sblock("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(matmul_intermediate[v_ax0, v_ax1, v_ax2], transformer_h_0_attn_c_attn_bias3[v_ax2])
                T.writes(T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2] =  T.Cast("float16", matmul_intermediate[v_ax0, v_ax1, v_ax2]) * in_scale[v_ax0, v_ax1] * scale[v_ax0, v_ax2] + transformer_h_0_attn_c_attn_bias3[v_ax2]

    @T.prim_func
    def expected(quant: T.Buffer((T.int64(512), T.int64(12288)), "uint32"), scale: T.Buffer((T.int64(1), T.int64(12288)), "float16"), p_rms_norm130: T.handle, in_scale_var: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12277),), "float16"), p_output0: T.handle):
        T.func_attr({"global_symbol": "before", "tir.is_scheduled": True, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(512)), "int8")
        in_scale = T.match_buffer(in_scale_var, (T.int64(1), seq_len), "float16")
        T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "float16")
        # with T.sblock("root"):
        dequantize_intermediate_intermediate_local_wmma_matrix_b = T.alloc_buffer((T.int64(12288), T.int64(4096)), "int8", scope="wmma.matrix_b")
        quant_local = T.alloc_buffer((T.int64(12288), T.int64(512)), "uint32", scope="local")
        rms_norm130_pad = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(512)), "int8")
        matmul_intermediate_pad_wmma_accumulator = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(12288)), "int32", scope="wmma.accumulator")
        rms_norm130_pad_global_wmma_matrix_a = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(512)), "int8", scope="wmma.matrix_a")
        matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(12288)), "int32", scope="local")
        dequantize_intermediate_intermediate_local_local = T.alloc_buffer((T.int64(12288), T.int64(4096)), "int8", scope="local")
        for i0 in T.thread_binding(T.int64(1), thread="blockIdx.y"):
            for i1_i2_0_fused_0 in T.thread_binding((seq_len + T.int64(127)) // T.int64(128) * T.int64(128), thread="blockIdx.x"):
                for i1_i2_0_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                    for i2_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                        for i2_2 in T.vectorized(T.int64(4)):
                            with T.sblock("rms_norm130_pad"):
                                v0 = T.axis.spatial(T.int64(1), i0)
                                v1 = T.axis.spatial((seq_len + T.int64(127)) // T.int64(128) * T.int64(128), (i1_i2_0_fused_0 * T.int64(4) + i1_i2_0_fused_1) // T.int64(4))
                                v2 = T.axis.spatial(T.int64(512), (i1_i2_0_fused_0 * T.int64(4) + i1_i2_0_fused_1) % T.int64(4) * T.int64(128) + i2_1 * T.int64(4) + i2_2)
                                T.reads(rms_norm130[v0, v1, v2])
                                T.writes(rms_norm130_pad[v0, v1, v2])
                                T.sblock_attr({"buffer_dim_align": [[0, 1, 512, 64]]})
                                rms_norm130_pad[v0, v1, v2] = T.if_then_else(v1 < seq_len, rms_norm130[v0, v1, v2], T.int8(0))
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0 in T.thread_binding((seq_len + T.int64(127)) // T.int64(128), thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(T.int64(96), thread="blockIdx.y"):
                    for ax1_2 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(1), thread="vthread.y"):
                            for ax1_1 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                                for ax2_1 in T.thread_binding(T.int64(2), thread="threadIdx.z", annotations={"pragma_auto_unroll_max_step": 8, "pragma_unroll_explicit": 1}):
                                    with T.sblock("matmul_init_o"):
                                        v0_o = T.axis.spatial(T.int64(1), ax0)
                                        v1_o = T.axis.spatial(T.int64(2) * ((seq_len + T.int64(127)) // T.int64(128)), ax1_0 * T.int64(2) + ax1_1 + ax1_2)
                                        v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                        T.reads()
                                        T.writes(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                        C = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "int32", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                        T.tvm_fill_fragment(C.data, 64, 64, 32, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), T.float32(0.0))
                                    for ax3_0, ax3_1 in T.grid(T.int64(64), T.int64(2)):
                                        for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                            for ax1 in T.unroll(T.int64(4)):
                                                with T.sblock("quant_local"):
                                                    v0 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_1)
                                                    v1 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(8) + ax3_1 * T.int64(4) + ax1)
                                                    T.reads(quant[v1, v0])
                                                    T.writes(quant_local[v0, v1])
                                                    quant_local[v0, v1] = quant[v1, v0]
                                        for ax3_2 in range(T.int64(1)):
                                            with T.sblock("rms_norm130_pad_global_wmma.matrix_a_o"):
                                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1_o = T.axis.spatial(T.int64(2) * ((seq_len + T.int64(127)) // T.int64(128)), ax1_0 * T.int64(2) + ax1_1)
                                                v2_o = T.axis.spatial(T.int64(16), ax3_0 * T.int64(2) + ax3_1)
                                                T.reads(rms_norm130_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(32):v2_o * T.int64(32) + T.int64(32)])
                                                T.writes(rms_norm130_pad_global_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(32):v2_o * T.int64(32) + T.int64(32)])
                                                A = T.match_buffer(rms_norm130_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(32):v2_o * T.int64(32) + T.int64(32)], (T.int64(64), T.int64(32)), "int8", strides=("A_s0", "A_s1"), offset_factor=32)
                                                C = T.match_buffer(rms_norm130_pad_global_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(32):v2_o * T.int64(32) + T.int64(32)], (T.int64(64), T.int64(32)), "int8", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=32)
                                                for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                    T.tvm_load_matrix_sync(C.data, 64, 64, 32, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(32)) + C.elem_offset % C.strides[0] // T.int64(32), T.tvm_access_ptr(T.type_annotation("int8"), A.data, A.elem_offset, A.strides[0] * T.int64(64), 1), A.strides[0], "row_major")
                                            for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                for ax1 in T.unroll(T.int64(32)):
                                                    with T.sblock("dequantize"):
                                                        v0 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_1)
                                                        v1 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(64) + ax3_1 * T.int64(32) + ax1)
                                                        T.reads(quant_local[v0, v1 // T.int64(8)])
                                                        T.writes(dequantize_intermediate_intermediate_local_local[v0, v1])
                                                        dequantize_intermediate_intermediate_local_local[v0, v1] = T.Cast("int8", T.bitwise_and(T.shift_right(quant_local[v0, v1 // T.int64(8)], T.Cast("uint32", v1 % T.int64(8) * T.int64(4))), T.uint32(15))) - T.int8(7)
                                            for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                                with T.sblock("dequantize_intermediate_intermediate_local_local_o"):
                                                    v0_o = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax0_1)
                                                    v1_o = T.axis.spatial(T.int64(128), ax3_0 * T.int64(2) + ax3_1)
                                                    T.reads(dequantize_intermediate_intermediate_local_local[v0_o, v1_o * T.int64(32):v1_o * T.int64(32) + T.int64(32)])
                                                    T.writes(dequantize_intermediate_intermediate_local_wmma_matrix_b[v0_o, v1_o * T.int64(32):v1_o * T.int64(32) + T.int64(32)])
                                                    A = T.match_buffer(dequantize_intermediate_intermediate_local_local[v0_o, v1_o * T.int64(32):v1_o * T.int64(32) + T.int64(32)], (T.int64(32),), "int8", scope="local", offset_factor=32)
                                                    C = T.match_buffer(dequantize_intermediate_intermediate_local_wmma_matrix_b[v0_o, v1_o * T.int64(32):v1_o * T.int64(32) + T.int64(32)], (T.int64(32),), "int8", scope="wmma.matrix_b", offset_factor=32)
                                                    T.tvm_construct_coopmat_qcom(C.data, 64, 64, 32, A.data)
                                            with T.sblock("matmul_update_o"):
                                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                                v1_o = T.axis.spatial(T.int64(2) * ((seq_len + T.int64(127)) // T.int64(128)), ax1_0 * T.int64(2) + ax1_1 + ax1_2)
                                                v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                                v3_o = T.axis.reduce(T.int64(128), ax3_0 * T.int64(2) + ax3_1 + ax3_2)
                                                T.reads(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], rms_norm130_pad_global_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v3_o * T.int64(32):v3_o * T.int64(32) + T.int64(32)], dequantize_intermediate_intermediate_local_wmma_matrix_b[v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), v3_o * T.int64(32):v3_o * T.int64(32) + T.int64(32)])
                                                T.writes(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                                A = T.match_buffer(rms_norm130_pad_global_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v3_o * T.int64(32):v3_o * T.int64(32) + T.int64(32)], (T.int64(64), T.int64(32)), "int8", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=32)
                                                B = T.match_buffer(dequantize_intermediate_intermediate_local_wmma_matrix_b[v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), v3_o * T.int64(32):v3_o * T.int64(32) + T.int64(32)], (T.int64(64), T.int64(32)), "int8", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=32)
                                                C = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "int32", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                                T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), A.data, A.elem_offset // A.strides[0] // T.int64(64) * (A.strides[0] // T.int64(32)) + A.elem_offset % A.strides[0] // T.int64(32), B.data, B.elem_offset // B.strides[0] // T.int64(64) * (B.strides[0] // T.int64(32)) + B.elem_offset % B.strides[0] // T.int64(32), C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64))
                                    for ax0_1 in range(T.int64(1)):
                                        for ax1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                            with T.sblock("matmul_intermediate_pad_local_o"):
                                                v0_o = T.axis.spatial(T.int64(1), ax0_1)
                                                v1_o = T.axis.spatial(T.int64(128) * ((seq_len + T.int64(127)) // T.int64(128)), ax1_0 * T.int64(128) + ax1_1 * T.int64(64) + ax1_2 * T.int64(64) + ax1)
                                                v2_o = T.axis.spatial(T.int64(192), ax2_0 * T.int64(2) + ax2_1 + ax2_2)
                                                T.reads(matmul_intermediate_pad_wmma_accumulator[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                                T.writes(matmul_intermediate_pad_local[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                                A = T.match_buffer(matmul_intermediate_pad_wmma_accumulator[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64),), "int32", scope="wmma.accumulator", offset_factor=64)
                                                C = T.match_buffer(matmul_intermediate_pad_local[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64),), "int32", scope="local", offset_factor=64)
                                                T.tvm_deconstruct_coopmat_qcom(A.data, 64, 64, 32, C.data)
                                    for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                        for ax1_0_1 in T.unroll(T.int64(16)):
                                            for ax1_1_1 in range(T.int64(4)):
                                                with T.sblock("matmul_intermediate_pad"):
                                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                    v1 = T.axis.spatial(seq_len, ax1_0 * T.int64(128) + ax1_1 * T.int64(64) + ax0_1)
                                                    v2 = T.axis.spatial(T.int64(12288), ax2_0 * T.int64(128) + ax2_1 * T.int64(64) + ax1_0_1 * T.int64(4) + ax1_1_1)
                                                    T.where((ax1_0 - (seq_len + T.int64(127)) // T.int64(128) < T.int64(0) or ax1_0 * T.int64(128) + ax1_1 * T.int64(64) + ax0_1 == T.int64(0)) and ax1_0 * T.int64(128) + ax1_1 * T.int64(64) + ax0_1 < seq_len)
                                                    T.reads(matmul_intermediate_pad_local[v0, v1, v2], transformer_h_0_attn_c_attn_bias3[v2])
                                                    T.writes(T_add_intermediate_intermediate[v0, v1, v2])
                                                    T_add_intermediate_intermediate[v0, v1, v2] = T.Cast("float16", matmul_intermediate_pad_local[v0, v1, v2]) * in_scale[v0, v1] * scale[v0, v2] + transformer_h_0_attn_c_attn_bias3[v2]
    # fmt: on
    mod = IRModule({"main": before})
    vk_target = {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_khr_cooperative_matrix": True,
        "supports_qcom_cooperative_matrix_conversion": True,
    }
    with Target(vk_target):
        mod = dl.ApplyDefaultSchedule(dl.adreno.DequantMatmulTensorization()).transform_module(
            mod, False
        )
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_matmul_no_pad():
    # fmt: off
    @T.prim_func
    def before(A: T.Buffer((T.int64(1), T.int64(64), T.int64(16)), "float16"), B: T.Buffer((T.int64(16), T.int64(64)), "float16"), C: T.Buffer((T.int64(1), T.int64(64), T.int64(64)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.sblock("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(64), T.int64(64), T.int64(16)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_k, v_i2])
                T.writes(C[v_i0, v_i1, v_i2])
                with T.init():
                    C[v_i0, v_i1, v_i2] = T.float16(0)
                C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_k, v_i2]

    @T.prim_func
    def expected(A: T.Buffer((T.int64(1), T.int64(64), T.int64(16)), "float16"), B: T.Buffer((T.int64(16), T.int64(64)), "float16"), C: T.Buffer((T.int64(1), T.int64(64), T.int64(64)), "float16")):
        T.func_attr({"global_symbol": "before", "tir.is_scheduled": True, "tir.noalias": T.bool(True)})
        # with T.sblock("root"):
        A_wmma_matrix_a = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(16)), "float16", scope="wmma.matrix_a")
        B_wmma_matrix_b = T.alloc_buffer((T.int64(16), T.int64(64)), "float16", scope="wmma.matrix_b")
        C_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(64)), "float16", scope="wmma.accumulator")
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(T.int64(1), thread="blockIdx.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                        for ax2_1 in T.thread_binding(T.int64(1), thread="threadIdx.z"):
                            with T.sblock("matmul_init_o"):
                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                v1_o = T.axis.spatial(T.int64(1), ax1_0 + ax1_1)
                                v2_o = T.axis.spatial(T.int64(1), ax2_0 + ax2_1)
                                T.reads()
                                T.writes(C_wmma_accumulator[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                                C_1 = T.match_buffer(C_wmma_accumulator[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                T.tvm_fill_fragment(C_1.data, 64, 64, 16, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(64)) + C_1.elem_offset % C_1.strides[0] // T.int64(64), T.float32(0.0))
                            for ax3_0 in range(T.int64(1)):
                                with T.sblock("A_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(A[v0_o, T.int64(0):T.int64(64), T.int64(0):T.int64(16)])
                                    T.writes(A_wmma_matrix_a[v0_o, T.int64(0):T.int64(64), T.int64(0):T.int64(16)])
                                    A_1 = T.match_buffer(A[v0_o, T.int64(0):T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), offset_factor=16)
                                    C_1 = T.match_buffer(A_wmma_matrix_a[v0_o, T.int64(0):T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                        T.tvm_load_matrix_sync(C_1.data, 64, 64, 16, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(64), 1), A_1.strides[0], "row_major")
                                with T.sblock("B_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(B[T.int64(0):T.int64(16), T.int64(0):T.int64(64)])
                                    T.writes(B_wmma_matrix_b[T.int64(0):T.int64(16), T.int64(0):T.int64(64)])
                                    A_1 = T.match_buffer(B[T.int64(0):T.int64(16), T.int64(0):T.int64(64)], (T.int64(16), T.int64(64)), "float16", strides=("A_s0", "A_s1"), offset_factor=64)
                                    C_1 = T.match_buffer(B_wmma_matrix_b[T.int64(0):T.int64(16), T.int64(0):T.int64(64)], (T.int64(16), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=64)
                                    for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                        T.tvm_load_matrix_sync(C_1.data, 64, 64, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(64)) + C_1.elem_offset % C_1.strides[0] // T.int64(64), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                                with T.sblock("matmul_update_o"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0)
                                    v1_o = T.axis.spatial(T.int64(1), ax1_0 + ax1_1)
                                    v2_o = T.axis.spatial(T.int64(1), ax2_0 + ax2_1)
                                    v3_o = T.axis.reduce(T.int64(1), ax3_0)
                                    T.reads(C_wmma_accumulator[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(64)], A_wmma_matrix_a[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(16)], B_wmma_matrix_b[T.int64(0):T.int64(16), T.int64(0):T.int64(64)])
                                    T.writes(C_wmma_accumulator[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                                    A_1 = T.match_buffer(A_wmma_matrix_a[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B_1 = T.match_buffer(B_wmma_matrix_b[T.int64(0):T.int64(16), T.int64(0):T.int64(64)], (T.int64(16), T.int64(64)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=64)
                                    C_1 = T.match_buffer(C_wmma_accumulator[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(64)) + C_1.elem_offset % C_1.strides[0] // T.int64(64), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(64) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), B_1.data, B_1.elem_offset // B_1.strides[0] // T.int64(16) * (B_1.strides[0] // T.int64(64)) + B_1.elem_offset % B_1.strides[0] // T.int64(64), C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(64)) + C_1.elem_offset % C_1.strides[0] // T.int64(64))
                            with T.sblock("C_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v2_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads(C_wmma_accumulator[v0_o, T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                                T.writes(C[v0_o, T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                                A_1 = T.match_buffer(C_wmma_accumulator[v0_o, T.int64(0):T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=64)
                                C_1 = T.match_buffer(C[v0_o, T.int64(0):T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), offset_factor=64)
                                T.tvm_store_matrix_sync(A_1.data, 64, 64, 16, A_1.elem_offset // A_1.strides[0] // T.int64(64) * (A_1.strides[0] // T.int64(64)) + A_1.elem_offset % A_1.strides[0] // T.int64(64), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * T.int64(64), 2), C_1.strides[0], "row_major")
    # fmt: on

    mod = IRModule({"main": before})
    vk_target = {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_khr_cooperative_matrix": True,
    }
    with Target(vk_target):
        mod = dl.ApplyDefaultSchedule(dl.adreno.MatmulTensorization()).transform_module(mod, False)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_matmul_no_pad_trans():
    # fmt: off
    @T.prim_func
    def before(A: T.Buffer((T.int64(1), T.int64(16), T.int64(64)), "float16"), B: T.Buffer((T.int64(64), T.int64(16)), "float16"), C: T.Buffer((T.int64(1), T.int64(64), T.int64(64)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.sblock("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(64), T.int64(64), T.int64(16)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_k, v_i1], B[v_i2, v_k])
                T.writes(C[v_i0, v_i1, v_i2])
                with T.init():
                    C[v_i0, v_i1, v_i2] = T.float16(0.0)
                C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + A[v_i0, v_k, v_i1] * B[v_i2, v_k]

    @T.prim_func
    def expected(A: T.Buffer((T.int64(1), T.int64(16), T.int64(64)), "float16"), B: T.Buffer((T.int64(64), T.int64(16)), "float16"), C: T.Buffer((T.int64(1), T.int64(64), T.int64(64)), "float16")):
        T.func_attr({"global_symbol": "before", "tir.is_scheduled": True, "tir.noalias": T.bool(True)})
        # with T.sblock("root"):
        A_wmma_matrix_a = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(64)), "float16", scope="wmma.matrix_a")
        B_wmma_matrix_b = T.alloc_buffer((T.int64(64), T.int64(16)), "float16", scope="wmma.matrix_b")
        C_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(64)), "float16", scope="wmma.accumulator")
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(T.int64(1), thread="blockIdx.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                        for ax2_1 in T.thread_binding(T.int64(1), thread="threadIdx.z"):
                            with T.sblock("matmul_init_o"):
                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                v1_o = T.axis.spatial(T.int64(1), ax1_0 + ax1_1)
                                v2_o = T.axis.spatial(T.int64(1), ax2_0 + ax2_1)
                                T.reads()
                                T.writes(C_wmma_accumulator[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                                C_1 = T.match_buffer(C_wmma_accumulator[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                T.tvm_fill_fragment(C_1.data, 64, 64, 16, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(64)) + C_1.elem_offset % C_1.strides[0] // T.int64(64), T.float32(0.0))
                            for ax3_0 in range(T.int64(1)):
                                with T.sblock("A_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(A[v0_o, T.int64(0):T.int64(16), T.int64(0):T.int64(64)])
                                    T.writes(A_wmma_matrix_a[v0_o, T.int64(0):T.int64(16), T.int64(0):T.int64(64)])
                                    A_1 = T.match_buffer(A[v0_o, T.int64(0):T.int64(16), T.int64(0):T.int64(64)], (T.int64(16), T.int64(64)), "float16", strides=("A_s0", "A_s1"), offset_factor=64)
                                    C_1 = T.match_buffer(A_wmma_matrix_a[v0_o, T.int64(0):T.int64(16), T.int64(0):T.int64(64)], (T.int64(16), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=64)
                                    for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                        T.tvm_load_matrix_sync(C_1.data, 64, 64, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(64)) + C_1.elem_offset % C_1.strides[0] // T.int64(64), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "col_major")
                                with T.sblock("B_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(B[T.int64(0):T.int64(64), T.int64(0):T.int64(16)])
                                    T.writes(B_wmma_matrix_b[T.int64(0):T.int64(64), T.int64(0):T.int64(16)])
                                    A_1 = T.match_buffer(B[T.int64(0):T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), offset_factor=16)
                                    C_1 = T.match_buffer(B_wmma_matrix_b[T.int64(0):T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                        T.tvm_load_matrix_sync(C_1.data, 64, 64, 16, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(64), 1), A_1.strides[0], "col_major")
                                with T.sblock("matmul_update_o"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0)
                                    v1_o = T.axis.spatial(T.int64(1), ax1_0 + ax1_1)
                                    v2_o = T.axis.spatial(T.int64(1), ax2_0 + ax2_1)
                                    v3_o = T.axis.reduce(T.int64(1), ax3_0)
                                    T.reads(C_wmma_accumulator[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(64)], A_wmma_matrix_a[T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(64)], B_wmma_matrix_b[T.int64(0):T.int64(64), T.int64(0):T.int64(16)])
                                    T.writes(C_wmma_accumulator[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                                    A_1 = T.match_buffer(A_wmma_matrix_a[T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(64)], (T.int64(16), T.int64(64)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B_1 = T.match_buffer(B_wmma_matrix_b[T.int64(0):T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_wmma_accumulator[T.int64(0), T.int64(0):T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(64)) + C_1.elem_offset % C_1.strides[0] // T.int64(64), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(64)) + A_1.elem_offset % A_1.strides[0] // T.int64(64), B_1.data, B_1.elem_offset // B_1.strides[0] // T.int64(64) * (B_1.strides[0] // T.int64(16)) + B_1.elem_offset % B_1.strides[0] // T.int64(16), C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(64)) + C_1.elem_offset % C_1.strides[0] // T.int64(64))
                            with T.sblock("C_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v2_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads(C_wmma_accumulator[v0_o, T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                                T.writes(C[v0_o, T.int64(0):T.int64(64), T.int64(0):T.int64(64)])
                                A_1 = T.match_buffer(C_wmma_accumulator[v0_o, T.int64(0):T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=64)
                                C_1 = T.match_buffer(C[v0_o, T.int64(0):T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), offset_factor=64)
                                T.tvm_store_matrix_sync(A_1.data, 64, 64, 16, A_1.elem_offset // A_1.strides[0] // T.int64(64) * (A_1.strides[0] // T.int64(64)) + A_1.elem_offset % A_1.strides[0] // T.int64(64), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * T.int64(64), 2), C_1.strides[0], "row_major")
    # fmt: on

    mod = IRModule({"main": before})
    vk_target = {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_khr_cooperative_matrix": True,
    }
    with Target(vk_target):
        mod = dl.ApplyDefaultSchedule(dl.adreno.MatmulTensorization()).transform_module(mod, False)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_matmul_pad():
    # fmt: off
    @T.prim_func
    def before(A: T.Buffer((T.int64(1), T.int64(111), T.int64(3)), "float16"), B: T.Buffer((T.int64(3), T.int64(111)), "float16"), C: T.Buffer((T.int64(1), T.int64(111), T.int64(111)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.sblock("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(111), T.int64(111), T.int64(3)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_k, v_i2])
                T.writes(C[v_i0, v_i1, v_i2])
                with T.init():
                    C[v_i0, v_i1, v_i2] = T.float16(0.0)
                C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_k, v_i2]

    @T.prim_func
    def expected(A: T.Buffer((T.int64(1), T.int64(111), T.int64(3)), "float16"), B: T.Buffer((T.int64(3), T.int64(111)), "float16"), C: T.Buffer((T.int64(1), T.int64(111), T.int64(111)), "float16")):
        T.func_attr({"global_symbol": "before", "tir.is_scheduled": True, "tir.noalias": T.bool(True)})
        # with T.sblock("root"):
        A_global_pad = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(16)), "float16")
        B_global_pad = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(16)), "float16")
        C_pad_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(128)), "float16", scope="shared")
        A_global_pad_wmma_matrix_a = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(16)), "float16", scope="wmma.matrix_a")
        B_global_pad_wmma_matrix_b = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(16)), "float16", scope="wmma.matrix_b")
        C_pad_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(128)), "float16", scope="wmma.accumulator")
        for i0_i1_i2_0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for i0_i1_i2_0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for i2_1 in T.vectorized(T.int64(4)):
                    with T.sblock("A_global_pad"):
                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v1 = T.axis.spatial(T.int64(128), (i0_i1_i2_0_fused_0 * T.int64(256) + i0_i1_i2_0_fused_1) // T.int64(4))
                        v2 = T.axis.spatial(T.int64(16), (i0_i1_i2_0_fused_0 * T.int64(256) + i0_i1_i2_0_fused_1) % T.int64(4) * T.int64(4) + i2_1)
                        T.reads(A[v0, v1, v2])
                        T.writes(A_global_pad[T.int64(0), v1, v2])
                        A_global_pad[T.int64(0), v1, v2] = T.if_then_else(v1 < T.int64(111) and v2 < T.int64(3), A[v0, v1, v2], T.float16(0.0))
        for i1_i0_0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for i1_i0_0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for i0_1 in T.vectorized(T.int64(4)):
                    with T.sblock("B_global_pad"):
                        v0 = T.axis.spatial(T.int64(16), (i1_i0_0_fused_0 * T.int64(256) + i1_i0_0_fused_1) % T.int64(4) * T.int64(4) + i0_1)
                        v1 = T.axis.spatial(T.int64(128), (i1_i0_0_fused_0 * T.int64(256) + i1_i0_0_fused_1) // T.int64(4))
                        T.reads(B[v0, v1])
                        T.writes(B_global_pad[T.int64(0), v1, v0])
                        B_global_pad[T.int64(0), v1, v0] = T.if_then_else(v0 < T.int64(3) and v1 < T.int64(111), B[v0, v1], T.float16(0.0))
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(T.int64(1), thread="blockIdx.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                        for ax2_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                            with T.sblock("matmul_init_o"):
                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                v1_o = T.axis.spatial(T.int64(2), ax1_0 + ax1_1)
                                v2_o = T.axis.spatial(T.int64(2), ax2_0 * T.int64(2) + ax2_1)
                                T.reads()
                                T.writes(C_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                C_1 = T.match_buffer(C_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                T.tvm_fill_fragment(C_1.data, 64, 64, 16, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(64)) + C_1.elem_offset % C_1.strides[0] // T.int64(64), T.float32(0.0))
                            for ax3_0 in range(T.int64(1)):
                                with T.sblock("A_global_pad_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(2), ax1_0)
                                    v2_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(A_global_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)])
                                    T.writes(A_global_pad_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)])
                                    A_1 = T.match_buffer(A_global_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), offset_factor=16)
                                    C_1 = T.match_buffer(A_global_pad_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                        T.tvm_load_matrix_sync(C_1.data, 64, 64, 16, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(64), 1), A_1.strides[0], "row_major")
                                with T.sblock("B_global_pad_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(2), ax2_1)
                                    v2_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(B_global_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)])
                                    T.writes(B_global_pad_wmma_matrix_b[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)])
                                    A_1 = T.match_buffer(B_global_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), offset_factor=16)
                                    C_1 = T.match_buffer(B_global_pad_wmma_matrix_b[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                        T.tvm_load_matrix_sync(C_1.data, 64, 64, 16, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(64), 1), A_1.strides[0], "col_major")
                                with T.sblock("matmul_update_o"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0)
                                    v1_o = T.axis.spatial(T.int64(2), ax1_0 + ax1_1)
                                    v2_o = T.axis.spatial(T.int64(2), ax2_0 * T.int64(2) + ax2_1)
                                    v3_o = T.axis.reduce(T.int64(1), ax3_0)
                                    T.reads(C_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], A_global_pad_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)], B_global_pad_wmma_matrix_b[T.int64(0), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)])
                                    T.writes(C_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                    A_1 = T.match_buffer(A_global_pad_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B_1 = T.match_buffer(B_global_pad_wmma_matrix_b[T.int64(0), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(64)) + C_1.elem_offset % C_1.strides[0] // T.int64(64), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(64) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), B_1.data, B_1.elem_offset // B_1.strides[0] // T.int64(64) * (B_1.strides[0] // T.int64(16)) + B_1.elem_offset % B_1.strides[0] // T.int64(16), C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(64)) + C_1.elem_offset % C_1.strides[0] // T.int64(64))
                            with T.sblock("C_pad_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o, v2_o = T.axis.remap("SS", [ax1_0, ax2_1])
                                T.reads(C_pad_wmma_accumulator[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                T.writes(C_pad_shared[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                A_1 = T.match_buffer(C_pad_wmma_accumulator[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=64)
                                C_1 = T.match_buffer(C_pad_shared[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="shared", offset_factor=64)
                                T.tvm_store_matrix_sync(A_1.data, 64, 64, 16, A_1.elem_offset // A_1.strides[0] // T.int64(64) * (A_1.strides[0] // T.int64(64)) + A_1.elem_offset % A_1.strides[0] // T.int64(64), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * T.int64(64), 2), C_1.strides[0], "row_major")
                            for ax0_0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                for ax0_1, ax1 in T.grid(T.int64(1), T.int64(64)):
                                    with T.sblock("C_pad"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(111), ax1_0 * T.int64(64) + ax0_0 + ax0_1)
                                        v2 = T.axis.spatial(T.int64(111), ax2_1 * T.int64(64) + ax1)
                                        T.where(ax1_0 * T.int64(64) + (ax0_0 + ax0_1) < T.int64(111) and ax2_1 * T.int64(64) + ax1 < T.int64(111))
                                        T.reads(C_pad_shared[v0, v1, v2])
                                        T.writes(C[v0, v1, v2])
                                        C[v0, v1, v2] = C_pad_shared[v0, v1, v2]
    # fmt: on

    mod = IRModule({"main": before})
    vk_target = {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_khr_cooperative_matrix": True,
    }
    with Target(vk_target):
        mod = dl.ApplyDefaultSchedule(dl.adreno.MatmulTensorization()).transform_module(mod, False)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_matmul_pad_trans():
    # fmt: off
    @T.prim_func
    def before(A: T.Buffer((T.int64(1), T.int64(3), T.int64(111)), "float16"), B: T.Buffer((T.int64(111), T.int64(3)), "float16"), C: T.Buffer((T.int64(1), T.int64(111), T.int64(111)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.sblock("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(111), T.int64(111), T.int64(3)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_k, v_i1], B[v_i2, v_k])
                T.writes(C[v_i0, v_i1, v_i2])
                with T.init():
                    C[v_i0, v_i1, v_i2] = T.float16(0.0)
                C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + A[v_i0, v_k, v_i1] * B[v_i2, v_k]

    @T.prim_func
    def expected(A: T.Buffer((T.int64(1), T.int64(3), T.int64(111)), "float16"), B: T.Buffer((T.int64(111), T.int64(3)), "float16"), C: T.Buffer((T.int64(1), T.int64(111), T.int64(111)), "float16")):
        T.func_attr({"global_symbol": "before", "tir.is_scheduled": True, "tir.noalias": T.bool(True)})
        # with T.sblock("root"):
        A_global_pad = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(16)), "float16")
        B_global_pad = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(16)), "float16")
        C_pad_local = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(128)), "float16", scope="local")
        A_global_pad_wmma_matrix_a = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(16)), "float16", scope="wmma.matrix_a")
        B_global_pad_wmma_matrix_b = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(16)), "float16", scope="wmma.matrix_b")
        C_pad_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(128)), "float16", scope="wmma.accumulator")
        for i0_i2_i1_0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for i0_i2_i1_0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for i1_1 in T.vectorized(T.int64(4)):
                    with T.sblock("A_global_pad"):
                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v1 = T.axis.spatial(T.int64(16), (i0_i2_i1_0_fused_0 * T.int64(256) + i0_i2_i1_0_fused_1) % T.int64(4) * T.int64(4) + i1_1)
                        v2 = T.axis.spatial(T.int64(128), (i0_i2_i1_0_fused_0 * T.int64(256) + i0_i2_i1_0_fused_1) // T.int64(4))
                        T.reads(A[v0, v1, v2])
                        T.writes(A_global_pad[T.int64(0), v2, v1])
                        A_global_pad[T.int64(0), v2, v1] = T.if_then_else(v1 < T.int64(3) and v2 < T.int64(111), A[v0, v1, v2], T.float16(0.0))
        for i0_i1_0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for i0_i1_0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for i1_1 in T.vectorized(T.int64(4)):
                    with T.sblock("B_global_pad"):
                        v0 = T.axis.spatial(T.int64(128), (i0_i1_0_fused_0 * T.int64(256) + i0_i1_0_fused_1) // T.int64(4))
                        v1 = T.axis.spatial(T.int64(16), (i0_i1_0_fused_0 * T.int64(256) + i0_i1_0_fused_1) % T.int64(4) * T.int64(4) + i1_1)
                        T.reads(B[v0, v1])
                        T.writes(B_global_pad[T.int64(0), v0, v1])
                        B_global_pad[T.int64(0), v0, v1] = T.if_then_else(v0 < T.int64(111) and v1 < T.int64(3), B[v0, v1], T.float16(0.0))
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(T.int64(1), thread="blockIdx.y"):
                    for ax1_1 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax2_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                            with T.sblock("matmul_init_o"):
                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                v1_o = T.axis.spatial(T.int64(2), ax1_0 * T.int64(2) + ax1_1)
                                v2_o = T.axis.spatial(T.int64(2), ax2_0 * T.int64(2) + ax2_1)
                                T.reads()
                                T.writes(C_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                C_1 = T.match_buffer(C_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                T.tvm_fill_fragment(C_1.data, 64, 64, 16, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(64)) + C_1.elem_offset % C_1.strides[0] // T.int64(64), T.float32(0.0))
                            for ax3_0 in range(T.int64(1)):
                                with T.sblock("A_global_pad_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(2), ax1_1)
                                    v2_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(A_global_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)])
                                    T.writes(A_global_pad_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)])
                                    A_1 = T.match_buffer(A_global_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), offset_factor=16)
                                    C_1 = T.match_buffer(A_global_pad_wmma_matrix_a[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                        T.tvm_load_matrix_sync(C_1.data, 64, 64, 16, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(64), 1), A_1.strides[0], "row_major")
                                with T.sblock("B_global_pad_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(2), ax2_1)
                                    v2_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(B_global_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)])
                                    T.writes(B_global_pad_wmma_matrix_b[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)])
                                    A_1 = T.match_buffer(B_global_pad[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), offset_factor=16)
                                    C_1 = T.match_buffer(B_global_pad_wmma_matrix_b[v0_o, v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                        T.tvm_load_matrix_sync(C_1.data, 64, 64, 16, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(64), 1), A_1.strides[0], "col_major")
                                with T.sblock("matmul_update_o"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0)
                                    v1_o = T.axis.spatial(T.int64(2), ax1_0 * T.int64(2) + ax1_1)
                                    v2_o = T.axis.spatial(T.int64(2), ax2_0 * T.int64(2) + ax2_1)
                                    v3_o = T.axis.reduce(T.int64(1), ax3_0)
                                    T.reads(C_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], A_global_pad_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)], B_global_pad_wmma_matrix_b[T.int64(0), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)])
                                    T.writes(C_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                    A_1 = T.match_buffer(A_global_pad_wmma_matrix_a[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B_1 = T.match_buffer(B_global_pad_wmma_matrix_b[T.int64(0), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(16)], (T.int64(64), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_pad_wmma_accumulator[T.int64(0), v1_o * T.int64(64):v1_o * T.int64(64) + T.int64(64), v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64), T.int64(64)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(64)) + C_1.elem_offset % C_1.strides[0] // T.int64(64), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(64) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), B_1.data, B_1.elem_offset // B_1.strides[0] // T.int64(64) * (B_1.strides[0] // T.int64(16)) + B_1.elem_offset % B_1.strides[0] // T.int64(16), C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(64) * (C_1.strides[0] // T.int64(64)) + C_1.elem_offset % C_1.strides[0] // T.int64(64))
                            for ax0_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                with T.sblock("C_pad_wmma.accumulator_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(128), ax1_1 * T.int64(64) + ax0_1)
                                    v2_o = T.axis.spatial(T.int64(2), ax2_1)
                                    T.reads(C_pad_wmma_accumulator[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                    T.writes(C_pad_local[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)])
                                    A_1 = T.match_buffer(C_pad_wmma_accumulator[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64),), "float16", scope="wmma.accumulator", offset_factor=64)
                                    C_1 = T.match_buffer(C_pad_local[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64)], (T.int64(64),), "float16", scope="local", offset_factor=64)
                                    T.tvm_deconstruct_coopmat_qcom(A_1.data, 64, 64, 16, C_1.data)
                            for ax0_0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                for ax0_1, ax1 in T.grid(T.int64(1), T.int64(64)):
                                    with T.sblock("C_pad"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(111), ax1_1 * T.int64(64) + ax0_0 + ax0_1)
                                        v2 = T.axis.spatial(T.int64(111), ax2_1 * T.int64(64) + ax1)
                                        T.where(ax1_1 * T.int64(64) + (ax0_0 + ax0_1) < T.int64(111) and ax2_1 * T.int64(64) + ax1 < T.int64(111))
                                        T.reads(C_pad_local[v0, v1, v2])
                                        T.writes(C[v0, v1, v2])
                                        C[v0, v1, v2] = C_pad_local[v0, v1, v2]
    # fmt: on

    mod = IRModule({"main": before})
    # Additionally test for QCOM
    vk_target = {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_khr_cooperative_matrix": True,
        "supports_qcom_cooperative_matrix_conversion": True,
    }
    with Target(vk_target):
        mod = dl.ApplyDefaultSchedule(dl.adreno.MatmulTensorization()).transform_module(mod, False)
    tvm.ir.assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
