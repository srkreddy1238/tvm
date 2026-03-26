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
# ruff: noqa: E501
import tvm
import tvm.testing
from tvm.s_tir import dlight as dl
from tvm.script import tir as T
from tvm.target import Target


def test_conv2d_tensorization_khr():
    # fmt: off
    @T.prim_func(private=True)
    def before(lv: T.Buffer((T.int64(1), T.int64(16), T.int64(223), T.int64(223), T.int64(4)), "float32"), lv1: T.Buffer((T.int64(48), T.int64(16), T.int64(3), T.int64(3), T.int64(4)), "float32"), p_lv2: T.handle, T_add_intermediate: T.Buffer((T.int64(1), T.int64(12), T.int64(112), T.int64(112), T.int64(4)), "float32")):
        lv2 = T.match_buffer(p_lv2, (T.int64(1), T.int64(12), T.int64(1), T.int64(1), T.int64(4)), scope="global.texture-weight")
        # with T.sblock("root"):
        data_pad = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(227), T.int64(227), T.int64(4)))
        input_imed = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(3), T.int64(12544), T.int64(64)))
        weight_imed = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(3), T.int64(64), T.int64(64)))
        conv_matrix = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(12544), T.int64(64)))
        conv_reinterpret_intermediate = T.alloc_buffer((T.int64(1), T.int64(12), T.int64(112), T.int64(112), T.int64(4)))
        for i0, i1, i2, i3, i4 in T.grid(T.int64(1), T.int64(16), T.int64(227), T.int64(227), T.int64(4)):
            with T.sblock("data_pad"):
                v_i0, v_i1, v_i2, v_i3, v_i4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(lv[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2), v_i4])
                T.writes(data_pad[v_i0, v_i1, v_i2, v_i3, v_i4])
                data_pad[v_i0, v_i1, v_i2, v_i3, v_i4] = T.if_then_else(T.int64(2) <= v_i2 and v_i2 < T.int64(225) and T.int64(2) <= v_i3 and v_i3 < T.int64(225), lv[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2), v_i4], T.float32(0.0))
        for nn, gg, kh, kw, pad_oh_ow, kc in T.grid(T.int64(1), T.int64(1), T.int64(3), T.int64(3), T.int64(12544), T.int64(64)):
            with T.sblock("input_imed"):
                v_nn, v_gg, v_kh, v_kw, v_pad_oh_ow, v_kc = T.axis.remap("SSSSSS", [nn, gg, kh, kw, pad_oh_ow, kc])
                T.reads(data_pad[v_nn, v_gg * T.int64(16) + v_kc // T.int64(4), v_pad_oh_ow // T.int64(112) * T.int64(2) + v_kh * T.int64(2), v_kw * T.int64(2) + v_pad_oh_ow % T.int64(112) * T.int64(2), v_kc % T.int64(4)])
                T.writes(input_imed[v_nn, v_gg, v_kh, v_kw, v_pad_oh_ow, v_kc])
                input_imed[v_nn, v_gg, v_kh, v_kw, v_pad_oh_ow, v_kc] = T.if_then_else(v_pad_oh_ow < T.int64(12544), data_pad[v_nn, v_gg * T.int64(16) + v_kc // T.int64(4), v_pad_oh_ow // T.int64(112) * T.int64(2) + v_kh * T.int64(2), v_kw * T.int64(2) + v_pad_oh_ow % T.int64(112) * T.int64(2), v_kc % T.int64(4)], T.float32(0.0))
        for gg, kh, kw, pad_gg_oc, kc in T.grid(T.int64(1), T.int64(3), T.int64(3), T.int64(64), T.int64(64)):
            with T.sblock("weight_imed"):
                v_gg, v_kh, v_kw, v_pad_gg_oc, v_kc = T.axis.remap("SSSSS", [gg, kh, kw, pad_gg_oc, kc])
                T.reads(lv1[v_gg * T.int64(48) + v_pad_gg_oc, v_kc // T.int64(4), v_kh, v_kw, v_kc % T.int64(4)])
                T.writes(weight_imed[v_gg, v_kh, v_kw, v_pad_gg_oc, v_kc])
                weight_imed[v_gg, v_kh, v_kw, v_pad_gg_oc, v_kc] = T.if_then_else(v_pad_gg_oc < T.int64(48), lv1[v_gg * T.int64(48) + v_pad_gg_oc, v_kc // T.int64(4), v_kh, v_kw, v_kc % T.int64(4)], T.float32(0.0))
        for nn, gg, pad_oh_ow, pad_gg_oc, kh, kw, kc in T.grid(T.int64(1), T.int64(1), T.int64(12544), T.int64(64), T.int64(3), T.int64(3), T.int64(64)):
            with T.sblock("conv_matrix"):
                v_nn, v_gg, v_pad_oh_ow, v_pad_gg_oc, v_kh, v_kw, v_kc = T.axis.remap("SSSSRRR", [nn, gg, pad_oh_ow, pad_gg_oc, kh, kw, kc])
                T.reads(input_imed[v_nn, v_gg, v_kh, v_kw, v_pad_oh_ow, v_kc], weight_imed[v_gg, v_kh, v_kw, v_pad_gg_oc, v_kc])
                T.writes(conv_matrix[v_nn, v_gg, v_pad_oh_ow, v_pad_gg_oc])
                with T.init():
                    conv_matrix[v_nn, v_gg, v_pad_oh_ow, v_pad_gg_oc] = T.float32(0.0)
                conv_matrix[v_nn, v_gg, v_pad_oh_ow, v_pad_gg_oc] = conv_matrix[v_nn, v_gg, v_pad_oh_ow, v_pad_gg_oc] + input_imed[v_nn, v_gg, v_kh, v_kw, v_pad_oh_ow, v_kc] * weight_imed[v_gg, v_kh, v_kw, v_pad_gg_oc, v_kc]
        for nn, occ, oh, ow, ocb in T.grid(T.int64(1), T.int64(12), T.int64(112), T.int64(112), T.int64(4)):
            with T.sblock("conv_reinterpret"):
                v_nn, v_occ, v_oh, v_ow, v_ocb = T.axis.remap("SSSSS", [nn, occ, oh, ow, ocb])
                T.reads(conv_matrix[v_nn, (v_occ * T.int64(4) + v_ocb) // T.int64(48), v_oh * T.int64(112) + v_ow, (v_occ * T.int64(4) + v_ocb) % T.int64(48)])
                T.writes(conv_reinterpret_intermediate[v_nn, v_occ, v_oh, v_ow, v_ocb])
                conv_reinterpret_intermediate[v_nn, v_occ, v_oh, v_ow, v_ocb] = conv_matrix[v_nn, (v_occ * T.int64(4) + v_ocb) // T.int64(48), v_oh * T.int64(112) + v_ow, (v_occ * T.int64(4) + v_ocb) % T.int64(48)]
        for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(12), T.int64(112), T.int64(112), T.int64(4)):
            with T.sblock("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                T.reads(conv_reinterpret_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], lv2[v_ax0, v_ax1, T.int64(0), T.int64(0), v_ax4])
                T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = conv_reinterpret_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] + lv2[v_ax0, v_ax1, T.int64(0), T.int64(0), v_ax4]

    @T.prim_func(private=True)
    def expected(lv: T.Buffer((T.int64(1), T.int64(16), T.int64(223), T.int64(223), T.int64(4)), "float32"), lv1: T.Buffer((T.int64(48), T.int64(16), T.int64(3), T.int64(3), T.int64(4)), "float32"), p_lv2: T.handle, T_add_intermediate: T.Buffer((T.int64(1), T.int64(12), T.int64(112), T.int64(112), T.int64(4)), "float32")):
        T.func_attr({"tir.is_scheduled": True})
        lv2 = T.match_buffer(p_lv2, (T.int64(1), T.int64(12), T.int64(1), T.int64(1), T.int64(4)), scope="global.texture-weight")
        # with T.sblock("root"):
        data_pad = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(227), T.int64(227), T.int64(4)))
        input_imed_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(3), T.int64(12544), T.int64(64)), scope="shared")
        weight_imed_shared = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(3), T.int64(64), T.int64(64)), scope="shared")
        conv_matrix = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(12544), T.int64(64)))
        input_imed_shared_wmma_matrix_a = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(3), T.int64(12544), T.int64(64)), scope="wmma.matrix_a")
        weight_imed_shared_wmma_matrix_b = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(3), T.int64(64), T.int64(64)), scope="wmma.matrix_b")
        conv_matrix_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(12544), T.int64(64)), scope="wmma.accumulator")
        conv_matrix_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(12544), T.int64(64)), scope="shared")
        for i0_i1_i2_i3_i4_0_fused_0 in T.thread_binding(T.int64(3221), thread="blockIdx.x"):
            for i0_i1_i2_i3_i4_0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for i4_1 in T.vectorized(T.int64(4)):
                    with T.sblock("data_pad"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(16), (i0_i1_i2_i3_i4_0_fused_0 * T.int64(256) + i0_i1_i2_i3_i4_0_fused_1) // T.int64(51529))
                        v_i2 = T.axis.spatial(T.int64(227), (i0_i1_i2_i3_i4_0_fused_0 * T.int64(256) + i0_i1_i2_i3_i4_0_fused_1) % T.int64(51529) // T.int64(227))
                        v_i3 = T.axis.spatial(T.int64(227), (i0_i1_i2_i3_i4_0_fused_0 * T.int64(256) + i0_i1_i2_i3_i4_0_fused_1) % T.int64(227))
                        v_i4 = T.axis.spatial(T.int64(4), i4_1)
                        T.where(i0_i1_i2_i3_i4_0_fused_0 * T.int64(256) + i0_i1_i2_i3_i4_0_fused_1 < T.int64(824464))
                        T.reads(lv[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2), v_i4])
                        T.writes(data_pad[v_i0, v_i1, v_i2, v_i3, v_i4])
                        data_pad[v_i0, v_i1, v_i2, v_i3, v_i4] = T.if_then_else(T.int64(2) <= v_i2 and v_i2 < T.int64(225) and T.int64(2) <= v_i3 and v_i3 < T.int64(225), lv[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2), v_i4], T.float32(0.0))
        for nn in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for gg in T.thread_binding(T.int64(1), thread="blockIdx.y"):
                for pad_oh_ow_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
                    for pad_gg_oc_0 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                        with T.sblock("conv_matrix_init_o"):
                            v_nn_o, v_gg_o, v_pad_oh_ow_o, v_pad_gg_oc_o = T.axis.remap("SSSS", [nn, gg, pad_oh_ow_0, pad_gg_oc_0])
                            T.reads()
                            T.writes(conv_matrix_wmma_accumulator[v_nn_o, v_gg_o, v_pad_oh_ow_o * T.int64(64):v_pad_oh_ow_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)])
                            C = T.match_buffer(conv_matrix_wmma_accumulator[v_nn_o, v_gg_o, v_pad_oh_ow_o * T.int64(64):v_pad_oh_ow_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                            T.tvm_fill_fragment(C.data, 64, 64, 8, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), T.float32(0.0))
                        for kh, kw, kc_0 in T.grid(T.int64(3), T.int64(3), T.int64(8)):
                            for ax0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                for ax1_0 in range(T.int64(2)):
                                    for ax1_1 in T.vectorized(T.int64(4)):
                                        with T.sblock("input_imed"):
                                            v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                                            v_gg = T.axis.spatial(T.int64(1), T.int64(0))
                                            v_kh, v_kw = T.axis.remap("SS", [kh, kw])
                                            v_pad_oh_ow = T.axis.spatial(T.int64(12544), pad_oh_ow_0 * T.int64(64) + ax0)
                                            v_kc = T.axis.spatial(T.int64(64), kc_0 * T.int64(8) + ax1_0 * T.int64(4) + ax1_1)
                                            T.reads(data_pad[v_nn, v_gg * T.int64(16) + v_kc // T.int64(4), v_pad_oh_ow // T.int64(112) * T.int64(2) + v_kh * T.int64(2), v_kw * T.int64(2) + v_pad_oh_ow % T.int64(112) * T.int64(2), v_kc % T.int64(4)])
                                            T.writes(input_imed_shared[v_nn, v_gg, v_kh, v_kw, v_pad_oh_ow, v_kc])
                                            input_imed_shared[v_nn, v_gg, v_kh, v_kw, v_pad_oh_ow, v_kc] = T.if_then_else(v_pad_oh_ow < T.int64(12544), data_pad[v_nn, v_gg * T.int64(16) + v_kc // T.int64(4), v_pad_oh_ow // T.int64(112) * T.int64(2) + v_kh * T.int64(2), v_kw * T.int64(2) + v_pad_oh_ow % T.int64(112) * T.int64(2), v_kc % T.int64(4)], T.float32(0.0))
                            for ax0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                for ax1_0 in range(T.int64(2)):
                                    for ax1_1 in T.vectorized(T.int64(4)):
                                        with T.sblock("weight_imed"):
                                            v_gg = T.axis.spatial(T.int64(1), T.int64(0))
                                            v_kh, v_kw, v_pad_gg_oc = T.axis.remap("SSS", [kh, kw, ax0])
                                            v_kc = T.axis.spatial(T.int64(64), kc_0 * T.int64(8) + ax1_0 * T.int64(4) + ax1_1)
                                            T.reads(lv1[v_gg * T.int64(48) + v_pad_gg_oc, v_kc // T.int64(4), v_kh, v_kw, v_kc % T.int64(4)])
                                            T.writes(weight_imed_shared[v_gg, v_kh, v_kw, v_pad_gg_oc, v_kc])
                                            weight_imed_shared[v_gg, v_kh, v_kw, v_pad_gg_oc, v_kc] = T.if_then_else(v_pad_gg_oc < T.int64(48), lv1[v_gg * T.int64(48) + v_pad_gg_oc, v_kc // T.int64(4), v_kh, v_kw, v_kc % T.int64(4)], T.float32(0.0))
                            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                                with T.sblock("input_imed_shared_wmma.matrix_a_o"):
                                    v0_o, v1_o = T.axis.remap("SS", [ax0, ax1])
                                    v2_o = T.axis.spatial(T.int64(3), kh + ax2)
                                    v3_o = T.axis.spatial(T.int64(3), kw + ax3)
                                    v4_o, v5_o = T.axis.remap("SS", [pad_oh_ow_0, kc_0])
                                    T.reads(input_imed_shared[v0_o, v1_o, v2_o, v3_o, v4_o * T.int64(64):v4_o * T.int64(64) + T.int64(64), v5_o * T.int64(8):v5_o * T.int64(8) + T.int64(8)])
                                    T.writes(input_imed_shared_wmma_matrix_a[v0_o, v1_o, v2_o, v3_o, v4_o * T.int64(64):v4_o * T.int64(64) + T.int64(64), v5_o * T.int64(8):v5_o * T.int64(8) + T.int64(8)])
                                    A = T.match_buffer(input_imed_shared[v0_o, v1_o, v2_o, v3_o, v4_o * T.int64(64):v4_o * T.int64(64) + T.int64(64), v5_o * T.int64(8):v5_o * T.int64(8) + T.int64(8)], (T.int64(64), T.int64(8)), strides=("A_s0", "A_s1"), scope="shared", offset_factor=8)
                                    C = T.match_buffer(input_imed_shared_wmma_matrix_a[v0_o, v1_o, v2_o, v3_o, v4_o * T.int64(64):v4_o * T.int64(64) + T.int64(64), v5_o * T.int64(8):v5_o * T.int64(8) + T.int64(8)], (T.int64(64), T.int64(8)), strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=8)
                                    for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                        T.tvm_load_matrix_sync(C.data, 64, 64, 8, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(8)) + C.elem_offset % C.strides[0] // T.int64(8), T.tvm_access_ptr(T.type_annotation("float32"), A.data, A.elem_offset, A.strides[0] * T.int64(64), 1), A.strides[0], "row_major")
                            for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                                for ax3_0 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                    with T.sblock("weight_imed_shared_wmma.matrix_b_o"):
                                        v0_o = T.axis.spatial(T.int64(1), ax0)
                                        v1_o = T.axis.spatial(T.int64(3), kh + ax1)
                                        v2_o = T.axis.spatial(T.int64(3), kw + ax2)
                                        v3_o, v4_o = T.axis.remap("SS", [ax3_0, kc_0])
                                        T.reads(weight_imed_shared[v0_o, v1_o, v2_o, T.int64(0):T.int64(64), v4_o * T.int64(8):v4_o * T.int64(8) + T.int64(8)])
                                        T.writes(weight_imed_shared_wmma_matrix_b[v0_o, v1_o, v2_o, T.int64(0):T.int64(64), v4_o * T.int64(8):v4_o * T.int64(8) + T.int64(8)])
                                        A = T.match_buffer(weight_imed_shared[v0_o, v1_o, v2_o, T.int64(0):T.int64(64), v4_o * T.int64(8):v4_o * T.int64(8) + T.int64(8)], (T.int64(64), T.int64(8)), strides=("A_s0", "A_s1"), scope="shared", offset_factor=8)
                                        C = T.match_buffer(weight_imed_shared_wmma_matrix_b[v0_o, v1_o, v2_o, T.int64(0):T.int64(64), v4_o * T.int64(8):v4_o * T.int64(8) + T.int64(8)], (T.int64(64), T.int64(8)), strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=8)
                                        for _ in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                            T.tvm_load_matrix_sync(C.data, 64, 64, 8, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(8)) + C.elem_offset % C.strides[0] // T.int64(8), T.tvm_access_ptr(T.type_annotation("float32"), A.data, A.elem_offset, A.strides[0] * T.int64(64), 1), A.strides[0], "col_major")
                            with T.sblock("conv_matrix_update_o"):
                                v_nn_o, v_gg_o, v_pad_oh_ow_o, v_pad_gg_oc_o, v_kh_o, v_kw_o, v_kc_o = T.axis.remap("SSSSRRR", [nn, gg, pad_oh_ow_0, pad_gg_oc_0, kh, kw, kc_0])
                                T.reads(conv_matrix_wmma_accumulator[v_nn_o, v_gg_o, v_pad_oh_ow_o * T.int64(64):v_pad_oh_ow_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)], input_imed_shared_wmma_matrix_a[v_nn_o, v_gg_o, v_kh_o, v_kw_o, v_pad_oh_ow_o * T.int64(64):v_pad_oh_ow_o * T.int64(64) + T.int64(64), v_kc_o * T.int64(8):v_kc_o * T.int64(8) + T.int64(8)], weight_imed_shared_wmma_matrix_b[v_gg_o, v_kh_o, v_kw_o, T.int64(0):T.int64(64), v_kc_o * T.int64(8):v_kc_o * T.int64(8) + T.int64(8)])
                                T.writes(conv_matrix_wmma_accumulator[v_nn_o, v_gg_o, v_pad_oh_ow_o * T.int64(64):v_pad_oh_ow_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)])
                                A = T.match_buffer(input_imed_shared_wmma_matrix_a[v_nn_o, v_gg_o, v_kh_o, v_kw_o, v_pad_oh_ow_o * T.int64(64):v_pad_oh_ow_o * T.int64(64) + T.int64(64), v_kc_o * T.int64(8):v_kc_o * T.int64(8) + T.int64(8)], (T.int64(64), T.int64(8)), strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=8)
                                B = T.match_buffer(weight_imed_shared_wmma_matrix_b[v_gg_o, v_kh_o, v_kw_o, T.int64(0):T.int64(64), v_kc_o * T.int64(8):v_kc_o * T.int64(8) + T.int64(8)], (T.int64(64), T.int64(8)), strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=8)
                                C = T.match_buffer(conv_matrix_wmma_accumulator[v_nn_o, v_gg_o, v_pad_oh_ow_o * T.int64(64):v_pad_oh_ow_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), A.data, A.elem_offset // A.strides[0] // T.int64(64) * (A.strides[0] // T.int64(8)) + A.elem_offset % A.strides[0] // T.int64(8), B.data, B.elem_offset // B.strides[0] // T.int64(64) * (B.strides[0] // T.int64(8)) + B.elem_offset % B.strides[0] // T.int64(8), C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64))
                        with T.sblock("conv_matrix_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                            v1_o = T.axis.spatial(T.int64(1), T.int64(0))
                            v2_o = T.axis.spatial(T.int64(196), pad_oh_ow_0)
                            v3_o = T.axis.spatial(T.int64(1), T.int64(0))
                            T.reads(conv_matrix_wmma_accumulator[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)])
                            T.writes(conv_matrix_shared[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)])
                            A = T.match_buffer(conv_matrix_wmma_accumulator[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=64)
                            C = T.match_buffer(conv_matrix_shared[v0_o, v1_o, v2_o * T.int64(64):v2_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), strides=("C_s0", "C_s1"), scope="shared", offset_factor=64)
                            T.tvm_store_matrix_sync(A.data, 64, 64, 8, A.elem_offset // A.strides[0] // T.int64(64) * (A.strides[0] // T.int64(64)) + A.elem_offset % A.strides[0] // T.int64(64), T.tvm_access_ptr(T.type_annotation("float32"), C.data, C.elem_offset, C.strides[0] * T.int64(64), 2), C.strides[0], "row_major")
                        for ax0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                            for ax1_0, ax1_1 in T.grid(T.int64(16), T.int64(4)):
                                with T.sblock("conv_matrix_shared"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2 = T.axis.spatial(T.int64(12544), pad_oh_ow_0 * T.int64(64) + ax0)
                                    v3 = T.axis.spatial(T.int64(64), ax1_0 * T.int64(4) + ax1_1)
                                    T.reads(conv_matrix_shared[v0, v1, v2, v3])
                                    T.writes(conv_matrix[v0, v1, v2, v3])
                                    conv_matrix[v0, v1, v2, v3] = conv_matrix_shared[v0, v1, v2, v3]
        for nn_occ_oh_ow_ocb_0_fused_0 in T.thread_binding(T.int64(588), thread="blockIdx.x"):
            for nn_occ_oh_ow_ocb_0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ocb_1 in T.vectorized(T.int64(4)):
                    with T.sblock("conv_reinterpret"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_occ = T.axis.spatial(T.int64(12), (nn_occ_oh_ow_ocb_0_fused_0 * T.int64(256) + nn_occ_oh_ow_ocb_0_fused_1) // T.int64(12544))
                        v_oh = T.axis.spatial(T.int64(112), (nn_occ_oh_ow_ocb_0_fused_0 * T.int64(256) + nn_occ_oh_ow_ocb_0_fused_1) % T.int64(12544) // T.int64(112))
                        v_ow = T.axis.spatial(T.int64(112), (nn_occ_oh_ow_ocb_0_fused_0 * T.int64(256) + nn_occ_oh_ow_ocb_0_fused_1) % T.int64(112))
                        v_ocb = T.axis.spatial(T.int64(4), ocb_1)
                        T.reads(conv_matrix[v_nn, (v_occ * T.int64(4) + v_ocb) // T.int64(48), v_oh * T.int64(112) + v_ow, (v_occ * T.int64(4) + v_ocb) % T.int64(48)], lv2[v_nn, v_occ, T.int64(0), T.int64(0), v_ocb])
                        T.writes(T_add_intermediate[v_nn, v_occ, v_oh, v_ow, v_ocb])
                        T_add_intermediate[v_nn, v_occ, v_oh, v_ow, v_ocb] = conv_matrix[v_nn, (v_occ * T.int64(4) + v_ocb) // T.int64(48), v_oh * T.int64(112) + v_ow, (v_occ * T.int64(4) + v_ocb) % T.int64(48)] + lv2[v_nn, v_occ, T.int64(0), T.int64(0), v_ocb]
    # fmt: on

    mod = tvm.IRModule({"main": before})
    vk_target = {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_khr_cooperative_matrix": True,
    }
    with Target(vk_target):
        mod = dl.ApplyDefaultSchedule(dl.adreno.Conv2DTensorization()).transform_module(mod, False)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_conv2d_tensorization_qcom():
    # fmt: off
    @T.prim_func(private=True)
    def before(lv: T.Buffer((T.int64(1), T.int64(16), T.int64(223), T.int64(223), T.int64(4)), "float32"), lv1: T.Buffer((T.int64(48), T.int64(16), T.int64(3), T.int64(3), T.int64(4)), "float32"), p_lv2: T.handle, T_add_intermediate: T.Buffer((T.int64(1), T.int64(12), T.int64(112), T.int64(112), T.int64(4)), "float32")):
        lv2 = T.match_buffer(p_lv2, (T.int64(1), T.int64(12), T.int64(1), T.int64(1), T.int64(4)), scope="global.texture-weight")
        # with T.sblock("root"):
        data_pad = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(227), T.int64(227), T.int64(4)))
        input_imed = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(3), T.int64(12544), T.int64(64)))
        weight_imed = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(3), T.int64(64), T.int64(64)))
        conv_matrix = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(12544), T.int64(64)))
        conv_reinterpret_intermediate = T.alloc_buffer((T.int64(1), T.int64(12), T.int64(112), T.int64(112), T.int64(4)))
        for i0, i1, i2, i3, i4 in T.grid(T.int64(1), T.int64(16), T.int64(227), T.int64(227), T.int64(4)):
            with T.sblock("data_pad"):
                v_i0, v_i1, v_i2, v_i3, v_i4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(lv[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2), v_i4])
                T.writes(data_pad[v_i0, v_i1, v_i2, v_i3, v_i4])
                data_pad[v_i0, v_i1, v_i2, v_i3, v_i4] = T.if_then_else(T.int64(2) <= v_i2 and v_i2 < T.int64(225) and T.int64(2) <= v_i3 and v_i3 < T.int64(225), lv[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2), v_i4], T.float32(0.0))
        for nn, gg, kh, kw, pad_oh_ow, kc in T.grid(T.int64(1), T.int64(1), T.int64(3), T.int64(3), T.int64(12544), T.int64(64)):
            with T.sblock("input_imed"):
                v_nn, v_gg, v_kh, v_kw, v_pad_oh_ow, v_kc = T.axis.remap("SSSSSS", [nn, gg, kh, kw, pad_oh_ow, kc])
                T.reads(data_pad[v_nn, v_gg * T.int64(16) + v_kc // T.int64(4), v_pad_oh_ow // T.int64(112) * T.int64(2) + v_kh * T.int64(2), v_kw * T.int64(2) + v_pad_oh_ow % T.int64(112) * T.int64(2), v_kc % T.int64(4)])
                T.writes(input_imed[v_nn, v_gg, v_kh, v_kw, v_pad_oh_ow, v_kc])
                input_imed[v_nn, v_gg, v_kh, v_kw, v_pad_oh_ow, v_kc] = T.if_then_else(v_pad_oh_ow < T.int64(12544), data_pad[v_nn, v_gg * T.int64(16) + v_kc // T.int64(4), v_pad_oh_ow // T.int64(112) * T.int64(2) + v_kh * T.int64(2), v_kw * T.int64(2) + v_pad_oh_ow % T.int64(112) * T.int64(2), v_kc % T.int64(4)], T.float32(0.0))
        for gg, kh, kw, pad_gg_oc, kc in T.grid(T.int64(1), T.int64(3), T.int64(3), T.int64(64), T.int64(64)):
            with T.sblock("weight_imed"):
                v_gg, v_kh, v_kw, v_pad_gg_oc, v_kc = T.axis.remap("SSSSS", [gg, kh, kw, pad_gg_oc, kc])
                T.reads(lv1[v_gg * T.int64(48) + v_pad_gg_oc, v_kc // T.int64(4), v_kh, v_kw, v_kc % T.int64(4)])
                T.writes(weight_imed[v_gg, v_kh, v_kw, v_pad_gg_oc, v_kc])
                weight_imed[v_gg, v_kh, v_kw, v_pad_gg_oc, v_kc] = T.if_then_else(v_pad_gg_oc < T.int64(48), lv1[v_gg * T.int64(48) + v_pad_gg_oc, v_kc // T.int64(4), v_kh, v_kw, v_kc % T.int64(4)], T.float32(0.0))
        for nn, gg, pad_oh_ow, pad_gg_oc, kh, kw, kc in T.grid(T.int64(1), T.int64(1), T.int64(12544), T.int64(64), T.int64(3), T.int64(3), T.int64(64)):
            with T.sblock("conv_matrix"):
                v_nn, v_gg, v_pad_oh_ow, v_pad_gg_oc, v_kh, v_kw, v_kc = T.axis.remap("SSSSRRR", [nn, gg, pad_oh_ow, pad_gg_oc, kh, kw, kc])
                T.reads(input_imed[v_nn, v_gg, v_kh, v_kw, v_pad_oh_ow, v_kc], weight_imed[v_gg, v_kh, v_kw, v_pad_gg_oc, v_kc])
                T.writes(conv_matrix[v_nn, v_gg, v_pad_oh_ow, v_pad_gg_oc])
                with T.init():
                    conv_matrix[v_nn, v_gg, v_pad_oh_ow, v_pad_gg_oc] = T.float32(0.0)
                conv_matrix[v_nn, v_gg, v_pad_oh_ow, v_pad_gg_oc] = conv_matrix[v_nn, v_gg, v_pad_oh_ow, v_pad_gg_oc] + input_imed[v_nn, v_gg, v_kh, v_kw, v_pad_oh_ow, v_kc] * weight_imed[v_gg, v_kh, v_kw, v_pad_gg_oc, v_kc]
        for nn, occ, oh, ow, ocb in T.grid(T.int64(1), T.int64(12), T.int64(112), T.int64(112), T.int64(4)):
            with T.sblock("conv_reinterpret"):
                v_nn, v_occ, v_oh, v_ow, v_ocb = T.axis.remap("SSSSS", [nn, occ, oh, ow, ocb])
                T.reads(conv_matrix[v_nn, (v_occ * T.int64(4) + v_ocb) // T.int64(48), v_oh * T.int64(112) + v_ow, (v_occ * T.int64(4) + v_ocb) % T.int64(48)])
                T.writes(conv_reinterpret_intermediate[v_nn, v_occ, v_oh, v_ow, v_ocb])
                conv_reinterpret_intermediate[v_nn, v_occ, v_oh, v_ow, v_ocb] = conv_matrix[v_nn, (v_occ * T.int64(4) + v_ocb) // T.int64(48), v_oh * T.int64(112) + v_ow, (v_occ * T.int64(4) + v_ocb) % T.int64(48)]
        for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(12), T.int64(112), T.int64(112), T.int64(4)):
            with T.sblock("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                T.reads(conv_reinterpret_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], lv2[v_ax0, v_ax1, T.int64(0), T.int64(0), v_ax4])
                T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = conv_reinterpret_intermediate[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] + lv2[v_ax0, v_ax1, T.int64(0), T.int64(0), v_ax4]

    @T.prim_func(private=True)
    def expected(lv: T.Buffer((T.int64(1), T.int64(16), T.int64(223), T.int64(223), T.int64(4)), "float32"), lv1: T.Buffer((T.int64(48), T.int64(16), T.int64(3), T.int64(3), T.int64(4)), "float32"), p_lv2: T.handle, T_add_intermediate: T.Buffer((T.int64(1), T.int64(12), T.int64(112), T.int64(112), T.int64(4)), "float32")):
        T.func_attr({"tir.is_scheduled": True})
        lv2 = T.match_buffer(p_lv2, (T.int64(1), T.int64(12), T.int64(1), T.int64(1), T.int64(4)), scope="global.texture-weight")
        # with T.sblock("root"):
        data_pad = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(227), T.int64(227), T.int64(4)))
        input_imed_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(3), T.int64(12544), T.int64(64)), scope="local")
        weight_imed_local = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(3), T.int64(64), T.int64(64)), scope="local")
        conv_matrix = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(12544), T.int64(64)))
        input_imed_local_wmma_matrix_a = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(3), T.int64(12544), T.int64(64)), scope="wmma.matrix_a")
        weight_imed_local_wmma_matrix_b = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(3), T.int64(64), T.int64(64)), scope="wmma.matrix_b")
        conv_matrix_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(12544), T.int64(64)), scope="wmma.accumulator")
        conv_matrix_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(12544), T.int64(64)), scope="local")
        for i0_i1_i2_i3_i4_0_fused_0 in T.thread_binding(T.int64(3221), thread="blockIdx.x"):
            for i0_i1_i2_i3_i4_0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for i4_1 in T.vectorized(T.int64(4)):
                    with T.sblock("data_pad"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(16), (i0_i1_i2_i3_i4_0_fused_0 * T.int64(256) + i0_i1_i2_i3_i4_0_fused_1) // T.int64(51529))
                        v_i2 = T.axis.spatial(T.int64(227), (i0_i1_i2_i3_i4_0_fused_0 * T.int64(256) + i0_i1_i2_i3_i4_0_fused_1) % T.int64(51529) // T.int64(227))
                        v_i3 = T.axis.spatial(T.int64(227), (i0_i1_i2_i3_i4_0_fused_0 * T.int64(256) + i0_i1_i2_i3_i4_0_fused_1) % T.int64(227))
                        v_i4 = T.axis.spatial(T.int64(4), i4_1)
                        T.where(i0_i1_i2_i3_i4_0_fused_0 * T.int64(256) + i0_i1_i2_i3_i4_0_fused_1 < T.int64(824464))
                        T.reads(lv[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2), v_i4])
                        T.writes(data_pad[v_i0, v_i1, v_i2, v_i3, v_i4])
                        data_pad[v_i0, v_i1, v_i2, v_i3, v_i4] = T.if_then_else(T.int64(2) <= v_i2 and v_i2 < T.int64(225) and T.int64(2) <= v_i3 and v_i3 < T.int64(225), lv[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2), v_i4], T.float32(0.0))
        for nn in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for gg in T.thread_binding(T.int64(1), thread="blockIdx.y"):
                for pad_oh_ow_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
                    for pad_gg_oc_0 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                        with T.sblock("conv_matrix_init_o"):
                            v_nn_o, v_gg_o, v_pad_oh_ow_o, v_pad_gg_oc_o = T.axis.remap("SSSS", [nn, gg, pad_oh_ow_0, pad_gg_oc_0])
                            T.reads()
                            T.writes(conv_matrix_wmma_accumulator[v_nn_o, v_gg_o, v_pad_oh_ow_o * T.int64(64):v_pad_oh_ow_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)])
                            C = T.match_buffer(conv_matrix_wmma_accumulator[v_nn_o, v_gg_o, v_pad_oh_ow_o * T.int64(64):v_pad_oh_ow_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                            T.tvm_fill_fragment(C.data, 64, 64, 8, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), T.float32(0.0))
                        for kh, kw, kc_0 in T.grid(T.int64(3), T.int64(3), T.int64(8)):
                            for ax0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                for ax1_0 in range(T.int64(2)):
                                    for ax1_1 in T.vectorized(T.int64(4)):
                                        with T.sblock("input_imed"):
                                            v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                                            v_gg = T.axis.spatial(T.int64(1), T.int64(0))
                                            v_kh, v_kw = T.axis.remap("SS", [kh, kw])
                                            v_pad_oh_ow = T.axis.spatial(T.int64(12544), pad_oh_ow_0 * T.int64(64) + ax0)
                                            v_kc = T.axis.spatial(T.int64(64), kc_0 * T.int64(8) + ax1_0 * T.int64(4) + ax1_1)
                                            T.reads(data_pad[v_nn, v_gg * T.int64(16) + v_kc // T.int64(4), v_pad_oh_ow // T.int64(112) * T.int64(2) + v_kh * T.int64(2), v_kw * T.int64(2) + v_pad_oh_ow % T.int64(112) * T.int64(2), v_kc % T.int64(4)])
                                            T.writes(input_imed_local[v_nn, v_gg, v_kh, v_kw, v_pad_oh_ow, v_kc])
                                            input_imed_local[v_nn, v_gg, v_kh, v_kw, v_pad_oh_ow, v_kc] = T.if_then_else(v_pad_oh_ow < T.int64(12544), data_pad[v_nn, v_gg * T.int64(16) + v_kc // T.int64(4), v_pad_oh_ow // T.int64(112) * T.int64(2) + v_kh * T.int64(2), v_kw * T.int64(2) + v_pad_oh_ow % T.int64(112) * T.int64(2), v_kc % T.int64(4)], T.float32(0.0))
                            for ax0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                for ax1_0 in range(T.int64(2)):
                                    for ax1_1 in T.vectorized(T.int64(4)):
                                        with T.sblock("weight_imed"):
                                            v_gg = T.axis.spatial(T.int64(1), T.int64(0))
                                            v_kh, v_kw, v_pad_gg_oc = T.axis.remap("SSS", [kh, kw, ax0])
                                            v_kc = T.axis.spatial(T.int64(64), kc_0 * T.int64(8) + ax1_0 * T.int64(4) + ax1_1)
                                            T.reads(lv1[v_gg * T.int64(48) + v_pad_gg_oc, v_kc // T.int64(4), v_kh, v_kw, v_kc % T.int64(4)])
                                            T.writes(weight_imed_local[v_gg, v_kh, v_kw, v_pad_gg_oc, v_kc])
                                            weight_imed_local[v_gg, v_kh, v_kw, v_pad_gg_oc, v_kc] = T.if_then_else(v_pad_gg_oc < T.int64(48), lv1[v_gg * T.int64(48) + v_pad_gg_oc, v_kc // T.int64(4), v_kh, v_kw, v_kc % T.int64(4)], T.float32(0.0))
                            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                                for ax4 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    with T.sblock("input_imed_local_wmma.matrix_a_o"):
                                        v0_o, v1_o = T.axis.remap("SS", [ax0, ax1])
                                        v2_o = T.axis.spatial(T.int64(3), kh + ax2)
                                        v3_o = T.axis.spatial(T.int64(3), kw + ax3)
                                        v4_o = T.axis.spatial(T.int64(12544), pad_oh_ow_0 * T.int64(64) + ax4)
                                        v5_o = T.axis.spatial(T.int64(8), kc_0)
                                        T.reads(input_imed_local[v0_o, v1_o, v2_o, v3_o, v4_o, v5_o * T.int64(8):v5_o * T.int64(8) + T.int64(8)])
                                        T.writes(input_imed_local_wmma_matrix_a[v0_o, v1_o, v2_o, v3_o, v4_o, v5_o * T.int64(8):v5_o * T.int64(8) + T.int64(8)])
                                        A = T.match_buffer(input_imed_local[v0_o, v1_o, v2_o, v3_o, v4_o, v5_o * T.int64(8):v5_o * T.int64(8) + T.int64(8)], (T.int64(8),), scope="local", offset_factor=8)
                                        C = T.match_buffer(input_imed_local_wmma_matrix_a[v0_o, v1_o, v2_o, v3_o, v4_o, v5_o * T.int64(8):v5_o * T.int64(8) + T.int64(8)], (T.int64(8),), scope="wmma.matrix_a", offset_factor=8)
                                        T.tvm_construct_coopmat_qcom(C.data, 64, 64, 8, A.data)
                            for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                                for ax3 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                                    with T.sblock("weight_imed_local_wmma.matrix_b_o"):
                                        v0_o = T.axis.spatial(T.int64(1), ax0)
                                        v1_o = T.axis.spatial(T.int64(3), kh + ax1)
                                        v2_o = T.axis.spatial(T.int64(3), kw + ax2)
                                        v3_o, v4_o = T.axis.remap("SS", [ax3, kc_0])
                                        T.reads(weight_imed_local[v0_o, v1_o, v2_o, v3_o, v4_o * T.int64(8):v4_o * T.int64(8) + T.int64(8)])
                                        T.writes(weight_imed_local_wmma_matrix_b[v0_o, v1_o, v2_o, v3_o, v4_o * T.int64(8):v4_o * T.int64(8) + T.int64(8)])
                                        A = T.match_buffer(weight_imed_local[v0_o, v1_o, v2_o, v3_o, v4_o * T.int64(8):v4_o * T.int64(8) + T.int64(8)], (T.int64(8),), scope="local", offset_factor=8)
                                        C = T.match_buffer(weight_imed_local_wmma_matrix_b[v0_o, v1_o, v2_o, v3_o, v4_o * T.int64(8):v4_o * T.int64(8) + T.int64(8)], (T.int64(8),), scope="wmma.matrix_b", offset_factor=8)
                                        T.tvm_construct_coopmat_qcom(C.data, 64, 64, 8, A.data)
                            with T.sblock("conv_matrix_update_o"):
                                v_nn_o, v_gg_o, v_pad_oh_ow_o, v_pad_gg_oc_o, v_kh_o, v_kw_o, v_kc_o = T.axis.remap("SSSSRRR", [nn, gg, pad_oh_ow_0, pad_gg_oc_0, kh, kw, kc_0])
                                T.reads(conv_matrix_wmma_accumulator[v_nn_o, v_gg_o, v_pad_oh_ow_o * T.int64(64):v_pad_oh_ow_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)], input_imed_local_wmma_matrix_a[v_nn_o, v_gg_o, v_kh_o, v_kw_o, v_pad_oh_ow_o * T.int64(64):v_pad_oh_ow_o * T.int64(64) + T.int64(64), v_kc_o * T.int64(8):v_kc_o * T.int64(8) + T.int64(8)], weight_imed_local_wmma_matrix_b[v_gg_o, v_kh_o, v_kw_o, T.int64(0):T.int64(64), v_kc_o * T.int64(8):v_kc_o * T.int64(8) + T.int64(8)])
                                T.writes(conv_matrix_wmma_accumulator[v_nn_o, v_gg_o, v_pad_oh_ow_o * T.int64(64):v_pad_oh_ow_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)])
                                A = T.match_buffer(input_imed_local_wmma_matrix_a[v_nn_o, v_gg_o, v_kh_o, v_kw_o, v_pad_oh_ow_o * T.int64(64):v_pad_oh_ow_o * T.int64(64) + T.int64(64), v_kc_o * T.int64(8):v_kc_o * T.int64(8) + T.int64(8)], (T.int64(64), T.int64(8)), strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=8)
                                B = T.match_buffer(weight_imed_local_wmma_matrix_b[v_gg_o, v_kh_o, v_kw_o, T.int64(0):T.int64(64), v_kc_o * T.int64(8):v_kc_o * T.int64(8) + T.int64(8)], (T.int64(64), T.int64(8)), strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=8)
                                C = T.match_buffer(conv_matrix_wmma_accumulator[v_nn_o, v_gg_o, v_pad_oh_ow_o * T.int64(64):v_pad_oh_ow_o * T.int64(64) + T.int64(64), T.int64(0):T.int64(64)], (T.int64(64), T.int64(64)), strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=64)
                                T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64), A.data, A.elem_offset // A.strides[0] // T.int64(64) * (A.strides[0] // T.int64(8)) + A.elem_offset % A.strides[0] // T.int64(8), B.data, B.elem_offset // B.strides[0] // T.int64(64) * (B.strides[0] // T.int64(8)) + B.elem_offset % B.strides[0] // T.int64(8), C.data, C.elem_offset // C.strides[0] // T.int64(64) * (C.strides[0] // T.int64(64)) + C.elem_offset % C.strides[0] // T.int64(64))
                        for ax0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                            with T.sblock("conv_matrix_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v2_o = T.axis.spatial(T.int64(12544), pad_oh_ow_0 * T.int64(64) + ax0)
                                v3_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads(conv_matrix_wmma_accumulator[v0_o, v1_o, v2_o, T.int64(0):T.int64(64)])
                                T.writes(conv_matrix_local[v0_o, v1_o, v2_o, T.int64(0):T.int64(64)])
                                A = T.match_buffer(conv_matrix_wmma_accumulator[v0_o, v1_o, v2_o, T.int64(0):T.int64(64)], (T.int64(64),), scope="wmma.accumulator", offset_factor=64)
                                C = T.match_buffer(conv_matrix_local[v0_o, v1_o, v2_o, T.int64(0):T.int64(64)], (T.int64(64),), scope="local", offset_factor=64)
                                T.tvm_deconstruct_coopmat_qcom(A.data, 64, 64, 8, C.data)
                        for ax0 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                            for ax1_0, ax1_1 in T.grid(T.int64(16), T.int64(4)):
                                with T.sblock("conv_matrix_local"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2 = T.axis.spatial(T.int64(12544), pad_oh_ow_0 * T.int64(64) + ax0)
                                    v3 = T.axis.spatial(T.int64(64), ax1_0 * T.int64(4) + ax1_1)
                                    T.reads(conv_matrix_local[v0, v1, v2, v3])
                                    T.writes(conv_matrix[v0, v1, v2, v3])
                                    conv_matrix[v0, v1, v2, v3] = conv_matrix_local[v0, v1, v2, v3]
        for nn_occ_oh_ow_ocb_0_fused_0 in T.thread_binding(T.int64(588), thread="blockIdx.x"):
            for nn_occ_oh_ow_ocb_0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ocb_1 in T.vectorized(T.int64(4)):
                    with T.sblock("conv_reinterpret"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_occ = T.axis.spatial(T.int64(12), (nn_occ_oh_ow_ocb_0_fused_0 * T.int64(256) + nn_occ_oh_ow_ocb_0_fused_1) // T.int64(12544))
                        v_oh = T.axis.spatial(T.int64(112), (nn_occ_oh_ow_ocb_0_fused_0 * T.int64(256) + nn_occ_oh_ow_ocb_0_fused_1) % T.int64(12544) // T.int64(112))
                        v_ow = T.axis.spatial(T.int64(112), (nn_occ_oh_ow_ocb_0_fused_0 * T.int64(256) + nn_occ_oh_ow_ocb_0_fused_1) % T.int64(112))
                        v_ocb = T.axis.spatial(T.int64(4), ocb_1)
                        T.reads(conv_matrix[v_nn, (v_occ * T.int64(4) + v_ocb) // T.int64(48), v_oh * T.int64(112) + v_ow, (v_occ * T.int64(4) + v_ocb) % T.int64(48)], lv2[v_nn, v_occ, T.int64(0), T.int64(0), v_ocb])
                        T.writes(T_add_intermediate[v_nn, v_occ, v_oh, v_ow, v_ocb])
                        T_add_intermediate[v_nn, v_occ, v_oh, v_ow, v_ocb] = conv_matrix[v_nn, (v_occ * T.int64(4) + v_ocb) // T.int64(48), v_oh * T.int64(112) + v_ow, (v_occ * T.int64(4) + v_ocb) % T.int64(48)] + lv2[v_nn, v_occ, T.int64(0), T.int64(0), v_ocb]
    # fmt: on

    mod = tvm.IRModule({"main": before})
    vk_target = {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_khr_cooperative_matrix": True,
        "supports_qcom_cooperative_matrix_conversion": True,
    }
    with Target(vk_target):
        mod = dl.ApplyDefaultSchedule(dl.adreno.Conv2DTensorization()).transform_module(mod, False)
    tvm.ir.assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
