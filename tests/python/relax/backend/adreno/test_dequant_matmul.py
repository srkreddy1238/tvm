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
import os

import pytest
from utils import requires_adreno_vulkan, verify_results

import tvm
from tvm import relax, tir
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

vk_target = tvm.target.Target(
    {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_float16": 1,
        "supports_16bit_buffer": 1,
        "supports_int64": 1,
        "supports_int8": 1,
        "supports_8bit_buffer": 1,
        "supports_storage_buffer_storage_class": 1,
        "supports_khr_cooperative_matrix": 1,
        "max_shared_memory_per_block": 32768,
    }
)
vk_target_qcom = tvm.target.Target(
    {
        "kind": "vulkan",
        "keys": ["adreno", "gpu"],
        "supports_float16": 1,
        "supports_16bit_buffer": 1,
        "supports_int64": 1,
        "supports_int8": 1,
        "supports_8bit_buffer": 1,
        "supports_storage_buffer_storage_class": 1,
        "supports_khr_cooperative_matrix": 1,
        "supports_qcom_cooperative_matrix_conversion": 1,
    }
)

TARGET_SUPPORTS_EXTENSION = os.getenv("ADRENO_TARGET_COOP", "").strip().lower() == "yes"


def check_codegen_pipeline(mod, target):
    relax_pipeline = relax.get_default_pipeline(target)
    tir_pipeline = tir.get_default_tir_pipeline(target)
    ex = tvm.compile(mod, target, tir_pipeline=tir_pipeline, relax_pipeline=relax_pipeline)
    source_str = ex.mod.imports[0].imports[0].inspect_source()
    assert "OpCooperativeMatrixMulAddKHR" in source_str


@requires_adreno_vulkan
@pytest.mark.skipif(not TARGET_SUPPORTS_EXTENSION, reason="Device not supported.")
@pytest.mark.parametrize("use_qcom", [False, True])
@pytest.mark.parametrize("seq_len", [128, 512])
def test_dequant_matmul(seq_len: int, use_qcom: bool):
    # fmt: off
    @I.ir_module
    class DequantMatmul:
        @T.prim_func
        def dequant_matmul(quant: T.Buffer((T.int64(512), T.int64(12288)), "uint32"), scale: T.Buffer((T.int64(128), T.int64(12288)), "float16"), p_rms_norm130: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12288),), "float16"), p_output0: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            seq_len = T.int64()
            rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(4096)), "float16")
            compute = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
            dequantize_intermediate_intermediate = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
            matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(12288)), "float16")
            T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "float16")
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
        @R.function
        def main(quant: R.Tensor((T.int64(512), T.int64(12288)), "uint32"), scale: R.Tensor((T.int64(128), T.int64(12288)), "float16"), rms_norm130: R.Tensor((T.int64(1), T.int64(seq_len), T.int64(4096)), "float16"), transformer_h_0_attn_c_attn_bias3: R.Tensor((T.int64(12288),), "float16")):
            cls = DequantMatmul
            with R.dataflow():
                out = R.call_tir(cls.dequant_matmul, (quant, scale, rms_norm130, transformer_h_0_attn_c_attn_bias3), R.Tensor((T.int64(1), T.int64(seq_len), T.int64(12288)), "float16"))
                R.output(out)
            return out

    # fmt: on

    if use_qcom:
        target = vk_target_qcom
    else:
        target = vk_target

    check_codegen_pipeline(DequantMatmul, target)
    verify_results(DequantMatmul, target, "opencl", atol=0.0, rtol=1e-2)


@requires_adreno_vulkan
@pytest.mark.skipif(not TARGET_SUPPORTS_EXTENSION, reason="Device not supported.")
@pytest.mark.parametrize("use_qcom", [False, True])
@pytest.mark.parametrize("seq_len", [128, 512])
def test_dequant_matmul_trans(seq_len: int, use_qcom: bool):
    # fmt: off
    @I.ir_module
    class DequantMatmulTrans:
        @T.prim_func
        def dequant_matmul_trans(quant: T.Buffer((T.int64(12288), T.int64(512)), "uint32"), scale: T.Buffer((T.int64(12288), T.int64(128)), "float16"), p_rms_norm130: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12288),), "float16"), p_output0: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            seq_len = T.int64()
            rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(4096)), "float16")
            T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "float16")
            # with T.sblock("root"):
            compute = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16")
            dequantize_intermediate_intermediate = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16")
            matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(12288)), "float16")
            for i0, i1 in T.grid(T.int64(12288), T.int64(4096)):
                with T.sblock("compute"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(quant[v_i0, v_i1 // T.int64(8)])
                    T.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(quant[v_i0, v_i1 // T.int64(8)], T.Cast("uint32", v_i1 % T.int64(8) * T.int64(4))), T.uint32(15)))
            for i0, i1 in T.grid(T.int64(12288), T.int64(4096) ):
                with T.sblock("dequantize"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(compute[v_i0, v_i1], scale[v_i0, v_i1 // T.int64(32)])
                    T.writes(dequantize_intermediate_intermediate[v_i0, v_i1])
                    dequantize_intermediate_intermediate[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * scale[v_i0, v_i1 // T.int64(32)]
            for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(12288), T.int64(4096)):
                with T.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(rms_norm130[v_i0, v_i1, v_k], dequantize_intermediate_intermediate[v_i2, v_k])
                    T.writes(matmul_intermediate[v_i0, v_i1, v_i2])
                    with T.init():
                        matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                    matmul_intermediate[v_i0, v_i1, v_i2] = matmul_intermediate[v_i0, v_i1, v_i2] + rms_norm130[v_i0, v_i1, v_k] * dequantize_intermediate_intermediate[v_i2, v_k]
            for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(12288)):
                with T.sblock("T_add"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(matmul_intermediate[v_ax0, v_ax1, v_ax2], transformer_h_0_attn_c_attn_bias3[v_ax2])
                    T.writes(T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2])
                    T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2] = matmul_intermediate[v_ax0, v_ax1, v_ax2] + transformer_h_0_attn_c_attn_bias3[v_ax2]
        @R.function
        def main(quant: R.Tensor((T.int64(12288), T.int64(512)), "uint32"), scale: R.Tensor((T.int64(12288), T.int64(128)), "float16"), rms_norm130: R.Tensor((T.int64(1), T.int64(seq_len), T.int64(4096)), "float16"), transformer_h_0_attn_c_attn_bias3: R.Tensor((T.int64(12288),), "float16")):
            cls = DequantMatmulTrans
            with R.dataflow():
                out = R.call_tir(cls.dequant_matmul_trans, (quant, scale, rms_norm130, transformer_h_0_attn_c_attn_bias3), R.Tensor((T.int64(1), T.int64(seq_len), T.int64(12288)), "float16"))
                R.output(out)
            return out
    # fmt: on

    if use_qcom:
        target = vk_target_qcom
    else:
        target = vk_target

    check_codegen_pipeline(DequantMatmulTrans, target)
    verify_results(DequantMatmulTrans, target, "opencl", atol=0.0, rtol=1e-2)


@requires_adreno_vulkan
@pytest.mark.skipif(not TARGET_SUPPORTS_EXTENSION, reason="Device not supported.")
@pytest.mark.parametrize("use_qcom", [False, True])
@pytest.mark.parametrize("seq_len", [128, 512])
def test_dequant_matmul_int8(seq_len: int, use_qcom: bool):
    # fmt: off
    @I.ir_module
    class DequantMatmulInt8:
        @T.prim_func
        def dequant_matmul(quant: T.Buffer((T.int64(512), T.int64(12288)), "uint32"), scale: T.Buffer((1, T.int64(12288)), "float16"), p_rms_norm130: T.handle, in_scale_var: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12288),), "float16"), p_output0: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            seq_len = T.int64()
            rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(4096)), "int8")
            in_scale = T.match_buffer(in_scale_var, (T.int64(1), seq_len), "float16")
            T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "int32")
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
                    T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2] = T.Cast("float16", matmul_intermediate[v_ax0, v_ax1, v_ax2]) * in_scale[v_ax0, v_ax1] * scale[v_ax0, v_ax2] + transformer_h_0_attn_c_attn_bias3[v_ax2]

        @R.function
        def main(input: R.Tensor((1, T.int64(seq_len), 4096), dtype="int8"), in_scale: R.Tensor((1, T.int64(seq_len)), dtype="float16"), weight: R.Tensor((512, 12288), dtype="uint32"), scale: R.Tensor((1, 12288), dtype="float16"), bias: R.Tensor((12288,), dtype="float16"),):
            cls = DequantMatmulInt8
            with R.dataflow():
                gv: R.Tensor((1, T.int64(seq_len), 12288), dtype="int32") = relax.call_tir(
                    cls.dequant_matmul,
                    (weight, scale, input, in_scale, bias),
                    out_sinfo=R.Tensor((1, T.int64(seq_len), 12288), dtype="int32"),
                )
                R.output(gv)
            return gv
    # fmt: on

    if use_qcom:
        target = vk_target_qcom
    else:
        target = vk_target

    check_codegen_pipeline(DequantMatmulInt8, target)
    verify_results(DequantMatmulInt8, target, "opencl")


if __name__ == "__main__":
    tvm.testing.main()
