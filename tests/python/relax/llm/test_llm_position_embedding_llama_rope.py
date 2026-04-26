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
"""Tests for llama_rope (the Relax-level fused-rope op).

Each test calls ``llama_rope`` inside a tiny ``Module``, exports it via
``export_tvm``, and compares the resulting IRModule against a hand-written
expected IRModule in TVMScript – exactly the same pattern used throughout
``tests/python/relax/test_frontend_nn_op.py``.

NOTE: llama_rope's internal ``_rope`` helper accesses ``rope_scaling["rope_type"]``
without a key guard, so an empty dict ``{}`` cannot be used with ``llama_rope``.
All tests therefore use configs that include a ``"rope_type"`` key.

Config used across all tests
-----------------------------
* batch_size = 1, seq_len = 4
* num_q_heads = 2, num_kv_heads = 2  (MHA)
* head_dim = 8, dtype = "float16"
* theta = 10000.0, scale = 1.0
"""

import tvm
import tvm.testing
from tvm import tir
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax.frontend.nn import Module, Tensor, spec
from tvm.relax.frontend.nn.llm.position_embedding import llama_rope

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
BATCH = 1
SEQ = 4
NUM_Q = 2
NUM_KV = 2
HEAD_DIM = 8
FUSED = NUM_Q + NUM_KV * 2  # 6
DTYPE = "float16"
THETA = 10000.0
SCALE = 1.0

# llama_rope requires rope_scaling to contain "rope_type".
ROPE_SCALING_GPTJ = {"rope_type": "gptj"}
ROPE_SCALING_LLAMA3 = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


# ---------------------------------------------------------------------------
# Helper: build the IRModule for a given rope_scaling / rotary_dim
# ---------------------------------------------------------------------------

def _build(rope_scaling, num_q=NUM_Q, num_kv=NUM_KV, rotary_dim=None):
    fused = num_q + num_kv * 2

    class M(Module):
        def forward(self, qkv: Tensor, total_seq_len: tir.Var):
            q, k, v = llama_rope(
                qkv,
                total_seq_len,
                theta=THETA,
                scale=SCALE,
                num_q_heads=num_q,
                num_kv_heads=num_kv,
                rope_scaling=rope_scaling,
                rotary_dim=rotary_dim,
            )
            return q, k, v

    mod, _ = M().export_tvm(
        spec={
            "forward": {
                "qkv": spec.Tensor([BATCH, SEQ, fused, HEAD_DIM], DTYPE),
                "total_seq_len": int,
            }
        },
        debug=True,
    )
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_llama_rope_llama3_scaling_idempotent():
    """rope_scaling=llama3 – structural equality with expected IRModule."""
    mod = _build(ROPE_SCALING_LLAMA3)

    # fmt: off
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def llama_rope(var_qkv: T.handle, var_q: T.handle, var_k: T.handle, var_v: T.handle, total_seq_len: T.int64):
            T.func_attr({"op_pattern": 8, "tir.noalias": True})
            batch_size, seq_len = T.int64(), T.int64()
            qkv = T.match_buffer(var_qkv, (batch_size, seq_len, 6, 8), "float16")
            q = T.match_buffer(var_q, (batch_size, seq_len, 2, 8), "float16")
            k = T.match_buffer(var_k, (batch_size, seq_len, 2, 8), "float16")
            v = T.match_buffer(var_v, (batch_size, seq_len, 2, 8), "float16")
            for iters_0, iters_1, iters_2, iters_3 in T.grid(batch_size, seq_len, 6, 8):
                with T.sblock("llama_fused_rope"):
                    b, s, h, d = T.axis.remap("SSSS", [iters_0, iters_1, iters_2, iters_3])
                    T.reads(qkv[b, s, h, d - 4:d - 4 + 9])
                    T.writes(q[b, s, h, d], k[b, s, h - 2, d], v[b, s, h - 4, d])
                    if h < 2:
                        orig_freq = T.float32()
                        smoothed_freq = T.float32()
                        q[b, s, h, d] = T.if_then_else(d < 8, T.Let(T.Let(T.Cast("float16", T.cos(smoothed_freq)) * qkv[b, s, h, d] + T.Cast("float16", T.sin(smoothed_freq)) * T.if_then_else(d < 4, qkv[b, s, h, d + 4] * T.float16(-1.0), qkv[b, s, h, d - 4]), where={smoothed_freq: T.Cast("float32", T.Cast("float16", s + (total_seq_len - seq_len))) * ((T.float32(1.0) - T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331)))) * orig_freq * T.float32(0.125) + T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331))) * orig_freq)}), where={orig_freq: T.float32(1.0) / T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 8) / T.float32(8.0))}), qkv[b, s, h, d])
                    else:
                        if h < 4:
                            orig_freq = T.float32()
                            smoothed_freq = T.float32()
                            k[b, s, h - 2, d] = T.if_then_else(d < 8, T.Let(T.Let(T.Cast("float16", T.cos(smoothed_freq)) * qkv[b, s, h, d] + T.Cast("float16", T.sin(smoothed_freq)) * T.if_then_else(d < 4, qkv[b, s, h, d + 4] * T.float16(-1.0), qkv[b, s, h, d - 4]), where={smoothed_freq: T.Cast("float32", T.Cast("float16", s + (total_seq_len - seq_len))) * ((T.float32(1.0) - T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331)))) * orig_freq * T.float32(0.125) + T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331))) * orig_freq)}), where={orig_freq: T.float32(1.0) / T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 8) / T.float32(8.0))}), qkv[b, s, h, d])
                        else:
                            v[b, s, h - 4, d] = qkv[b, s, h, d]

        @R.function
        def _initialize_effect() -> R.Tuple(R.Object):
            with R.dataflow():
                _io: R.Object = R.null_value()
                lv: R.Tuple(R.Object) = (_io,)
                gv: R.Tuple(R.Object) = lv
                R.output(gv)
            return gv

        @R.function
        def forward(qkv: R.Tensor((1, 4, 6, 8), dtype="float16"), total_seq_len_1: R.Shape(["total_seq_len"]), _io: R.Object) -> R.Tuple(R.Tuple(R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16")), R.Tuple(R.Object)):
            total_seq_len = T.int64()
            R.func_attr({"num_input": 3})
            cls = Expected
            with R.dataflow():
                lv1 = R.call_tir(cls.llama_rope, (qkv,), out_sinfo=[R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16")], tir_vars=R.shape([total_seq_len]))
                llama_rope_0: R.Tensor((1, 4, 2, 8), dtype="float16") = lv1[0]
                llama_rope_1: R.Tensor((1, 4, 2, 8), dtype="float16") = lv1[1]
                llama_rope_2: R.Tensor((1, 4, 2, 8), dtype="float16") = lv1[2]
                gv1: R.Tuple(R.Tuple(R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16")), R.Tuple(R.Object)) = (llama_rope_0, llama_rope_1, llama_rope_2), (_io,)
                R.output(gv1)
            return gv1
    # fmt: on

    tvm.ir.assert_structural_equal(mod, Expected)


def test_llama_rope_gptj_scaling_idempotent():
    """rope_scaling=gptj – structural equality with expected IRModule."""
    mod = _build(ROPE_SCALING_GPTJ)

    # fmt: off
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def llama_rope(var_qkv: T.handle, var_q: T.handle, var_k: T.handle, var_v: T.handle, total_seq_len: T.int64):
            T.func_attr({"op_pattern": 8, "tir.noalias": True})
            batch_size, seq_len = T.int64(), T.int64()
            qkv = T.match_buffer(var_qkv, (batch_size, seq_len, 6, 8), "float16")
            q = T.match_buffer(var_q, (batch_size, seq_len, 2, 8), "float16")
            k = T.match_buffer(var_k, (batch_size, seq_len, 2, 8), "float16")
            v = T.match_buffer(var_v, (batch_size, seq_len, 2, 8), "float16")
            for iters_0, iters_1, iters_2, iters_3 in T.grid(batch_size, seq_len, 6, 8):
                with T.sblock("llama_fused_rope"):
                    b, s, h, d = T.axis.remap("SSSS", [iters_0, iters_1, iters_2, iters_3])
                    T.reads(qkv[b, s, h, d - 1:d - 1 + 3])
                    T.writes(q[b, s, h, d], k[b, s, h - 2, d], v[b, s, h - 4, d])
                    if h < 2:
                        freq = T.float32()
                        q[b, s, h, d] = T.if_then_else(d < 8, T.Let(T.Cast("float16", T.cos(freq)) * qkv[b, s, h, d] + T.Cast("float16", T.sin(freq)) * T.if_then_else(d % 2 == 0, qkv[b, s, h, d + 1] * T.float16(-1.0), qkv[b, s, h, d - 1]), where={freq: T.Cast("float32", T.Cast("float16", s + (total_seq_len - seq_len))) / T.pow(T.float32(10000.0), T.Cast("float32", 2 * (d // 2) % 8) / T.float32(8.0))}), qkv[b, s, h, d])
                    else:
                        if h < 4:
                            freq = T.float32()
                            k[b, s, h - 2, d] = T.if_then_else(d < 8, T.Let(T.Cast("float16", T.cos(freq)) * qkv[b, s, h, d] + T.Cast("float16", T.sin(freq)) * T.if_then_else(d % 2 == 0, qkv[b, s, h, d + 1] * T.float16(-1.0), qkv[b, s, h, d - 1]), where={freq: T.Cast("float32", T.Cast("float16", s + (total_seq_len - seq_len))) / T.pow(T.float32(10000.0), T.Cast("float32", 2 * (d // 2) % 8) / T.float32(8.0))}), qkv[b, s, h, d])
                        else:
                            v[b, s, h - 4, d] = qkv[b, s, h, d]

        @R.function
        def _initialize_effect() -> R.Tuple(R.Object):
            with R.dataflow():
                _io: R.Object = R.null_value()
                lv: R.Tuple(R.Object) = (_io,)
                gv: R.Tuple(R.Object) = lv
                R.output(gv)
            return gv

        @R.function
        def forward(qkv: R.Tensor((1, 4, 6, 8), dtype="float16"), total_seq_len_1: R.Shape(["total_seq_len"]), _io: R.Object) -> R.Tuple(R.Tuple(R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16")), R.Tuple(R.Object)):
            total_seq_len = T.int64()
            R.func_attr({"num_input": 3})
            cls = Expected
            with R.dataflow():
                lv1 = R.call_tir(cls.llama_rope, (qkv,), out_sinfo=[R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16")], tir_vars=R.shape([total_seq_len]))
                llama_rope_0: R.Tensor((1, 4, 2, 8), dtype="float16") = lv1[0]
                llama_rope_1: R.Tensor((1, 4, 2, 8), dtype="float16") = lv1[1]
                llama_rope_2: R.Tensor((1, 4, 2, 8), dtype="float16") = lv1[2]
                gv1: R.Tuple(R.Tuple(R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16")), R.Tuple(R.Object)) = (llama_rope_0, llama_rope_1, llama_rope_2), (_io,)
                R.output(gv1)
            return gv1
    # fmt: on

    tvm.ir.assert_structural_equal(mod, Expected)


def test_llama_rope_gqa():
    """GQA variant: num_q_heads=4, num_kv_heads=2 with llama3 scaling – structural equality with expected IRModule."""
    mod = _build(ROPE_SCALING_LLAMA3, num_q=4, num_kv=2)

    # fmt: off
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def llama_rope(var_qkv: T.handle, var_q: T.handle, var_k: T.handle, var_v: T.handle, total_seq_len: T.int64):
            T.func_attr({"op_pattern": 8, "tir.noalias": True})
            batch_size, seq_len = T.int64(), T.int64()
            qkv = T.match_buffer(var_qkv, (batch_size, seq_len, 8, 8), "float16")
            q = T.match_buffer(var_q, (batch_size, seq_len, 4, 8), "float16")
            k = T.match_buffer(var_k, (batch_size, seq_len, 2, 8), "float16")
            v = T.match_buffer(var_v, (batch_size, seq_len, 2, 8), "float16")
            for iters_0, iters_1, iters_2, iters_3 in T.grid(batch_size, seq_len, 8, 8):
                with T.sblock("llama_fused_rope"):
                    b, s, h, d = T.axis.remap("SSSS", [iters_0, iters_1, iters_2, iters_3])
                    T.reads(qkv[b, s, h, d - 4:d - 4 + 9])
                    T.writes(q[b, s, h, d], k[b, s, h - 4, d], v[b, s, h - 6, d])
                    if h < 4:
                        orig_freq = T.float32()
                        smoothed_freq = T.float32()
                        q[b, s, h, d] = T.if_then_else(d < 8, T.Let(T.Let(T.Cast("float16", T.cos(smoothed_freq)) * qkv[b, s, h, d] + T.Cast("float16", T.sin(smoothed_freq)) * T.if_then_else(d < 4, qkv[b, s, h, d + 4] * T.float16(-1.0), qkv[b, s, h, d - 4]), where={smoothed_freq: T.Cast("float32", T.Cast("float16", s + (total_seq_len - seq_len))) * ((T.float32(1.0) - T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331)))) * orig_freq * T.float32(0.125) + T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331))) * orig_freq)}), where={orig_freq: T.float32(1.0) / T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 8) / T.float32(8.0))}), qkv[b, s, h, d])
                    else:
                        if h < 6:
                            orig_freq = T.float32()
                            smoothed_freq = T.float32()
                            k[b, s, h - 4, d] = T.if_then_else(d < 8, T.Let(T.Let(T.Cast("float16", T.cos(smoothed_freq)) * qkv[b, s, h, d] + T.Cast("float16", T.sin(smoothed_freq)) * T.if_then_else(d < 4, qkv[b, s, h, d + 4] * T.float16(-1.0), qkv[b, s, h, d - 4]), where={smoothed_freq: T.Cast("float32", T.Cast("float16", s + (total_seq_len - seq_len))) * ((T.float32(1.0) - T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331)))) * orig_freq * T.float32(0.125) + T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331))) * orig_freq)}), where={orig_freq: T.float32(1.0) / T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 8) / T.float32(8.0))}), qkv[b, s, h, d])
                        else:
                            v[b, s, h - 6, d] = qkv[b, s, h, d]

        @R.function
        def _initialize_effect() -> R.Tuple(R.Object):
            with R.dataflow():
                _io: R.Object = R.null_value()
                lv: R.Tuple(R.Object) = (_io,)
                gv: R.Tuple(R.Object) = lv
                R.output(gv)
            return gv

        @R.function
        def forward(qkv: R.Tensor((1, 4, 8, 8), dtype="float16"), total_seq_len_1: R.Shape(["total_seq_len"]), _io: R.Object) -> R.Tuple(R.Tuple(R.Tensor((1, 4, 4, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16")), R.Tuple(R.Object)):
            total_seq_len = T.int64()
            R.func_attr({"num_input": 3})
            cls = Expected
            with R.dataflow():
                lv1 = R.call_tir(cls.llama_rope, (qkv,), out_sinfo=[R.Tensor((1, 4, 4, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16")], tir_vars=R.shape([total_seq_len]))
                llama_rope_0: R.Tensor((1, 4, 4, 8), dtype="float16") = lv1[0]
                llama_rope_1: R.Tensor((1, 4, 2, 8), dtype="float16") = lv1[1]
                llama_rope_2: R.Tensor((1, 4, 2, 8), dtype="float16") = lv1[2]
                gv1: R.Tuple(R.Tuple(R.Tensor((1, 4, 4, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16")), R.Tuple(R.Object)) = (llama_rope_0, llama_rope_1, llama_rope_2), (_io,)
                R.output(gv1)
            return gv1
    # fmt: on

    tvm.ir.assert_structural_equal(mod, Expected)


def test_llama_rope_partial_rotary_dim():
    """Partial rotary_dim = HEAD_DIM // 2 = 4 with llama3 scaling – structural equality with expected IRModule."""
    mod = _build(ROPE_SCALING_LLAMA3, rotary_dim=HEAD_DIM // 2)

    # fmt: off
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def llama_rope(var_qkv: T.handle, var_q: T.handle, var_k: T.handle, var_v: T.handle, total_seq_len: T.int64):
            T.func_attr({"op_pattern": 8, "tir.noalias": True})
            batch_size, seq_len = T.int64(), T.int64()
            qkv = T.match_buffer(var_qkv, (batch_size, seq_len, 6, 8), "float16")
            q = T.match_buffer(var_q, (batch_size, seq_len, 2, 8), "float16")
            k = T.match_buffer(var_k, (batch_size, seq_len, 2, 8), "float16")
            v = T.match_buffer(var_v, (batch_size, seq_len, 2, 8), "float16")
            for iters_0, iters_1, iters_2, iters_3 in T.grid(batch_size, seq_len, 6, 8):
                with T.sblock("llama_fused_rope"):
                    b, s, h, d = T.axis.remap("SSSS", [iters_0, iters_1, iters_2, iters_3])
                    T.reads(qkv[b, s, h, d - 2:d - 2 + 5])
                    T.writes(q[b, s, h, d], k[b, s, h - 2, d], v[b, s, h - 4, d])
                    if h < 2:
                        orig_freq = T.float32()
                        smoothed_freq = T.float32()
                        q[b, s, h, d] = T.if_then_else(d < 4, T.Let(T.Let(T.Cast("float16", T.cos(smoothed_freq)) * qkv[b, s, h, d] + T.Cast("float16", T.sin(smoothed_freq)) * T.if_then_else(d < 2, qkv[b, s, h, d + 2] * T.float16(-1.0), qkv[b, s, h, d - 2]), where={smoothed_freq: T.Cast("float32", T.Cast("float16", s + (total_seq_len - seq_len))) * ((T.float32(1.0) - T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331)))) * orig_freq * T.float32(0.125) + T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331))) * orig_freq)}), where={orig_freq: T.float32(1.0) / T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 4) / T.float32(4.0))}), qkv[b, s, h, d])
                    else:
                        if h < 4:
                            orig_freq = T.float32()
                            smoothed_freq = T.float32()
                            k[b, s, h - 2, d] = T.if_then_else(d < 4, T.Let(T.Let(T.Cast("float16", T.cos(smoothed_freq)) * qkv[b, s, h, d] + T.Cast("float16", T.sin(smoothed_freq)) * T.if_then_else(d < 2, qkv[b, s, h, d + 2] * T.float16(-1.0), qkv[b, s, h, d - 2]), where={smoothed_freq: T.Cast("float32", T.Cast("float16", s + (total_seq_len - seq_len))) * ((T.float32(1.0) - T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331)))) * orig_freq * T.float32(0.125) + T.max(T.float32(0.0), T.min(T.float32(1.0), T.float32(434.59909793626889) * orig_freq - T.float32(0.33333333333333331))) * orig_freq)}), where={orig_freq: T.float32(1.0) / T.pow(T.float32(10000.0), T.Cast("float32", d * 2 % 4) / T.float32(4.0))}), qkv[b, s, h, d])
                        else:
                            v[b, s, h - 4, d] = qkv[b, s, h, d]

        @R.function
        def _initialize_effect() -> R.Tuple(R.Object):
            with R.dataflow():
                _io: R.Object = R.null_value()
                lv: R.Tuple(R.Object) = (_io,)
                gv: R.Tuple(R.Object) = lv
                R.output(gv)
            return gv

        @R.function
        def forward(qkv: R.Tensor((1, 4, 6, 8), dtype="float16"), total_seq_len_1: R.Shape(["total_seq_len"]), _io: R.Object) -> R.Tuple(R.Tuple(R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16")), R.Tuple(R.Object)):
            total_seq_len = T.int64()
            R.func_attr({"num_input": 3})
            cls = Expected
            with R.dataflow():
                lv1 = R.call_tir(cls.llama_rope, (qkv,), out_sinfo=[R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16")], tir_vars=R.shape([total_seq_len]))
                llama_rope_0: R.Tensor((1, 4, 2, 8), dtype="float16") = lv1[0]
                llama_rope_1: R.Tensor((1, 4, 2, 8), dtype="float16") = lv1[1]
                llama_rope_2: R.Tensor((1, 4, 2, 8), dtype="float16") = lv1[2]
                gv1: R.Tuple(R.Tuple(R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16"), R.Tensor((1, 4, 2, 8), dtype="float16")), R.Tuple(R.Object)) = (llama_rope_0, llama_rope_1, llama_rope_2), (_io,)
                R.output(gv1)
            return gv1
    # fmt: on

    tvm.ir.assert_structural_equal(mod, Expected)



if __name__ == "__main__":
    tvm.testing.main()
