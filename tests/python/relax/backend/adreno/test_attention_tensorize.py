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
import os

import numpy as np
import pytest
from utils import SessionManager, requires_adreno_vulkan

import tvm
import tvm.testing
from tvm import IRModule, relax, s_tir
from tvm.relax.frontend.nn.llm.kv_cache import (
    _attention_prefill,
    _attention_prefill_ragged,
)

TARGET_SUPPORTS_EXTENSION = os.getenv("ADRENO_TARGET_COOP", "").strip().lower() == "yes"
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
        "supports_qcom_cooperative_matrix_conversion": 1,
        "host": {
            "kind": "llvm",
            "mtriple": "aarch64-linux-android",
        },
    }
)

ref_target = tvm.target.Target(
    {
        "kind": "opencl",
        "keys": ["adreno", "gpu"],
        "host": {
            "kind": "llvm",
            "mtriple": "aarch64-linux-android",
        },
    }
)


def build_and_run_func(mod, inputs_np, target, out_id):
    pipeline = relax.pipeline.get_default_pipeline(target)
    mod = pipeline(mod)
    ex = tvm.compile(mod, target)

    with SessionManager() as sess:
        rexec = sess.load_module(ex)
        dev = sess.device(target.kind.name)

        inputs = [inp if np.isscalar(inp) else tvm.runtime.tensor(inp, dev) for inp in inputs_np]
        func_profile = rexec.time_evaluator(rexec.entry_name, dev, number=3)
        time_f = func_profile(*inputs)
        print(
            f"Attention Time under Target({str(target.kind).split()[0]}): {np.mean(time_f.results) * 1000}ms"  # noqa: E501
        )
        out = inputs[out_id].numpy()
        return out


@requires_adreno_vulkan
@pytest.mark.skipif(not TARGET_SUPPORTS_EXTENSION, reason="Device not supported.")
@pytest.mark.parametrize(
    "h_q, h_kv, d_qk, d_v, seq_len",
    [(32, 32, 64, 64, 256), (32, 8, 64, 128, 1024), (32, 8, 64, 256, 1500), (32, 32, 96, 96, 256)],
)
def test_prefill_ragged_attention(h_q, h_kv, d_qk, d_v, seq_len):
    np.random.seed(42)

    mod_ref = IRModule(
        {"main": _attention_prefill_ragged(h_kv, h_q, d_qk, d_v, "float16", {}, ref_target)}
    )
    func = _attention_prefill_ragged(h_kv, h_q, d_qk, d_v, "float16", {}, vk_target)
    sch = s_tir.Schedule(func)
    func = sch.mod["main"].with_attr("global_symbol", "main")
    mod_org = IRModule({"main": func})

    query = np.random.uniform(-10.0, 10.0, size=(1, seq_len, h_q, d_qk)).astype("float16")
    key = np.random.uniform(-10.0, 10.0, size=(1, seq_len, h_kv, d_qk)).astype("float16")
    value = np.random.uniform(-10.0, 10.0, size=(1, seq_len, h_kv, d_v)).astype("float16")

    inputs = (
        query.reshape((query.shape[1], query.shape[2], query.shape[3])),
        np.array([0, seq_len], dtype="int32"),
        key.reshape((key.shape[1], key.shape[2], key.shape[3])),
        value.reshape((value.shape[1], value.shape[2], value.shape[3])),
        np.array([0, seq_len], dtype="int32"),
        np.random.randint(0, seq_len, size=(seq_len,)).astype("int32"),
        np.array([0], dtype="int32"),
        np.random.uniform(-1, 1, size=(seq_len, h_q, d_v)).astype("float16"),
        np.random.uniform(-1, 1, size=(seq_len, h_q)).astype("float32"),
        np.int32(1),
        np.int32(0),
        np.float32(0.0),
        np.float32(0.0),
        np.float32(0.125),
    )

    rs_ref = build_and_run_func(mod_ref, inputs, ref_target, 7)
    rs_org = build_and_run_func(mod_org, inputs, vk_target, 7)

    atol = max(rs_ref.flatten()) * 0.05
    close = np.isclose(rs_org.flatten(), rs_ref.flatten(), rtol=1e-3, atol=atol)
    mismatch_ratio = 1.0 - np.mean(close)
    max_mismatch_ratio = 0.01  # Allow up to 1% mismatch
    if mismatch_ratio > max_mismatch_ratio:
        raise AssertionError(
            f"Mismatch ratio {mismatch_ratio:.4f} exceeds allowed {max_mismatch_ratio}"
        )


@requires_adreno_vulkan
@pytest.mark.skipif(not TARGET_SUPPORTS_EXTENSION, reason="Device not supported.")
@pytest.mark.parametrize(
    "h_q, h_kv, d, seq_len, kv_len",
    [(32, 32, 64, 256, 256), (32, 8, 64, 256, 1024), (32, 16, 64, 512, 1024)],
)
def test_prefill_pagged_attention(h_q, h_kv, d, seq_len, kv_len):
    np.random.seed(42)
    page_size = 64
    mod_ref = IRModule(
        {"main": _attention_prefill(h_kv, h_q, d, "float16", False, {}, ref_target, 64)}
    )
    func = _attention_prefill(h_kv, h_q, d, "float16", False, {}, vk_target, 64)
    sch = s_tir.Schedule(func)
    func = sch.mod["main"].with_attr("global_symbol", "main")
    mod_org = IRModule({"main": func})

    query = np.random.uniform(-10.0, 10.0, size=(seq_len, h_q, d)).astype("float16")
    pages = np.random.uniform(
        -10.0, 10.0, size=((kv_len + page_size - 1) // page_size, 2, h_kv, page_size, d)
    ).astype("float16")

    inputs = (
        query,
        np.array([0, seq_len], dtype="int32"),
        pages,
        np.array([0, (kv_len + page_size - 1) // page_size], dtype="int32"),
        np.arange((kv_len + page_size - 1) // page_size).astype("int32"),
        np.array(
            [page_size - (((kv_len + page_size - 1) // page_size) * page_size - kv_len)],
            dtype="int32",
        ),
        np.array([0], dtype="int32"),
        np.random.randint(0, seq_len, size=(seq_len,)).astype("int32"),
        np.random.uniform(-1, 1, size=(seq_len, h_q, d)).astype("float16"),
        np.random.uniform(-1, 1, size=(seq_len, h_q)).astype("float32"),
        np.int32(1),
        np.int32(0),
        np.float32(0.0),
        np.float32(0.0),
        np.float32(0.125),
    )

    rs_ref = build_and_run_func(mod_ref, inputs, ref_target, 8)
    rs_org = build_and_run_func(mod_org, inputs, vk_target, 8)

    atol = max(rs_ref.flatten()) * 0.05
    close = np.isclose(rs_org.flatten(), rs_ref.flatten(), rtol=1e-3, atol=atol)
    mismatch_ratio = 1.0 - np.mean(close)
    max_mismatch_ratio = 0.01  # Allow up to 1% mismatch
    if mismatch_ratio > max_mismatch_ratio:
        raise AssertionError(
            f"Mismatch ratio {mismatch_ratio:.4f} exceeds allowed {max_mismatch_ratio}"
        )


if __name__ == "__main__":
    tvm.testing.main()
