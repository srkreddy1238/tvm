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
"""Tests for switch_rope_freq_func.

Each test calls the returned frequency function with concrete TIR vars and
compares the resulting (cos_freq, sin_freq, var_map) expressions against
hand-written expected TIR expressions using structural equality.

Config used across all tests
-----------------------------
* d_range = 8  (head_dim / rotary_dim)
* theta   = 10000.0
* dtype   = "float16"
"""

import tvm
import tvm.testing
from tvm import tir
from tvm.relax.frontend.nn.llm.position_embedding import switch_rope_freq_func

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
D_RANGE = 8
THETA = 10000.0
DTYPE = "float16"

ROPE_SCALING_NONE: dict = {}
ROPE_SCALING_GPTJ = {"rope_type": "gptj"}
ROPE_SCALING_LLAMA3 = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}
ROPE_SCALING_LONGROPE = {
    "rope_type": "longrope",
    "max_position_embeddings": 131072,
    "original_max_position_embeddings": 4096,
}


# ---------------------------------------------------------------------------
# Helper: call the freq func with fixed TIR vars
# ---------------------------------------------------------------------------


def _call(rope_scaling):
    """Call switch_rope_freq_func and return (cos_freq, sin_freq, var_map)."""
    s = tir.Var("s", "float32")
    d = tir.Var("d", "int64")
    return switch_rope_freq_func(rope_scaling)(s, d, D_RANGE, THETA, DTYPE), s, d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_switch_rope_freq_func_none_returns_default():
    """rope_scaling={} - structural equality with expected TIR expressions."""
    (cos_freq, sin_freq, var_map), s, d = _call(ROPE_SCALING_NONE)

    freq_var = tir.Var("freq", "float32")
    expected_cos = tir.Cast("float16", tir.cos(freq_var))
    expected_sin = tir.Cast("float16", tir.sin(freq_var))
    expected_freq = s / tir.pow(
        tir.const(10000.0, "float32"),
        tir.Cast("float32", d * tir.IntImm("int64", 2) % tir.IntImm("int64", 8))
        / tir.const(8.0, "float32"),
    )

    assert len(var_map) == 1
    ((actual_var, actual_val),) = var_map.items()
    tvm.ir.assert_structural_equal(cos_freq, expected_cos, map_free_vars=True)
    tvm.ir.assert_structural_equal(sin_freq, expected_sin, map_free_vars=True)
    tvm.ir.assert_structural_equal(actual_val, expected_freq, map_free_vars=True)


def test_switch_rope_freq_func_gptj_returns_gptj():
    """rope_type='gptj' - structural equality with expected TIR expressions."""
    (cos_freq, sin_freq, var_map), s, d = _call(ROPE_SCALING_GPTJ)

    freq_var = tir.Var("freq", "float32")
    expected_cos = tir.Cast("float16", tir.cos(freq_var))
    expected_sin = tir.Cast("float16", tir.sin(freq_var))
    expected_freq = s / tir.pow(
        tir.const(10000.0, "float32"),
        tir.Cast(
            "float32",
            tir.IntImm("int64", 2) * (d // tir.IntImm("int64", 2)) % tir.IntImm("int64", 8),
        )
        / tir.const(8.0, "float32"),
    )

    assert len(var_map) == 1
    ((actual_var, actual_val),) = var_map.items()
    tvm.ir.assert_structural_equal(cos_freq, expected_cos, map_free_vars=True)
    tvm.ir.assert_structural_equal(sin_freq, expected_sin, map_free_vars=True)
    tvm.ir.assert_structural_equal(actual_val, expected_freq, map_free_vars=True)


def test_switch_rope_freq_func_llama3_returns_callable():
    """rope_type='llama3' - structural equality with expected TIR expressions."""
    (cos_freq, sin_freq, var_map), s, d = _call(ROPE_SCALING_LLAMA3)

    smoothed_freq_var = tir.Var("smoothed_freq", "float32")
    orig_freq_var = tir.Var("orig_freq", "float32")
    expected_cos = tir.Cast("float16", tir.cos(smoothed_freq_var))
    expected_sin = tir.Cast("float16", tir.sin(smoothed_freq_var))
    expected_smoothed_freq = s * (
        (
            tir.const(1.0, "float32")
            - tir.max(
                tir.const(0.0, "float32"),
                tir.min(
                    tir.const(1.0, "float32"),
                    tir.const(434.59909793626889, "float32") * orig_freq_var
                    - tir.const(0.33333333333333331, "float32"),
                ),
            )
        )
        * orig_freq_var
        * tir.const(0.125, "float32")
        + tir.max(
            tir.const(0.0, "float32"),
            tir.min(
                tir.const(1.0, "float32"),
                tir.const(434.59909793626889, "float32") * orig_freq_var
                - tir.const(0.33333333333333331, "float32"),
            ),
        )
        * orig_freq_var
    )
    expected_orig_freq = tir.const(1.0, "float32") / tir.pow(
        tir.const(10000.0, "float32"),
        tir.Cast("float32", d * tir.IntImm("int64", 2) % tir.IntImm("int64", 8))
        / tir.const(8.0, "float32"),
    )

    assert len(var_map) == 2
    var_map_list = list(var_map.items())
    tvm.ir.assert_structural_equal(cos_freq, expected_cos, map_free_vars=True)
    tvm.ir.assert_structural_equal(sin_freq, expected_sin, map_free_vars=True)
    tvm.ir.assert_structural_equal(var_map_list[0][1], expected_smoothed_freq, map_free_vars=True)
    tvm.ir.assert_structural_equal(var_map_list[1][1], expected_orig_freq, map_free_vars=True)


def test_switch_rope_freq_func_longrope_returns_callable():
    """rope_type='longrope' - structural equality with expected TIR expressions."""
    (cos_freq, sin_freq, var_map), s, d = _call(ROPE_SCALING_LONGROPE)

    freq_var = tir.Var("freq", "float32")
    scaling_factor = tir.const(1.1902380714238083, "float32")
    expected_cos = tir.Cast("float16", tir.cos(freq_var) * scaling_factor)
    expected_sin = tir.Cast("float16", tir.sin(freq_var) * scaling_factor)
    expected_freq = s / tir.pow(
        tir.const(10000.0, "float32"),
        tir.Cast("float32", d * tir.IntImm("int64", 2) % tir.IntImm("int64", 8))
        / tir.const(8.0, "float32"),
    )

    assert len(var_map) == 1
    ((actual_var, actual_val),) = var_map.items()
    tvm.ir.assert_structural_equal(cos_freq, expected_cos, map_free_vars=True)
    tvm.ir.assert_structural_equal(sin_freq, expected_sin, map_free_vars=True)
    tvm.ir.assert_structural_equal(actual_val, expected_freq, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
