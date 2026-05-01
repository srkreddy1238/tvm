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

"""Thin Python wrappers for positional-embedding operators (e.g. RoPE).

All heavy-weight TIR construction lives in the C++ implementation at
  src/relax/frontend/nn/llm/position_embedding.cc

This module is a thin FFI bridge:
  1. Converts Python arguments (nn.Tensor -> relax.Var, dict -> ffi.Map, …).
  2. Calls the corresponding C++ FFI function.
  3. Wraps the returned relax.Var(s) back into nn.Tensor via
     ``_unwrap_ffi_result``.

The public API (``llama_rope``, ``llama_rope_with_position_map``,
``llama4_rope_with_position_map``, ``switch_rope_freq_func``) is identical
to the previous pure-Python implementation so all call-sites remain unchanged.
"""

from collections.abc import Callable
from typing import Any

from tvm import tir
from tvm.relax.frontend.nn import Tensor

import tvm.relax.frontend.nn.llm._ffi_api_llm_position_embedding as _ffi  # C++ FFI bridge

# pylint: disable=invalid-name


# ---------------------------------------------------------------------------
# switch_rope_freq_func – thin wrapper around the C++ implementation
#
# kv_cache.py and tree_attn.py call this as:
#
#   cos_freq, sin_freq, var_map = switch_rope_freq_func(rope_scaling)(
#       offset * scale, d, rotary_dim, theta, "float32"
#   )
#
# The C++ FFI function has signature:
#   switch_rope_freq_func(rope_scaling, s, d, d_range, theta, dtype)
#   -> [cos_freq, sin_freq, Array<Var> keys, Array<PrimExpr> vals]
#
# We wrap it so the returned callable matches the Python convention exactly.
# ---------------------------------------------------------------------------


def switch_rope_freq_func(rope_scaling: dict[str, Any]) -> Callable:
    """Return the RoPE inverse-frequency computation function for the given scaling config.

    The returned callable has signature::

        (s, d, d_range, theta, dtype) -> (cos_freq, sin_freq, var_map)

    where ``var_map`` is a ``{tir.Var: tir.PrimExpr}`` dict of intermediate
    variables that must be bound via ``tir.Let`` in the enclosing expression.

    All computation is delegated to the C++ ``SwitchRopeFreqFunc`` /
    ``RopeFreq*`` family via FFI.
    """

    def _call(s: tir.Var, d: tir.Var, d_range: int, theta, dtype: str) -> tuple:
        result = _ffi.switch_rope_freq_func(rope_scaling, s, d, int(d_range), theta, dtype)
        cos_freq = result[0]
        sin_freq = result[1]
        keys = result[2]   # Array<tir.Var>
        vals = result[3]   # Array<tir.PrimExpr>
        var_map = {k: v for k, v in zip(keys, vals)}
        return cos_freq, sin_freq, var_map

    return _call


# mypy: disable-error-code="attr-defined"


def llama_rope(  # pylint: disable=too-many-arguments
    qkv: Tensor,
    total_seq_len: tir.Var,
    theta: float,
    scale: float,
    num_q_heads: int,
    num_kv_heads: int,
    rope_scaling: dict[str, Any],
    rotary_dim: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Llama-style RoPE. Given a fused QKV tensor, it returns three tensors, Q, K, and V, where Q
    and K are rotated by RoPE while V remains unchanged.

    Parameters
    ----------
    qkv : Tensor
        The fused QKV tensor of shape: [batch_size, seq_len, #q_heads + #kv_heads * 2, head_dim]

    total_seq_len : tir.Var
        The total sequence length after being concatenated with KVCache. It is used to compute the
        offset of RoPE.

    theta : float
        The theta value, or "base" in RoPE, which controls the frequency.

    scale : float
        The RoPE scaling factor.

    num_q_heads : int
        The number of query heads.

    num_kv_heads : int
        The number of key/value heads. It differs from `num_q_heads` in group-query attention.

    rope_scaling : Dict
        The configuration of RoPE scaling.

    rotary_dim : Optional[int]
        The number of dimensions in the embedding that RoPE is applied to. By default, the
        rotary_dim is the same as head_dim.

    Returns
    -------
    q : Tensor
        The query tensor of shape [batch_size, seq_len, #q_heads, head_dim] w/ RoPE applied

    k : Tensor
        The key tensor of shape [batch_size, seq_len, #kv_heads, head_dim] w/ RoPE applied

                        v : Tensor
        The value tensor of shape [batch_size, seq_len, #kv_heads, head_dim] w/o RoPE applied
    """
    result = _ffi.llama_rope(
        qkv,
        total_seq_len,
        float(theta),
        float(scale),
        int(num_q_heads),
        int(num_kv_heads),
        rope_scaling,
        rotary_dim,
    )
    return result[0], result[1], result[2]  # type: ignore[return-value]


def llama_rope_with_position_map(  # pylint: disable=too-many-arguments
    theta: float,
    scale: float,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: str,
    rope_scaling: dict[str, Any],
    rotary_dim: int | None = None,
):
    """Return the TIR function that computes Llama-style RoPE with a position map.

    Parameters
    ----------
    theta : float
        The theta value, or "base" in RoPE, which controls the frequency.

    scale : float
        The RoPE scaling factor.

    head_dim : int
        The number of features on each head.

    num_q_heads : int
        The number of query heads.

    num_kv_heads : int
        The number of key/value heads. It differs from `num_q_heads` in group-query attention.

    dtype : str
        The dtype of qkv data.

    rope_scaling : Dict
        The configuration of RoPE scaling.

    rotary_dim : int
        The number of dimensions in the embedding that RoPE is applied to. By default, the
        rotary_dim is the same as head_dim.

    Returns
    -------
    fused_rope : tir.PrimFunc
        The TIR function implementing the fused RoPE kernel.  For longrope
        scaling the returned function accepts an ``ext_factors`` buffer as its
        last argument instead of ``apply_rope``.
    """
    return _ffi.llama_rope_with_position_map(
        float(theta),
        float(scale),
        int(head_dim),
        int(num_q_heads),
        int(num_kv_heads),
        dtype,
        rope_scaling,
        rotary_dim,
    )


def llama4_rope_with_position_map(  # pylint: disable=too-many-arguments
    theta: float,
    scale: float,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: str,
    rope_scaling: dict[str, Any],
    rotary_dim: int | None = None,
):
    """Return the TIR function that computes Llama-4-style RoPE with a position map.

    Parameters
    ----------
    theta : float
        The theta value, or "base" in RoPE, which controls the frequency.

    scale : float
        The RoPE scaling factor.

    head_dim : int
        The number of features on each head.

    num_q_heads : int
        The number of query heads.

    num_kv_heads : int
        The number of key/value heads. It differs from `num_q_heads` in group-query attention.

    dtype : str
        The dtype of qkv data.

    rope_scaling : Dict
        The configuration of RoPE scaling.

    rotary_dim : int
        The number of dimensions in the embedding that RoPE is applied to. By default, the
        rotary_dim is the same as head_dim.

    Returns
    -------
    fused_rope : tir.PrimFunc
        The TIR function implementing the fused Llama-4 RoPE kernel.
    """
    return _ffi.llama4_rope_with_position_map(
        float(theta),
        float(scale),
        int(head_dim),
        int(num_q_heads),
        int(num_kv_heads),
        dtype,
        rope_scaling,
        rotary_dim,
    )
