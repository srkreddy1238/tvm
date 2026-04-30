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
# ruff: noqa: E501, E731, RUF005, RUF012

"""Attention KV cache modeling."""

# pylint: disable=too-many-statements,too-many-lines,too-many-arguments,invalid-name
import enum
import math
from typing import Any, Literal

import tvm
from tvm import relax as rx
from tvm import s_tir, tir
from tvm.relax.frontend.nn import Object, Tensor
from tvm.runtime import DataType
from tvm.script import tir as T
from tvm.target import Target

from .position_embedding import llama_rope_with_position_map, switch_rope_freq_func
import tvm.relax.frontend.nn.llm._ffi_api_llm_kv_cache as _ffi_kv_cache  # C++ FFI bridge
from .tree_attn import (
    tree_attn,
    tree_attn_cpu,
    tree_attn_with_paged_kv_cache,
    tree_attn_with_paged_kv_cache_cpu,
)


def _var_cpu(dtype):
    return T.alloc_buffer((1,), dtype)


def get_max_num_threads_per_block(target: Target) -> int:
    """
    max(max_num_threads, max_threads_per_block); if latter does not exist, return max_num_threads.
    We add this method since some targets have both fields and `max_threads_per_block` is larger.
    """
    max_num_threads = int(target.attrs["max_num_threads"])
    max_threads_per_block = target.attrs.get("max_threads_per_block", None)
    if max_threads_per_block is None:
        return max_num_threads
    return max(max_num_threads, max_threads_per_block)


def check_thread_limits(target: Target, bdx: int, bdy: int, bdz: int, gdz: int):
    """
    Check whether max num threads exceeded given a target.

    Parameters
    ----------
    bdx: threadIdx.x
    bdy: threadIdx.y
    bdz: threadIdx.z
    gdz: blockIdx.z
    """
    max_num_threads_per_block = get_max_num_threads_per_block(target)

    assert bdx * bdy * bdz <= max_num_threads_per_block, (
        f"{target.kind} max num threads exceeded: {bdx}*{bdy}*{bdz}>{max_num_threads_per_block}"
    )

    if str(target.kind) == "webgpu":
        # https://gpuweb.github.io/gpuweb/#dom-supported-limits-maxcomputeworkgroupsizez
        assert bdz <= 64, f"webgpu's threadIdx.z cannot exceed 64, but got bdz={bdz}"
        assert gdz == 1, f"webgpu's blockIdx.z should be 1, but got gdz={gdz}"


class AttnKind(enum.IntEnum):
    """The attention kind class.
    MHA denotes multi-head attention, multi-query attention or grouped query attention.
    MLA denotes multi-head latent attention.
    """

    MHA = 0
    MLA = 1
    MHA_SLIDING = 3


class RopeMode(enum.IntEnum):
    """The RoPE mode of the Paged KV cache.
    If it is none, the KV cache will not apply RoPE to q and k.
    If it is normal, RoPE will be applied to k before adding k to cache.
    Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
    """

    NONE = 0
    NORMAL = 1
    INLINE = 2


class PagedKVCache(Object):  # pylint: disable=too-few-public-methods
    """The Paged KV Cache used in LLM batching for efficient attention computation."""

    extern_mods: list[tvm.runtime.Module] = []

    def attention_with_fused_qkv(
        self,
        layer_id: int,
        qkv: Tensor,
        num_qo_heads: int,
        sm_scale: float,
    ) -> Tensor:
        """Compute attention with the given fused q/k/v data and in-cache k/v data
        on the specified layer. Rotary position embeddings are applied to k/v
        within this function.

        - For prefill, the input qkv and output tensor have shape
        (1, total_seq_len) for the first two dimensions.
        - For decode, the input qkv and output tensor have shape
        (batch_size, 1) for the first two dimensions.
        - The input qkv have `2 * num_qo_heads + num_kv_heads` at the third dim.
        - The output tensor have `num_qo_heads` at the third dim.
        - The input qkv and output tensor have `head_dim` at the last dim.
        """
        # pylint: disable=protected-access
        b, s, _, d = qkv._expr.struct_info.shape
        qkv = qkv.reshape(b * s, qkv.shape[2], d)
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.call_dps_packed(
                    "vm.builtin.attention_kv_cache_attention_with_fused_qkv",
                    [
                        self._expr,
                        rx.PrimValue(layer_id),  # type: ignore[arg-type]
                        rx.PrimValue(sm_scale),
                        qkv._expr,
                    ],
                    out_sinfo=rx.TensorStructInfo((b * s, num_qo_heads, d), qkv.dtype),
                )
            )
        ).reshape(b, s, num_qo_heads, d)

    def self_attention(  # pylint: disable=too-many-locals
        self,
        layer_id: int,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        sm_scale: float,
    ) -> tuple[Tensor, Tensor]:
        """Fine-grained API that computes ragged self attention with Q/K/V data."""
        # pylint: disable=protected-access
        b, s, h_qo, d_qk = q._expr.struct_info.shape
        _, _, h_kv, d_v = v._expr.struct_info.shape
        q = q.reshape(b * s, h_qo, d_qk)
        k = k.reshape(b * s, h_kv, d_qk)
        v = v.reshape(b * s, h_kv, d_v)
        bb = rx.BlockBuilder.current()
        attn_results = bb.emit(
            rx.call_dps_packed(
                "vm.builtin.attention_kv_cache_self_attention",
                [
                    self._expr,
                    rx.PrimValue(layer_id),  # type: ignore[arg-type]
                    rx.PrimValue(sm_scale),
                    q._expr,
                    k._expr,
                    v._expr,
                ],
                out_sinfo=[
                    rx.TensorStructInfo((b * s, h_qo, d_v), q.dtype),
                    rx.TensorStructInfo((b * s, h_qo), "float32"),
                ],
            )
        )
        assert isinstance(attn_results.struct_info, rx.TupleStructInfo)
        assert len(attn_results.struct_info.fields) == 2
        o = Tensor(_expr=bb.emit(rx.TupleGetItem(attn_results, 0))).reshape(b, s, h_qo, d_v)
        lse = Tensor(_expr=bb.emit(rx.TupleGetItem(attn_results, 1))).reshape(b, s, h_qo)
        return o, lse

    def cross_attention(
        self,
        layer_id: int,
        q: Tensor,
        v_head_dim: int,
        sm_scale: float,
    ) -> tuple[Tensor, Tensor]:
        """Fine-grained API that computes paged cross attention with Q and in-cache KV data."""
        # pylint: disable=protected-access
        b, s, h_qo, d_qk = q._expr.struct_info.shape
        q = q.reshape(b * s, h_qo, d_qk)
        bb = rx.BlockBuilder.current()
        attn_results = bb.emit(
            rx.call_dps_packed(
                "vm.builtin.attention_kv_cache_cross_attention",
                [
                    self._expr,
                    rx.PrimValue(layer_id),  # type: ignore[arg-type]
                    rx.PrimValue(sm_scale),
                    q._expr,
                ],
                out_sinfo=[
                    rx.TensorStructInfo((b * s, h_qo, v_head_dim), q.dtype),
                    rx.TensorStructInfo((b * s, h_qo), "float32"),
                ],
            )
        )
        assert isinstance(attn_results.struct_info, rx.TupleStructInfo)
        assert len(attn_results.struct_info.fields) == 2
        o = Tensor(_expr=bb.emit(rx.TupleGetItem(attn_results, 0))).reshape(b, s, h_qo, v_head_dim)
        lse = Tensor(_expr=bb.emit(rx.TupleGetItem(attn_results, 1))).reshape(b, s, h_qo)
        return o, lse

    def append_mla_kv(self, layer_id: int, kv: Tensor) -> "PagedKVCache":
        """Fine-grained API that appends the MLA K/V data to KV cache."""
        # pylint: disable=protected-access
        b, s, _, d_qk = kv._expr.struct_info.shape
        kv = kv.reshape(b * s, d_qk)
        return PagedKVCache(
            _expr=rx.call_pure_packed(
                "vm.builtin.attention_kv_cache_append_mla_kv",
                self._expr,
                rx.PrimValue(layer_id),  # type: ignore[arg-type]
                kv._expr,
                sinfo_args=rx.ObjectStructInfo(),
            ),
            _name="paged_kv_cache",
        )

    def merge_attn_output_inplace(
        self,
        o_self_attn: Tensor,
        lse_self_attn: Tensor,
        o_cross_attn: Tensor,
        lse_cross_attn: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Fine-grained API that merges the attention output from two sources.
        The first two tensors will be inplace updated.
        """
        # pylint: disable=protected-access
        b, s, h_qo, d_v = o_self_attn._expr.struct_info.shape
        o_self_attn = o_self_attn.reshape(b * s, h_qo, d_v)
        lse_self_attn = lse_self_attn.reshape(b * s, h_qo)
        o_cross_attn = o_cross_attn.reshape(b * s, h_qo, d_v)
        lse_cross_attn = lse_cross_attn.reshape(b * s, h_qo)
        bb = rx.BlockBuilder.current()
        merge_results = bb.emit(
            rx.call_pure_packed(
                "vm.builtin.attention_kv_cache_merge_attn_output_inplace",
                self._expr,
                o_self_attn._expr,
                lse_self_attn._expr,
                o_cross_attn._expr,
                lse_cross_attn._expr,
                sinfo_args=rx.TupleStructInfo(
                    [o_self_attn._expr.struct_info, lse_self_attn._expr.struct_info]
                ),
            )
        )
        assert isinstance(merge_results.struct_info, rx.TupleStructInfo)
        assert len(merge_results.struct_info.fields) == 2
        o_self_attn = Tensor(_expr=bb.emit(rx.TupleGetItem(merge_results, 0))).reshape(
            b, s, h_qo, d_v
        )
        lse_self_attn = Tensor(_expr=bb.emit(rx.TupleGetItem(merge_results, 1))).reshape(b, s, h_qo)
        return o_self_attn, lse_self_attn

    def get_query_positions(self, total_length: tir.PrimExpr) -> Tensor:
        """Get the in-sequence positions of each slot in the query,
        which are needed for applying positional embeddings in some models.

        Parameters
        ----------
        total_length : tir.PrimExpr
            The summed-up total sequence length of queries in
            the batch being forwarded.

        Returns
        -------
        q_positions : Tensor
            The in-sequence query positions, in shape `(total_length,)`
        """
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.call_pure_packed(
                    "vm.builtin.attention_kv_cache_get_query_positions",
                    self._expr,
                    sinfo_args=rx.TensorStructInfo((total_length,), "int32"),
                )
            )
        )

    # pylint: enable=protected-access


def _prepare_yarn_rope_scaling(
    rope_scaling: dict[str, Any] | None,
    rope_theta: float | None,
) -> dict[str, Any] | None:
    """Ensure Yarn-specific scaling configs include the theta metadata."""
    if rope_scaling is None:
        return None
    if rope_scaling.get("rope_type") != "yarn":
        return rope_scaling

    rope_scaling_updated = dict(rope_scaling)
    if "inv_theta_log_scale" not in rope_scaling_updated and rope_theta is not None:
        theta_value = float(rope_theta)
        rope_scaling_updated["inv_theta_log_scale"] = 1.0 / (2 * math.log(theta_value))
    return rope_scaling_updated


class FlashInferPagedKVCache(PagedKVCache):  # pylint: disable=too-few-public-methods
    """Paged KV cache using FlashInfer (CUDA) kernels."""

    def __init__(  # pylint: disable=too-many-locals
        self,
        attn_kind: Literal["mha", "mla"] | list[Literal["mha", "mla", "mha_sliding"]],
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
        layer_partition: rx.ShapeExpr,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        mla_original_qk_head_dim: int,
        mla_original_v_head_dim: int,
        rope_mode: RopeMode,
        rope_scale: int,
        rope_theta: int,
        rope_scaling: dict[str, Any],
        rope_ext_factors: rx.Expr,
        rotary_dim: int,
        enable_disaggregation: bool,
        dtype: str,
        target: Target,
        name: str = "paged_kv_cache",
    ) -> None:
        """Create a paged KV cache object with FlashInfer kernels.

        Parameters
        ----------
        max_batch_size : tir.Var
            The maximum allowed batch size of the KV cache.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        max_total_seq_len : tir.Var
            The maximum allowed total sequence length of the KV cache.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        prefill_chunk_size : tir.Var
            The maximum total sequence length in a prefill.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        page_size : tir.Var
            The size (a.k.a. number of tokens) of each page.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        support_sliding_window : tir.Var
            0 or 1, denoting whether the KV cache supports sliding window.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        layer_partition : rx.ShapeExpr
            The KV cache layer partition for pipeline stages.
            It is an indptr array, denoting the starting layer of each pipeline stage.
        rope_mode : RopeMode
            The RoPE mode of the Paged KV cache.
            If it is normal, RoPE will be applied to k before adding k to cache.
            Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
        rope_scale : int
            The scale of rotary position embedding.
        rope_theta : int
            The base of rotary position embedding.
        rope_scaling: Dict[str, Any]
            The RoPE scaling information dict.
        rope_ext_factors: rx.Expr
            The RoPE extension factors when "longrope" mode RoPE scaling is enabled.
        rotary_dim : int
            The number of dimensions in the embedding that RoPE is applied to.
        enable_disaggregation : bool
            Whether to enable disaggregation in the KV cache.
        """
        assert rope_mode != RopeMode.INLINE, "FlashInfer RoPE does not support inline mode."
        rope_scaling = _prepare_yarn_rope_scaling(rope_scaling, rope_theta)

        attn_kind_single = attn_kind[0] if isinstance(attn_kind, list) else attn_kind
        if attn_kind_single == "mha_sliding":
            attn_kind_single = "mha"
        flashinfer_prefill_mods = rx.backend.cuda.flashinfer.gen_flashinfer_prefill_module(
            dtype_q=dtype,
            dtype_kv=dtype,
            dtype_o=dtype,
            qk_head_dim=(qk_head_dim if attn_kind_single == "mha" else mla_original_qk_head_dim),
            v_head_dim=(v_head_dim if attn_kind_single == "mha" else mla_original_v_head_dim),
            enable_inline_rope=False,
            return_static_libs=True,
        )
        flashinfer_decode_mods = (
            rx.backend.cuda.flashinfer.gen_flashinfer_decode_module(
                dtype_q=dtype,
                dtype_kv=dtype,
                dtype_o=dtype,
                qk_head_dim=qk_head_dim,
                v_head_dim=v_head_dim,
                enable_inline_rope=False,
                return_static_libs=True,
            )
            if attn_kind_single == "mha"
            else []
        )
        flashinfer_mla_mods = (
            rx.backend.cuda.flashinfer.gen_flashinfer_mla_module(
                dtype_q=dtype,
                dtype_kv=dtype,
                dtype_o=dtype,
                head_dim_ckv=v_head_dim,
                head_dim_kpe=qk_head_dim - v_head_dim,
                return_static_libs=True,
            )
            if attn_kind_single == "mla"
            else []
        )
        self.extern_mods = flashinfer_prefill_mods + flashinfer_decode_mods + flashinfer_mla_mods

        # fmt: off
        # pylint: disable=line-too-long
        bb = rx.BlockBuilder.current()
        mha_functions = (
            [
                rx.Tuple([rx.StringImm("flashinfer"), rx.ExternFunc("batch_prefill_paged_run"), rx.ExternFunc("batch_prefill_plan")]),
                rx.Tuple([rx.StringImm("flashinfer"), rx.ExternFunc("batch_decode_run"), rx.ExternFunc("batch_decode_plan")]),
                rx.Tuple([rx.StringImm("tir"), bb.add_func(_attention_prefill(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, True, rope_scaling, target), "tir_attention_prefill_sliding_window")]),
                rx.Tuple([rx.StringImm("tir"), bb.add_func(_attention_decode(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, True, rope_scaling, target), "tir_attention_decode_sliding_window")]),
                rx.Tuple([rx.StringImm("tir"), bb.add_func(tree_attn_with_paged_kv_cache(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, rope_scaling, target), "tir_attention_prefill_with_tree_mask_with_paged_kv_cache")]),
                rx.Tuple([rx.StringImm("tir"), bb.add_func(tree_attn(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, rope_scaling, target), "tir_attention_prefill_with_tree_mask")]),
            ]
            if attn_kind_single == "mha"
            else [rx.Tuple([]) for _ in range(6)]
        )
        ragged_prefill_function = rx.Tuple([rx.StringImm("flashinfer"), rx.ExternFunc("batch_prefill_ragged_run"), rx.ExternFunc("batch_prefill_plan")]) if attn_kind_single == "mha" else rx.Tuple([rx.StringImm("flashinfer"), rx.ExternFunc("batch_prefill_ragged_run"), rx.ExternFunc("batch_prefill_plan"), rx.PrimValue(mla_original_qk_head_dim), rx.PrimValue(mla_original_v_head_dim)])
        mla_function = rx.Tuple([rx.StringImm("flashinfer"), rx.ExternFunc("batch_mla_run"), rx.ExternFunc("batch_mla_plan")] if attn_kind_single == "mla" else [])
        attn_merge_functions = [
            bb.add_func(_merge_state_inplace(num_attention_heads, v_head_dim, dtype, target, "tir_attention_merge_state"), "tir_attention_merge_state"),
        ]
        if attn_kind_single == "mla":
            attn_merge_functions.append(bb.add_func(_merge_state_inplace(num_attention_heads, mla_original_v_head_dim, dtype, target, "tir_attention_merge_state_mla"), "tir_attention_merge_state_mla"))

        if isinstance(attn_kind, list):
            attn_kind = [int(getattr(AttnKind, layer_kind.upper())) for layer_kind in attn_kind]
        else:
            attn_kind = [int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]

        args = [
            rx.ShapeExpr(
                [
                    max_batch_size,
                    max_total_seq_len,
                    prefill_chunk_size,
                    page_size,
                    support_sliding_window,
                ]
            ),
            layer_partition,
            rx.PrimValue(num_attention_heads),
            rx.PrimValue(num_key_value_heads),
            rx.PrimValue(qk_head_dim),
            rx.PrimValue(v_head_dim),
            rx.ShapeExpr(attn_kind),
            rx.PrimValue(enable_disaggregation),
            rx.PrimValue(rope_mode),
            rx.PrimValue(rope_scale),
            rx.PrimValue(rope_theta),
            rope_ext_factors,
            rx.op.zeros((), dtype),
            bb.add_func(_kv_cache_transpose_append(num_key_value_heads, qk_head_dim, dtype), "kv_cache_transpose_append"),
            bb.add_func(_kv_cache_transpose_append_mla(qk_head_dim, dtype), "kv_cache_transpose_append_mla"),
            ragged_prefill_function,
            *mha_functions,
            mla_function,
            rx.Tuple(attn_merge_functions),
            bb.add_func(llama_rope_with_position_map(rope_theta, rope_scale, qk_head_dim, num_attention_heads, num_key_value_heads, dtype, rope_scaling, rotary_dim), "tir_split_rotary"),
            bb.add_func(_copy_single_page(num_key_value_heads, page_size, qk_head_dim, dtype, target) if attn_kind_single == "mha" else _copy_single_page_mla(page_size, qk_head_dim, dtype, target), "kv_cache_copy_single_page"),
            bb.add_func(_kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, qk_head_dim, dtype), "kv_cache_debug_get_kv"),
            bb.add_func(_compact_kv_copy(num_key_value_heads, qk_head_dim, dtype, target), "kv_cache_compact_kv_copy"),
            # pylint: enable=line-too-long
        ]
        super().__init__(
            _expr=rx.call_pure_packed(
                "vm.builtin.paged_attention_kv_cache_create",
                *args,
                sinfo_args=rx.ObjectStructInfo(),
            ),
            _name=name,
        )


class TIRPagedKVCache(PagedKVCache):  # pylint: disable=too-few-public-methods
    """Paged KV cache using TIR kernels."""

    def __init__(  # pylint: disable=too-many-locals
        self,
        attn_kind: Literal["mha", "mla"] | list[Literal["mha", "mla", "mha_sliding"]],
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
        layer_partition: rx.ShapeExpr,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        mla_original_qk_head_dim: int,
        mla_original_v_head_dim: int,
        rope_mode: RopeMode,
        rope_scale: int,
        rope_theta: int,
        rope_scaling: dict[str, Any],
        rope_ext_factors: rx.Expr,
        rotary_dim: int,
        enable_disaggregation: bool,
        dtype: str,
        target: Target,
        name: str = "paged_kv_cache",
    ) -> None:
        """Create a paged KV cache object with TIR kernels.

        Parameters
        ----------
        max_batch_size : tir.Var
            The maximum allowed batch size of the KV cache.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        max_total_seq_len : tir.Var
            The maximum allowed total sequence length of the KV cache.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        prefill_chunk_size : tir.Var
            The maximum total sequence length in a prefill.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        page_size : tir.Var
            The size (a.k.a. number of tokens) of each page.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        support_sliding_window : tir.Var
            0 or 1, denoting whether the KV cache supports sliding window.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        layer_partition : rx.ShapeExpr
            The KV cache layer partition for pipeline stages.
            It is an indptr array, denoting the starting layer of each pipeline stage.
        rope_mode : RopeMode
            The RoPE mode of the Paged KV cache.
            If it is normal, RoPE will be applied to k before adding k to cache.
            Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
        rope_scale : int
            The scale of rotary position embedding.
        rope_theta : int
            The base of rotary position embedding.
        rope_scaling: Dict[str, Any]
            The RoPE scaling information dict.
        rope_ext_factors: rx.Expr
            The RoPE extension factors when "longrope" mode RoPE scaling is enabled.
        rotary_dim : int
            The number of dimensions in the embedding that RoPE is applied to.
        enable_disaggregation : bool
            Whether to enable disaggregation in the KV cache.
        target : Target
            The target to build the model to.
        """
        rope_scaling = _prepare_yarn_rope_scaling(rope_scaling, rope_theta)
        attn_kind_single = attn_kind[0] if isinstance(attn_kind, list) else attn_kind
        if attn_kind_single == "mha_sliding":
            attn_kind_single = "mha"
        if isinstance(attn_kind, list):
            attn_kind = [int(getattr(AttnKind, layer_kind.upper())) for layer_kind in attn_kind]
        else:
            attn_kind = [
                int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)
            ]
        bb = rx.BlockBuilder.current()
        args = [
            rx.ShapeExpr(
                [
                    max_batch_size,
                    max_total_seq_len,
                    prefill_chunk_size,
                    page_size,
                    support_sliding_window,
                ]
            ),
            layer_partition,
            rx.PrimValue(num_attention_heads),
            rx.PrimValue(num_key_value_heads),
            rx.PrimValue(qk_head_dim),
            rx.PrimValue(v_head_dim),
            rx.ShapeExpr(attn_kind),
            rx.PrimValue(enable_disaggregation),
            rx.PrimValue(rope_mode),
            rx.PrimValue(rope_scale),
            rx.PrimValue(rope_theta),
            rope_ext_factors,
            rx.op.zeros((), dtype),
            # pylint: disable=line-too-long
            bb.add_func(
                _kv_cache_transpose_append(num_key_value_heads, qk_head_dim, dtype),
                "kv_cache_transpose_append",
            ),
            bb.add_func(
                _kv_cache_transpose_append_mla(qk_head_dim, dtype),
                "kv_cache_transpose_append_mla",
            ),
            # pylint: enable=line-too-long
        ]

        if str(target.kind) == "llvm":
            if attn_kind_single == "mla":
                raise ValueError("MLA is not supported in TIR kernels for now.")
            # pylint: disable=line-too-long
            # fmt: off
            args.extend(
                [
                    rx.Tuple([rx.StringImm("tir"), bb.add_func(_attention_prefill_ragged_cpu(num_key_value_heads, num_attention_heads, qk_head_dim, v_head_dim, dtype, rope_scaling), "tir_attention_prefill_ragged_cpu")]),
                    rx.Tuple([rx.StringImm("tir"), bb.add_func(_attention_prefill_cpu(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, False, rope_scaling), "tir_attention_prefill_cpu")]),
                    rx.Tuple([rx.StringImm("tir"), bb.add_func(_attention_decode_cpu(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, False, rope_scaling), "tir_attention_decode_cpu")]),
                    rx.Tuple([rx.StringImm("tir"), bb.add_func(_attention_prefill_cpu(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, True, rope_scaling), "tir_attention_prefill_cpu_sliding_window")]),
                    rx.Tuple([rx.StringImm("tir"), bb.add_func(_attention_decode_cpu(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, True, rope_scaling), "tir_attention_decode_cpu_sliding_window")]),
                    rx.Tuple([rx.StringImm("tir"), bb.add_func(tree_attn_cpu(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, rope_scaling), "tir_attention_prefill_with_tree_mask_cpu")]),
                    rx.Tuple([rx.StringImm("tir"), bb.add_func(tree_attn_with_paged_kv_cache_cpu(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, rope_scaling), "tir_attention_prefill_with_tree_mask_with_paged_kv_cache_cpu")]),
                    rx.Tuple([]),  # f_mla_prefill
                    rx.Tuple([bb.add_func(_merge_state_inplace_cpu(dtype), "tir_attention_merge_state_cpu")]),
                    bb.add_func(llama_rope_with_position_map(rope_theta, rope_scale, qk_head_dim, num_attention_heads, num_key_value_heads, dtype, rope_scaling, rotary_dim), "tir_split_rotary"),
                    bb.add_func(_copy_single_page_cpu(num_key_value_heads, page_size, qk_head_dim, dtype), "kv_cache_copy_single_page_cpu"),
                    bb.add_func(_kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, qk_head_dim, dtype), "kv_cache_debug_get_kv"),
                    bb.add_func(_compact_kv_copy_cpu(num_key_value_heads, qk_head_dim, dtype), "kv_cache_compact_kv_copy_cpu"),
                ]
            )
            # fmt: on
            # pylint: enable=line-too-long
        else:
            # pylint: disable=line-too-long
            # fmt: off
            ragged_qk_head_dim = qk_head_dim if attn_kind_single == "mha" else mla_original_qk_head_dim
            ragged_v_head_dim = v_head_dim if attn_kind_single == "mha" else mla_original_v_head_dim
            args.append(rx.Tuple([rx.StringImm("tir"), bb.add_func(_attention_prefill_ragged(num_key_value_heads if attn_kind_single == "mha" else num_attention_heads, num_attention_heads, ragged_qk_head_dim, ragged_v_head_dim, dtype, rope_scaling, target), "tir_attention_prefill_ragged")]))
            mha_functions = (
                [
                    rx.Tuple([rx.StringImm("tir"), bb.add_func(_attention_prefill(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, False, rope_scaling, target), "tir_attention_prefill")]),
                    rx.Tuple([rx.StringImm("tir"), bb.add_func(_attention_decode(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, False, rope_scaling, target), "tir_attention_decode")]),
                    rx.Tuple([rx.StringImm("tir"), bb.add_func(_attention_prefill(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, True, rope_scaling, target), "tir_attention_prefill_sliding_window")]),
                    rx.Tuple([rx.StringImm("tir"), bb.add_func(_attention_decode(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, True, rope_scaling, target), "tir_attention_decode_sliding_window")]),
                    rx.Tuple([rx.StringImm("tir"), bb.add_func(tree_attn_with_paged_kv_cache(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, rope_scaling, target), "tir_attention_prefill_with_tree_mask_with_paged_kv_cache")]),
                    rx.Tuple([rx.StringImm("tir"), bb.add_func(tree_attn(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, rope_scaling, target), "tir_attention_prefill_with_tree_mask")]),
                ]
                if attn_kind_single == "mha"
                else [rx.Tuple([]) for _ in range(6)]
            )
            mla_function = rx.Tuple([rx.StringImm("tir"), bb.add_func(_attention_prefill_mla(num_attention_heads, v_head_dim, qk_head_dim - v_head_dim, dtype, False, target), "tir_attention_prefill_mla")] if attn_kind_single == "mla" else [])
            attn_merge_functions = [
                bb.add_func(_merge_state_inplace(num_attention_heads, v_head_dim, dtype, target, "tir_attention_merge_state"), "tir_attention_merge_state"),
            ]
            if attn_kind_single == "mla":
                attn_merge_functions.append(bb.add_func(_merge_state_inplace(num_attention_heads, mla_original_v_head_dim, dtype, target, "tir_attention_merge_state_mla"), "tir_attention_merge_state_mla"))
            args.extend(mha_functions)
            args.append(mla_function)
            args.extend(
                [
                    rx.Tuple(attn_merge_functions),
                    bb.add_func(llama_rope_with_position_map(rope_theta, rope_scale, qk_head_dim, num_attention_heads, num_key_value_heads, dtype, rope_scaling, rotary_dim), "tir_split_rotary"),
                    bb.add_func(_copy_single_page(num_key_value_heads, page_size, qk_head_dim, dtype, target) if attn_kind_single == "mha" else _copy_single_page_mla(page_size, qk_head_dim, dtype, target), "kv_cache_copy_single_page"),
                    bb.add_func(_kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, qk_head_dim, dtype), "kv_cache_debug_get_kv"),
                    bb.add_func(_compact_kv_copy(num_key_value_heads, qk_head_dim, dtype, target), "kv_cache_compact_kv_copy"),
                ]
            )
            # fmt: on
            # pylint: enable=line-too-long

        super().__init__(
            _expr=rx.call_pure_packed(
                "vm.builtin.paged_attention_kv_cache_create",
                *args,
                sinfo_args=rx.ObjectStructInfo(),
            ),
            _name=name,
        )


# mypy: disable-error-code="attr-defined,valid-type,no-redef"
# pylint: disable=too-many-locals


def _kv_cache_transpose_append(num_key_value_heads, head_dim, dtype, page_size: int = 16):
    """Return the TIR function that appends new k/v data to PagedKVCache."""

    # pylint: disable=line-too-long
    # fmt: off
    @T.prim_func
    def tir_kv_cache_transpose_append(
        var_pages: T.handle,
        var_k_data: T.handle,
        var_v_data: T.handle,
        var_position_map: T.handle,
    ):
        T.func_attr({"tir.noalias": True})
        ntoken = T.SizeVar("num_tokens_excluding_cache", "int64")
        num_pages = T.int64()
        pages_elem_offset = T.int64()
        position_map_elem_offset = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_key_value_heads, page_size, head_dim), dtype, elem_offset=pages_elem_offset)
        k_data = T.match_buffer(var_k_data, (ntoken, num_key_value_heads, head_dim), dtype)
        v_data = T.match_buffer(var_v_data, (ntoken, num_key_value_heads, head_dim), dtype)
        position_map = T.match_buffer(
            var_position_map, (ntoken,), "int32", elem_offset=position_map_elem_offset
        )
        for global_pos, h, f in T.grid(ntoken, num_key_value_heads, head_dim):
            if position_map[global_pos] != T.int32(-1):
                with T.sblock("k_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                    T.writes(pages[position_map[vgpos] // page_size, 0, vh, position_map[vgpos] % page_size, vf])
                    position: T.int32 = position_map[vgpos]  # type: ignore
                    pages[T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vf] = k_data[vgpos, vh, vf]
                with T.sblock("v_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], v_data[vgpos, vh, vf])
                    T.writes(pages[position_map[vgpos] // page_size, 1, vh, position_map[vgpos] % page_size, vf])
                    position: T.int32 = position_map[vgpos] # type: ignore[name-defined,no-redef]
                    pages[T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vf] = v_data[vgpos, vh, vf]
    # fmt: on
    # pylint: enable=line-too-long

    return tir_kv_cache_transpose_append


def _kv_cache_transpose_append_mla(d_qk: int, dtype, page_size: int = 16):
    """Return the TIR function that appends new compressed KV data to PagedKVCache for MLA."""

    # pylint: disable=line-too-long
    # fmt: off
    @T.prim_func
    def tir_kv_cache_transpose_append_mla(
        var_pages: T.handle,
        var_kv_data: T.handle,
        var_position_map: T.handle,
    ):
        T.func_attr({"tir.noalias": True})
        ntoken = T.SizeVar("num_tokens_excluding_cache", "int64")
        num_pages = T.int64()
        pages_elem_offset = T.int64()
        position_map_elem_offset = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, page_size, d_qk), dtype, elem_offset=pages_elem_offset)
        kv_data = T.match_buffer(var_kv_data, (ntoken, d_qk), dtype)
        position_map = T.match_buffer(
            var_position_map, (ntoken,), "int32", elem_offset=position_map_elem_offset
        )
        for global_pos, f in T.grid(ntoken, d_qk):
            if position_map[global_pos] != T.int32(-1):
                with T.sblock("k_transpose_append"):
                    vgpos, vf = T.axis.remap("SS", [global_pos, f])
                    T.reads(position_map[vgpos], kv_data[vgpos, vf])
                    T.writes(pages[position_map[vgpos] // page_size, position_map[vgpos] % page_size, vf])
                    position: T.int32 = position_map[vgpos]  # type: ignore
                    pages[T.floordiv(position, page_size), T.floormod(position, page_size), vf] = kv_data[vgpos, vf]
    # fmt: on
    # pylint: enable=line-too-long

    return tir_kv_cache_transpose_append_mla


def _kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, head_dim, dtype):
    """Return the TIR function that fetches the k/v data on given positions and layer."""

    # pylint: disable=line-too-long
    # fmt: off
    @T.prim_func
    def tir_kv_cache_debug_get_kv(
        var_pages: T.handle,
        var_position_map: T.handle,
        var_k_data: T.handle,
        var_v_data: T.handle,
        layer_id: T.int64,
    ):
        T.func_attr({"tir.noalias": True})
        seqlen = T.SizeVar("num_tokens_including_cache", "int64")
        page_size = T.SizeVar("page_size", "int64")
        num_pages = T.int64()
        pages_elem_offset = T.int64()
        position_map_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_key_value_heads, page_size, head_dim), dtype,elem_offset=pages_elem_offset)
        position_map = T.match_buffer(
            var_position_map, (seqlen,), "int32", elem_offset=position_map_elem_offset
        )
        k_data = T.match_buffer(var_k_data, (num_hidden_layers, seqlen, num_key_value_heads, head_dim), dtype)
        v_data = T.match_buffer(var_v_data, (num_hidden_layers, seqlen, num_key_value_heads, head_dim), dtype)
        for p, h, d in T.grid(seqlen, num_key_value_heads, head_dim):
            with T.sblock("copy0"):
                vp, vh, vd = T.axis.remap("SSS", [p, h, d])
                T.reads(position_map[vp], pages[position_map[vp] // page_size, 0:2, vh, position_map[vp] % page_size, vd])
                T.writes(k_data[layer_id, vp, vh, vd], v_data[layer_id, vp, vh, vd])
                position: T.int32 = position_map[vp] # type: ignore[name-defined]
                k_data[layer_id, vp, vh, vd] = pages[T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vd]
                v_data[layer_id, vp, vh, vd] = pages[T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vd]
    # fmt: on
    # pylint: enable=line-too-long

    return tir_kv_cache_debug_get_kv


def _kv_cache_debug_get_kv_mla(num_hidden_layers, d_qk, dtype):
    """Return the TIR function that fetches the k/v data on given positions and layer."""

    # pylint: disable=line-too-long
    # fmt: off
    @T.prim_func
    def tir_kv_cache_debug_get_kv_mla(
        var_pages: T.handle,
        var_position_map: T.handle,
        var_compressed_kv_with_k_pe_data: T.handle,
        layer_id: T.int64,
    ):
        T.func_attr({"tir.noalias": True})
        seqlen = T.SizeVar("num_tokens_including_cache", "int64")
        page_size = T.SizeVar("page_size", "int64")
        num_pages = T.int64()
        pages_elem_offset = T.int64()
        position_map_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, page_size, d_qk), dtype, elem_offset=pages_elem_offset)
        position_map = T.match_buffer(
            var_position_map, (seqlen,), "int32", elem_offset=position_map_elem_offset
        )
        compressed_kv_with_k_pe_data = T.match_buffer(var_compressed_kv_with_k_pe_data, (num_hidden_layers, seqlen, d_qk), dtype)
        for p, d in T.grid(seqlen, d_qk):
            with T.sblock("copy0"):
                vp, vd = T.axis.remap("SS", [p, d])
                T.reads(position_map[vp], pages[position_map[vp] // page_size, position_map[vp] % page_size, vd])
                T.writes(compressed_kv_with_k_pe_data[layer_id, vp, vd])
                position: T.int32 = position_map[vp] # type: ignore[name-defined]
                compressed_kv_with_k_pe_data[layer_id, vp, vd] = pages[T.floordiv(position, page_size), T.floormod(position, page_size), vd]
    # fmt: on
    # pylint: enable=line-too-long

    return tir_kv_cache_debug_get_kv_mla


def _rope(
    buffer: T.Buffer,
    offset: tir.Var,
    rotary_dim: int,
    theta: tir.Var,
    scale: tir.Var,
    indices: tuple[tir.Var, ...],
    qkv_dtype: str,
    rope_scaling: dict[str, Any],
):
    d = indices[-1]
    cos_freq, sin_freq, var_map = switch_rope_freq_func(rope_scaling)(
        offset * scale, d, rotary_dim, theta, "float32"
    )
    cos = cos_freq * buffer[indices].astype("float32")
    sin = sin_freq * tir.if_then_else(
        d < rotary_dim // 2,
        -buffer[indices[:-1] + (d + rotary_dim // 2,)],
        buffer[indices[:-1] + (d - rotary_dim // 2,)],
    ).astype("float32")
    expr = (cos + sin).astype(qkv_dtype)
    for var, value in var_map.items():
        expr = tir.Let(var, value, expr)
    return expr


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


def _get_seq_offset(pos, seq_id, length_info, sliding_window):
    if not sliding_window:
        return pos
    # pos if pos < sink_size else pos - sink_size + sliding_window_offset
    return T.if_then_else(
        pos < length_info[2, seq_id],
        pos,
        pos - length_info[2, seq_id] + length_info[1, seq_id],
    )


def _attention_prefill_cpu(
    h_kv,
    h_q,
    d,
    dtype,
    sliding_window: bool,
    rope_scaling: dict[str, Any],
    page_size: int = 16,
):
    """CPU batched-prefill paged-KV kernel. Delegates to C++ via FFI."""
    return _ffi_kv_cache.attention_prefill_cpu(
        h_kv, h_q, d, dtype, sliding_window, rope_scaling, page_size,
    )


def _get_prefill_kernel_config(h_kv, h_q, d, dtype, target: Target):
    NUM_BLKS = 16
    LOAD_VEC = 8 // ((DataType(dtype).bits + 7) // 8)  # 8 bytes
    group_size = h_q // h_kv

    bdx = 32
    num_warps = 4
    tile_x, tile_y, tile_z = (
        64 // ((DataType(dtype).bits + 7) // 8) // max(d // 128, 1),
        d,
        64 // ((DataType(dtype).bits + 7) // 8) // max(d // 128, 1),
    )
    original_tile_y = tile_y
    original_tile_z = tile_z
    while (tile_x * tile_z) % (bdx * num_warps) != 0:
        tile_z += original_tile_z
    while (tile_x * tile_y) % (bdx * num_warps) != 0:
        tile_y += original_tile_y

    # Otherwise we would exceed maxComputeWorkgroupStorageSize
    if (
        str(target.kind) == "webgpu"
        and ((d + 127) // 128) * ((DataType(dtype).bits + 15) // 16) >= 4
    ):
        tile_z = 8
        num_warps = 2

    if ((target.kind.name == "opencl") or (target.kind.name == "vulkan")) and (
        ("android" in str(target.host)) or ("adreno" in str(target.keys))
    ):
        if target.kind.name == "opencl":
            LOAD_VEC = 16 // ((DataType(dtype).bits + 7) // 8)  # 16 bytes
        NUM_BLKS = group_size * 8
        tile_x = 32
        tile_z = 4
        if (tile_y * tile_z) % (bdx * num_warps) != 0:
            tile_z = 16

    check_thread_limits(target, bdx=bdx, bdy=num_warps, bdz=1, gdz=1)

    return NUM_BLKS, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z


def _schedule_prefill_kernel(
    sch: s_tir.Schedule,
    load_vec,
    bdx,
    num_warps,
    tile_x,
    tile_y,
    tile_z,
    transform_k_load: bool,
    merged_qk_load: bool,
) -> tvm.s_tir.Schedule:
    get_extent = lambda *lps: [int(sch.get(lp).extent) for lp in lps]

    def get_vecsize(extent):
        return min(load_vec, (extent & ~(extent - 1)))

    def getxy_vecsize(x, y, t):
        assert (x * y) % t == 0
        return min(get_vecsize(y), get_vecsize(x * y // t))

    def get_tile_size(x, y, t):
        cnt = (x * y) // t
        assert (x * y) % t == 0
        tile_y = math.ceil(math.sqrt(cnt))
        while (cnt % tile_y != 0 or y % tile_y != 0 or x % (cnt // tile_y) != 0) and tile_y <= cnt:
            tile_y += 1
        assert tile_y <= cnt
        tile_x = cnt // tile_y
        return tile_x, tile_y

    def apply_to_qkv_load(sch: s_tir.Schedule, block):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        x_extent, y_extent = get_extent(loop_x, loop_y)
        vec_size = getxy_vecsize(x_extent, y_extent, bdx * num_warps)
        yo, yv = sch.split(loop_y, [None, vec_size])
        yo_extent = y_extent // vec_size
        tile_x, tile_y = get_tile_size(x_extent, yo_extent, (bdx * num_warps))
        xo, xi = sch.split(loop_x, [tile_x, None])
        yo, yi = sch.split(yo, [tile_y, None])
        sch.reorder(xi, yi, xo, yo)
        t = sch.fuse(xi, yi)
        ty, tx = sch.split(t, [num_warps, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(yv)

    def apply_to_so_ewise(sch: s_tir.Schedule, block, tile):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        yiv_extent = get_vecsize(tile[1])
        yio, yiv = sch.split(yi, [None, yiv_extent])
        sch.unroll(yio)
        sch.vectorize(yiv)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    def apply_to_gemm(sch: s_tir.Schedule, block, tile, r_len=16, k_major=False):
        loop_x, loop_y, loop_z = sch.get_loops(block)[-3:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        ko, ki = sch.split(loop_z, factors=[None, r_len])
        if k_major:
            sch.reorder(ko, xi, yi, ki)
        else:
            sch.reorder(ko, ki, xi, yi)
        yiv_extent = get_vecsize(tile[1])
        yio, yiv = sch.split(yi, [None, yiv_extent])
        sch.unroll(yio)
        sch.vectorize(yiv)
        sch.unroll(xi)
        sch.decompose_reduction(block, ty)

    def apply_to_md(sch, block):
        loop = sch.get_loops(block)[-1]
        _, ty, tx = sch.split(loop, factors=[None, num_warps, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    if transform_k_load and not merged_qk_load:
        sch.transform_layout("K_load", ("write", 0), lambda i, j: (j, i))
    tile_s = get_tile_size(tile_x, tile_z, bdx * num_warps)
    tile_o = get_tile_size(tile_x, tile_y, bdx * num_warps)
    apply_to_gemm(sch, sch.get_sblock("S_gemm"), tile_s, k_major=True)
    apply_to_gemm(sch, sch.get_sblock("O_gemm"), tile_o, k_major=False)
    apply_to_so_ewise(sch, sch.get_sblock("S_store"), tile_s)
    apply_to_so_ewise(sch, sch.get_sblock("O_init"), tile_o)
    apply_to_so_ewise(sch, sch.get_sblock("O_store"), tile_o)
    apply_to_qkv_load(sch, sch.get_sblock("Q_load"))
    if not merged_qk_load:
        apply_to_qkv_load(sch, sch.get_sblock("K_load"))
        apply_to_qkv_load(sch, sch.get_sblock("V_load"))
    else:
        apply_to_qkv_load(sch, sch.get_sblock("KV_load"))
    apply_to_md(sch, sch.get_sblock("lse_store"))
    return sch


def _attention_prefill(
    h_kv,
    h_q,
    d,
    dtype,
    sliding_window: bool,
    rope_scaling: dict[str, Any],
    target: Target,
    page_size: int = 16,
):
    """GPU batched-prefill paged-KV kernel. Delegates to C++ via FFI."""
    return _ffi_kv_cache.attention_prefill(
        h_kv, h_q, d, dtype, sliding_window, rope_scaling, target, page_size,
    )


def _attention_decode_cpu(
    num_kv_heads,
    num_qo_heads,
    head_dim,
    qkv_dtype,
    sliding_window: bool,
    rope_scaling: dict[str, Any],
    page_size: int = 16,
):
    H_qo = num_qo_heads
    H_kv = num_kv_heads
    D = head_dim
    group_size = num_qo_heads // num_kv_heads

    global_symbol = "batch_decode_paged_kv_cpu"
    if sliding_window:
        global_symbol += "_sliding_window"

    # fmt: off
    # pylint: disable=line-too-long
    @T.prim_func(check_well_formed=False)
    def batch_decode_paged_kv(
        Q_handle: T.handle,
        pages_handle: T.handle,
        page_table_indptr_handle: T.handle,
        page_table_values_handle: T.handle,
        var_length_info: T.handle,  # [b] when sliding window = False, or otherwise [3, b]
        k_rope_pos_offset_handle: T.handle,
        q_rope_position_handle: T.handle,
        output_handle: T.handle,
        lse_handle: T.handle,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):
        T.func_attr({"tir.is_scheduled": True, "global_symbol": global_symbol})
        B = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)
        page_indptr_elem_offset = T.int32(is_size_var=True)
        page_values_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        length_info_elem_offset = T.int32(is_size_var=True)

        Q = T.match_buffer(Q_handle, (B, H_qo, D), qkv_dtype)
        pages = T.match_buffer(pages_handle, (max_num_pages, 2, H_kv, page_size, D), qkv_dtype)
        page_table_indptr = T.match_buffer(
            page_table_indptr_handle, (B + 1,), "int32", elem_offset=page_indptr_elem_offset
        )
        page_table_values = T.match_buffer(
            page_table_values_handle, (nnz_pages,), "int32", elem_offset=page_values_elem_offset
        )
        k_rope_pos_offset = T.match_buffer(
            k_rope_pos_offset_handle, (B,), "int32", elem_offset=k_rope_pos_offset_elem_offset
        )
        q_rope_position = T.match_buffer(
            q_rope_position_handle, (B,), "int32", elem_offset=q_rope_position_elem_offset
        )
        output = T.match_buffer(output_handle, (B, H_qo, D), qkv_dtype)
        lse = T.match_buffer(lse_handle, (B, H_qo), "float32")  # pylint: disable=unused-variable
        # The length information of the sequences.
        # - It is in shape `(3, batch_size)` when sliding window is enabled.
        #   For a sequence "i", location
        #   - "(0, i)" is the number of KV slots used in the last page of the seq ("last_page_len"),
        #   - "(1, i)" is the starting offset of the sliding window in the seq,
        #   - "(2, i)" is the attn sink length of the sequence.
        # - It is in shape `(batch_size,)` when sliding window is disabled,
        #   denoting the "last_page_len".
        length_info = _declare_length_info(
            var_length_info, B, sliding_window, length_info_elem_offset
        )

        for b in T.serial(B):
            with T.sblock("attn"):
                O_local = T.alloc_buffer((D,), "float32")
                Q_local = T.alloc_buffer((D,), "float32")
                K_local = T.alloc_buffer((D,), "float32")
                V_local = T.alloc_buffer((D,), "float32")

                kv_chunk_len = T.alloc_buffer((1,), "int32")

                m_val = T.alloc_buffer((1,), "float32")
                new_m = T.alloc_buffer((1,), "float32")
                d_val = T.alloc_buffer((1,), "float32")
                S_val = T.alloc_buffer((1,), "float32")
                scale_O = T.alloc_buffer((1,), "float32")
                factor = T.alloc_buffer((1,), "float32")

                cur_page_indptr_begin: T.int32 = page_table_indptr[b]
                cur_page_indptr_end: T.int32 = page_table_indptr[b + 1]

                kv_chunk_len[0] = T.if_then_else(
                    cur_page_indptr_begin != cur_page_indptr_end,
                    _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, page_size, b, length_info, sliding_window),
                    0,
                )

                for h_qo in T.serial(H_qo):
                    m_val[0] = -5e4
                    d_val[0] = 1.0

                    for d in T.serial(D):
                        O_local[d] = 0.0

                    for d in T.serial(D):
                        Q_local[d] = T.if_then_else(
                            rotary_mode == 1,
                            _rope(Q, q_rope_position[b], head_dim, rope_theta, rope_scale, (b, h_qo, d), qkv_dtype, rope_scaling),
                            Q[b, h_qo, d],
                        )

                    for row_idx in T.serial(kv_chunk_len[0]):
                        seq_offset: T.int32(is_size_var=True) = _get_seq_offset(row_idx, b, length_info, sliding_window)
                        page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + (seq_offset // page_size)]
                        page_offset: T.int32(is_size_var=True) = seq_offset % page_size

                        for d in T.serial(D):
                            K_local[d] = T.if_then_else(
                                rotary_mode == 1,
                                _rope(pages, k_rope_pos_offset[b] + row_idx, head_dim, rope_theta, rope_scale, (page_no, 0, h_qo // group_size, page_offset, d), qkv_dtype, rope_scaling),
                                pages[page_no, 0, h_qo // group_size, page_offset, d],
                            )
                        S_val[0] = 0.0
                        for d in T.serial(D):
                            S_val[0] += Q_local[d] * K_local[d]
                        S_val[0] *= sm_scale * math.log2(math.exp(1))

                        new_m[0] = T.max(m_val[0], S_val[0])
                        d_val[0] = (d_val[0] * T.exp2(m_val[0] - new_m[0])) + T.exp2(
                            S_val[0] - new_m[0]
                        )

                        scale_O[0] = T.exp2(m_val[0] - new_m[0])

                        for d in T.serial(D):
                            O_local[d] = O_local[d] * scale_O[0]

                        m_val[0] = new_m[0]
                        for d in T.serial(D):
                            V_local[d] = pages[page_no, 1, h_qo // group_size, page_offset, d]

                        factor[0] = T.exp2(S_val[0] - m_val[0])
                        for d in T.serial(D):
                            O_local[d] = O_local[d] + V_local[d] * factor[0]
                    for d in T.serial(D):
                        O_local[d] = O_local[d] / d_val[0]
                        output[b, h_qo, d] = O_local[d]
                    lse[b, h_qo] = m_val[0] + T.log2(d_val[0])
    # fmt: on
    # pylint: enable=line-too-long

    return batch_decode_paged_kv


def _attention_decode(
    num_kv_heads,
    num_qo_heads,
    head_dim,
    qkv_dtype,
    sliding_window: bool,
    rope_scaling: dict[str, Any],
    target: Target,
    page_size: int = 16,
):
    """GPU batched-decode paged-KV kernel. Delegates to C++ via FFI."""
    return _ffi_kv_cache.attention_decode(
        num_kv_heads, num_qo_heads, head_dim, qkv_dtype,
        sliding_window, rope_scaling, target, page_size,
    )

def _merge_state_inplace_cpu(v_dtype):
    @T.prim_func
    def merge_state_inplace_cpu(
        v: T.handle,
        s: T.handle,
        v_other: T.handle,
        s_other: T.handle,
    ):
        T.func_attr({"tir.is_scheduled": True})
        N = T.int32(is_size_var=True)
        H = T.int32(is_size_var=True)
        D = T.int32(is_size_var=True)

        V = T.match_buffer(v, (N, H, D), v_dtype)
        S = T.match_buffer(s, (N, H), "float32")
        V_other = T.match_buffer(v_other, (N, H, D), v_dtype)
        S_other = T.match_buffer(s_other, (N, H), "float32")

        for n in T.serial(N):
            for h in T.serial(H):
                with T.sblock("merge"):
                    s_val = _var_cpu("float32")
                    s_other_val = _var_cpu("float32")
                    s_max = _var_cpu("float32")
                    scale = _var_cpu("float32")
                    other_scale = _var_cpu("float32")

                    s_val[0] = S[n, h]
                    s_other_val[0] = S_other[n, h]
                    s_max[0] = T.max(s_val[0], s_other_val[0])
                    s_val[0] = T.exp2(s_val[0] - s_max[0])
                    s_other_val[0] = T.exp2(s_other_val[0] - s_max[0])
                    scale[0] = s_val[0] / (s_val[0] + s_other_val[0])
                    other_scale[0] = s_other_val[0] / (s_val[0] + s_other_val[0])
                    for d in T.serial(D):
                        V[n, h, d] = V[n, h, d] * scale[0] + V_other[n, h, d] * other_scale[0]
                    S[n, h] = T.log2(s_val[0] + s_other_val[0]) + s_max[0]

    return merge_state_inplace_cpu


def _merge_state_inplace(
    num_heads, head_dim, v_dtype, target: Target, global_symbol: str | None = None
):
    v_dtype_bytes = 2
    VEC_SIZE = min(max(8 // v_dtype_bytes, head_dim // 32), 4)
    bdx = head_dim // VEC_SIZE
    bdy = num_heads
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    while bdx * bdy > max_num_threads_per_block and bdy > 1:
        bdy //= 2
    gdy = num_heads // bdy
    check_thread_limits(target, bdx=bdx, bdy=bdy, bdz=1, gdz=1)

    @T.prim_func
    def merge_state_inplace(
        v: T.handle,
        s: T.handle,
        v_other: T.handle,
        s_other: T.handle,
    ):
        T.func_attr({"tir.is_scheduled": True})
        N = T.int32(is_size_var=True)
        H = T.int32(is_size_var=True)
        D = T.int32(is_size_var=True)

        V = T.match_buffer(v, (N, H, D), v_dtype)
        S = T.match_buffer(s, (N, H), "float32")
        V_other = T.match_buffer(v_other, (N, H, D), v_dtype)
        S_other = T.match_buffer(s_other, (N, H), "float32")

        for bx in T.thread_binding(N, thread="blockIdx.x"):
            for by in T.thread_binding(gdy, thread="blockIdx.y"):
                for ty in T.thread_binding(bdy, thread="threadIdx.y"):
                    for tx in T.thread_binding(bdx, thread="threadIdx.x"):
                        with T.sblock("merge"):
                            s_val = _var("float32")
                            s_other_val = _var("float32")
                            s_max = _var("float32")
                            scale = _var("float32")
                            other_scale = _var("float32")

                            v_vec = T.alloc_buffer((VEC_SIZE,), v_dtype, scope="local")
                            v_other_vec = T.alloc_buffer((VEC_SIZE,), v_dtype, scope="local")

                            s_val[0] = S[bx, ty + by * bdy]
                            s_other_val[0] = S_other[bx, ty + by * bdy]
                            s_max[0] = T.max(s_val[0], s_other_val[0])
                            s_val[0] = T.exp2(s_val[0] - s_max[0])
                            s_other_val[0] = T.exp2(s_other_val[0] - s_max[0])
                            scale[0] = s_val[0] / (s_val[0] + s_other_val[0])
                            other_scale[0] = s_other_val[0] / (s_val[0] + s_other_val[0])

                            # load v
                            for vec in T.vectorized(VEC_SIZE):
                                v_vec[vec] = V[bx, ty + by * bdy, tx * VEC_SIZE + vec]
                            # load v_other
                            for vec in T.vectorized(VEC_SIZE):
                                v_other_vec[vec] = V_other[bx, ty + by * bdy, tx * VEC_SIZE + vec]

                            # merge
                            for vec in T.serial(VEC_SIZE):
                                v_vec[vec] = (
                                    v_vec[vec] * scale[0] + v_other_vec[vec] * other_scale[0]
                                )

                            # store v
                            for vec in T.vectorized(VEC_SIZE):
                                V[bx, ty + by * bdy, tx * VEC_SIZE + vec] = v_vec[vec]

                            # store s
                            S[bx, ty + by * bdy] = T.log2(s_val[0] + s_other_val[0]) + s_max[0]

    func = merge_state_inplace
    if global_symbol:
        func = func.with_attr("global_symbol", global_symbol)
    return func


def _attention_sequence_prefill(h_kv, h_q, d, dtype, target: Target, causal=0, sm_scale=1.0):  # pylint: disable=line-too-long
    """GPU sequence-prefill kernel. Delegates to C++ via FFI."""
    return _ffi_kv_cache.attention_sequence_prefill(
        h_kv, h_q, d, d, dtype, {}, target, causal, sm_scale,
    )


def _attention_prefill_ragged_cpu(h_kv, h_q, d_qk, d_v, dtype, rope_scaling: dict[str, Any]):
    """CPU ragged-prefill paged-KV kernel. Delegates to C++ via FFI."""
    return _ffi_kv_cache.attention_prefill_ragged_cpu(
        h_kv, h_q, d_qk, d_v, dtype, rope_scaling,
    )


def _attention_prefill_ragged(
    h_kv, h_q, d_qk, d_v, dtype, rope_scaling: dict[str, Any], target: Target
):
    """GPU ragged-prefill paged-KV kernel. Delegates to C++ via FFI."""
    return _ffi_kv_cache.attention_prefill_ragged(
        h_kv, h_q, d_qk, d_v, dtype, rope_scaling, target,
    )


def _attention_prefill_mla(
    h_q,
    d_latent,
    d_rope,
    dtype,
    sliding_window: bool,
    target: Target,
    page_size: int = 16,
):
    """GPU MLA prefill kernel. Delegates to C++ via FFI."""
    # C++ signature: (num_heads, v_head_dim, qk_nope_head_dim, dtype, causal_flag, target)
    # sliding_window maps to causal_flag; page_size is baked into the C++ kernel.
    return _ffi_kv_cache.attention_prefill_mla(
        h_q, d_latent, d_rope, dtype, sliding_window, target,
    )


def _copy_single_page(num_heads, page_size, head_dim, dtype, target: Target):
    tx = get_max_num_threads_per_block(target)

    @T.prim_func
    def copy_single_page(
        var_pages: T.handle,
        src_page_id: T.int64,
        tgt_page_id: T.int64,
        copy_length: T.int64,
    ):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(
            var_pages,
            (num_pages, 2, num_heads, page_size, head_dim),
            dtype,
            elem_offset=pages_elem_offset,
        )

        for b in T.thread_binding(
            (copy_length * num_heads * head_dim + tx - 1) // tx, thread="blockIdx.x"
        ):
            for t in T.thread_binding(tx, thread="threadIdx.x"):
                with T.sblock("copy"):
                    T.where(b * tx + t < copy_length * num_heads * head_dim)
                    vh = T.axis.spatial(
                        num_heads,
                        T.Cast("int32", (b * tx + t) // (copy_length * head_dim)),
                    )
                    vp = T.axis.spatial(
                        copy_length,
                        (b * tx + t) % (copy_length * head_dim) // head_dim,
                    )
                    vd = T.axis.spatial(
                        head_dim,
                        T.Cast(
                            "int32",
                            (b * tx + t) % head_dim,
                        ),
                    )
                    pages[tgt_page_id, 0, vh, vp, vd] = pages[src_page_id, 0, vh, vp, vd]
                    pages[tgt_page_id, 1, vh, vp, vd] = pages[src_page_id, 1, vh, vp, vd]

    return copy_single_page


def _copy_single_page_mla(page_size, head_dim, dtype, target: Target):
    tx = get_max_num_threads_per_block(target)

    @T.prim_func
    def copy_single_page_mla(
        var_pages: T.handle,
        src_page_id: T.int64,
        tgt_page_id: T.int64,
        copy_length: T.int64,
    ):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(
            var_pages,
            (num_pages, page_size, head_dim),
            dtype,
            elem_offset=pages_elem_offset,
        )

        for b in T.thread_binding((copy_length * head_dim + tx - 1) // tx, thread="blockIdx.x"):
            for t in T.thread_binding(tx, thread="threadIdx.x"):
                with T.sblock("copy"):
                    T.where(b * tx + t < copy_length * head_dim)
                    vp = T.axis.spatial(copy_length, (b * tx + t) // head_dim)
                    vd = T.axis.spatial(head_dim, T.Cast("int32", (b * tx + t) % head_dim))
                    pages[tgt_page_id, vp, vd] = pages[src_page_id, vp, vd]

    return copy_single_page_mla


def _copy_single_page_cpu(num_heads, page_size, head_dim, dtype):
    tx = 1

    @T.prim_func
    def copy_single_page_cpu(
        var_pages: T.handle,
        src_page_id: T.int64,
        tgt_page_id: T.int64,
        copy_length: T.int64,
    ):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_heads, page_size, head_dim), dtype)

        for b in T.serial((copy_length * num_heads * head_dim + tx - 1) // tx):
            for t in T.serial(tx):
                with T.sblock("copy"):
                    T.where(b * tx + t < copy_length * num_heads * head_dim)
                    vh = T.axis.spatial(
                        num_heads,
                        T.Cast("int32", (b * tx + t) // (copy_length * head_dim)),
                    )
                    vp = T.axis.spatial(
                        copy_length,
                        (b * tx + t) % (copy_length * head_dim) // head_dim,
                    )
                    vd = T.axis.spatial(
                        head_dim,
                        T.Cast(
                            "int32",
                            (b * tx + t) % head_dim,
                        ),
                    )
                    pages[tgt_page_id, 0, vh, vp, vd] = pages[src_page_id, 0, vh, vp, vd]
                    pages[tgt_page_id, 1, vh, vp, vd] = pages[src_page_id, 1, vh, vp, vd]

    return copy_single_page_cpu


def _compact_kv_copy(num_heads, head_dim, dtype, target: Target, page_size: int = 16):
    tx = get_max_num_threads_per_block(target)

    @T.prim_func
    def compact_kv_copy(
        var_pages: T.handle,
        var_copy_length_indptr: T.handle,
        var_copy_src_dst_pos: T.handle,
        batch_size: T.int32,
    ):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        total_copy_length = T.int32()
        copy_length_indptr_elem_offset = T.int32()
        copy_src_dst_pos_elem_offset = T.int32()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(
            var_pages,
            (num_pages, 2, num_heads, page_size, head_dim),
            dtype,
            elem_offset=pages_elem_offset,
        )
        copy_length_indptr = T.match_buffer(
            var_copy_length_indptr,
            (batch_size + 1,),
            "int32",
            elem_offset=copy_length_indptr_elem_offset,
        )
        copy_src_dst_pos = T.match_buffer(
            var_copy_src_dst_pos,
            (2, total_copy_length),
            "int32",
            elem_offset=copy_src_dst_pos_elem_offset,
        )

        with T.sblock("root"):
            for bhd_o in T.thread_binding(
                (batch_size * num_heads * head_dim + tx - 1) // tx, thread="blockIdx.x"
            ):
                for bhd_i in T.thread_binding(tx, thread="threadIdx.x"):
                    b: T.int32 = (bhd_o * tx + bhd_i) // (num_heads * head_dim)
                    h: T.int32 = (bhd_o * tx + bhd_i) // head_dim % num_heads
                    d: T.int32 = (bhd_o * tx + bhd_i) % head_dim
                    if (bhd_o * tx + bhd_i) < batch_size * num_heads * head_dim:
                        for i in T.serial(copy_length_indptr[b + 1] - copy_length_indptr[b]):
                            src_pos: T.int32 = copy_src_dst_pos[0, copy_length_indptr[b] + i]
                            dst_pos: T.int32 = copy_src_dst_pos[1, copy_length_indptr[b] + i]
                            pages[dst_pos // page_size, 0, h, dst_pos % page_size, d] = pages[
                                src_pos // page_size, 0, h, src_pos % page_size, d
                            ]
                            pages[dst_pos // page_size, 1, h, dst_pos % page_size, d] = pages[
                                src_pos // page_size, 1, h, src_pos % page_size, d
                            ]

    return compact_kv_copy


def _compact_kv_copy_cpu(num_heads, head_dim, dtype, page_size: int = 16):
    tx = 8

    @T.prim_func
    def compact_kv_copy_cpu(
        var_pages: T.handle,
        var_copy_length_indptr: T.handle,
        var_copy_src_dst_pos: T.handle,
        batch_size: T.int32,
    ):
        T.func_attr({"tir.is_scheduled": True})
        num_pages = T.int32()
        total_copy_length = T.int32()
        copy_length_indptr_elem_offset = T.int32()
        copy_src_dst_pos_elem_offset = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_heads, page_size, head_dim), dtype)
        copy_length_indptr = T.match_buffer(
            var_copy_length_indptr,
            (batch_size + 1,),
            "int32",
            elem_offset=copy_length_indptr_elem_offset,
        )
        copy_src_dst_pos = T.match_buffer(
            var_copy_src_dst_pos,
            (2, total_copy_length),
            "int32",
            elem_offset=copy_src_dst_pos_elem_offset,
        )

        with T.sblock("root"):
            for bhd_o in T.serial((batch_size * num_heads * head_dim + tx - 1) // tx):
                for bhd_i in T.serial(tx):
                    b: T.int32 = (bhd_o * tx + bhd_i) // (num_heads * head_dim)
                    h: T.int32 = (bhd_o * tx + bhd_i) // head_dim % num_heads
                    d: T.int32 = (bhd_o * tx + bhd_i) % head_dim
                    if (bhd_o * tx + bhd_i) < batch_size * num_heads * head_dim:
                        for i in T.serial(copy_length_indptr[b + 1] - copy_length_indptr[b]):
                            src_pos: T.int32 = copy_src_dst_pos[0, copy_length_indptr[b] + i]
                            dst_pos: T.int32 = copy_src_dst_pos[1, copy_length_indptr[b] + i]
                            pages[dst_pos // page_size, 0, h, dst_pos % page_size, d] = pages[
                                src_pos // page_size, 0, h, src_pos % page_size, d
                            ]
                            pages[dst_pos // page_size, 1, h, dst_pos % page_size, d] = pages[
                                src_pos // page_size, 1, h, src_pos % page_size, d
                            ]

    return compact_kv_copy_cpu
