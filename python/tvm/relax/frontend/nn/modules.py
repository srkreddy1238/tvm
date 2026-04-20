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
# pylint: disable=too-many-arguments,invalid-name
"""Built-in nn.Module subclasses.

Every compute module (ReLU, SiLU, GELU, Linear, Embedding, LayerNorm,
RMSNorm, GroupNorm, Conv1D, Conv2D, Conv3D, ConvTranspose1D) is a native
C++ runtime::Object.  ALL members -- Parameters, scalars, strings -- live
inside the C++ object.  Python holds only the object handle.

__init__ is a single __init_handle_by_constructor__ call to a C++ Make*
factory that allocates Parameters, computes kernel shapes, and returns the
fully-constructed object.

Properties (weight, bias, epsilon, ...) are thin wrappers that read the
corresponding C++ field via __object_handle__.

Pure-Python modules (IOEffect, KVCache, Identity, TimestepEmbedding,
Timesteps, Attention) remain in Python because they manage Python-level
state (BlockBuilder effects, module composition).
"""

from collections.abc import Sequence

import tvm_ffi

from tvm import relax as rx
from tvm import tir

from . import _ffi_api, op
from .core import Effect, Module, ModuleList, Parameter, Tensor, get_default_dtype

# ===========================================================================
# IOEffect  (pure Python)
# ===========================================================================


class IOEffect(Effect):
    """Modeling IO side effect."""

    def __init__(self):
        self.effect = None

    def emit_init(self, name_hint, builder):
        return [builder.emit(rx.op.null_value(), f"{name_hint}.io")]

    def create(self, name_hint):
        self.effect = rx.Var(f"{name_hint}.io", struct_info=rx.ObjectStructInfo())
        return [self.effect]

    def set_state(self, state_vars):
        (self.effect,) = state_vars

    def finalize(self):
        result, self.effect = self.effect, None
        return [result]


# ===========================================================================
# Identity  (pure Python)
# ===========================================================================


class Identity(Module):
    """Pass-through module."""

    def forward(self, x: Tensor) -> Tensor:
        return x


# ===========================================================================
# ReLU
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.ReLU")
class ReLU(Module):
    """ReLU activation."""

    def __init__(self) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ReLU)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self.__object_handle__.forward(x._expr))


# ===========================================================================
# SiLU
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.SiLU")
class SiLU(Module):
    """SiLU activation."""

    def __init__(self) -> None:
        self.__init_handle_by_constructor__(_ffi_api.SiLU)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self.__object_handle__.forward(x._expr))


# ===========================================================================
# GELU
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.GELU")
class GELU(Module):
    """GELU activation."""

    def __init__(self, approximate: str = "") -> None:
        self.__init_handle_by_constructor__(_ffi_api.GELU, approximate)

    @property
    def approximate(self) -> str:
        return str(self.__object_handle__.approximate)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self.__object_handle__.forward(x._expr))


# ===========================================================================
# Linear
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Linear")
class Linear(Module):
    """Linear layer: out = x @ W^T + b.

    All state (weight Parameter, bias Parameter, out_dtype) lives in C++.
    """

    def __init__(
        self,
        in_features: int | str | tir.PrimExpr,
        out_features: int | str | tir.PrimExpr,
        bias: bool = True,
        dtype: str | None = None,
        out_dtype: str | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MakeLinear,
            in_features,
            out_features,
            bias,
            dtype,
            out_dtype,
        )

    # -- properties read directly from C++ fields --
    @property
    def weight(self) -> Parameter:
        return self.__object_handle__.weight

    @property
    def bias(self) -> Parameter | None:
        return self.__object_handle__.bias  # Optional[NNParameter] -> None or Parameter

    @property
    def out_dtype(self) -> str | None:
        v = self.__object_handle__.out_dtype
        return str(v) if v is not None else None

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self.__object_handle__.forward(x._expr))

    def to(self, dtype: str | None = None) -> None:
        if dtype is not None:
            self.weight.to(dtype=dtype)
            if self.bias is not None and self.out_dtype is None:
                self.bias.to(dtype=dtype)


# ===========================================================================
# Embedding
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Embedding")
class Embedding(Module):
    """Embedding lookup. All state lives in C++."""

    def __init__(
        self,
        num: int | str | tir.PrimExpr,
        dim: int | str | tir.PrimExpr,
        dtype: str | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(_ffi_api.MakeEmbedding, num, dim, dtype)

    @property
    def weight(self) -> Parameter:
        return self.__object_handle__.weight

    def forward(self, x: Tensor) -> Tensor:
        out_shape = [] if x.ndim == 1 else [*list(x.shape), self.weight.shape[1]]
        return Tensor(_expr=self.__object_handle__.forward(x._expr, out_shape))


# ===========================================================================
# LayerNorm
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.LayerNorm")
class LayerNorm(Module):
    """Layer Normalization. All state lives in C++."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float | None = 1e-5,
        elementwise_affine: bool = True,
        dtype: str | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MakeLayerNorm,
            normalized_shape,
            float(eps if eps is not None else 1e-5),
            elementwise_affine,
            dtype,
        )

    @property
    def weight(self) -> Parameter | None:
        return self.__object_handle__.weight

    @property
    def bias(self) -> Parameter | None:
        return self.__object_handle__.bias

    @property
    def eps(self) -> float:
        return float(self.__object_handle__.epsilon)

    @property
    def elementwise_affine(self) -> bool:
        return bool(self.__object_handle__.elementwise_affine)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self.__object_handle__.forward(x._expr))


# ===========================================================================
# RMSNorm
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.RMSNorm")
class RMSNorm(Module):
    """RMS Normalization. All state lives in C++."""

    def __init__(
        self,
        hidden_size: int,
        axes: int | list[int],
        epsilon: float = 1e-5,
        bias: bool = True,
        dtype: str | None = None,
    ) -> None:
        ax = [axes] if isinstance(axes, int) else list(axes)
        self.__init_handle_by_constructor__(
            _ffi_api.MakeRMSNorm,
            hidden_size,
            ax,
            float(epsilon),
            bias,
            dtype,
        )

    @property
    def weight(self) -> Parameter:
        return self.__object_handle__.weight

    @property
    def bias(self) -> Parameter | None:
        return self.__object_handle__.bias

    @property
    def epsilon(self) -> float:
        return float(self.__object_handle__.epsilon)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self.__object_handle__.forward(x._expr))


# ===========================================================================
# GroupNorm
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.GroupNorm")
class GroupNorm(Module):
    """Group Normalization. All state lives in C++."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        dtype: str | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MakeGroupNorm,
            num_groups,
            num_channels,
            float(eps),
            affine,
            dtype,
        )

    @property
    def num_groups(self) -> int:
        return int(self.__object_handle__.num_groups)

    @property
    def weight(self) -> Parameter | None:
        return self.__object_handle__.weight

    @property
    def bias(self) -> Parameter | None:
        return self.__object_handle__.bias

    @property
    def eps(self) -> float:
        return float(self.__object_handle__.epsilon)

    def forward(self, x: Tensor, channel_axis: int = 1, axes: list[int] | None = None) -> Tensor:
        if axes is None:
            axes = list(range(2, len(x._expr.struct_info.shape)))
        return Tensor(_expr=self.__object_handle__.forward(x._expr, channel_axis, axes))


# ===========================================================================
# Conv1D
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Conv1D")
class Conv1D(Module):
    """1D Convolution. All state lives in C++."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: str | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MakeConv1D,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            dtype,
        )

    @property
    def weight(self) -> Parameter:
        return self.__object_handle__.weight

    @property
    def bias(self) -> Parameter | None:
        return self.__object_handle__.bias

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self.__object_handle__.forward(x._expr))


# ===========================================================================
# Conv2D
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Conv2D")
class Conv2D(Module):
    """2D Convolution. All state lives in C++."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int] | int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: str | None = None,
        data_layout: str = "NCHW",
    ) -> None:
        ks = [kernel_size, kernel_size] if isinstance(kernel_size, int) else list(kernel_size)
        self.__init_handle_by_constructor__(
            _ffi_api.MakeConv2D,
            in_channels,
            out_channels,
            ks,
            stride,
            padding,
            dilation,
            groups,
            bias,
            dtype,
            data_layout,
        )

    @property
    def weight(self) -> Parameter:
        return self.__object_handle__.weight

    @property
    def bias(self) -> Parameter | None:
        return self.__object_handle__.bias

    @property
    def data_layout(self) -> str:
        return str(self.__object_handle__.data_layout)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self.__object_handle__.forward(x._expr))


# ===========================================================================
# Conv3D
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Conv3D")
class Conv3D(Module):
    """3D Convolution. All state lives in C++."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int] | int,
        stride: list[int] | int = 1,
        padding: list[int] | int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: str | None = None,
        data_layout: str = "NCDHW",
    ) -> None:
        ks = [kernel_size] * 3 if isinstance(kernel_size, int) else list(kernel_size)
        s = stride[0] if isinstance(stride, list | tuple) else stride
        p = padding[0] if isinstance(padding, list | tuple) else padding
        self.__init_handle_by_constructor__(
            _ffi_api.MakeConv3D,
            in_channels,
            out_channels,
            ks,
            s,
            p,
            dilation,
            groups,
            bias,
            dtype,
            data_layout,
        )

    @property
    def weight(self) -> Parameter:
        return self.__object_handle__.weight

    @property
    def bias(self) -> Parameter | None:
        return self.__object_handle__.bias

    @property
    def data_layout(self) -> str:
        return str(self.__object_handle__.data_layout)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self.__object_handle__.forward(x._expr))


# ===========================================================================
# ConvTranspose1D
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.ConvTranspose1D")
class ConvTranspose1D(Module):
    """1D Transposed Convolution. All state lives in C++."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: str | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MakeConvTranspose1D,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            bias,
            dtype,
        )

    @property
    def weight(self) -> Parameter:
        return self.__object_handle__.weight

    @property
    def bias(self) -> Parameter | None:
        return self.__object_handle__.bias

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self.__object_handle__.forward(x._expr))


# ===========================================================================
# KVCache  (pure Python)
# ===========================================================================


class KVCache(Effect):
    """KVCache effect for attention layers."""

    def __init__(self, init_seq_len: int, unit_shape: Sequence[int], dtype: str | None = None):
        if dtype is None:
            dtype = get_default_dtype()
        self.init_seq_len = init_seq_len
        self.unit_shape = [int(i) for i in unit_shape]
        self.dtype = dtype
        self.cache = None

    def emit_init(self, name_hint, bb):
        init_shape = rx.ShapeExpr([self.init_seq_len, *self.unit_shape])
        return [
            bb.emit(
                rx.op.call_pure_packed(
                    "vm.builtin.attention_kv_cache_create",
                    rx.op.zeros(init_shape, self.dtype),
                    init_shape,
                    rx.PrimValue(0),
                    sinfo_args=rx.ObjectStructInfo(),
                ),
                name_hint=name_hint,
            )
        ]

    def create(self, name_hint):
        self.cache = rx.Var(name_hint, struct_info=rx.ObjectStructInfo())
        return [self.cache]

    def set_state(self, state_vars):
        (self.cache,) = state_vars

    def finalize(self):
        result, self.cache = self.cache, None
        return [result]

    def to(self, dtype=None):
        if dtype is not None:
            self.dtype = dtype

    def view(self, seq_len) -> Tensor:
        shape = rx.ShapeExpr([seq_len, *self.unit_shape])
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.op.call_pure_packed(
                    "vm.builtin.attention_kv_cache_view",
                    self.cache,
                    shape,
                    sinfo_args=rx.TensorStructInfo(shape, self.dtype),
                )
            )
        )

    def append(self, new_element: Tensor) -> None:
        if new_element.dtype != self.dtype:
            raise TypeError(f'KVCache dtype "{self.dtype}" != "{new_element.dtype}"')
        self.cache = rx.BlockBuilder.current().emit(
            rx.op.call_inplace_packed(
                "vm.builtin.attention_kv_cache_append",
                self.cache,
                new_element._expr,
                inplace_indices=[0],
                sinfo_args=rx.ObjectStructInfo(),
            )
        )


# ===========================================================================
# TimestepEmbedding  (pure Python)
# ===========================================================================


class TimestepEmbedding(Module):
    """HF TimestepEmbedding layer."""

    def __init__(
        self,
        in_channels,
        time_embed_dim,
        act_fn="silu",
        out_dim=None,
        post_act_fn=None,
        cond_proj_dim=None,
    ):
        self.linear_1 = Linear(in_channels, time_embed_dim)
        self.cond_proj = (
            Linear(cond_proj_dim, in_channels, bias=False) if cond_proj_dim is not None else None
        )
        assert act_fn == "silu"
        self.act = SiLU()
        self.linear_2 = Linear(time_embed_dim, out_dim if out_dim else time_embed_dim)
        self.post_act = SiLU() if post_act_fn == "silu" else None

    def forward(self, sample: Tensor, condition: Tensor | None = None) -> Tensor:
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.act(self.linear_1(sample))
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


# ===========================================================================
# Timesteps  (pure Python)
# ===========================================================================


class Timesteps(Module):
    """HF Timesteps layer."""

    def __init__(self, num_channels, flip_sin_to_cos=False, downscale_freq_shift=1):
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, x: Tensor) -> Tensor:
        return op.get_timestep_embedding(
            x,
            embedding_dim=self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


# ===========================================================================
# Attention  (pure Python)
# ===========================================================================


class Attention(Module):
    """Cross-attention layer."""

    def __init__(
        self,
        query_dim,
        cross_attention_dim=None,
        heads=8,
        dim_head=64,
        bias=False,
        norm_num_groups=None,
        out_bias=True,
        scale_qk=True,
    ):
        self.heads = heads
        self.inner_dim = dim_head * heads
        cross_dim = cross_attention_dim if cross_attention_dim else query_dim
        self.to_q = Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = Linear(cross_dim, self.inner_dim, bias=bias)
        self.to_v = Linear(cross_dim, self.inner_dim, bias=bias)
        self.group_norm = (
            GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, affine=True)
            if norm_num_groups is not None
            else None
        )
        self.to_out = ModuleList([Linear(self.inner_dim, query_dim, bias=out_bias)])

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        assert attention_mask is None
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states, channel_axis=2, axes=[1])
        q = self.to_q(hidden_states)
        enc = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k, v = self.to_k(enc), self.to_v(enc)
        head_dim = int(self.inner_dim // self.heads)
        q = op.reshape(q, [0, -1, self.heads, head_dim])
        k = op.reshape(k, [0, -1, self.heads, head_dim])
        v = op.reshape(v, [0, -1, self.heads, head_dim])
        out = op.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = op.reshape(out, (0, -1, self.heads * head_dim))
        return self.to_out[0](out)
