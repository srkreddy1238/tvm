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

from tvm import tir

from . import _ffi_api
from .core import Effect, Module, Tensor, get_default_dtype

# ===========================================================================
# Identity
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Identity")
class Identity(tvm_ffi.Object, Module):
    """Pass-through module."""

    def __init__(self) -> None:
        self.__ffi_init__()

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))


# ===========================================================================
# IOEffect
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.IOEffect")
class IOEffect(tvm_ffi.Object, Effect):
    """Modeling IO side effect — backed by native C++."""

    def __init__(self) -> None:
        self.__ffi_init__()

    def emit_init(self, name_hint, builder):
        return list(self._cpp_emit_init(name_hint, builder))

    def create(self, name_hint):
        return list(self._cpp_create(name_hint))

    def set_state(self, state_vars):
        self._cpp_set_state(list(state_vars))

    def finalize(self):
        return list(self._cpp_finalize())


# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.ReLU")
class ReLU(tvm_ffi.Object, Module):
    """ReLU activation."""

    def __init__(self) -> None:
        self.__ffi_init__()

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))


# ===========================================================================
# SiLU
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.SiLU")
class SiLU(tvm_ffi.Object, Module):
    """SiLU activation."""

    def __init__(self) -> None:
        self.__ffi_init__()

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))


# ===========================================================================
# GELU
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.GELU")
class GELU(tvm_ffi.Object, Module):
    """GELU activation."""

    def __init__(self, approximate: str = "") -> None:
        self.__ffi_init__(approximate)

    # approximate is a C++ field; register_object sets it as a property directly.

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))


# ===========================================================================
# Linear
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Linear")
class Linear(tvm_ffi.Object, Module):
    """Linear layer: out = x @ W^T + b.

    All state (weight Parameter, bias Parameter, out_dtype) lives in C++.
    weight, bias, out_dtype are set as properties by register_object.
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

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))

    def to(self, dtype: str | None = None) -> None:
        if dtype is not None:
            self.weight.to(dtype=dtype)
            if self.bias is not None and self.out_dtype is None:
                self.bias.to(dtype=dtype)


# ===========================================================================
# Embedding
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Embedding")
class Embedding(tvm_ffi.Object, Module):
    """Embedding lookup. All state lives in C++.
    weight is set as a property by register_object.
    """

    def __init__(
        self,
        num: int | str | tir.PrimExpr,
        dim: int | str | tir.PrimExpr,
        dtype: str | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(_ffi_api.MakeEmbedding, num, dim, dtype)

    def forward(self, x: Tensor) -> Tensor:
        out_shape = [] if x.ndim == 1 else [*list(x.shape), self.weight.shape[1]]
        return Tensor(_expr=self._forward(x._expr, out_shape))


# ===========================================================================
# LayerNorm
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.LayerNorm")
class LayerNorm(tvm_ffi.Object, Module):
    """Layer Normalization. All state lives in C++.
    weight, bias, axes, epsilon, elementwise_affine set by register_object.
    """

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

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))


# ===========================================================================
# RMSNorm
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.RMSNorm")
class RMSNorm(tvm_ffi.Object, Module):
    """RMS Normalization. All state lives in C++.
    weight, bias, axes, epsilon set by register_object.
    """

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

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))


# ===========================================================================
# GroupNorm
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.GroupNorm")
class GroupNorm(tvm_ffi.Object, Module):
    """Group Normalization. All state lives in C++.
    num_groups, weight, bias, epsilon set by register_object.
    """

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

    def forward(self, x: Tensor, channel_axis: int = 1, axes: list[int] | None = None) -> Tensor:
        if axes is None:
            axes = list(range(2, len(x._expr.struct_info.shape)))
        return Tensor(_expr=self._forward(x._expr, channel_axis, axes))


# ===========================================================================
# Conv1D
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Conv1D")
class Conv1D(tvm_ffi.Object, Module):
    """1D Convolution. All state lives in C++.
    weight, bias, stride, padding, dilation, groups set by register_object.
    """

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

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))


# ===========================================================================
# Conv2D
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Conv2D")
class Conv2D(tvm_ffi.Object, Module):
    """2D Convolution. All state lives in C++.
    weight, bias, stride, padding, dilation, groups, data_layout set by register_object.
    """

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

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))


# ===========================================================================
# Conv3D
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Conv3D")
class Conv3D(tvm_ffi.Object, Module):
    """3D Convolution. All state lives in C++.
    weight, bias, stride, padding, dilation, groups, data_layout set by register_object.
    """

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

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))


# ===========================================================================
# ConvTranspose1D
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.ConvTranspose1D")
class ConvTranspose1D(tvm_ffi.Object, Module):
    """1D Transposed Convolution. All state lives in C++.
    weight, bias, stride, padding, output_padding, dilation, groups set by register_object.
    """

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

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))


# ===========================================================================
# KVCache
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.KVCache")
class KVCache(tvm_ffi.Object, Effect):
    """KVCache effect — backed by native C++."""

    def __init__(
        self, init_seq_len: int, unit_shape: Sequence[int], dtype: str | None = None
    ) -> None:
        if dtype is None:
            dtype = get_default_dtype()
        self.__init_handle_by_constructor__(
            _ffi_api.MakeKVCache, int(init_seq_len), [int(i) for i in unit_shape], dtype
        )

    def emit_init(self, name_hint, bb):
        return list(self._cpp_emit_init(name_hint, bb))

    def create(self, name_hint):
        return list(self._cpp_create(name_hint))

    def set_state(self, state_vars):
        self._cpp_set_state(list(state_vars))

    def finalize(self):
        return list(self._cpp_finalize())

    def to(self, dtype=None):
        if dtype is not None:
            self._cpp_to(dtype)

    def view(self, seq_len) -> Tensor:
        return Tensor(_expr=self._view(int(seq_len))._expr)

    def append(self, new_element: Tensor) -> None:
        self._append(new_element)


# ===========================================================================
# TimestepEmbedding
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.TimestepEmbedding")
class TimestepEmbedding(tvm_ffi.Object, Module):
    """HF TimestepEmbedding layer — backed by native C++."""

    def __init__(
        self,
        in_channels,
        time_embed_dim,
        act_fn="silu",
        out_dim=None,
        post_act_fn=None,
        cond_proj_dim=None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MakeTimestepEmbedding,
            int(in_channels),
            int(time_embed_dim),
            str(act_fn),
            int(out_dim) if out_dim is not None else None,
            str(post_act_fn) if post_act_fn is not None else None,
            int(cond_proj_dim) if cond_proj_dim is not None else None,
        )

    def forward(self, sample: Tensor, condition: Tensor | None = None) -> Tensor:
        cond_var = condition._expr if condition is not None else None
        return Tensor(_expr=self._forward(sample._expr, cond_var))


# ===========================================================================
# Timesteps
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Timesteps")
class Timesteps(tvm_ffi.Object, Module):
    """HF Timesteps layer — backed by native C++."""

    def __init__(self, num_channels, flip_sin_to_cos=False, downscale_freq_shift=1) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MakeTimesteps,
            int(num_channels),
            bool(flip_sin_to_cos),
            float(downscale_freq_shift),
        )

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))


# ===========================================================================
# Attention
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Attention")
class Attention(tvm_ffi.Object, Module):
    """Cross-attention layer — backed by native C++."""

    def __init__(
        self,
        query_dim,
        cross_attention_dim=None,
        heads=8,
        dim_head=64,
        bias=False,
        norm_num_groups=None,
        out_bias=True,
        scale_qk=True,  # kept for API compat, unused
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MakeAttention,
            int(query_dim),
            int(cross_attention_dim) if cross_attention_dim is not None else None,
            int(heads),
            int(dim_head),
            bool(bias),
            int(norm_num_groups) if norm_num_groups is not None else None,
            bool(out_bias),
        )

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        assert attention_mask is None
        enc = encoder_hidden_states._expr if encoder_hidden_states is not None else None
        return Tensor(_expr=self._forward(hidden_states._expr, enc))
