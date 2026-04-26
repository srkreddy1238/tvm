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
    """Pass-through module that returns its input unchanged.

    Useful as a no-op placeholder in model architectures where an
    activation or projection layer is optional.
    """

    def __init__(self) -> None:
        self.__ffi_init__()

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))


# ===========================================================================
# IOEffect
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.IOEffect")
class IOEffect(tvm_ffi.Object, Effect):
    """Side-effect token for debug I/O operations.

    Represents a sequenced IO side-effect in the Relax IR.  When
    ``debug=True`` is passed to :meth:`~core.Module.export_tvm`, an
    ``IOEffect`` is prepended to every compiled method's argument list so
    that :func:`~op.debug_func` and :func:`~op.print_` calls are correctly
    ordered in the IR.  Backed by a native C++ object.
    """

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
    """Rectified Linear Unit activation: ``max(0, x)``.

    Applies the element-wise function :math:`\\text{ReLU}(x) = \\max(0, x)`.
    """

    def __init__(self) -> None:
        self.__ffi_init__()

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))


# ===========================================================================
# SiLU
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.SiLU")
class SiLU(tvm_ffi.Object, Module):
    """Sigmoid Linear Unit activation: ``x * sigmoid(x)``.

    Applies the element-wise function
    :math:`\\text{SiLU}(x) = x \\cdot \\sigma(x)` where
    :math:`\\sigma` is the logistic sigmoid.
    """

    def __init__(self) -> None:
        self.__ffi_init__()

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_expr=self._forward(x._expr))


# ===========================================================================
# GELU
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.GELU")
class GELU(tvm_ffi.Object, Module):
    """Gaussian Error Linear Unit activation.

    Applies the element-wise GELU function.  When *approximate* is
    ``"tanh"``, the fast tanh approximation from Hendrycks & Gimpel (2016)
    is used instead of the exact ``erf``-based formula.

    Parameters
    ----------
    approximate : str
        Approximation method.  Pass ``"tanh"`` for the tanh approximation
        or ``""`` (default) for the exact computation.
    """

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
    """Fully-connected linear layer: ``out = x @ weight.T + bias``.

    All state (``weight`` Parameter, ``bias`` Parameter, ``out_dtype``) lives
    in C++.  ``weight``, ``bias``, and ``out_dtype`` are set as properties by
    ``register_object``.

    Parameters
    ----------
    in_features : int | str | tir.PrimExpr
        Size of each input sample.
    out_features : int | str | tir.PrimExpr
        Size of each output sample.
    bias : bool
        If ``True`` (default), add a learnable bias term.
    dtype : str | None
        Data type for the weight (and bias when *out_dtype* is ``None``).
        Defaults to the current default dtype.
    out_dtype : str | None
        If set, the output is cast to this dtype after the matmul.  The
        bias (if any) is kept in *dtype* and added before the cast.
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
    """Lookup table that maps integer indices to dense vectors.

    All state lives in C++.  ``weight`` is set as a property by
    ``register_object``.

    Parameters
    ----------
    num : int | str | tir.PrimExpr
        Size of the embedding dictionary (vocabulary size).
    dim : int | str | tir.PrimExpr
        Dimensionality of each embedding vector.
    dtype : str | None
        Data type for the weight matrix.  Defaults to the current default
        dtype.
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
    """Layer Normalization (Ba et al., 2016).

    Normalises the last *normalized_shape* dimensions of the input tensor,
    then applies a learnable affine transform when *elementwise_affine* is
    ``True``.  All state lives in C++.  ``weight``, ``bias``, ``axes``,
    ``epsilon``, and ``elementwise_affine`` are set by ``register_object``.

    Parameters
    ----------
    normalized_shape : int
        Number of features in the last dimension to normalise.
    eps : float | None
        Small constant added to the variance for numerical stability.
        Defaults to ``1e-5``.
    elementwise_affine : bool
        If ``True`` (default), learn per-element scale and shift parameters.
    dtype : str | None
        Data type for the weight and bias.  Defaults to the current default
        dtype.
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
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalises the input by its root mean square without subtracting the
    mean, then applies a learnable scale.  All state lives in C++.
    ``weight``, ``bias``, ``axes``, and ``epsilon`` are set by
    ``register_object``.

    Parameters
    ----------
    hidden_size : int
        Number of features to normalise (size of the last dimension).
    axes : int | list[int]
        Axis or axes over which to compute the RMS.
    epsilon : float
        Small constant added to the RMS for numerical stability.
    bias : bool
        If ``True`` (default), add a learnable bias term.
    dtype : str | None
        Data type for the weight and bias.  Defaults to the current default
        dtype.
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
    """Group Normalization (Wu & He, 2018).

    Divides the channels into *num_groups* groups and normalises each group
    independently.  All state lives in C++.  ``num_groups``, ``weight``,
    ``bias``, and ``epsilon`` are set by ``register_object``.

    Parameters
    ----------
    num_groups : int
        Number of groups to divide the channels into.
    num_channels : int
        Total number of channels (must be divisible by *num_groups*).
    eps : float
        Small constant added to the variance for numerical stability.
    affine : bool
        If ``True`` (default), learn per-channel scale and shift parameters.
    dtype : str | None
        Data type for the weight and bias.  Defaults to the current default
        dtype.
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
    """1D convolution layer.

    Applies a 1D convolution over an input signal of shape
    ``(N, C_in, L)``.  All state lives in C++.  ``weight``, ``bias``,
    ``stride``, ``padding``, ``dilation``, and ``groups`` are set by
    ``register_object``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels produced by the convolution.
    kernel_size : int
        Size of the convolving kernel.
    stride : int
        Stride of the convolution.  Defaults to ``1``.
    padding : int
        Zero-padding added to both sides of the input.  Defaults to ``0``.
    dilation : int
        Spacing between kernel elements.  Defaults to ``1``.
    groups : int
        Number of blocked connections from input to output channels.
        Defaults to ``1``.
    bias : bool
        If ``True`` (default), add a learnable bias term.
    dtype : str | None
        Data type for the weight and bias.  Defaults to the current default
        dtype.
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
    """2D convolution layer.

    Applies a 2D convolution over an input image of shape
    ``(N, C_in, H, W)`` (NCHW layout by default).  All state lives in C++.
    ``weight``, ``bias``, ``stride``, ``padding``, ``dilation``, ``groups``,
    and ``data_layout`` are set by ``register_object``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels produced by the convolution.
    kernel_size : list[int] | int
        Size of the convolving kernel.  A single ``int`` is used for both
        height and width.
    stride : int
        Stride of the convolution.  Defaults to ``1``.
    padding : int
        Zero-padding added to all spatial sides.  Defaults to ``0``.
    dilation : int
        Spacing between kernel elements.  Defaults to ``1``.
    groups : int
        Number of blocked connections from input to output channels.
        Defaults to ``1``.
    bias : bool
        If ``True`` (default), add a learnable bias term.
    dtype : str | None
        Data type for the weight and bias.  Defaults to the current default
        dtype.
    data_layout : str
        Layout of the input tensor.  Defaults to ``"NCHW"``.
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
    """3D convolution layer.

    Applies a 3D convolution over a volumetric input of shape
    ``(N, C_in, D, H, W)`` (NCDHW layout by default).  All state lives in
    C++.  ``weight``, ``bias``, ``stride``, ``padding``, ``dilation``,
    ``groups``, and ``data_layout`` are set by ``register_object``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels produced by the convolution.
    kernel_size : list[int] | int
        Size of the convolving kernel.  A single ``int`` is broadcast to
        all three spatial dimensions.
    stride : list[int] | int
        Stride of the convolution.  Defaults to ``1``.
    padding : list[int] | int
        Zero-padding added to all spatial sides.  Defaults to ``0``.
    dilation : int
        Spacing between kernel elements.  Defaults to ``1``.
    groups : int
        Number of blocked connections from input to output channels.
        Defaults to ``1``.
    bias : bool
        If ``True`` (default), add a learnable bias term.
    dtype : str | None
        Data type for the weight and bias.  Defaults to the current default
        dtype.
    data_layout : str
        Layout of the input tensor.  Defaults to ``"NCDHW"``.
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
    """1D transposed convolution layer (fractionally-strided convolution).

    Applies a 1D transposed convolution over an input of shape
    ``(N, C_in, L)``.  All state lives in C++.  ``weight``, ``bias``,
    ``stride``, ``padding``, ``output_padding``, ``dilation``, and
    ``groups`` are set by ``register_object``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels produced by the transposed convolution.
    kernel_size : int
        Size of the convolving kernel.
    stride : int
        Stride of the convolution.  Defaults to ``1``.
    padding : int
        Zero-padding added to both sides of the input.  Defaults to ``0``.
    output_padding : int
        Additional size added to one side of the output shape.  Defaults
        to ``0``.
    dilation : int
        Spacing between kernel elements.  Defaults to ``1``.
    groups : int
        Number of blocked connections from input to output channels.
        Defaults to ``1``.
    bias : bool
        If ``True`` (default), add a learnable bias term.
    dtype : str | None
        Data type for the weight and bias.  Defaults to the current default
        dtype.
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
    """Key-value cache effect for autoregressive inference.

    Maintains a growing cache of key/value tensors across decoding steps.
    Backed by a native C++ object.  The cache is initialised with
    *init_seq_len* slots of shape *unit_shape* and grows dynamically via
    :meth:`append`.

    Parameters
    ----------
    init_seq_len : int
        Initial number of sequence positions to pre-allocate.
    unit_shape : Sequence[int]
        Shape of a single cache entry (excluding the sequence dimension),
        e.g. ``[num_heads, head_dim]``.
    dtype : str | None
        Data type for the cached tensors.  Defaults to the current default
        dtype.
    """

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
        """Return a view of the first *seq_len* entries in the cache.

        Parameters
        ----------
        seq_len : int | tir.PrimExpr
            Number of sequence positions to include in the view.

        Returns
        -------
        result : Tensor
            Tensor of shape ``(seq_len, *unit_shape)``.
        """
        return Tensor(_expr=self._view(seq_len)._expr)

    def append(self, new_element: Tensor) -> None:
        """Append *new_element* to the end of the cache.

        Parameters
        ----------
        new_element : Tensor
            Tensor of shape ``(1, *unit_shape)`` to append.
        """
        self._append(new_element)


# ===========================================================================
# TimestepEmbedding
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.TimestepEmbedding")
class TimestepEmbedding(tvm_ffi.Object, Module):
    """Hugging Face-compatible timestep embedding layer.

    Projects a scalar timestep through two linear layers with an
    activation in between, matching the ``TimestepEmbedding`` module from
    ``diffusers``.  Backed by a native C++ object.

    Parameters
    ----------
    in_channels : int
        Dimensionality of the input timestep embedding.
    time_embed_dim : int
        Dimensionality of the output embedding.
    act_fn : str
        Activation function name between the two linear layers.
        Defaults to ``"silu"``.
    out_dim : int | None
        If set, add a second linear projection to this output size.
    post_act_fn : str | None
        Optional activation applied after the final linear layer.
    cond_proj_dim : int | None
        If set, add a conditioning projection of this size.
    """

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
    """Hugging Face-compatible sinusoidal timestep projection layer.

    Converts scalar timestep indices to sinusoidal embeddings, matching
    the ``Timesteps`` module from ``diffusers``.  Backed by a native C++
    object.

    Parameters
    ----------
    num_channels : int
        Dimensionality of the output embedding.
    flip_sin_to_cos : bool
        If ``True``, place cosine features before sine features.
        Defaults to ``False``.
    downscale_freq_shift : float
        Shift applied to the log-frequency before exponentiation.
        Defaults to ``1``.
    """

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
    """Multi-head cross-attention layer.

    Implements scaled dot-product attention with optional cross-attention
    (separate query and key/value sources) and optional group normalization
    on the query.  Matches the ``Attention`` module from ``diffusers``.
    Backed by a native C++ object.

    Parameters
    ----------
    query_dim : int
        Dimensionality of the query input.
    cross_attention_dim : int | None
        Dimensionality of the key/value input for cross-attention.  When
        ``None``, self-attention is used (key/value come from the query).
    heads : int
        Number of attention heads.  Defaults to ``8``.
    dim_head : int
        Dimensionality of each attention head.  Defaults to ``64``.
    bias : bool
        If ``True``, add bias to the query/key/value projections.
        Defaults to ``False``.
    norm_num_groups : int | None
        If set, apply group normalization with this many groups to the
        query before computing attention.
    out_bias : bool
        If ``True`` (default), add bias to the output projection.
    scale_qk : bool
        Kept for API compatibility; currently unused.
    """

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
