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
# pylint: disable=too-many-lines,invalid-name,protected-access,redefined-outer-name
# pylint: disable=redefined-builtin
"""nn.Tensor operators - thin Python wrappers over C++ FFI (relax.frontend.nn.op.*).

Every public function here:
  1. Unwraps nn.Tensor arguments to their underlying relax.Var (_expr).
  2. Calls the corresponding C++ FFI function.
  3. Wraps the returned relax.Var back into nn.Tensor via _unwrap_ffi_result.

Heavy-weight logic (shape inference, op construction, BlockBuilder emission)
lives entirely in src/relax/frontend/nn/op.cc.
"""

import inspect
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np

from tvm import tir as _tir

from ... import expr as rx
from ... import op as _op
from . import _ffi_api
from .core import Tensor, _unwrap_ffi_result, get_default_dtype, wrap_nested

IntExpr = int | _tir.PrimExpr

# ---------------------------------------------------------------------------
# FFI API - populated by init_ffi_api("relax.frontend.nn.op") at import time.
# Each attribute corresponds to a C++ global registered as
# "relax.frontend.nn.op.<Name>" (the dot-free suffix becomes the attribute).
# ---------------------------------------------------------------------------
from . import _ffi_api_op as _ffi_op  # noqa: E402

# ---------------------------------------------------------------------------
# Public API exported by ``from .op import *``
# Internal helpers (_v, _t, IntExpr, OutType) are intentionally excluded.
# ---------------------------------------------------------------------------
__all__ = [
    "add",
    "arange",
    "argsort",
    "astype",
    "broadcast_to",
    "ccl_allgather",
    "ccl_allreduce",
    "ccl_broadcast_from_worker0",
    "chunk",
    "clip",
    "concat",
    "conv1d",
    "conv1d_transpose",
    "conv2d",
    "conv3d",
    "cumsum",
    "debug_func",
    "divide",
    "empty",
    "equal",
    "exp",
    "extern",
    "floor",
    "full",
    "gelu",
    "get_timestep_embedding",
    "greater",
    "greater_equal",
    "group_norm",
    "interpolate",
    "layer_norm",
    "less",
    "less_equal",
    "log",
    "matmul",
    "max",
    "maximum",
    "min",
    "minimum",
    "multinomial_from_uniform",
    "multiply",
    "negative",
    "not_equal",
    "ones",
    "pad",
    "permute",
    "permute_dims",
    "prelu",
    "print_",
    "relu",
    "relu6",
    "renormalize_top_p_top_k_prob",
    "repeat",
    "reshape",
    "rms_norm",
    "sample_top_p_top_k_from_sorted_prob",
    "scaled_dot_product_attention",
    "sigmoid",
    "silu",
    "softmax",
    "softplus",
    "sort",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "subtract",
    "sum",
    "take",
    "tanh",
    "tensor_expr_op",
    "tensor_ir_inplace_op",
    "tensor_ir_op",
    "topk",
    "triu",
    "unsqueeze",
    "where",
    "zeros",
]


def _v(t: Tensor) -> rx.Var:
    """Unwrap nn.Tensor to its underlying relax.Var."""
    return t._expr  # pylint: disable=protected-access


def _t(result) -> "Tensor | tuple":
    """Wrap a C++ FFI result (Var or Array) back into nn.Tensor / tuple."""
    return _unwrap_ffi_result(result)


def unsqueeze(x: Tensor, dim: int, name: str = "unsqueeze") -> Tensor:
    """Add a new axis of size 1 at position *dim*.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int
        Position at which to insert the new axis.  Negative values count
        from the end of the shape.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Tensor with ``x.ndim + 1`` dimensions.
    """
    return _t(_ffi_op.unsqueeze(_v(x), dim, name))


def concat(x: list[Tensor], dim: int, name: str = "concat") -> Tensor:
    """Concatenate a list of tensors along *dim*.

    Parameters
    ----------
    x : list[Tensor]
        Tensors to concatenate.  All tensors must have the same shape
        except along *dim*.
    dim : int
        Axis along which to concatenate.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Concatenated tensor.
    """
    return _t(_ffi_op.concat([_v(t) for t in x], dim, name))


def add(a: Tensor, b: Tensor, name: str = "add") -> Tensor:
    """Element-wise addition with numpy-style broadcasting.

    Parameters
    ----------
    a : Tensor
        First operand.
    b : Tensor
        Second operand.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Element-wise sum ``a + b``.
    """
    return _t(_ffi_op.add(_v(a), _v(b), name))


def subtract(a: Tensor, b: Tensor, name: str = "subtract") -> Tensor:
    """Element-wise subtraction with numpy-style broadcasting.

    Parameters
    ----------
    a : Tensor
        Minuend.
    b : Tensor
        Subtrahend.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Element-wise difference ``a - b``.
    """
    return _t(_ffi_op.subtract(_v(a), _v(b), name))


def multiply(a: Tensor, b: Tensor, name: str = "mul") -> Tensor:
    """Element-wise multiplication with numpy-style broadcasting.

    Parameters
    ----------
    a : Tensor
        First operand.
    b : Tensor
        Second operand.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Element-wise product ``a * b``.
    """
    return _t(_ffi_op.multiply(_v(a), _v(b), name))


def divide(a: Tensor, b: Tensor, name: str = "divide") -> Tensor:
    """Element-wise true division with numpy-style broadcasting.

    Parameters
    ----------
    a : Tensor
        Dividend.
    b : Tensor
        Divisor.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Element-wise quotient ``a / b``.
    """
    return _t(_ffi_op.divide(_v(a), _v(b), name))


def chunk(x: Tensor, chunks: int, dim: int = 0, name: str = "chunk") -> Tensor:
    """Split *x* along *dim* into *chunks* equal-sized sub-tensors.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    chunks : int
        Number of chunks to split into.  The last chunk may be smaller
        if the dimension size is not evenly divisible.
    dim : int
        Axis along which to split.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : tuple[Tensor, ...]
        Tuple of sub-tensors.
    """
    return _t(_ffi_op.chunk(_v(x), chunks, dim, name))


def sum(
    x: Tensor,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
    name: str = "sum",
) -> Tensor:
    """Compute the sum of tensor elements over the given axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int | list[int] | None
        Axis or axes to reduce.  ``None`` reduces over all axes.
    keepdims : bool
        If ``True``, retain reduced axes as size-1 dimensions.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Reduced tensor.
    """
    ax = [axis] if isinstance(axis, int) else (axis or [])
    return _t(_ffi_op.sum(_v(x), ax, keepdims, name))


def max(
    x: Tensor,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
    name: str = "max",
) -> Tensor:
    """Compute the maximum of tensor elements over the given axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int | list[int] | None
        Axis or axes to reduce.  ``None`` reduces over all axes.
    keepdims : bool
        If ``True``, retain reduced axes as size-1 dimensions.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Reduced tensor.
    """
    ax = [axis] if isinstance(axis, int) else (axis or [])
    return _t(_ffi_op.max(_v(x), ax, keepdims, name))


def min(
    x: Tensor,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
    name: str = "min",
) -> Tensor:
    """Compute the minimum of tensor elements over the given axes.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int | list[int] | None
        Axis or axes to reduce.  ``None`` reduces over all axes.
    keepdims : bool
        If ``True``, retain reduced axes as size-1 dimensions.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Reduced tensor.
    """
    ax = [axis] if isinstance(axis, int) else (axis or [])
    return _t(_ffi_op.min(_v(x), ax, keepdims, name))


def matmul(a: Tensor, b: Tensor, out_dtype: str | None = None, name: str = "matmul") -> Tensor:
    """General matrix multiplication with broadcasting on batched dimensions.

    Parameters
    ----------
    a : Tensor
        Left-hand operand.  Must have at least 1 dimension.
    b : Tensor
        Right-hand operand.  Must have at least 1 dimension.
    out_dtype : str | None
        Optional output dtype.  Defaults to the common dtype of *a* and *b*.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Matrix product, with batch dimensions broadcast.
    """
    return _t(_ffi_op.matmul(_v(a), _v(b), out_dtype, name))


def conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple | None = 1,
    padding: int | tuple | str | None = 0,
    dilation: int | tuple | None = 1,
    groups: int | None = 1,
    name: str = "conv1d",
) -> Tensor:
    """Apply a 1D convolution over an input signal.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(N, C_in, L)``.
    weight : Tensor
        Convolution kernel of shape ``(C_out, C_in/groups, kL)``.
    bias : Tensor | None
        Optional bias of shape ``(C_out,)``.
    stride : int | tuple | None
        Stride of the convolution.
    padding : int | tuple | str | None
        Zero-padding added to both sides of the input.
    dilation : int | tuple | None
        Spacing between kernel elements.
    groups : int | None
        Number of blocked connections from input to output channels.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Output tensor of shape ``(N, C_out, L_out)``.
    """
    return _t(
        _ffi_op.conv1d(
            _v(x), _v(weight), _v(bias) if bias else None, stride, padding, dilation, groups, name
        )
    )


def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple | None = 1,
    padding: int | tuple | str | None = 0,
    dilation: int | tuple | None = 1,
    groups: int | None = 1,
    data_layout: str | None = "NCHW",
    name: str = "conv2d",
) -> Tensor:
    """Apply a 2D convolution over an input image.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(N, C_in, H, W)`` (NCHW layout).
    weight : Tensor
        Convolution kernel of shape ``(C_out, C_in/groups, kH, kW)``.
    bias : Tensor | None
        Optional bias of shape ``(C_out,)``.
    stride : int | tuple | None
        Stride of the convolution.
    padding : int | tuple | str | None
        Zero-padding added to both spatial sides.
    dilation : int | tuple | None
        Spacing between kernel elements.
    groups : int | None
        Number of blocked connections from input to output channels.
    data_layout : str | None
        Layout of the input tensor.  Defaults to ``"NCHW"``.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Output tensor of shape ``(N, C_out, H_out, W_out)``.
    """
    return _t(
        _ffi_op.conv2d(
            _v(x),
            _v(weight),
            _v(bias) if bias else None,
            stride,
            padding,
            dilation,
            groups,
            data_layout,
            name,
        )
    )


def conv3d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple | None = 1,
    padding: int | tuple | str | None = 0,
    dilation: int | tuple | None = 1,
    groups: int | None = 1,
    data_layout: str | None = "NCDHW",
    name: str = "conv3d",
) -> Tensor:
    """Apply a 3D convolution over a volumetric input.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(N, C_in, D, H, W)`` (NCDHW layout).
    weight : Tensor
        Convolution kernel of shape ``(C_out, C_in/groups, kD, kH, kW)``.
    bias : Tensor | None
        Optional bias of shape ``(C_out,)``.
    stride : int | tuple | None
        Stride of the convolution.
    padding : int | tuple | str | None
        Zero-padding added to all spatial sides.
    dilation : int | tuple | None
        Spacing between kernel elements.
    groups : int | None
        Number of blocked connections from input to output channels.
    data_layout : str | None
        Layout of the input tensor.  Defaults to ``"NCDHW"``.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Output tensor of shape ``(N, C_out, D_out, H_out, W_out)``.
    """
    return _t(
        _ffi_op.conv3d(
            _v(x),
            _v(weight),
            _v(bias) if bias else None,
            stride,
            padding,
            dilation,
            groups,
            data_layout,
            name,
        )
    )


def conv1d_transpose(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int] | None = 1,
    padding: int | tuple[int, ...] | None = 0,
    output_padding: int | tuple[int] | None = 0,
    dilation: int | tuple | None = 1,
    groups: int | None = 1,
    name: str = "conv1d_transpose",
) -> Tensor:
    """Apply a 1D transposed convolution (fractionally-strided convolution).

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(N, C_in, L)``.
    weight : Tensor
        Kernel of shape ``(C_in, C_out/groups, kL)``.
    bias : Tensor | None
        Optional bias of shape ``(C_out,)``.
    stride : int | tuple[int] | None
        Stride of the convolution.
    padding : int | tuple[int, ...] | None
        Zero-padding added to both sides of the input.
    output_padding : int | tuple[int] | None
        Additional size added to one side of the output shape.
    dilation : int | tuple | None
        Spacing between kernel elements.
    groups : int | None
        Number of blocked connections from input to output channels.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Output tensor of shape ``(N, C_out, L_out)``.
    """
    return _t(
        _ffi_op.conv1d_transpose(
            _v(x),
            _v(weight),
            _v(bias) if bias else None,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            name,
        )
    )


def maximum(x1: Tensor, x2: Tensor, name: str = "maximum"):
    """Element-wise maximum."""
    return _t(_ffi_op.maximum(_v(x1), _v(x2), name))


def minimum(x1: Tensor, x2: Tensor, name: str = "minimum"):
    """Element-wise minimum."""
    return _t(_ffi_op.minimum(_v(x1), _v(x2), name))


def broadcast_to(x: Tensor, shape: Sequence[IntExpr], name: str = "broadcast_to") -> Tensor:
    """Broadcasts a tensor to a specified shape."""
    return _t(_ffi_op.broadcast_to(_v(x), list(shape), name))


def permute_dims(x: Tensor, axes: list[int] | None = None, name: str | None = None) -> Tensor:
    """Permutes the dimensions of an array.

    When *name* is ``None`` the binding name is derived from the input
    tensor's name_hint by the C++ implementation (``NNPermuteDims`` in
    ``op.cc``), mirroring the rule: if ``"linear"`` appears in the name,
    replace it with ``"matmul"``; otherwise use ``"permute_dims"``.
    """
    return _t(_ffi_op.permute_dims(_v(x), axes, name))


def reshape(x: Tensor, shape: Sequence[IntExpr], name="reshape") -> Tensor:
    """Reshape the input array."""
    return _t(_ffi_op.reshape(_v(x), list(shape), name))


def repeat(x: Tensor, repeats: int, axis: int | None = None, name="repeat") -> Tensor:
    """Repeats elements of an array."""
    return _t(_ffi_op.repeat(_v(x), repeats, axis, name))


def squeeze(x: Tensor, axis: int = -1, name: str = "squeeze") -> Tensor:
    """Squeeze axes in the array."""
    return _t(_ffi_op.squeeze(_v(x), axis, name))


def take(x: Tensor, indices: Tensor, axis: int | None = None, name="take") -> Tensor:
    """Take elements from a tensor along an axis."""
    return _t(_ffi_op.take(_v(x), _v(indices), axis, name))


def astype(x: Tensor, dtype: str, name: str = "astype") -> Tensor:
    """Cast input tensor to the given data type."""
    return _t(_ffi_op.astype(_v(x), dtype, name))


def relu(x: Tensor, name: str = "relu") -> Tensor:
    """Rectified Linear Unit (ReLU) activation function."""
    return _t(_ffi_op.relu(_v(x), name))


def relu6(x: Tensor, name: str = "relu6") -> Tensor:
    """ReLU6 activation function."""
    return _t(_ffi_op.relu6(_v(x), name))


def silu(x: Tensor, name: str = "silu") -> Tensor:
    """Sigmoid Linear Unit function."""
    return _t(_ffi_op.silu(_v(x), name))


def gelu(x: Tensor, approximate: str | None = None, name: str = "gelu") -> Tensor:
    """Applies the Gaussian Error Linear Units function."""
    return _t(_ffi_op.gelu(_v(x), approximate, name))


def sigmoid(x: Tensor, name: str = "sigmoid") -> Tensor:
    """Computes sigmoid."""
    return _t(_ffi_op.sigmoid(_v(x), name))


def softmax(x: Tensor, axis: int = -1, name: str = "softmax") -> Tensor:
    """Computes softmax."""
    return _t(_ffi_op.softmax(_v(x), axis, name))


def softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0, name: str = "softplus"):
    """Softplus activation function."""
    return _t(_ffi_op.softplus(_v(x), beta, threshold, name))


def prelu(x: Tensor, alpha: Tensor, name: str = "prelu"):
    """Parametric ReLU activation function."""
    return _t(_ffi_op.prelu(_v(x), _v(alpha), name))


def tanh(x: Tensor, name: str = "tanh") -> Tensor:
    """Applies the hyperbolic tangent function."""
    return _t(_ffi_op.tanh(_v(x), name))


def exp(x: Tensor, name: str = "exp") -> Tensor:
    """Applies the exponential function."""
    return _t(_ffi_op.exp(_v(x), name))


def log(x: Tensor, name: str = "log") -> Tensor:
    """Applies the natural logarithm function."""
    return _t(_ffi_op.log(_v(x), name))


def floor(x: Tensor, name: str = "floor") -> Tensor:
    """Computes the floor of the input tensor."""
    return _t(_ffi_op.floor(_v(x), name))


def arange(
    start: int,
    end: int | None = None,
    step: int = 1,
    dtype: str | None = "float32",
    name: str = "arange",
) -> Tensor:
    """Construct a tensor with evenly spaced elements."""
    return _t(_ffi_op.arange(start, end, step, dtype, name))


def permute(x: Tensor, axes: list[int] | None, name: str = "permute") -> Tensor:
    """Permutes the dimensions of the input tensor."""
    return _t(_ffi_op.permute_dims(_v(x), axes, name))


def negative(x: Tensor, name: str = "neg") -> Tensor:
    """Numerical negative of the input tensor."""
    return _t(_ffi_op.negative(_v(x), name))


def layer_norm(
    x: Tensor,
    normalized_shape: int | list[int],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
    name: str = "layer_norm",
) -> Tensor:
    """Apply Layer Normalization over the last ``len(normalized_shape)`` dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    normalized_shape : int | list[int]
        Shape of the sub-tensor to normalise.  A single ``int`` is treated
        as a one-element list.
    weight : Tensor | None
        Optional learnable per-element scale (gamma).  Defaults to all-ones
        when ``None``.
        bias : Tensor | None
        Optional learnable per-element shift (beta).  Defaults to all-zeros
        when ``None``.
    eps : float
        Small constant added to the variance for numerical stability.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Normalised tensor with the same shape as *x*.
    """
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    dim_num = len(normalized_shape)
    axes = list(range(-dim_num, 0))
    dtype = x._expr.struct_info.dtype
    if weight is None:
        weight = Tensor.from_const(np.ones(normalized_shape, dtype=dtype))
    if bias is None:
        bias = Tensor.from_const(np.zeros(normalized_shape, dtype=dtype))
    return _t(_ffi_op.layer_norm(_v(x), axes, _v(weight), _v(bias), eps, name))


def rms_norm(
    x: Tensor,
    weight: Tensor,
    axes: int | list[int],
    epsilon: float = 1e-5,
    name: str = "rms_norm",
) -> Tensor:
    """Apply Root Mean Square Layer Normalization.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    weight : Tensor
        Learnable per-element scale (gamma).
    axes : int | list[int]
        Axis or axes over which to compute the RMS.  Typically ``-1``.
    epsilon : float
        Small constant added to the RMS for numerical stability.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Normalised tensor with the same shape as *x*.
    """
    ax = [axes] if isinstance(axes, int) else axes
    return _t(_ffi_op.rms_norm(_v(x), _v(weight), ax, epsilon, name))


def group_norm(
    x: Tensor,
    num_groups: int,
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float = 1e-5,
    channel_axis: int = 1,
    axes: list[int] | None = None,
    name: str = "group_norm",
) -> Tensor:
    """Apply Group Normalization.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(N, C, *spatial)``.
    num_groups : int
        Number of groups to divide the channels into.
    weight : Tensor | None
        Optional learnable per-channel scale (gamma).
    bias : Tensor | None
        Optional learnable per-channel shift (beta).
    eps : float
        Small constant added to the variance for numerical stability.
    channel_axis : int
        Axis that holds the channel dimension.  Defaults to ``1`` (NCHW).
    axes : list[int] | None
        Spatial axes to normalise over.  Defaults to all axes after
        *channel_axis* (i.e. ``range(2, x.ndim)``).
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Normalised tensor with the same shape as *x*.
    """
    dim = len(x._expr.struct_info.shape)
    if axes is None:
        axes = list(range(2, dim))
    return _t(
        _ffi_op.group_norm(
            _v(x),
            _v(weight) if weight is not None else None,
            _v(bias) if bias is not None else None,
            num_groups,
            channel_axis,
            axes,
            eps,
            name,
        )
    )


def triu(x: Tensor, diagonal: int = 0, name: str = "triu") -> Tensor:
    """Return the upper triangular part of a matrix or a batch of matrices."""
    return _t(_ffi_op.triu(_v(x), diagonal, name))


def full(
    shape: Sequence[IntExpr],
    fill_value: Tensor,
    dtype: str = "float32",
    name: str = "full",
) -> Tensor:
    """Fill array with scalar value."""
    if isinstance(fill_value, _tir.FloatImm | _tir.IntImm):
        fill_value = Tensor.from_scalar(fill_value.value, dtype=dtype)
    elif isinstance(fill_value, int | float):
        fill_value = Tensor.from_scalar(fill_value, dtype=dtype)
    return _t(_ffi_op.full(list(shape), _v(fill_value), dtype, name))


def zeros(
    shape: Sequence[IntExpr],
    dtype: str = "float32",
    name: str = "zeros",
) -> Tensor:
    """Construct a tensor of all zeros."""
    return _t(_ffi_op.zeros(list(shape), dtype, name))


def ones(
    shape: Sequence[IntExpr],
    dtype: str = "float32",
    name: str = "ones",
) -> Tensor:
    """Construct a tensor of all ones."""
    return _t(_ffi_op.ones(list(shape), dtype, name))


def empty(
    shape: Sequence[IntExpr],
    dtype: str = "float32",
    name: str = "empty",
) -> Tensor:
    """Construct an uninitialized tensor."""
    return wrap_nested(
        _op.builtin.alloc_tensor(
            rx.ShapeExpr(shape),  # type: ignore
            dtype,
            runtime_device_index=0,
        ),
        name,
    )


def split(
    ary: Tensor,
    indices_or_sections: int | Sequence[int],
    axis: int = 0,
    name: str = "split",
) -> tuple[Tensor, ...]:
    """Split an array into multiple sub-arrays."""
    return _t(_ffi_op.split(_v(ary), indices_or_sections, axis, name))


def pad(
    x: Tensor,
    pad: list[int],
    mode: str = "constant",
    value: float = 0.0,
    name: str = "pad",
) -> Tensor:
    """Apply spatial padding to the input tensor."""
    return _t(_ffi_op.pad(_v(x), pad, mode, value, name))


def square(x: Tensor, name: str = "square") -> Tensor:
    """Computes the element-wise square of the input tensor."""
    return _t(_ffi_op.square(_v(x), name))


def sqrt(x: Tensor, name: str = "sqrt") -> Tensor:
    """Computes the element-wise sqrt of the input tensor."""
    return _t(_ffi_op.sqrt(_v(x), name))


def clip(x: Tensor, min_val: float, max_val: float, name: str = "clip") -> Tensor:
    """Clips tensor values to the range [min_val, max_val]."""
    return _t(_ffi_op.clip(_v(x), float(min_val), float(max_val), name))


def get_timestep_embedding(
    x: Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
    name: str = "get_timestep_embedding",
) -> Tensor:
    """Compute sinusoidal timestep embeddings as in DDPM.

    Implements the timestep embedding described in *Denoising Diffusion
    Probabilistic Models* (Ho et al., 2020).

    Parameters
    ----------
    x : Tensor
        1-D integer tensor of timestep indices, shape ``(batch,)``.
    embedding_dim : int
        Dimensionality of the output embedding.
    flip_sin_to_cos : bool
        If ``True``, place cosine features before sine features in the
        output channel dimension.
    downscale_freq_shift : float
        Shift applied to the log-frequency before exponentiation.
    scale : float
        Multiplicative scale applied to the sinusoidal inputs.
    max_period : int
        Controls the minimum frequency of the embeddings.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Embedding tensor of shape ``(batch, embedding_dim)``.
    """
    dtype = get_default_dtype()
    return _t(
        _ffi_op.get_timestep_embedding(
            _v(x),
            embedding_dim,
            flip_sin_to_cos,
            downscale_freq_shift,
            scale,
            max_period,
            dtype,
            name,
        )
    )


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    is_causal: bool | None = False,
    scale: float | None = None,
    name: str = "scaled_dot_product_attention",
):
    """Compute scaled dot-product attention.

    Computes ``softmax(Q @ K^T / sqrt(d_k)) @ V``, optionally with a
    causal mask.

    Parameters
    ----------
    query : Tensor
        Query tensor of shape ``(batch, heads, seq_q, d_k)``.
    key : Tensor
        Key tensor of shape ``(batch, heads, seq_k, d_k)``.
    value : Tensor
        Value tensor of shape ``(batch, heads, seq_k, d_v)``.
    attn_mask : Tensor | None
        Additive attention mask.  Currently unsupported; must be ``None``.
    is_causal : bool | None
        If ``True``, apply a causal (top-left) mask so each query position
        can only attend to earlier key positions.
    scale : float | None
        Explicit scale factor.  Defaults to ``1 / sqrt(d_k)``.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Attention output of shape ``(batch, heads, seq_q, d_v)``.
    """
    assert attn_mask is None, "attn_mask not yet supported."
    causal_mask = "TopLeft" if is_causal else None
    return _t(
        _ffi_op.scaled_dot_product_attention(
            _v(query), _v(key), _v(value), causal_mask, scale, name
        )
    )


def interpolate(
    x: Tensor,
    size: int | tuple[int] | None = None,
    scale_factor: float | tuple[float] | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    recompute_scale_factor: bool | None = None,
    antialias: bool | None = None,
    data_layout: str | None = "NCHW",
    name: str = "interpolate",
):
    """Resize a spatial tensor using the specified interpolation mode.

    Parameters
    ----------
    x : Tensor
        Input tensor with layout described by *data_layout*.
    size : int | tuple[int] | None
        Output spatial size.  Mutually exclusive with *scale_factor*.
    scale_factor : float | tuple[float] | None
        Multiplier for each spatial dimension.  Mutually exclusive with
        *size*.
    mode : str
        Interpolation algorithm: ``"nearest"``, ``"bilinear"``,
        ``"bicubic"``, etc.
    align_corners : bool | None
        If ``True``, align the corner pixels of input and output tensors.
        Only meaningful for ``"bilinear"`` and ``"bicubic"`` modes.
    recompute_scale_factor : bool | None
        Unsupported; must be ``None``.
    antialias : bool | None
        Unsupported; must be ``None``.
    data_layout : str | None
        Layout string describing the input tensor axes, e.g. ``"NCHW"``.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Resized tensor.
    """
    assert recompute_scale_factor is None, "recompute_scale_factor is not supported."
    assert antialias is None, "antialias is not supported."
    if size is None:
        size = []
        for i, dim in enumerate(data_layout):
            if dim not in ["N", "C"]:
                if isinstance(scale_factor, list | tuple):
                    size.append(int(x.shape[i] * scale_factor[len(size)]))
                else:
                    size.append(int(x.shape[i] * scale_factor))
    if mode.startswith("nearest"):
        mode = "nearest_neighbor"
    elif mode[0:2] == "bi":
        mode = mode[2:]
    if mode == "nearest_neighbor":
        coord_trans = "asymmetric"
    elif align_corners:
        coord_trans = "align_corners"
    else:
        coord_trans = "half_pixel"
    return _t(_ffi_op.interpolate(_v(x), size, data_layout, mode, coord_trans, name))


def ccl_allreduce(x: Tensor, op_type: str = "sum", in_group: bool = True, name="ccl_allreduce"):
    """Perform a collective all-reduce across workers.

    Parameters
    ----------
    x : Tensor
        Local tensor to reduce.
    op_type : str
        Reduction operation: ``"sum"``, ``"prod"``, ``"min"``, ``"max"``,
        or ``"avg"``.
    in_group : bool
        If ``True``, restrict the reduction to the current worker group.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Reduced tensor with the same shape as *x*.
    """
    return _t(_ffi_op.ccl_allreduce(_v(x), op_type, in_group, name))


def ccl_allgather(x: Tensor, num_workers: int, name="ccl_allgather"):
    """Gather tensors from all workers and concatenate along axis 0.

    Parameters
    ----------
    x : Tensor
        Local tensor to contribute.
    num_workers : int
        Total number of workers participating in the gather.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Concatenated tensor with leading dimension scaled by *num_workers*.
    """
    return _t(_ffi_op.ccl_allgather(_v(x), num_workers, name))


def ccl_broadcast_from_worker0(x: Tensor, name="broadcast_from_worker"):
    """Broadcast *x* from worker 0 to all other workers.

    Parameters
    ----------
    x : Tensor
        Tensor on worker 0 to broadcast.  On other workers the value is
        ignored and overwritten with the broadcast result.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor
        Tensor with the same shape and values as *x* on worker 0.
    """
    return _t(_ffi_op.ccl_broadcast_from_worker0(_v(x), name))


def tensor_expr_op(
    tensor_expr_func: Callable,
    name_hint: str,
    args: list[Tensor | _tir.Var | int],
    *,
    attrs: dict[str, Any] | None = None,
):
    """Build a ``te.compute`` kernel from *tensor_expr_func* and emit it.

    Parameters
    ----------
    tensor_expr_func : Callable
        A function ``(*te.Tensor) -> te.Tensor | list[te.Tensor]`` that
        describes the computation using TVM Tensor Expressions.
    name_hint : str
        Name hint for the generated ``PrimFunc`` and the emitted binding.
    args : list[Tensor | tir.Var | int]
        Inputs to the kernel.  ``nn.Tensor`` values are unwrapped to their
        underlying ``relax.Var``; ``tir.Var`` and ``int`` are passed as-is.
    attrs : dict[str, Any] | None
        Optional attribute dictionary attached to the generated
        ``PrimFunc``.

    Returns
    -------
    result : Tensor | tuple[Tensor, ...]
        The output tensor(s) produced by the kernel.
    """

    def _convert(arg):
        if isinstance(arg, Tensor):
            return arg._expr  # pylint: disable=protected-access
        return arg

    def _wrapped_func(te_inputs):
        """Unpack Array<te.Tensor>, call user func, repack result into Array."""
        unpacked = list(te_inputs)
        result = tensor_expr_func(*unpacked)
        if isinstance(result, list | tuple):
            return list(result)
        return [result]

    return _t(
        _ffi_op.tensor_expr_op(
            _wrapped_func,
            name_hint,
            [_convert(arg) for arg in args],
            attrs,
        )
    )


OutType = TypeVar("OutType", bound=Tensor | Sequence[Tensor])


def tensor_ir_op(
    func: _tir.PrimFunc,
    name_hint: str,
    args: Tensor | Sequence[Tensor | rx.ShapeExpr | _tir.PrimExpr],
    out: OutType,
) -> OutType:
    """Emit a ``call_tir`` binding that invokes a pre-built ``PrimFunc``.

    Parameters
    ----------
    func : tir.PrimFunc
        The TIR primitive function to call.
    name_hint : str
        Name hint for the emitted binding.
    args : Tensor | Sequence[Tensor | ShapeExpr | PrimExpr]
        Input arguments.  ``nn.Tensor`` values are unwrapped;
        ``ShapeExpr`` and ``PrimExpr`` are wrapped in a ``ShapeExpr``.
    out : Tensor | Sequence[Tensor]
        Pre-allocated output tensor(s) that describe the output shape and
        dtype.  Passed to ``call_tir`` as the destination.

    Returns
    -------
    result : Tensor | tuple[Tensor, ...]
        The output tensor(s), matching the type of *out*.
    """
    if not isinstance(args, tuple | list):
        args = [args]

    def _convert(arg):
        if isinstance(arg, Tensor):
            return arg._expr
        if isinstance(arg, rx.Var):
            return arg
        if isinstance(arg, rx.ShapeExpr):
            return arg
        if isinstance(arg, _tir.PrimExpr):
            return rx.ShapeExpr([arg])
        raise TypeError(f"tensor_ir_op: unsupported arg type {type(arg)}")

    if isinstance(out, Tensor):
        out_list = [out]
    else:
        out_list = list(out)

    return _t(
        _ffi_op.tensor_ir_op(
            func,
            name_hint,
            [_convert(a) for a in args],
            [o._expr for o in out_list],
        )
    )


def tensor_ir_inplace_op(
    func: _tir.PrimFunc,
    name_hint: str,
    args: Tensor | Sequence[Tensor | rx.ShapeExpr | _tir.PrimExpr],
    inplace_indices: int | list[int],
    out: OutType,
) -> OutType:
    """Emit a ``call_tir_inplace`` binding that writes results back into
    existing buffers.

    Parameters
    ----------
    func : tir.PrimFunc
        The TIR primitive function to call.
    name_hint : str
        Name hint for the emitted binding.
    args : Tensor | Sequence[Tensor | ShapeExpr | PrimExpr]
        Input arguments.  ``nn.Tensor`` values are unwrapped;
        ``ShapeExpr`` and ``PrimExpr`` are wrapped in a ``ShapeExpr``.
    inplace_indices : int | list[int]
        Index or indices into *args* that are written in-place.
    out : Tensor | Sequence[Tensor]
        Pre-allocated output tensor(s) describing the output shape and
        dtype.

    Returns
    -------
    result : Tensor | tuple[Tensor, ...]
        The output tensor(s), matching the type of *out*.
    """
    if not isinstance(args, tuple | list):
        args = [args]
    if isinstance(inplace_indices, int):
        inplace_indices = [inplace_indices]

    def _convert(arg):
        if isinstance(arg, Tensor):
            return arg._expr
        if isinstance(arg, rx.Var):
            return arg
        if isinstance(arg, rx.ShapeExpr):
            return arg
        if isinstance(arg, _tir.PrimExpr):
            return rx.ShapeExpr([arg])
        raise TypeError(f"tensor_ir_inplace_op: unsupported arg type {type(arg)}")

    if isinstance(out, Tensor):
        out_list = [out]
    else:
        out_list = list(out)

    return _t(
        _ffi_op.tensor_ir_inplace_op(
            func,
            name_hint,
            [_convert(a) for a in args],
            inplace_indices,
            [o._expr for o in out_list],
        )
    )


def extern(
    name: str,
    args: Sequence[Tensor | _tir.PrimExpr | int | float | str],
    out: OutType,
) -> OutType:
    """Invoke a named external (packed) function at runtime.

    Parameters
    ----------
    name : str
        Registered name of the external function (e.g. a TVM packed-func
        name or a symbol from an :class:`~extern.ExternModule`).
    args : Sequence[Tensor | PrimExpr | int | float | str]
        Arguments forwarded to the external function.  ``nn.Tensor`` values
        are unwrapped to their underlying ``relax.Var``; scalars and strings
        are passed as-is.
    out : Tensor | Sequence[Tensor]
        Pre-allocated output tensor(s) describing the expected output shape
        and dtype (destination-passing style).

    Returns
    -------
    result : Tensor | tuple[Tensor, ...]
        The output tensor(s), matching the type of *out*.
    """

    def _convert(arg):
        if isinstance(arg, Tensor):
            return arg._expr
        return arg  # int, float, str, PrimExpr passed as-is to C++

    if isinstance(out, Tensor):
        out_list = [out]
    else:
        out_list = list(out)

    return _t(
        _ffi_op.extern(
            name,
            [_convert(a) for a in args],
            [o._expr for o in out_list],
        )
    )


def debug_func(
    name: str,
    *args: Tensor | _tir.PrimExpr | int | float | str,
    _line_info: str | None = None,
):
    """Call a debug callback function at runtime, threading the IO effect token.

    The IO effect token is obtained from the C++ thread-local set by
    ``ExportToIRModule``, or from the Python ``Exporter`` when running on
    the Python export path.  Subsequent calls chain the updated token so
    that debug calls are sequenced correctly in the IR.

    Parameters
    ----------
    name : str
        Registered name of the debug function to call (e.g.
        ``"vm.builtin.debug_print"``).
    *args : Tensor | PrimExpr | int | float | str
        Arguments forwarded to the debug function.  ``nn.Tensor`` values
        are unwrapped to their underlying ``relax.Var``.
    _line_info : str | None
        Source location string ``"filename:lineno"`` injected into the IR.
        Auto-detected from the caller's frame when ``None``.
    """
    # Get the current _io var from the C++ thread-local (set by ExportToIRModule)
    # or fall back to the Python Exporter if available.
    io_var = _ffi_api.GetCurrentIOVar()
    if io_var is None:
        # Python Exporter path
        from .exporter import Exporter  # pylint: disable=import-outside-toplevel

        if not hasattr(Exporter._tls, "current"):
            raise RuntimeError("Debugging is only supported when debug mode is on.")
        io_var = Exporter.current().io_effect.effect

    if _line_info is None:
        filename, line_number = inspect.getframeinfo(inspect.currentframe().f_back)[:2]
        _line_info = f"{filename}:{line_number}"

    def _convert(arg):
        if isinstance(arg, Tensor):
            return arg._expr
        return arg  # int, float, str, PrimExpr passed as-is to C++

    new_io = _ffi_op.debug_func(
        name,
        [_convert(a) for a in args],
        io_var,
        _line_info,
    )
    # Update the C++ thread-local so subsequent debug_func calls chain correctly
    _ffi_api.SetCurrentIOVar(new_io)


def print_(tensor: Tensor):
    """Print *tensor* values to stdout at runtime via the debug IO effect.

    Equivalent to ``debug_func("vm.builtin.debug_print", tensor)`` with
    the caller's source location automatically recorded.

    Parameters
    ----------
    tensor : Tensor
        The tensor whose values to print.
    """
    filename, line_number = inspect.getframeinfo(inspect.currentframe().f_back)[:2]
    line_info = f"{filename}:{line_number}"
    debug_func("vm.builtin.debug_print", tensor, _line_info=line_info)


def less(a: Tensor, b: Tensor, name: str = "less") -> Tensor:
    """Broadcasted element-wise comparison for (lhs < rhs)."""
    return _t(_ffi_op.less(_v(a), _v(b), name))


def less_equal(a: Tensor, b: Tensor, name: str = "less_equal") -> Tensor:
    """Broadcasted element-wise comparison for (lhs <= rhs)."""
    return _t(_ffi_op.less_equal(_v(a), _v(b), name))


def greater(a: Tensor, b: Tensor, name: str = "greater") -> Tensor:
    """Broadcasted element-wise comparison for (lhs > rhs)."""
    return _t(_ffi_op.greater(_v(a), _v(b), name))


def greater_equal(a: Tensor, b: Tensor, name: str = "greater_equal") -> Tensor:
    """Broadcasted element-wise comparison for (lhs >= rhs)."""
    return _t(_ffi_op.greater_equal(_v(a), _v(b), name))


def equal(a: Tensor, b: Tensor, name: str = "equal") -> Tensor:
    """Broadcasted element-wise comparison for (lhs == rhs)."""
    return _t(_ffi_op.equal(_v(a), _v(b), name))


def not_equal(a: Tensor, b: Tensor, name: str = "not_equal") -> Tensor:
    """Broadcasted element-wise comparison for (lhs != rhs)."""
    return _t(_ffi_op.not_equal(_v(a), _v(b), name))


def where(condition: Tensor, x1: Tensor, x2: Tensor, name: str = "where") -> Tensor:
    """Selecting elements from either the input tensors depending on the value of the condition."""
    return _t(_ffi_op.where(_v(condition), _v(x1), _v(x2), name))


def cumsum(
    data: Tensor,
    axis: int | None = None,
    dtype: str | None = None,
    exclusive: bool | None = None,
    name: str = "cumsum",
) -> Tensor:
    """Numpy style cumsum op."""
    return _t(_ffi_op.cumsum(_v(data), axis, dtype, exclusive, name))


def sort(x: Tensor, axis: int = -1, descending: bool = False, name="sort"):
    """Performs sorting along the given axis."""
    return _t(_ffi_op.sort(_v(x), axis, descending, name))


def argsort(
    data: Tensor, axis: int = -1, descending: bool = False, dtype: str = "int32", name="argsort"
):
    """Performs sorting and returns indices."""
    return _t(_ffi_op.argsort(_v(data), axis, descending, dtype, name))


def topk(
    data: Tensor,
    k: int = 1,
    axis: int = -1,
    ret_type: str = "both",
    largest: bool = True,
    dtype: str = "int32",
    name: str = "topk",
):
    """Get the top k elements in an input tensor along the given axis."""
    return _t(_ffi_op.topk(_v(data), k, axis, ret_type, largest, dtype, name))


def multinomial_from_uniform(
    prob: Tensor,
    uniform_sample: Tensor,
    sample_indices: Tensor | None = None,
    dtype: str = "int64",
    name: str = "multinomial_from_uniform",
):
    """Returns a tensor where each row contains the index sampled from the multinomial
    probability distribution.
    """
    out_batch = uniform_sample.shape[0]
    if sample_indices is not None:
        assert sample_indices.shape == uniform_sample.shape, (
            "The shape of sample_indices must match the shape of uniform_sample."
        )
    else:
        assert prob.shape[0] == uniform_sample.shape[0], (
            "Number of samples must match the number of probability distributions."
        )
        sample_indices = Tensor.from_const(np.arange(out_batch).reshape(out_batch, 1))
    return _t(
        _ffi_op.multinomial_from_uniform(
            _v(prob), _v(uniform_sample), _v(sample_indices), dtype, name
        )
    )


def sample_top_p_top_k_from_sorted_prob(
    sorted_prob: Tensor,
    sorted_index: Tensor,
    top_p: Tensor,
    top_k: Tensor,
    uniform_sample: Tensor,
    sample_indices: Tensor | None = None,
):
    """Samples indices from a sorted probability tensor based on top_p and top_k criteria.

    Notes
    -----
    For accurate results, ensure probabilities are between 0 and 1 and sum to 1.

    Parameters
    ----------
    sorted_prob : Tensor
        A 2-D tensor, with shape (batch, vocab_size), contains probabilities
        sorted in descending order.

    sorted_index: Tensor
        The indices tensor with shape (batch, vocab_size), corresponding to the
        sorted_prob. Potentially from applying argsort on the original probability
        tensor in descending order.

    top_p : Tensor
        The cumulative probability threshold with shape (batch, 1) for nucleus sampling.

    top_k :Tensor
        A tensor with shape (batch, 1), representing the number of top probabilities
        to consider for top-k sampling.

    uniform_sample : Tensor
        Uniformly sampled values with shape (n, 1) are used to select the output indices.

    sample_indices : Optional[Tensor]
        The 2-D tensor with the shape [n, 1], which indicates the specific
        probability distribution to sample from. The value of sample_indices[i]
        determines that the ith token should be sampled from the sample_indices[i]th
        probability distribution. For instance, if there are 3 distinct probability
        distributions and the requirement is to sample 2, 3, and 4 tokens from each,
        then sample_indices would be [0, 0, 1, 1, 1, 2, 2, 2, 2].

    Returns
    -------
    result : Tensor
        The selected indices with shape (n, 1).

    Examples
    --------
    .. code-block:: python

        prob = [[0.1 , 0.4, 0.5],
                [0.3, 0.3, 0.4]]
        sorted_prob = [[0.5, 0.4, 0.1],
                       [0.4, 0.3, 0.3]]
        sorted_index = [[2, 1, 0],
                        [2, 0, 1]]
        top_p = [[0.6],[0.9]]
        top_k = [[3],[2]]
        uniform_sample = [[0.5], [0.6]]
        sample_indices = [[0], [1]]

        sample_top_p_top_k_from_sorted_prob(
            sorted_prob, sorted_index,top_p, top_k, uniform_sample, sample_indices)
        -> [2, 0]

    """
    return _t(
        _ffi_op.sample_top_p_top_k_from_sorted_prob(
            _v(sorted_prob),
            _v(sorted_index),
            _v(top_p),
            _v(top_k),
            _v(uniform_sample),
            _v(sample_indices) if sample_indices is not None else None,
        )
    )


def renormalize_top_p_top_k_prob(prob, sorted_prob, top_p, top_k):
    """Renormalizes probabilities after filtering with top_p and top_k, ensuring
    they sum up to 1.

    Notes
    -----
    For accurate results, ensure probabilities are between 0 and 1 and sum to 1.

    Parameters
    ----------
    prob : Tensor
        A 2-D tensor of shape (batch, vocab_size) representing probability distributions.

    sorted_prob : Tensor
        Probabilities sorted in descending order.

    top_p : Tensor
        The cumulative probability threshold with shape (batch, 1) for nucleus sampling.

    top_k :Tensor
        A tensor with shape (batch, 1), representing the number of top probabilities
        to consider for top-k sampling.

    Returns
    -------
    result : Tensor
        The filtered and nomalized tensor with the sampe shape as input prob.
    """
    return _t(_ffi_op.renormalize_top_p_top_k_prob(_v(prob), _v(sorted_prob), _v(top_p), _v(top_k)))
