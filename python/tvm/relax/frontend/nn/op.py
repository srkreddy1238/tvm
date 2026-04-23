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


def _v(t: Tensor) -> rx.Var:
    """Unwrap nn.Tensor to its underlying relax.Var."""
    return t._expr  # pylint: disable=protected-access


def _t(result) -> "Tensor | tuple":
    """Wrap a C++ FFI result (Var or Array) back into nn.Tensor / tuple."""
    return _unwrap_ffi_result(result)


def unsqueeze(x: Tensor, dim: int, name: str = "unsqueeze") -> Tensor:
    """Add a new axis to a tensor"""
    return _t(_ffi_op.unsqueeze(_v(x), dim, name))


def concat(x: list[Tensor], dim: int, name: str = "concat") -> Tensor:
    """Concatenate a list of tensors along an axis."""
    return _t(_ffi_op.concat([_v(t) for t in x], dim, name))


def add(a: Tensor, b: Tensor, name: str = "add") -> Tensor:
    """Addition with numpy-style broadcasting."""
    return _t(_ffi_op.add(_v(a), _v(b), name))


def subtract(a: Tensor, b: Tensor, name: str = "subtract") -> Tensor:
    """Subtraction with numpy-style broadcasting."""
    return _t(_ffi_op.subtract(_v(a), _v(b), name))


def multiply(a: Tensor, b: Tensor, name: str = "mul") -> Tensor:
    """Multiplication with numpy-style broadcasting."""
    return _t(_ffi_op.multiply(_v(a), _v(b), name))


def divide(a: Tensor, b: Tensor, name: str = "divide") -> Tensor:
    """Division with numpy-style broadcasting."""
    return _t(_ffi_op.divide(_v(a), _v(b), name))


def chunk(x: Tensor, chunks: int, dim: int = 0, name: str = "chunk") -> Tensor:
    """Split a tensor along dim into the specified number of chunks."""
    return _t(_ffi_op.chunk(_v(x), chunks, dim, name))


def sum(
    x: Tensor,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
    name: str = "sum",
) -> Tensor:
    """Computes the sum of tensor elements over given axes."""
    ax = [axis] if isinstance(axis, int) else (axis or [])
    return _t(_ffi_op.sum(_v(x), ax, keepdims, name))


def max(
    x: Tensor,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
    name: str = "max",
) -> Tensor:
    """Computes the max of tensor elements over given axes."""
    ax = [axis] if isinstance(axis, int) else (axis or [])
    return _t(_ffi_op.max(_v(x), ax, keepdims, name))


def min(
    x: Tensor,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
    name: str = "min",
) -> Tensor:
    """Computes the min of tensor elements over given axes."""
    ax = [axis] if isinstance(axis, int) else (axis or [])
    return _t(_ffi_op.min(_v(x), ax, keepdims, name))


def matmul(a: Tensor, b: Tensor, out_dtype: str | None = None, name: str = "matmul") -> Tensor:
    """General matrix multiplication of two tensors, with broadcasting on batched dimensions."""
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
    """1D convolution."""
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
    """Applies a 2D convolution."""
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
    """Applies a 3D convolution."""
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
    """1D transposed convolution operator."""
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
    """Layer normalization."""
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
    """Root mean square normalization."""
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
    """Group normalization."""
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
    """Timestep calculation as described in Denoising Diffusion Probabilistic Models."""
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
    """Computes a scaled dot product attention."""
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
    """Resize a tensor using the specified mode."""
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
    """CCL Allreduce operator."""
    return _t(_ffi_op.ccl_allreduce(_v(x), op_type, in_group, name))


def ccl_allgather(x: Tensor, num_workers: int, name="ccl_allgather"):
    """CCL Allgather operator."""
    return _t(_ffi_op.ccl_allgather(_v(x), num_workers, name))


def ccl_broadcast_from_worker0(x: Tensor, name="broadcast_from_worker"):
    """Broadcast data from worker-0 to all other workers."""
    return _t(_ffi_op.ccl_broadcast_from_worker0(_v(x), name))


def tensor_expr_op(
    tensor_expr_func: Callable,
    name_hint: str,
    args: list[Tensor | _tir.Var | int],
    *,
    attrs: dict[str, Any] | None = None,
):
    """Build the given tensor_expr_func with te."""

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
    """Create a `call_tir` binding with given PrimFunc."""
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
    """Create a `call_tir_inplace` binding with given PrimFunc."""
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
    """Invoke an extern function during runtime."""

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
    """Call a debug function during runtime."""
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
    """Debug printing a Tensor during runtime."""
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
