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
"""Relax Neural Network (QNN) operators"""

from tvm import DataType

from ...expr import Expr
from . import _ffi_api


def conv2d(
    data: Expr,
    weight: Expr,
    data_zero_point: Expr,
    weight_zero_point: Expr,
    data_scale: Expr | None,
    weight_scale: Expr | None,
    strides: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, ...] = (0, 0),
    dilation: int | tuple[int, int] = (1, 1),
    groups: int = 1,
    data_layout: str = "NCHW",
    kernel_layout: str = "OIHW",
    out_layout: str | None = None,
    out_dtype: str | DataType | None = None,
) -> Expr:
    r"""2D convolution.

    This operator takes the weight as the convolution kernel
    and convolves it with data to produce an output.


    In the default case, where the data_layout is `NCHW`
    and kernel_layout is `OIHW`, conv2d takes in
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    and a weight Tensor with shape `(channels, in_channels, kernel_h, kernel_w)`,
    where `kernel_h` and `kernel_w` is the lengths of the `H` and `W` kernel dimensions,
    to produce an output Tensor with the following rule:

    .. math::

        \mbox{out}[b, c, y, x] = \sum_{dy, dx, k}
           \mbox{data}[b, k, \mbox{strides}[0] * y  + dy, \mbox{strides}[1] * x + dx] *
           \mbox{weight}[c, k, dy, dx]

    Padding and dilation are applied to data and weight respectively before the computation.
    This operator accepts data layout specification.
    Semantically, the operator will convert the layout to the canonical layout
    (`NCHW` for data and `OIHW` for weight), perform the computation,
    then convert to the out_layout.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    weight : relax.Expr
        The weight expressions.

    data_zero_point: relax.Expr
        The zero point of the quantized data distribution.

    kernel_zero_point: relax.Expr
        The zero point of the quantized kernel distribution.

    input_scale: relax.Expr
        Optional scale of input tensor.

    weight_scale: relax.Expr
        Optional scale of weight tensor.

    strides : Union[int, Tuple[int, int]]
        The strides of convolution. It is required to have length either 1 or 2.

    padding : Union[int, Tuple[int, ...]]
        The padding of convolution on both sides of inputs before convolution.
        It is required to have length either 1, 2 or 4.

    dilation : Union[int, Tuple[int, int]]
        Specifies the dilation rate to be used for dilated convolution.
        It is required to have length either 1 or 2.

    groups : int
        Number of groups to split the input into for grouped convolution.
        The number of input and output channels should be divisible by the number of groups.

    data_layout : str
        Layout of the input.

    kernel_layout : str
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output. If not specified, it is the same as data_layout

    out_dtype : Optional[Union[str, DataType]]
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)

    return _ffi_api.conv2d(  # type: ignore
        data,
        weight,
        data_zero_point,
        weight_zero_point,
        data_scale,
        weight_scale,
        strides,
        padding,
        dilation,
        groups,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )
