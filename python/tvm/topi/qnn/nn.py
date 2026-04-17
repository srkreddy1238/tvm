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
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin
"""TOPI implementations of quantized neural network operators."""

import tvm
from tvm import te
from tvm.topi.nn import conv, conv2d_transpose_nchw, conv2d_transpose_nhwc

from ..utils import get_const_tuple
from .utils import is_scalar_tensor, subtract_zero_point


def conv2d(  # Conv2d inputs
    data,
    weight,
    # Conv2d quantization params:
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    # Conv2d attributes:
    strides,
    padding,
    dilation,
    groups,
    out_dtype,
    data_layout,
    kernel_layout,
):
    """Compute for qnn.conv2d with NCHWc layout."""

    # Subtract zero point from input and weights.
    weight = subtract_zero_point(weight, kernel_zero_point, "weight_zp")
    data = subtract_zero_point(data, input_zero_point, "data_zp")

    strides = get_const_tuple(strides)
    padding = get_const_tuple(padding)
    dilation = get_const_tuple(dilation)

    # Handling Default Case
    if len(data_layout) == 4 and len(kernel_layout) == 4:
        out = conv(
            data, weight, strides, padding, dilation, groups, data_layout, kernel_layout, "int32"
        )
    else:
        raise ValueError(
            f"Can't handle legalization for the given layouts {data_layout} and {kernel_layout}"
        )

    # Handle Scale if supplied
    if input_scale is not None:
        assert is_scalar_tensor(input_scale)
        out = te.compute(
            out.shape,
            lambda *indices: tvm.tir.multiply(out(*indices), input_scale).astype(out_dtype),
            name="input_scale",
        )

    if kernel_scale is not None:
        if is_scalar_tensor(kernel_scale):
            out = te.compute(
                out.shape,
                lambda *indices: tvm.tir.multiply(out(*indices), kernel_scale).astype(out_dtype),
                name="kernel_scale",
            )
        else:
            oc_idx = tvm.s_tir.layout(data_layout).index_of("C")
            out = te.compute(
                out.shape,
                lambda *indices: tvm.tir.multiply(
                    out(*indices), kernel_scale[indices[oc_idx]]
                ).astype(out_dtype),
                name="kernel_scale",
            )

    return out


def conv2d_transpose(
    data,
    weight,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    strides,
    padding,
    output_padding,
    dilation,
    groups: int,
    out_dtype: str,
    data_layout: str,
    kernel_layout: str,
):
    """
    TOPI compute for qnn.conv2d_transpose.

    """

    weight = subtract_zero_point(weight, kernel_zero_point, "weight_zp")
    data = subtract_zero_point(data, input_zero_point, "data_zp")

    strides = get_const_tuple(strides)
    padding = get_const_tuple(padding)
    output_padding = get_const_tuple(output_padding)
    dilation = get_const_tuple(dilation)

    if data_layout == "NCHW" and kernel_layout == "IOHW":
        out = conv2d_transpose_nchw(
            data,
            weight,
            strides,
            padding,
            out_dtype,
            output_padding,
        )
    elif data_layout == "NHWC" and kernel_layout == "OHWI":
        out = conv2d_transpose_nhwc(
            data,
            weight,
            strides,
            padding,
            out_dtype,
            output_padding,
        )
    else:
        raise ValueError(
            f"qnn_conv2d_transpose: cannot handle layouts "
            f"data_layout={data_layout!r}, kernel_layout={kernel_layout!r}. "
            f"Only 4-D layouts (e.g. NCHW / IOHW ; NHWC / OIHW) are currently supported."
        )

    if input_scale is not None and kernel_scale is not None:
        out = te.compute(
            out.shape,
            lambda *i: tvm.tir.multiply(
                out(*i),
                tvm.tir.multiply(input_scale, kernel_scale),
            ).astype(out_dtype),
            name="scale",
        )

    return out
