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
"""Relax QNN Dense operator"""

from ...expr import Expr
from . import _ffi_api


def dense(
    data: Expr,
    weight: Expr,
    input_scale: Expr,
    input_zero_point: Expr,
    kernel_scale: Expr,
    kernel_zero_point: Expr,
    out_dtype="int32",
) -> Expr:
    """Quantized dense operator.

    Computes quantized dense (fully connected) layer without using floating point.
    The computation is: output_int32 = (data - data_zp) @ (weight - weight_zp)

    Note: This operator does NOT apply output quantization. The output is int32/int64
    with effective scale = input_scale * kernel_scale and zero_point = 0.
    Use requantize() afterward if you need int8 output.

    Parameters
    ----------
    data : Expr
        The input quantized tensor, shape: [batch, in_features] or [in_features]
        Typically int8 or uint8

    weight : Expr
        The weight quantized tensor, shape: [units, in_features]
        Must be already transposed (TFLite convention)
        Typically int8 or uint8

    input_scale : Expr
        Scale for the input tensor (scalar float32)

    input_zero_point : Expr
        Zero point for the input tensor (scalar, same dtype as data)

    kernel_scale : Expr
        Scale for the weight/kernel tensor (scalar float32)

    kernel_zero_point : Expr
        Zero point for the weight/kernel tensor (scalar, same dtype as weight)

    out_dtype : str
        Output data type, typically "int32" or "int64" for accumulation
        Default: "int32"

    """
    return _ffi_api.dense(  # type: ignore
        data,
        weight,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        out_dtype,
    )
