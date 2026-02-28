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

from tvm import relax

from ...expr import Expr
from . import _ffi_api


def requantize(
    data: Expr,
    input_scale: Expr,
    input_zero_point: Expr,
    output_scale: Expr,
    output_zero_point: Expr,
    axis: int = -1,
    out_dtype: str = "int8",
) -> Expr:
    r"""Requantize op per channel
    Requantization typically happens after a computation like qnn.conv2d.
    It rescales from one quantized range (input scale/zero point) to another
    (output scale/zero point).
    Q_output = clamp(round((inp_tensor - inp_zp) * (inp_scale / out_scale)) + out_zp,
                     out_dtype::min, out_dtype::max)

    Parameters
    ----------
    data : tvm.relax.Expr
        The input tensor to be requantized (typically int32).

    input_scale : tvm.relax.Expr
        The input scale.

    input_zero_point : tvm.relax.Expr
        The input zero_point.

    output_scale : tvm.relax.Expr
        The output scale.

    output_zero_point : tvm.relax.Expr
        The output zero_point.

    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.

    out_dtype : str, optional
        The data type of the output tensor.

    Returns
    -------
    result : tvm.relax.Expr
        The requantized output tensor.
    """

    if isinstance(input_scale, list | tuple):
        input_scale = relax.Tuple(list(input_scale))
    if isinstance(input_zero_point, list | tuple):
        input_zero_point = relax.Tuple(list(input_zero_point))

    return _ffi_api.requantize(
        data, input_scale, input_zero_point, output_scale, output_zero_point, axis, out_dtype
    )
