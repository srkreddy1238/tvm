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
"""Relax QNN Manipulate operators"""

from tvm import relax

from . import _ffi_api


def concat(data, input_scales, input_zero_points, output_scale, output_zero_point, axis):
    """Concatenate the quantized input tensors along the given axis.

    Parameters
    ----------
    data : List[relax.Expr]
        The list of quantized tensors.

    input_scales : List[relax.Expr]
        The list of scales of input quantized tensors.

    input_zero_points : List[relax.Expr]
        The list of zero points of input quantized tensors.

    output_scale : relax.Expr
        The scale of the output quantized tensor.

    output_zero_point : relax.Expr
        The zero point of the output quantized tensor.

    axis : int
        The axis along which the tensors are concatenated.

    Returns
    -------
    result: relax.Expr
        The concatenated quantized tensor.
    """

    if isinstance(data, list | tuple):
        data = relax.Tuple(list(data))
    if isinstance(input_scales, list | tuple):
        input_scales = relax.Tuple(list(input_scales))
    if isinstance(input_zero_points, list | tuple):
        input_zero_points = relax.Tuple(list(input_zero_points))

    return _ffi_api.concat(
        data, input_scales, input_zero_points, output_scale, output_zero_point, axis
    )
