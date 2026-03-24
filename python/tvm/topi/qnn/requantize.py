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
"""QNN Requantize operator"""

from tvm import te, tir

from .utils import get_qnn_param, saturate


def requantize(
    data: te.Tensor,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    axis=-1,
    out_dtype="int8",
):
    """Compute for qnn.requantize
     If both input and output scales are constant scalars then we convert scale to fixed point value
    and use integer arithmetic only for performance optimization purpose.
    But this is a tradeoff between performance and accuracy, since we use int16 data type to
    represent fixed point values (against QNN lowering approach where we use int32 for that).

    if input and/or output scales are not constant scalars then we use the following formula:
        Q_output = zp_output + round((scale_input)/(scale_output) * (Q_input - zp_input))
    """

    if isinstance(input_scale, float) and isinstance(output_scale, float):

        def _compute(*indices):
            value = data(*indices)
            sub = te.subtract(value, tir.Cast("int16", input_zero_point))
            scale = input_scale / output_scale
            mul = te.multiply(scale, sub)
            val = te.add(te.round(mul), output_zero_point)
            return saturate(val, out_dtype).astype(out_dtype)

        return te.compute(data.shape, _compute, name="requantize_scalar")
    else:
        # Generic compute def
        def _compute(*indices):
            value = data(*indices)
            iscale = get_qnn_param(input_scale, indices, axis)
            inp_zp = get_qnn_param(input_zero_point, indices, axis)
            sub = te.subtract(value, tir.Cast("int16", inp_zp))
            scale = te.div(iscale, output_scale)
            mul = te.multiply(scale, sub)
            val = te.add(te.round(mul), output_zero_point)
            return saturate(val, out_dtype).astype(out_dtype)

        return te.compute(data.shape, _compute, name="requantize")
