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

from tvm import te

from .utils import get_fixed_point_value, get_qnn_param, saturate


def requantize(
    data: te.Tensor,
    input_scale,
    input_zp,
    output_scale,
    output_zp,
    axis=-1,
    out_dtype="int8",
    scale_fixed_point_list=None,
    rsh_list=None,
):
    """Compute for qnn.requantize
     If both input and output scales are constant scalars then we convert scale to fixed point value
    and use integer arithmetic only for performance optimization purpose.
    But this is a tradeoff between performance and accuracy, since we use int16 data type to
    represent fixed point values (against QNN lowering approach where we use int32 for that).

    if input and/or output scales are not constant scalars then we use the following formula:
        Q_output = zp_output + round((scale_input)/(scale_output) * (Q_input - zp_input))
    """

    if isinstance(input_scale, (float)) and isinstance(output_scale, (float)):
        scale = input_scale / output_scale
        scale_fixed_point, rsh = get_fixed_point_value(scale)

        def _compute(*indices):
            value = data(*indices)
            # Subtract input zero point (scalar expr)
            sub = te.subtract(value, input_zp)
            # Fixed point multiply + round-to-nearest via bias (1 << (rsh - 1))
            mul = (sub * scale_fixed_point + (1 << (rsh - 1))) >> rsh
            # Add output zero point + clip + cast
            return saturate(te.add(mul, output_zp), out_dtype).astype(out_dtype)

        return te.compute(data.shape, _compute, name="requantize_scalar")

    else:
        # find the scale fixed point and rsh for each tensor along axis
        def _compute(*indices):
            value = data(*indices)
            scale_fixed_point = get_qnn_param(scale_fixed_point_list, indices, axis)
            rsh = get_qnn_param(rsh_list, indices, axis)
            inp_zp = get_qnn_param(input_zp, indices, axis)

            sub = te.subtract(value, inp_zp)
            # Fixed point multiply + round-to-nearest via bias (1 << (rsh - 1))
            mul = (sub * scale_fixed_point + (1 << (rsh - 1))) >> rsh
            # Add output zero point + clip + cast
            return saturate(te.add(mul, output_zp), out_dtype).astype(out_dtype)

        return te.compute(data.shape, _compute, name="requantize_vector")
