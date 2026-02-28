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
# pylint: disable=invalid-name,unused-argument,unused-import
"""Default legalization function for quantized neural network operators."""

from tvm import relax, topi

from .....topi.qnn.utils import get_fixed_point_value
from ....block_builder import BlockBuilder
from ....expr import Call, Expr
from ..common import register_legalize
from .utils import get_values_from_expr


@register_legalize("relax.qnn.requantize")
def _qnn_requantize(bb: BlockBuilder, call: Call) -> Expr:
    # Get the attributes from the call
    args = call.args
    input_scale = args[1]
    input_zp = args[2]
    output_scale = get_values_from_expr(args[3])
    output_zp = get_values_from_expr(args[4])

    scale_fixed_point_list, rsh_list = [], []
    # for non scalar input scales
    if input_scale.struct_info.ndim != 0:
        i_scale = input_scale.data.numpy()
        # precompute the scale fixed point and rsh per axis/channel
        for i in range(len(i_scale)):
            scale = i_scale[i] / output_scale
            scale_fixed_point, rsh = get_fixed_point_value(scale)
            scale_fixed_point_list.append(scale_fixed_point)
            rsh_list.append(rsh)

        # convert to relax const tensor before passing to call_te()
        scale_fixed_point_list = relax.const(scale_fixed_point_list)
        rsh_list = relax.const(rsh_list)
    else:
        # for scalar input scale extract value and send to call_te()
        input_scale = get_values_from_expr(input_scale)
        input_zp = get_values_from_expr(input_zp)

    return bb.call_te(
        topi.qnn.requantize,
        data=args[0],
        input_scale=input_scale,
        input_zp=input_zp,
        output_scale=output_scale,
        output_zp=output_zp,
        axis=call.attrs.axis,
        out_dtype=call.attrs.out_dtype,
        scale_fixed_point_list=scale_fixed_point_list,
        rsh_list=rsh_list,
        primfunc_name_hint="qnn_requantize",
    )
