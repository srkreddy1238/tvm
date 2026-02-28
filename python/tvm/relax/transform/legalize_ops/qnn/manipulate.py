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

from tvm import topi

from ....block_builder import BlockBuilder
from ....expr import Call, Expr
from ..common import register_legalize


@register_legalize("relax.qnn.concat")
def _qnn_concat(bb: BlockBuilder, call: Call) -> Expr:
    # Get the attributes from the call
    args = call.args
    output_dtype = args[0].fields[0].struct_info.dtype

    in_scales = [s.data.numpy().item() for s in args[1]]
    in_zps = [z.data.numpy().item() for z in args[2]]

    out_scale = args[3].data.numpy().item()
    out_zp = args[4].data.numpy().item()

    return bb.call_te(
        topi.qnn.concat,
        data=list(args[0]),
        input_scales=in_scales,
        input_zero_points=in_zps,
        o_scale=out_scale,
        o_zp=out_zp,
        axis=call.attrs.axis,
        out_dtype=output_dtype,
        primfunc_name_hint="qnn_concat",
    )
