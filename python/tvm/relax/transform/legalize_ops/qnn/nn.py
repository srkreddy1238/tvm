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
import logging
from tvm import tir, topi
from ....block_builder import BlockBuilder
from ....expr import Call, Expr
from ..common import register_legalize


@register_legalize("relax.qnn.conv2d")
def _qnn_conv2d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.data_layout:
        logging.info(
            "TOPI conv2d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call
    if len(call.attrs.data_layout) != 4 or len(call.attrs.kernel_layout) != 4:
        logging.info(
            "Conv2D where data layout or kernel layout have channel chunk "
            "cannot be legalized by TOPI at this moment."
        )
        return call
    if call.attrs.groups != 1:
        data_layout = tir.layout(call.attrs.data_layout)
        kernel_layout = tir.layout(call.attrs.kernel_layout)
        ic = call.args[0].struct_info.shape.values[data_layout.index_of("C")]
        oc = call.args[1].struct_info.shape.values[kernel_layout.index_of("O")]
        if not isinstance(ic, tir.IntImm) or not isinstance(oc, tir.IntImm):
            logging.info(
                "Conv2D where number of groups is more than one and input or output "
                "channel size is symbolic cannot be legalized by TOPI at this moment."
            )
            return call

    return bb.call_te(
        topi.qnn.conv2d,
        data=call.args[0],
        weight=call.args[1],
        input_zero_point=call.args[2],
        kernel_zero_point=call.args[3],
        input_scale=call.args[4] if len(call.args) > 4 else None,
        kernel_scale=call.args[5] if len(call.args) > 4 else None,
        strides=call.attrs.strides,
        padding=call.attrs.padding,
        dilation=call.attrs.dilation,
        groups=call.attrs.groups,
        out_dtype=call.attrs.out_dtype,
        data_layout=call.attrs.data_layout,
        kernel_layout=call.attrs.kernel_layout,
        primfunc_name_hint="qnn_conv2d",
    )
