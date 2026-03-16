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
"""Default legalization function for BatchMatmul quantized neural network operator."""

from tvm import topi

from ....block_builder import BlockBuilder
from ....expr import Call, Expr
from ..common import register_legalize


@register_legalize("relax.qnn.batch_matmul")
def _qnn_batch_matmul(bb: BlockBuilder, call: Call) -> Expr:
    """
    Legalize relax.qnn.batch_matmul to TOPI qnn.batch_matmul.
    - args[0]: x (first input tensor)
    - args[1]: y (second input tensor)
    - args[2]: x_zero_point
    - args[3]: y_zero_point
    - args[4]: x_scale
    - args[5]: y_scale
    - attrs.out_dtype: output data type
    """
    args = call.args
    out_dtype = call.attrs.out_dtype
    x_zp_val = args[2].data.numpy().item()
    y_zp_val = args[3].data.numpy().item()

    return bb.call_te(
        topi.qnn.batch_matmul,
        tensor_x=args[0],
        tensor_y=args[1],
        x_zp_val=x_zp_val,
        y_zp_val=y_zp_val,
        out_dtype=out_dtype,
        primfunc_name_hint="qnn_batch_matmul",
    )
