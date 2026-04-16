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
"""Legalization function for quantized dense operator."""

from tvm import topi

from ....block_builder import BlockBuilder
from ....expr import Call, Expr
from ..common import register_legalize


@register_legalize("relax.qnn.dense")
def _qnn_dense(bb: BlockBuilder, call: Call) -> Expr:
    """Legalize QNN dense operation."""

    data = call.args[0]
    weight = call.args[1]
    input_zero_point = call.args[3]
    kernel_zero_point = call.args[5]

    out_dtype = call.attrs.out_dtype if hasattr(call.attrs, "out_dtype") else "int32"

    input_zero_point = input_zero_point.data.numpy().item()
    kernel_zero_point = kernel_zero_point.data.numpy().item()

    return bb.call_te(
        topi.qnn.dense,
        data,
        weight,
        input_zero_point,
        kernel_zero_point,
        out_dtype,
    )
