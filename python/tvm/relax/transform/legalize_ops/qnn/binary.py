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
from .utils import qnn_binary_ops_extract_params


@register_legalize("relax.qnn.add")
def _qnn_add(bb: BlockBuilder, call: Call) -> Expr:
    res = qnn_binary_ops_extract_params(call, "qnn.add")
    if not res:
        return call
    lhs, rhs, l_sc, l_zp, r_sc, r_zp, o_sc, o_zp = res

    return bb.call_te(topi.qnn.add, lhs, rhs, l_sc, l_zp, r_sc, r_zp, o_sc, o_zp)


@register_legalize("relax.qnn.subtract")
def _qnn_subtract(bb: BlockBuilder, call: Call) -> Expr:
    res = qnn_binary_ops_extract_params(call, "qnn.subtract")
    if not res:
        return call
    lhs, rhs, l_sc, l_zp, r_sc, r_zp, o_sc, o_zp = res

    return bb.call_te(topi.qnn.subtract, lhs, rhs, l_sc, l_zp, r_sc, r_zp, o_sc, o_zp)


@register_legalize("relax.qnn.multiply")
def _qnn_mul(bb: BlockBuilder, call: Call) -> Expr:
    res = qnn_binary_ops_extract_params(call, "qnn.mul")
    if not res:
        return call
    lhs, rhs, l_sc, l_zp, r_sc, r_zp, o_sc, o_zp = res

    return bb.call_te(topi.qnn.mul, lhs, rhs, l_sc, l_zp, r_sc, r_zp, o_sc, o_zp)
