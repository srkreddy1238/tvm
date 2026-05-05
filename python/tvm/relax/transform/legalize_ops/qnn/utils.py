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
"""Common functionality for legalization."""

import logging

from tvm import relax

from ....expr import Call, Constant


def extract_relax_const(expr, param_name, op_name):
    """
    Extract the constant value from a Relax expression.
    """
    if isinstance(expr, Constant):
        return expr.data.numpy().item()
    logging.warning(f"{op_name} parameter '{param_name}' must be constant. Got {type(expr)}.")
    return None


def qnn_binary_ops_extract_params(call: Call, op_name: str):
    """Common validation and parameter extraction for QNN binary ops."""
    if len(call.args) != 8:
        logging.warning(f"{op_name} expects 8 args, got {len(call.args)}.")
        return None

    lhs, rhs = call.args[0], call.args[1]
    l_sinfo, r_sinfo = lhs.struct_info, rhs.struct_info

    if l_sinfo.shape is None or r_sinfo.shape is None:
        return None

    dtype = l_sinfo.dtype
    if dtype not in ["int8", "uint8"] and dtype != r_sinfo.dtype:
        return None

    # Extract all 6 constants
    vals = []
    names = ["lhs_scale", "lhs_zp", "rhs_scale", "rhs_zp", "out_scale", "out_zp"]
    for i, name in enumerate(names):
        v = extract_relax_const(call.args[i + 2], name, op_name)
        if v is None:
            return None
        vals.append(v)

    l_sc, l_zp, r_sc, r_zp, o_sc, o_zp = vals

    if any(s <= 0 for s in [l_sc, r_sc, o_sc]):
        return None

    # Zero point range check
    zp_min, zp_max = (-128, 127) if dtype == "int8" else (0, 255)
    for zero_point, name in zip([l_zp, r_zp, o_zp], ["lhs_zp", "rhs_zp", "out_zp"]):
        if not (zp_min <= zero_point <= zp_max):
            return None

    return (lhs, rhs, float(l_sc), int(l_zp), float(r_sc), int(r_zp), float(o_sc), int(o_zp))


def get_values_from_expr(expr):
    """Get the values from an relax expression."""

    if isinstance(expr, relax.Constant):
        sinfo = expr.struct_info
        if sinfo.ndim > 0:
            return expr.data.numpy().tolist()
        return expr.data.numpy().item()
    elif isinstance(expr, relax.Expr):
        return [input.data.numpy().item() for input in expr]
    else:
        raise ValueError(f"Unsupported expression type: {type(expr)}")
