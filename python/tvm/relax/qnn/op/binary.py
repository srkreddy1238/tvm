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
"""Relax QNN Binary operators"""

from ...expr import Expr
from . import _ffi_api


def add(
    lhs: Expr,
    rhs: Expr,
    lhs_scale: Expr,
    lhs_zero_point: Expr,
    rhs_scale: Expr,
    rhs_zero_point: Expr,
    output_scale: Expr,
    output_zero_point: Expr,
    lhs_axis=-1,
    rhs_axis=-1,
) -> Expr:
    """Quantized add operator."""
    return _ffi_api.add(  # type: ignore
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        lhs_axis,
        rhs_axis,
    )


def subtract(
    lhs: Expr,
    rhs: Expr,
    lhs_scale: Expr,
    lhs_zero_point: Expr,
    rhs_scale: Expr,
    rhs_zero_point: Expr,
    output_scale: Expr,
    output_zero_point: Expr,
    lhs_axis=-1,
    rhs_axis=-1,
) -> Expr:
    """Quantized subtract operator."""
    return _ffi_api.subtract(  # type: ignore
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        lhs_axis,
        rhs_axis,
    )


def multiply(
    lhs: Expr,
    rhs: Expr,
    lhs_scale: Expr,
    lhs_zero_point: Expr,
    rhs_scale: Expr,
    rhs_zero_point: Expr,
    output_scale: Expr,
    output_zero_point: Expr,
    lhs_axis=-1,
    rhs_axis=-1,
) -> Expr:
    """Quantized multiply operator."""
    return _ffi_api.multiply(  # type: ignore
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        lhs_axis,
        rhs_axis,
    )
