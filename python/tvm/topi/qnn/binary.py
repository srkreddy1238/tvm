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

"""QNN Binary operators TOPI implementations."""

from tvm import te

from .utils import broadcast_axis, saturate


def add(
    lhs,
    rhs,
    lhs_scale,
    rhs_scale,
    rsh,
    corr,
):
    """Compute quantized add with broadcasting"""
    A_broadcast, B_broadcast = broadcast_axis(lhs, rhs)
    n_a, h_a, w_a, c_a = A_broadcast
    n_b, h_b, w_b, c_b = B_broadcast

    dtype = lhs.dtype
    output_shape = []
    for i in range(len(lhs.shape)):
        output_shape.append(te.max(lhs.shape[i], rhs.shape[i]))

    return te.compute(
        output_shape,
        lambda n, h, w, c: saturate(
            (
                (
                    (lhs[n * n_a, h * h_a, w * w_a, c * c_a] * lhs_scale)
                    + (rhs[n * n_b, h * h_b, w * w_b, c * c_b] * rhs_scale)
                    + corr
                )
                >> rsh
            ),
            dtype,
        ).astype(dtype),
    )


def subtract(
    lhs,
    rhs,
    lhs_scale,
    rhs_scale,
    rsh,
    corr,
):
    """Compute quantized subtract with broadcasting"""
    A_broadcast, B_broadcast = broadcast_axis(lhs, rhs)
    n_a, h_a, w_a, c_a = A_broadcast
    n_b, h_b, w_b, c_b = B_broadcast

    dtype = lhs.dtype
    output_shape = []
    for i in range(len(lhs.shape)):
        output_shape.append(te.max(lhs.shape[i], rhs.shape[i]))

    return te.compute(
        output_shape,
        lambda n, h, w, c: saturate(
            (
                (
                    (lhs[n * n_a, h * h_a, w * w_a, c * c_a] * lhs_scale)
                    - (rhs[n * n_b, h * h_b, w * w_b, c * c_b] * rhs_scale)
                    + corr
                )
                >> rsh
            ),
            dtype,
        ).astype(dtype),
    )


def mul(
    lhs,
    rhs,
    lhs_zp_val,
    rhs_zp_val,
    scale_int,
    rsh,
    corr,
):
    """Compute quantized multiply with broadcasting"""
    A_broadcast, B_broadcast = broadcast_axis(lhs, rhs)
    n_a, h_a, w_a, c_a = A_broadcast
    n_b, h_b, w_b, c_b = B_broadcast

    dtype = lhs.dtype
    output_shape = []
    for i in range(len(lhs.shape)):
        output_shape.append(te.max(lhs.shape[i], rhs.shape[i]))

    return te.compute(
        output_shape,
        lambda n, h, w, c: saturate(
            (
                (
                    scale_int
                    * (lhs[n * n_a, h * h_a, w * w_a, c * c_a] - lhs_zp_val)
                    * (rhs[n * n_b, h * h_b, w * w_b, c * c_b] - rhs_zp_val)
                    + corr
                )
                >> rsh
            ),
            dtype,
        ).astype(dtype),
    )
