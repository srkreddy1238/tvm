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

from .utils import broadcast_axis, build_broadcast_compute, saturate


def add(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
):
    """Compute quantized add"""
    A_broadcast, B_broadcast = broadcast_axis(lhs, rhs)
    output_shape, lhs_idx, rhs_idx = build_broadcast_compute(lhs, rhs, A_broadcast, B_broadcast)

    dtype = lhs.dtype

    def compute_fn(*indices):
        lhs_val = lhs(*lhs_idx(indices))
        rhs_val = rhs(*rhs_idx(indices))
        return saturate(
            (
                (lhs_val.astype("float32") - lhs_zero_point) * lhs_scale
                + (rhs_val.astype("float32") - rhs_zero_point) * rhs_scale
            )
            / output_scale
            + output_zero_point,
            dtype,
        ).astype(dtype)

    return te.compute(output_shape, compute_fn, name="qnn_add")


def subtract(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
):
    """Compute quantized subtract"""
    A_broadcast, B_broadcast = broadcast_axis(lhs, rhs)
    output_shape, lhs_idx, rhs_idx = build_broadcast_compute(lhs, rhs, A_broadcast, B_broadcast)

    dtype = lhs.dtype

    def compute_fn(*indices):
        lhs_val = lhs(*lhs_idx(indices))
        rhs_val = rhs(*rhs_idx(indices))
        return saturate(
            (
                (lhs_val.astype("float32") - lhs_zero_point) * lhs_scale
                - (rhs_val.astype("float32") - rhs_zero_point) * rhs_scale
            )
            / output_scale
            + output_zero_point,
            dtype,
        ).astype(dtype)

    return te.compute(output_shape, compute_fn, name="qnn_subtract")


def mul(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
):
    """Compute quantized multiply"""
    A_broadcast, B_broadcast = broadcast_axis(lhs, rhs)
    output_shape, lhs_idx, rhs_idx = build_broadcast_compute(lhs, rhs, A_broadcast, B_broadcast)

    dtype = lhs.dtype

    def compute_fn(*indices):
        lhs_val = lhs(*lhs_idx(indices))
        rhs_val = rhs(*rhs_idx(indices))
        return saturate(
            (
                (lhs_val.astype("float32") - lhs_zero_point)
                * (rhs_val.astype("float32") - rhs_zero_point)
                * (lhs_scale * rhs_scale)
                / output_scale
            )
            + output_zero_point,
            dtype,
        ).astype(dtype)

    return te.compute(output_shape, compute_fn, name="qnn_multiply")
