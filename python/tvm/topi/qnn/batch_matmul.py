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
"""QNN BatchMatmul Operator"""

from tvm import te

from ..nn.dense import matmul


def batch_matmul(
    tensor_x,
    tensor_y,
    x_zp_val,
    y_zp_val,
    out_dtype,
    transpose_x=False,
    transpose_y=False,
):
    xy = matmul(
        tensor_x,
        tensor_y,
        bias=None,
        out_dtype=out_dtype,
        transpose_a=transpose_x,
        transpose_b=transpose_y,
    )

    # tensor_x : [B, M, K], tensor_y : [B, K, N]
    B, M, K = tensor_x.shape
    _, _, N = tensor_y.shape

    k_axis = te.reduce_axis((0, K), name="k")
    zp_correction = x_zp_val * y_zp_val * K
    x_row_sums, y_col_sums = None, None

    if y_zp_val:
        x_row_sums = te.compute(
            (B, M),
            lambda b, m: te.sum(tensor_x[b, m, k_axis].astype(out_dtype), axis=k_axis),
            name="x_row_sums",
        )

    if x_zp_val:
        y_col_sums = te.compute(
            (B, N),
            lambda b, n: te.sum(tensor_y[b, k_axis, n].astype(out_dtype), axis=k_axis),
            name="y_col_sums",
        )

    out = te.compute(
        (B, M, N),
        lambda b, m, n: (
            xy[b, m, n]
            - ((y_zp_val * x_row_sums[b, m]) if y_zp_val else 0)
            - ((x_zp_val * y_col_sums[b, n]) if x_zp_val else 0)
            + zp_correction
        ),
        name="qnn_batch_matmul",
    )

    return out
