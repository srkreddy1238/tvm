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

"""QNN Dense operator TOPI implementation."""

from tvm import te

from ..nn.dense import matmul


def dense(
    data,
    weight,
    data_zero_point,
    kernel_zero_point,
    out_dtype="int32",
):
    """
    TOPI compute definition for QNN dense
    """
    assert len(data.shape) <= 2, f"Expected data with <= 2 dimensions, got shape {data.shape}"
    if len(data.shape) == 2:
        batch_size, in_features = data.shape
    else:
        batch_size = 1
        in_features = data.shape[0]
    weight_in_features, units = weight.shape
    assert in_features == weight_in_features, (
        f"Dimension mismatch: data features {in_features} != weight features {weight_in_features}"
    )

    # Raw dot product
    k = te.reduce_axis((0, in_features), name="k")
    data_weight_dot_product = matmul(
        data,
        weight,
        bias=None,
        out_dtype=out_dtype,
    )

    zp_correction = data_zero_point * kernel_zero_point * in_features
    row_sums, col_sums = None, None

    if kernel_zero_point:
        row_sums = te.compute(
            (batch_size,),
            lambda i: te.sum(data[i, k].astype(out_dtype), axis=k),
            name="data_row_sums",
        )

    if data_zero_point:
        col_sums = te.compute(
            (units,),
            lambda j: te.sum(weight[k, j].astype(out_dtype), axis=k),
            name="weight_col_sums",
        )

    output = te.compute(
        (batch_size, units),
        lambda i, j: (
            data_weight_dot_product[i, j]
            - ((kernel_zero_point * row_sums[i]) if kernel_zero_point else 0)
            - ((data_zero_point * col_sums[j]) if data_zero_point else 0)
            + zp_correction
        ),
        name="dense_out",
    )
    return output
