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

from tvm import te


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
    if len(data.shape) == 2:
        batch_size, in_features = data.shape
    else:
        batch_size = 1
        in_features = data.shape[0]

    weight_in_features, units = weight.shape
    assert in_features == weight_in_features, (
        f"Dimension mismatch: data features {in_features} != weight features {weight_in_features}"
    )

    k = te.reduce_axis((0, in_features), name="k")

    data_zp = (
        te.const(data_zero_point, "int32")
        if isinstance(data_zero_point, (int | float))
        else data_zero_point
    )
    kernel_zp = (
        te.const(kernel_zero_point, "int32")
        if isinstance(kernel_zero_point, (int | float))
        else kernel_zero_point
    )

    output = te.compute(
        (batch_size, units),
        lambda i, j: te.sum(
            (data[i, k] - data_zp) * (weight[k, j] - kernel_zp),
            axis=k,
        ),
        name="dense_out",
    )

    return output
