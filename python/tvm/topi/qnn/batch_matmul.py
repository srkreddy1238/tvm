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
"""QNN Operators"""

from tvm import te

from ..nn.dense import matmul


def batch_matmul(
    tensor_x,
    tensor_y,
    x_zero_point,
    y_zero_point,
    out_dtype,
    transpose_x=False,
    transpose_y=False,
):
    """Compute for qnn.batch_matmul"""

    # Preprocess tensor_a: subtract zp
    x_sub_zp = te.compute(
        tensor_x.shape, lambda *indices: te.subtract(tensor_x(*indices), x_zero_point)
    )
    # Preprocess tensor_b: subtract zp
    y_sub_zp = te.compute(
        tensor_y.shape, lambda *indices: te.subtract(tensor_y(*indices), y_zero_point)
    )

    return matmul(
        x_sub_zp,
        y_sub_zp,
        bias=None,
        out_dtype=out_dtype,
        transpose_a=transpose_x,
        transpose_b=transpose_y,
    )
