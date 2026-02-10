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

from .. import cpp
from .requantize import requantize


def concat(data, input_scales, input_zero_points, o_scale, o_zp, axis, out_dtype="int8"):
    """Compute for qnn.concatenate

    Parameters
    ----------
    data: Sequence[te.Tensor]
        The input quantized tensors.

    input_scales: Sequence[te.Tensor or tir.Imm]
        Per-input scales (scalar, 0-D tensor, or per-channel tensor).

    input_zero_points: Sequence[te.Tensor or tir.Imm]
        Per-input zero-points (scalar or 0-D tensor).

    o_scale: te.Tensor or tir.Imm
        Output scale.

    o_zp: te.Tensor or tir.Imm
        Output zero-point.

    axis: int
        Concatenation axis.

    out_dtype: str
        Output dtype (e.g., "int8", "uint8").
    """
    args_num = len(data)
    args = []

    for i in range(args_num):
        # Get next tensor and its quantization parameters.
        tensor = data[i]
        i_scale = input_scales[i]
        i_zp = input_zero_points[i]

        # IMPORTANT: pass `axis=axis` down to requantize so per-channel indexing works.
        rq = requantize(tensor, i_scale, i_zp, o_scale, o_zp, axis=axis, out_dtype=out_dtype)
        args.append(rq)

    # Call generic concatenate from TOPI C++ (handles negative axis as in standard TOPI)
    return cpp.concatenate(args, axis)
