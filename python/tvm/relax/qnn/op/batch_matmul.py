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
"""Relax QNN BatchMatmul operator"""

from . import _ffi_api


def batch_matmul(x, y, x_zero_point, y_zero_point, x_scale, y_scale, out_dtype="int32"):
    """Quantized batch matrix multiplication.

    Computes batch matrix multiplication between quantized tensors x and y:

    out[i, :, :] = matmul(x[i, :, :], y[i, :, :]^T)

    The output is quantized in the specified output dtype.

    Parameters
    ----------
    x : tvm.relax.Expr
        First quantized input tensor with shape (batch, M, K)
    y : tvm.relax.Expr
        Second quantized input tensor with shape (batch, N, K)
    x_zero_point : tvm.relax.Expr
        Zero point of first input (scalar or 1D)
    y_zero_point : tvm.relax.Expr
        Zero point of second input (scalar or 1D)
    x_scale : tvm.relax.Expr
        The scale for the first input tensor.
    y_scale : tvm.relax.Expr
        The scale for the second input tensor.
    out_dtype : str, optional
        Output data type, can be "int32" or "int16" (default: "int32")

    Returns
    -------
    result : tvm.relax.Expr
        Quantized batch matrix multiplication result with shape (batch, M, N)
        in the specified out_dtype
    """
    return _ffi_api.batch_matmul(x, y, x_zero_point, y_zero_point, x_scale, y_scale, out_dtype)
