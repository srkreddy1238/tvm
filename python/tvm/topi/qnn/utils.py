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
# pylint: disable=unused-argument, redefined-
"""QNN utility functions"""

import math
import struct

import tvm
from tvm import te, tir, topi


def is_scalar_tensor(t: te.Tensor) -> bool:
    return len(t.shape) == 0


def get_qnn_param(param, indices, axis):
    # Account scalar and 1D quantization parameters:
    if is_scalar_tensor(param):
        return param

    param_idx = tir.indexmod(indices[axis], topi.shape(param)[0])
    return param[param_idx]


def subtract_zero_point(
    tensor: te.Tensor,
    zero_point: te.Tensor | tvm.tir.IntImm,
    name: str,
):
    """
    Subtract zero point from given tensor. If zero point is scalar constant and is equal to 0, then
    it can be optimized and return tensor as it is.
    This new block is marked with 'meta_schedule.inline_rule = disable' attribute to disable inline.
    Otherwise, inline prevents from tensorization and leveraging vrmpy intrinsic
    """
    return te.compute(
        tensor.shape,
        lambda *i: te.subtract(tensor(*i), zero_point).astype(tensor.dtype),
        name=name,
        attrs={"meta_schedule.inline_rule": "disable"},
    )


def broadcast_axis(tensor_A, tensor_B):
    """Find out the indices that will have broadcasting"""
    A_broadcast = []
    B_broadcast = []

    for i in range(len(tensor_A.shape)):
        if tensor_A.shape[i] == tensor_B.shape[i]:
            A_broadcast.append(1)
            B_broadcast.append(1)
        elif tensor_A.shape[i] == 1:
            A_broadcast.append(0)
            B_broadcast.append(1)
        elif tensor_B.shape[i] == 1:
            A_broadcast.append(1)
            B_broadcast.append(0)
    return A_broadcast, B_broadcast


def saturate(x: te.Tensor, dtype: str):
    """Saturate value for the specified data type"""
    return te.max(te.min_value(dtype), te.min(x, te.max_value(dtype)))


def get_fixed_point_value(flp: float, dtype: str = "int16") -> tuple[int, int]:
    """
    Return fixed-point value and the corresponding log2 of the scale factor used to compute
    this value.

    Parameters
    ----------
    flp : float
        Floating-point value to be converted
    dtype : str
        Type of the resulting fixed-point value. By default, it's set to "int16"

    Returns
    -------
    fixed_point_value : int
        Fixed-point value for the given floating-point value
    exp_scale_factor : int
        log2 of the scale factor

    Convert floating-point value into fixed-point number. This is done by
    multiplying the value by a scaling factor and then rounding it to the nearest
    integer value.

    As per IEEE-754 standard, a floating-point value can be represented as follows
    [see: https://en.wikipedia.org/wiki/IEEE_754-1985]:
        (-1)^S * M * 2^(E-Bias)

    Here,
    * S is the signed bit (0 or 1).
    * M is the mantissa. It's composed of an implicit 1 for the normalized floating-point
      values or 0 for the denormalized values, and the fraction part. This ensures that
      mantissa is always within [0, 2) range. Please note that this function doesn't
      handle denormalized values.
    * E is the exponent.

    In single precision, 23 bits are used to represent the fraction part of
    the mantissa (and therefore, '23' shows up in one of the computations below) and
    8 bits are used for the exponent. Since exponent field needs to reperesent both
    positive and negative values, a bias (127 for single precision) is added to the actual
    value. Therefore, to compute the actual exponent, 127 must be subtracted from the stored
    value.

    As mentioned above, to find the corresponding fixed-point number, we multiply the
    value with a scaling factor and then round it to the nearest integer. The scaling factor
    is chosen to be a power for 2 and it's the largest value that can be safely multiplied
    to the floating-point value, without causing the resulting value to overflow the range
    of the integer type used to represent the fixed-point value.

    So, if we assume the scaling factor to be 2^x, the resulting fixed-point value will be:
        round((-1)^S * (M) * 2^(E-Bias) * 2^x)

    This can be simplified to:
        round((-1)^S * M * 2^(E-Bias+x)

    Now, if 'int16' is used for fixed-point value, then it has to be >= -(2 * 2^14)
    and <= (2 * 2^14) - 1. Since M (Mantissa) is always < 2, in order for the fixed-point value
    to be within this range, 2^(E - Bias + x) must be <= 2^14 - 1.
    And, if we ignore -1, (E - Bias + x) should be <= 14. Note: if mantissa gets too close to 2,
    this will cause the resulting value to go out of range and require it to be saturated.
    In the following implementation, we perform range check and adjust the scale to avoid
    saturation.
    For most cases, 2^x, where x = 14 - (E - Bias) or 14 - (E - 127) for single precision, is the
    best scaling factor for 'int16' type that can be used to convert the floating-point value to
    fixed-point with the least amount of precision loss.


    Here is a more rigorous explanation of the above, for non-negative scale values, which are of
    interest. M < 2, so M * 2^(E-Bias+x) < 2 ^ (E-Bias+x+1)   [Note: LHS is a fraction, RHS int]
    => round(M * 2^(E-Bias+x)) <= 2 ^ (E-Bias+x+1)  [Note the "<=", not "<"]
    We want x s.t. round(M * 2^(E-Bias+x)) <= 2^15 - 1
    We know round(M * 2^(E-Bias+x)) <= 2^(E-Bias+x+1)
    It will be sufficient to choose x s.t. 2^(E-Bias+x+1) <= 2^15 - 1
    That is, max x. s.t. 2^(E-Bias+x+1) < 2^15
    E-Bias+x+1 < 15
    E-Bias+x+1 <= 14
    Max x will make E-Bias+x+1 = 14
    x = 13 - E + Bias

    Additonal notes on various floating-point values:
    ------------------------------------------------
    1) Denormalized values: causes assertion failure. The problem with the denormalized values
        is that they require a very large scale factor (>= 2^127) to be converted to a fixed-point
        value. As the denormalzied values get smaller, the scale factor becomes too large to be
        represented as a IEEE-754 floating point value (as being done in the computaton below)
        and therefore, the denormalized values aren't being handled here.
    2) NaN and INF: assertion failure
    """

    def within_range(val, dtype):
        if dtype == "int16":
            return -32768 <= val <= 32767
        raise RuntimeError(f"Unsupported dtype, {dtype}'")

    # Make sure that 'flp' isn't NaN or infinity
    if math.isnan(flp) or math.isinf(flp):
        raise RuntimeError("NaN or INF can not be represented as fixed-point")

    flp_f = struct.pack("f", flp)
    flp_i = struct.unpack("I", flp_f)
    exp_stored_value = (flp_i[0] >> 23) & 0xFF

    if exp_stored_value == 0:
        raise RuntimeError(
            "Denormalized values are not considered for float -> fixed-point conversion!"
        )

    exp_value = ((flp_i[0] >> 23) & 0xFF) - 127
    if dtype == "int16":
        max_bits = 14
    else:
        raise RuntimeError(f"Unsupported dtype, {dtype}'")

    exp_scale_factor = max_bits - exp_value  # log2 of the scale_factor

    if exp_scale_factor > 127:
        raise RuntimeError("Value too small for fixed-point conversion!")

    # Scaling factor = 2^exp_scale_factor
    # Since exp_scale_factor can be -ve or +ve, scaling factor is calculated by first
    # representing the value in the binary format as per IEEE floating-point standand and then
    # reinterpreting it as a float using struct.pack and struct.unpack functions.
    # struct.pack returns a bytes object packed as integer and struct.unpack
    # unpacks this bytes object into float.
    scale = ((exp_scale_factor + 127) & 0xFF) << 23
    scale_i = struct.pack("I", scale)
    scale_f = struct.unpack("f", scale_i)
    fixed_point_value = round(flp * scale_f[0])

    if not within_range(fixed_point_value, dtype):
        # Adjust scale factor to avoid overflow.
        exp_scale_factor -= 1
        scale = ((exp_scale_factor + 127) & 0xFF) << 23
        scale_i = struct.pack("I", scale)
        scale_f = struct.unpack("f", scale_i)
        fixed_point_value = round(flp * scale_f[0])

    return fixed_point_value, exp_scale_factor


def get_int_scale(
    scale_A: float,
    scale_B: float,
    scale_M: float,
    zero_point_A: int,
    zero_point_B: int,
    zero_point_M: int,
    op: str,
):
    """
    Depending on the op, this function uses exp_scale_factor(log2 of the scale factor)
    to adjust the output's zero_point.
    """

    C_recip = 1 / scale_M

    if op == "qmul":
        scale = scale_A * scale_B * C_recip
        scale_fixed_point, rsh = get_fixed_point_value(scale, "int16")

        # We need to adjust output's zero point value since the compute for the op is multiplied
        # by a scaling factor.
        # The scaling factor is 2^x where x is the exp_scale_factor which is assigned to rsh here.
        # Since zero_point_M is multipled by 2^rsh while converting floating-point scale value
        # into fixed-point number, we left shift it by rsh in our compute to reflect that.

        corr = zero_point_M << rsh

        return scale_fixed_point, rsh, corr

    a_scale_f = scale_A * C_recip
    b_scale_f = scale_B * C_recip
    scale_fixed_point_a, rsh_a = get_fixed_point_value(a_scale_f, "int16")
    scale_fixed_point_b, rsh_b = get_fixed_point_value(b_scale_f, "int16")

    # Here we have two exp_scale_factors rsh_a and rsh_b.
    # To avoid complexity, we want to use a common exp_scale_factor and
    # we want to use the lowest of the two.

    # Since, either of scale_fixed_point_a or scale_fixed_point_b has already been multiplied
    # by 2^max(rsh_a, rsh_b)
    # we want to undo that by right shifting that scale_fixed_point value
    # by the difference of rsh_a and rsh_b.

    # This results into having a common exp_scale_factor for both scale_fixed_point_a
    # and scale_fixed_point_b.

    # We also set rsh here which is used to adjust the zero_point_M and compute the corr value,
    # computation of which comes from the original equation of the op's compute.

    if rsh_a > rsh_b:
        scale_fixed_point_a = scale_fixed_point_a >> (rsh_a - rsh_b)
        rsh = rsh_b
    else:
        scale_fixed_point_b = scale_fixed_point_b >> (rsh_b - rsh_a)
        rsh = rsh_a

    if op == "qadd":
        corr = (zero_point_M << rsh) - (
            zero_point_A * scale_fixed_point_a + zero_point_B * scale_fixed_point_b
        )
    else:
        corr = (zero_point_M << rsh) - (
            zero_point_A * scale_fixed_point_a - zero_point_B * scale_fixed_point_b
        )

    return scale_fixed_point_a, scale_fixed_point_b, rsh, corr
