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
"""A Convolution impl for Adreno GPU."""

from tvm import relax, s_tir, topi
from tvm.topi.utils import get_const_tuple


def get_in_out_dtype(call: relax.Call):
    data_dtype, weight_dtype = call.args[0].struct_info.dtype, call.args[1].struct_info.dtype
    out_dtype = call.struct_info.dtype
    assert data_dtype == weight_dtype, "Invalid Convolution Op"

    return (data_dtype, out_dtype)


def supports_conv2d_NCHWc_OIHWo(call: relax.Call, disable_layout_checks: bool = False):
    data_layout, kernel_layout = call.attrs.data_layout, call.attrs.kernel_layout
    out_layout = call.attrs.out_layout
    data_dtype, out_dtype = get_in_out_dtype(call)

    data_layout_mp = s_tir.bijective_layout(call.attrs.data_layout, "NCHW")
    out_layout_mp = s_tir.bijective_layout(call.attrs.out_layout, "NCHW")

    data_shape = data_layout_mp.forward_shape(
        get_const_tuple(call.args[0].struct_info.shape.values)
    )
    out_shape = out_layout_mp.forward_shape(get_const_tuple(call.struct_info.shape.values))

    input_channel = data_shape[1]
    out_channel = out_shape[1]

    conditions = []

    # Texture Checks
    conditions.append(input_channel % 4 == 0)
    conditions.append(out_channel % 4 == 0)
    conditions.append(data_dtype == out_dtype)
    conditions.append(data_dtype == "float16" or data_dtype == "float32")

    if not disable_layout_checks:
        conditions.append(data_layout == "NCHW4c")
        conditions.append(kernel_layout == "OIHW4o")
        conditions.append(out_layout == "NCHW4c")

    return all(conditions)


def conv2d_NCHWc_OIHWo(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    if supports_conv2d_NCHWc_OIHWo(call):
        return bb.call_te(
            topi.nn.conv2d_NCHWc_OIHWo,
            data=call.args[0],
            kernel=call.args[1],
            stride=call.attrs.strides,
            padding=call.attrs.padding,
            dilation=call.attrs.dilation,
            layout=call.attrs.data_layout,
            out_layout=call.attrs.out_layout,
            out_dtype=str(call.struct_info.dtype),
            sinfo_args=call.sinfo_args,
            primfunc_name_hint="conv2d_NCHWc_OIHWo",
        )
    return call


def supports_conv2d_matmul(call: relax.Call, disable_layout_checks: bool = False):
    data_layout, kernel_layout = call.attrs.data_layout, call.attrs.kernel_layout
    out_layout = call.attrs.out_layout
    data_dtype, out_dtype = get_in_out_dtype(call)

    data_layout_mp, kernel_layout_mp = (
        s_tir.bijective_layout(call.attrs.data_layout, "NCHW"),
        s_tir.bijective_layout(call.attrs.kernel_layout, "OIHW"),
    )
    out_layout_mp = s_tir.bijective_layout(call.attrs.out_layout, "NCHW")

    data_shape, weight_shape = (
        data_layout_mp.forward_shape(get_const_tuple(call.args[0].struct_info.shape.values)),
        kernel_layout_mp.forward_shape(get_const_tuple(call.args[1].struct_info.shape.values)),
    )
    out_shape = out_layout_mp.forward_shape(get_const_tuple(call.struct_info.shape.values))

    from tvm.s_tir.tensor_intrin.adreno import get_wmma_tile_sizes

    PROFILE = get_wmma_tile_sizes(data_dtype, out_dtype)
    if PROFILE is None:
        return False

    M_tile, N_tile, K_tile = PROFILE

    M = out_shape[2] * out_shape[3]  # H * W
    N = out_shape[1] // call.attrs.groups  # Group-Out-Channel
    K = weight_shape[1]  # I

    input_channel = data_shape[1]  # "C"
    kernel_channel = weight_shape[1]  # "I"
    out_channel = out_shape[1]  # "O"

    pad_threshold = 0.75

    conditions = []

    # Intrinsic Checks
    conditions.append(M >= pad_threshold * M_tile)
    conditions.append(N >= pad_threshold * N_tile)
    conditions.append(K % K_tile == 0)

    # Texture Checks
    conditions.append(input_channel % 4 == 0)
    conditions.append(kernel_channel % 4 == 0)
    conditions.append(out_channel % 4 == 0)

    if not disable_layout_checks:
        conditions.append(data_layout == "NCHW4c")
        conditions.append(kernel_layout == "OIHW4i")
        conditions.append(out_layout == "NCHW4c")

    return all(conditions)


def conv2d_matmul(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    if supports_conv2d_matmul(call, disable_layout_checks=False):
        in_dtype, out_dtype = get_in_out_dtype(call)
        from tvm.s_tir.tensor_intrin.adreno import get_wmma_tile_sizes

        PROFILE = get_wmma_tile_sizes(in_dtype, out_dtype)
        return bb.call_te(
            topi.nn.conv2d_matmul,
            data=call.args[0],
            weight=call.args[1],
            tiles_size=PROFILE,
            strides=call.attrs.strides,
            padding=call.attrs.padding,
            dilation=call.attrs.dilation,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
        )
    return call


def conv2d_convert_layout(use_matmul: bool):
    def _layout_function(call: relax.Call):
        if call.op.name != "relax.nn.conv2d":
            return {}

        if use_matmul and supports_conv2d_matmul(call, disable_layout_checks=True):
            return {"relax.nn.conv2d": ["NCHW4c", "OIHW4i", "NCHW4c"]}
        elif supports_conv2d_NCHWc_OIHWo(call, disable_layout_checks=True):
            return {"relax.nn.conv2d": ["NCHW4c", "OIHW4o", "NCHW4c"]}
        return {}

    return _layout_function
