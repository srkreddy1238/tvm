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
# pylint: disable=invalid-name,missing-function-docstring,unused-variable,unused-import
"""Intrinsics for tensorization on Adreno GPU."""
from typing import Dict, Literal, Optional, Tuple

from tvm.runtime import convert
from tvm.script import tir as T
from tvm.tir import Cast, IntImm, TensorIntrin
from tvm.tir.function import PrimFunc


def get_dotprod_intrin(in_dtype, out_dtype):
    assert in_dtype == "int8" and out_dtype == "int32"

    @T.prim_func
    def dp4a_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4,), align=4, dtype="int8", offset_factor=1, scope="local")
        B = T.match_buffer(b, (4,), align=4, dtype="int8", offset_factor=1, scope="local")
        C = T.match_buffer(c, (1,), align=1, dtype="int32", offset_factor=1, scope="local")

        with T.block("root"):
            T.reads(C[0], A[0:4], B[0:4])
            T.writes(C[0])
            for lj in range(4):
                with T.block("dp4a_update"):
                    j = T.axis.reduce(4, lj)
                    C[0] = C[0] + T.Cast("int32", A[j]) * T.Cast("int32", B[j])

    @T.prim_func
    def dp4a_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4,), align=4, dtype="int8", offset_factor=1, scope="local")
        B = T.match_buffer(b, (4,), align=4, dtype="int8", offset_factor=1, scope="local")
        C = T.match_buffer(c, (1,), align=1, dtype="int32", offset_factor=1, scope="local")

        with T.block("dp4a_update"):
            T.reads(C[0], A[0:4], B[0:4])
            T.writes(C[0])
            C[0] = T.dp4a(A[0:4], B[0:4], C[0])

    return dp4a_desc, dp4a_intrin


ADRENO_DP4A_i8i8i32_INTRIN = "dp4a_i8i8i32_adreno"

TensorIntrin.register(ADRENO_DP4A_i8i8i32_INTRIN, *get_dotprod_intrin("int8", "int32"))
