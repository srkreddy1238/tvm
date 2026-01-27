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


def get_wmma_fragment_index(buffer, stride, m_dim, n_dim):
    """Compute wmma fragment index using elem_offset of the buffer"""
    frag_index_m = buffer.elem_offset // stride // m_dim
    frag_index_n = buffer.elem_offset % stride // n_dim

    num_fragments_per_row = stride // n_dim
    return frag_index_m * num_fragments_per_row + frag_index_n


def get_wmma_fill_intrin(
    m_dim: int, n_dim: int, k_dim: int, dtype: str
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_fill intrins"""
    zero = IntImm("int32", 0).astype(dtype)
    offset_factor = n_dim

    @T.prim_func
    def wmma_fill_desc(c: T.handle) -> None:
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="wmma.accumulator",
        )
        with T.block("root"):
            T.reads()
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("init"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = zero

    @T.prim_func
    def wmma_fill_impl(c: T.handle) -> None:
        d1 = T.int32()
        d0 = T.int32()
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="wmma.accumulator",
            strides=[d1, d0],
        )
        with T.block("root"):
            T.reads()
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_fill_fragment(
                    C.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    get_wmma_fragment_index(C, d1, m_dim, n_dim),
                    T.float32(0),
                    dtype="handle",
                )
            )

    return wmma_fill_desc, wmma_fill_impl


def get_wmma_load_intrin(
    m_dim: int,
    n_dim: int,
    k_dim: int,
    dtype: str,
    shared_scope: str,
    is_b: bool,
    is_col_major: bool,
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_load intrins"""
    wmma_fragment_scope = f"wmma.matrix_{'b' if is_b else 'a'}"
    layout = "col_major" if is_col_major else "row_major"

    frag_m, frag_n = (k_dim, n_dim) if is_b else (m_dim, k_dim)
    if is_col_major:
        frag_m, frag_n = frag_n, frag_m
    offset_factor = frag_n

    @T.prim_func
    def wmma_load_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (frag_m, frag_n), dtype, align=64, offset_factor=offset_factor, scope=shared_scope
        )
        C = T.match_buffer(
            c,
            (frag_m, frag_n),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope=wmma_fragment_scope,
        )
        with T.block("root"):
            T.reads(A[0:frag_m, 0:frag_n])
            T.writes(C[0:frag_m, 0:frag_n])
            for i, j in T.grid(frag_m, frag_n):
                with T.block("load"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_load_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.int32()
        s0 = T.int32()
        d1 = T.int32()
        d0 = T.int32()
        A = T.match_buffer(
            a,
            (frag_m, frag_n),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope=shared_scope,
            strides=[s1, s0],
        )
        C = T.match_buffer(
            c,
            (frag_m, frag_n),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope=wmma_fragment_scope,
            strides=[d1, d0],
        )
        with T.block("root"):
            T.reads(A[0:frag_m, 0:frag_n])
            T.writes(C[0:frag_m, 0:frag_n])
            T.evaluate(
                T.tvm_load_matrix_sync(
                    C.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    get_wmma_fragment_index(C, d1, frag_m, frag_n),
                    A.access_ptr("r"),
                    s1,
                    layout,
                    dtype="handle",
                )
            )

    return wmma_load_desc, wmma_load_impl


def get_wmma_store_intrin(
    m_dim: int, n_dim: int, k_dim: int, dtype: str, scope: str
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_store intrins"""
    offset_factor = n_dim

    @T.prim_func
    def wmma_store_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="wmma.accumulator",
        )
        C = T.match_buffer(
            c, (m_dim, n_dim), dtype, align=64, offset_factor=offset_factor, scope=scope
        )
        with T.block("root"):
            T.reads(A[0:m_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("store"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_store_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.int32()
        s0 = T.int32()
        d1 = T.int32()
        d0 = T.int32()
        A = T.match_buffer(
            a,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="wmma.accumulator",
            strides=[d1, d0],
        )
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope=scope,
            strides=[s1, s0],
        )
        with T.block("root"):
            T.reads(A[0:m_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_store_matrix_sync(
                    A.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    get_wmma_fragment_index(A, d1, m_dim, n_dim),
                    C.access_ptr("w"),
                    s1,
                    "row_major",
                    dtype="handle",
                )
            )

    return wmma_store_desc, wmma_store_impl


def get_wmma_sync_intrin(
    m_dim: int, n_dim: int, k_dim: int, in_dtype: str, out_dtype: str, b_transposed: bool
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_sync intrins"""

    def maybe_cast(v):
        if in_dtype != out_dtype:
            return Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    b_shape_0, b_shape_1 = maybe_swap(k_dim, n_dim)

    A_offset_factor = k_dim
    B_offset_factor = b_shape_1
    out_offset_factor = n_dim

    @T.prim_func
    def wmma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            (m_dim, k_dim),
            in_dtype,
            align=64,
            offset_factor=A_offset_factor,
            scope="wmma.matrix_a",
        )
        B = T.match_buffer(
            b,
            maybe_swap(k_dim, n_dim),
            in_dtype,
            align=64,
            offset_factor=B_offset_factor,
            scope="wmma.matrix_b",
        )
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            out_dtype,
            align=64,
            offset_factor=out_offset_factor,
            scope="wmma.accumulator",
        )

        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:b_shape_0, 0:b_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j, k in T.grid(m_dim, n_dim, k_dim):
                with T.block(""):
                    vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                    B_index_0, B_index_1 = T.meta_var(maybe_swap(vkk, vjj))
                    C[vii, vjj] = C[vii, vjj] + maybe_cast(A[vii, vkk]) * maybe_cast(
                        B[B_index_0, B_index_1]
                    )

    @T.prim_func
    def wmma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        a1 = T.int32()
        a0 = T.int32()
        b1 = T.int32()
        b0 = T.int32()
        c1 = T.int32()
        c0 = T.int32()

        A = T.match_buffer(
            a,
            (m_dim, k_dim),
            in_dtype,
            align=64,
            offset_factor=A_offset_factor,
            scope="wmma.matrix_a",
            strides=[a1, a0],
        )
        B = T.match_buffer(
            b,
            maybe_swap(k_dim, n_dim),
            in_dtype,
            align=64,
            offset_factor=B_offset_factor,
            scope="wmma.matrix_b",
            strides=[b1, b0],
        )
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            out_dtype,
            align=64,
            offset_factor=out_offset_factor,
            scope="wmma.accumulator",
            strides=[c1, c0],
        )

        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:b_shape_0, 0:b_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_mma_sync(
                    C.data,
                    get_wmma_fragment_index(C, c1, m_dim, n_dim),
                    A.data,
                    get_wmma_fragment_index(A, a1, m_dim, k_dim),
                    B.data,
                    get_wmma_fragment_index(B, b1, b_shape_0, b_shape_1),
                    C.data,
                    get_wmma_fragment_index(C, c1, m_dim, n_dim),
                    dtype="handle",
                )
            )

    return wmma_sync_desc, wmma_sync_impl


WMMA_STORE_64x64x16_F32_GLOBAL_INTRIN = "wmma_store_64x64x16_f32_global"
TensorIntrin.register(
    WMMA_STORE_64x64x16_F32_GLOBAL_INTRIN, *get_wmma_store_intrin(64, 64, 16, "float32", "global")
)

WMMA_SYNC_64x64x16_f16f16f32_INTRIN = "wmma_sync_64x64x16_f16f16f32"
TensorIntrin.register(
    WMMA_SYNC_64x64x16_f16f16f32_INTRIN,
    *get_wmma_sync_intrin(64, 64, 16, "float16", "float32", False),
)

WMMA_LOAD_64x64x16_F16_A_INTRIN = "wmma_load_64x64x16_f16_a_shared"
TensorIntrin.register(
    WMMA_LOAD_64x64x16_F16_A_INTRIN,
    *get_wmma_load_intrin(64, 64, 16, "float16", "shared", False, False),
)

WMMA_LOAD_64x64x16_F16_B_INTRIN = "wmma_load_64x64x16_f16_b_shared"
TensorIntrin.register(
    WMMA_LOAD_64x64x16_F16_B_INTRIN,
    *get_wmma_load_intrin(64, 64, 16, "float16", "shared", True, False),
)

WMMA_STORE_64x64x16_F16_GLOBAL_INTRIN = "wmma_store_64x64x16_f16_global"
TensorIntrin.register(
    WMMA_STORE_64x64x16_F16_GLOBAL_INTRIN, *get_wmma_store_intrin(64, 64, 16, "float16", "global")
)

WMMA_SYNC_64x64x16_f16f16f16_INTRIN = "wmma_sync_64x64x16_f16f16f16"
TensorIntrin.register(
    WMMA_SYNC_64x64x16_f16f16f16_INTRIN,
    *get_wmma_sync_intrin(64, 64, 16, "float16", "float16", False),
)

WMMA_FILL_64x64x16_F16_INTRIN = "wmma_fill_64x64x16_f16"
TensorIntrin.register(WMMA_FILL_64x64x16_F16_INTRIN, *get_wmma_fill_intrin(64, 64, 16, "float16"))

WMMA_LOAD_64x64x16_F32_A_INTRIN = "wmma_load_64x64x16_f32_a_shared"
TensorIntrin.register(
    WMMA_LOAD_64x64x16_F32_A_INTRIN,
    *get_wmma_load_intrin(64, 64, 16, "float32", "shared", False, False),
)

# ----------------------- [ fp32 ] ------------------------

WMMA_LOAD_64x64x8_F32_A_INTRIN = "wmma_load_64x64x8_f32_a_shared"
TensorIntrin.register(
    WMMA_LOAD_64x64x8_F32_A_INTRIN,
    *get_wmma_load_intrin(64, 64, 8, "float32", "shared", False, False),
)

WMMA_LOAD_64x64x8_F32_B_INTRIN = "wmma_load_64x64x8_f32_b_shared"
TensorIntrin.register(
    WMMA_LOAD_64x64x8_F32_B_INTRIN,
    *get_wmma_load_intrin(64, 64, 8, "float32", "shared", True, False),
)

WMMA_FILL_64x64x8_F32_INTRIN = "wmma_fill_64x64x8_f32"
TensorIntrin.register(WMMA_FILL_64x64x8_F32_INTRIN, *get_wmma_fill_intrin(64, 64, 8, "float32"))

WMMA_STORE_64x64x8_F32_GLOBAL_INTRIN = "wmma_store_64x64x8_f32_global"
TensorIntrin.register(
    WMMA_STORE_64x64x8_F32_GLOBAL_INTRIN, *get_wmma_store_intrin(64, 64, 8, "float32", "global")
)

WMMA_SYNC_64x64x8_f32f32f32_INTRIN = "wmma_sync_64x64x8_f32f32f32"
TensorIntrin.register(
    WMMA_SYNC_64x64x8_f32f32f32_INTRIN,
    *get_wmma_sync_intrin(64, 64, 8, "float32", "float32", False),
)

# --------------- [int8] ------------------------
WMMA_LOAD_64x64x32_I8_A_INTRIN = "wmma_load_64x64x32_i8_a_shared"
TensorIntrin.register(
    WMMA_LOAD_64x64x32_I8_A_INTRIN,
    *get_wmma_load_intrin(64, 64, 32, "int8", "shared", False, False),
)

WMMA_LOAD_64x64x32_I8_B_INTRIN = "wmma_load_64x64x32_i8_b_shared"
TensorIntrin.register(
    WMMA_LOAD_64x64x32_I8_B_INTRIN,
    *get_wmma_load_intrin(64, 64, 32, "int8", "shared", True, False),
)

WMMA_FILL_64x64x32_I8_INTRIN = "wmma_fill_64x64x32_i8"
TensorIntrin.register(WMMA_FILL_64x64x32_I8_INTRIN, *get_wmma_fill_intrin(64, 64, 32, "int8"))

WMMA_STORE_64x64x32_I8_GLOBAL_INTRIN = "wmma_store_64x64x32_i8_global"
TensorIntrin.register(
    WMMA_STORE_64x64x32_I8_GLOBAL_INTRIN, *get_wmma_store_intrin(64, 64, 32, "int8", "global")
)

WMMA_SYNC_64x64x32_i8i8i8_INTRIN = "wmma_sync_64x64x32_i8i8i8"
TensorIntrin.register(
    WMMA_SYNC_64x64x32_i8i8i8_INTRIN,
    *get_wmma_sync_intrin(64, 64, 32, "int8", "int8", False),
)

# Supported shape profiles
SUPPORTED_PROFILES = {
    ("float16", "float16", "float16"): (64, 64, 16),
    ("float16", "float16", "float32"): (64, 64, 16),
    ("float32", "float32", "float32"): (64, 64, 8),
    ("int8", "int8", "int8"): (64, 64, 32),
}


def get_shorthand_dtype(dtype: str) -> str:
    """Get the shorthand dtype for a given dtype.
    Parameters
    ----------
    dtype : str
        The dtype to get the shorthand for.

    Returns
    -------
    str
        The shorthand dtype.
    """
    if dtype == "float16":
        return "f16"
    elif dtype == "float32":
        return "f32"
    elif dtype == "int8":
        return "i8"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_adreno_wmma_intrin_group(
    load_scope: Literal["shared"],
    store_scope: Literal["global"],
    dtype: str,
    out_dtype: Optional[str] = None,
    trans_a: bool = False,
    trans_b: bool = False,
    m: int = 64,
    n: int = 64,
    k: int = 16,
) -> Dict[str, str]:
    """Get a group of WMMA intrinsics for tensorization on Adreno GPU.

    Parameters
    ----------
    load_scope : Literal["shared"]
        The memory scope of the input buffers.

    store_scope : Literal["global"]
        The memory scope of the result buffer.

    dtype : str
        The data type of the input matrices A and B.

    out_dtype : Optional[str]
        The data type of the output matrix C.

    trans_a : bool
        Whether matrix A is transposed.

    trans_b : bool
        Whether matrix B is transposed.

    m, n, k : int
        Dimensions for WMMA operation.

    Returns
    -------
    ret : Dict[str, str]
        A group of WMMA tensor intrinsics.
    """
    assert load_scope == "shared"
    assert store_scope == "global"
    if out_dtype is None:
        out_dtype = dtype

    # Validate requested shape
    if (dtype, dtype, out_dtype) not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported dtype profile: {(dtype, dtype, out_dtype)}")

    expected_m, expected_n, expected_k = SUPPORTED_PROFILES[(dtype, dtype, out_dtype)]
    if (m, n, k) != (expected_m, expected_n, expected_k):
        raise ValueError(f"Unsupported shape: {(m, n, k)} for dtype {dtype}")

    trans_a_suffix = "_a" if not trans_a else "_a_trans"
    trans_b_suffix = "_b" if not trans_b else "_b_trans"

    dtype = get_shorthand_dtype(dtype)
    out_dtype = get_shorthand_dtype(out_dtype)

    shape_str = f"{m}x{n}x{k}"
    load_a_intrin = f"wmma_load_{shape_str}_{dtype}{trans_a_suffix}_{load_scope}"
    load_b_intrin = f"wmma_load_{shape_str}_{dtype}{trans_b_suffix}_{load_scope}"
    compute_intrin = f"wmma_sync_{shape_str}_{dtype}{dtype}{out_dtype}"
    init_intrin = f"wmma_fill_{shape_str}_{out_dtype}"
    store_intrin = f"wmma_store_{shape_str}_{out_dtype}_{store_scope}"

    return {
        "init": init_intrin,
        "load_a": load_a_intrin,
        "load_b": load_b_intrin,
        "compute": compute_intrin,
        "store": store_intrin,
    }
