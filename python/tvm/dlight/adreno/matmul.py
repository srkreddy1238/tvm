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

# pylint: disable=missing-docstring
from typing import Optional, Tuple, List

from tvm import tir
from tvm.target import Target
from tvm.tir import IterVar
from tvm.tir.schedule.schedule import BlockRV

from .base import AdrenoScheduleRule
from .. import analysis

# Supported shape profiles
SUPPORTED_PROFILES = {
    ("float16", "float16", "float16"): (64, 64, 16),
    ("float16", "float16", "float32"): (64, 64, 16),
    ("float32", "float32", "float32"): (64, 64, 8),
    ("int8", "int8", "int8"): (64, 64, 32),
}

# vector_size = 32 bits / bits per element
VECTOR_SIZE = {
    "float16": 2,
    "float32": 1,
    "int8": 4,
}

# wave size is 64 for Adreno
WAVE_SIZE = 64


def get_vector_size(dtype: str) -> int:
    return VECTOR_SIZE[dtype]


def extract_shapes_and_dtypes(func: tir.PrimFunc) -> Tuple[List[List[tir.PrimExpr]], List[str]]:
    shapes = [func.buffer_map[param].shape for param in func.params]
    dtypes = [func.buffer_map[param].dtype for param in func.params]
    return shapes, dtypes


def get_reduction_blocks(sch, blocks) -> bool:
    # Get the main computation block
    def is_reduction(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.CommReduce, IterVar.DataPar}

    def is_spatial(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.DataPar}

    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all([is_reduction(block) or is_spatial(block) for block in blocks]):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction(block)]
    if len(reduction_blocks) != 1:
        return None

    return reduction_blocks


def validate_supported_shape(
    shapes: List[List[tir.PrimExpr]], dtypes: List[str]
) -> Optional[Tuple[int, int, int]]:
    # pylint: disable=invalid-name
    if len(shapes) < 3 or len(dtypes) < 3:
        return None

    dtype_profile = tuple(dtypes[:3])
    expected_shape = SUPPORTED_PROFILES.get(dtype_profile)
    if expected_shape is None:
        return None

    a_shape, _, c_shape = shapes[:3]
    is_batch = len(c_shape) == 3

    if is_batch:
        M, N, K = int(c_shape[1]), int(c_shape[2]), int(a_shape[2])
    else:
        M, N, K = int(c_shape[0]), int(c_shape[1]), int(a_shape[1])

    EM, EN, EK = expected_shape
    if M % EM == 0 and N % EN == 0 and K % EK == 0:
        return EM, EN, EK
    return None


class MatmulTensorization(AdrenoScheduleRule):
    """Schedule rule for Adreno cooperative matrix matmul using tensor intrinsics"""

    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        # pylint: disable=invalid-name

        # skip openCl as of now
        if "opencl" in target.kind.name or "opencl" in target.keys:
            return None

        # skip if supports_khr_cooperative_matrix is not present for vulkan
        if "vulkan" in target.kind.name and "supports_khr_cooperative_matrix" not in target.attrs:
            return None

        from tvm.tir.tensor_intrin.adreno import (  # pylint: disable=import-outside-toplevel
            get_adreno_wmma_intrin_group,
        )

        if not isinstance(func, tir.PrimFunc):
            return None

        sch = tir.Schedule(func)
        root = sch.get_block(name="root", func_name="main")
        blocks = sch.get_child_blocks(root)
        reduction_blocks = get_reduction_blocks(sch, blocks)

        def is_matmul(blk):
            block_info = analysis.get_block_info(sch, blk)
            return "matmul" in block_info.name

        if (
            reduction_blocks is None
            or len(reduction_blocks) != 1
            or not is_matmul(reduction_blocks[0])
        ):
            return None

        # validate the shape and dtype of the matmul
        shapes, dtypes = extract_shapes_and_dtypes(func)
        valid_tile = validate_supported_shape(shapes, dtypes)
        if valid_tile is None:
            return None

        M_tile, N_tile, K_tile = valid_tile
        dtype_a, dtype_b, dtype_c = dtypes[:3]

        block = reduction_blocks[0]
        loops = sch.get_loops(block)
        if len(loops) == 4:
            b, i, j, k = loops
            # sch.bind(b, "blockIdx.z")  # Optional: parallelize batch
        elif len(loops) == 3:
            i, j, k = loops
            b = None
        else:
            return None  # Unexpected pattern

        i_outer, i_inner = sch.split(i, factors=[None, M_tile])
        j_outer, j_inner = sch.split(j, factors=[None, N_tile])
        k_outer, k_inner = sch.split(k, factors=[None, K_tile])
        sch.reorder(*(filter(None, [b])), i_outer, j_outer, k_outer, i_inner, j_inner, k_inner)

        fused = sch.fuse(i_outer, j_outer)
        if b:
            fused = sch.fuse(b, fused)
        sch.bind(fused, "blockIdx.x")

        # Cache reads to shared memory
        def fetch_to_shared(block, idx, vector_size):
            block_read = sch.cache_read(block, idx, "shared")
            sch.compute_at(block_read, k_outer)

            fused = sch.fuse(*sch.get_loops(block_read)[-2:])
            _, tx, vec = sch.split(fused, factors=[None, WAVE_SIZE, vector_size])
            sch.bind(tx, "threadIdx.x")
            sch.vectorize(vec)

        fetch_to_shared(block, 0, get_vector_size(dtype_a))
        fetch_to_shared(block, 1, get_vector_size(dtype_b))

        a_scope = "wmma.matrix_a"
        b_scope = "wmma.matrix_b"
        c_scope = "wmma.accumulator"

        A_wmma = sch.cache_read(block, 0, a_scope)
        B_wmma = sch.cache_read(block, 1, b_scope)
        C_wmma = sch.cache_write(block, 0, c_scope)
        sch.reverse_compute_at(C_wmma, fused)

        # Tensorize loading
        def tensorize_load(block, shape):
            loops = sch.get_loops(block)
            i, j = loops[-2:]
            i0, i1 = sch.split(i, factors=[None, shape[0]])
            j0, j1 = sch.split(j, factors=[None, shape[1]])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            return i1

        loop_a = tensorize_load(A_wmma, [M_tile, K_tile])
        loop_b = tensorize_load(B_wmma, [K_tile, N_tile])

        # Tensorize init, compute, store
        init_block = sch.decompose_reduction(block, sch.get_loops(block)[1])
        intrin_group = get_adreno_wmma_intrin_group(
            load_scope="shared",
            store_scope="global",
            dtype=dtype_a,
            out_dtype=dtype_c,
            trans_a=False,
            trans_b=False,
            m=M_tile,
            n=N_tile,
            k=K_tile,
        )

        sch.tensorize(loop_a, intrin_group["load_a"])
        sch.tensorize(loop_b, intrin_group["load_b"])
        sch.tensorize(sch.get_loops(init_block)[1], intrin_group["init"])
        sch.tensorize(sch.get_loops(C_wmma)[1], intrin_group["store"])
        sch.tensorize(sch.get_loops(block)[2], intrin_group["compute"])

        return sch
