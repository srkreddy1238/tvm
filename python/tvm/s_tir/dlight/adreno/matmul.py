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
# pylint: disable=missing-docstring, invalid-name
"""A GEMM schedule rule for Adreno."""

from dataclasses import dataclass
from enum import Enum

from tvm import s_tir, tir
from tvm.ir import Range
from tvm.target import Target
from tvm.tir import PrimExpr, Var
from tvm.tir.stmt import SBlock

from ...dlight import analysis
from ...dlight.analysis import SBlockInfo
from .base import AdrenoScheduleRule
from .utils import get_matmul_supported_profile, schedule_default


class IterKind(Enum):
    """Iter kinds for GEMM-liked programs.
    We can simplify the computation to C[S, I, J] += A[S, I, K] * B[S, J, K],
    where `I, J, K` are fundamental axes for gemm and `S` represents all
    other spatial axes (e.g. batches)
    kIter_S: spatial axes
    kIter_I: I axes
    kIter_J: J axes
    kIter_K: K axes
    kIter_T: trivial axes (i.e. with extent 1)
    """

    kIter_S = 0
    kIter_I = 1
    kIter_J = 2
    kIter_K = 3
    kIter_T = 4


@dataclass
class IterTrait:
    kind: IterKind
    extent: PrimExpr


def _is_one(x: PrimExpr) -> bool:
    return isinstance(x, tir.IntImm) and x.value == 1


def make_iter_fusion_index_map(
    traits: list[IterTrait],
    kind_order: list[IterKind],
) -> tir.IndexMap:
    fused_iters: dict[IterKind, PrimExpr] = {}
    input_iters: list[tir.Var] = []
    for i, trait in enumerate(traits):
        v_i = tir.Var(f"i{i}", trait.extent.dtype)
        input_iters.append(v_i)
        if trait.kind == IterKind.kIter_T:
            continue
        if trait.kind not in kind_order:
            raise ValueError(f"Unknown iter kind {trait.kind}")
        if trait.kind in fused_iters:
            fused_iters[trait.kind] = fused_iters[trait.kind] * trait.extent + v_i
        else:
            fused_iters[trait.kind] = v_i

    final_indices: list[tir.PrimExpr] = [
        fused_iters.get(kind, tir.IntImm(traits[0].extent.dtype, 0)) for kind in kind_order
    ]

    return tir.IndexMap(input_iters, final_indices, None)


def detect_iter_traits(block: SBlock) -> tuple[list[IterTrait]] | None:
    """Detect iter traits based on the pattern C[S, I, J] += A[S, I, K] * B[S, J, K]

    Parameters
    ----------
    block : s_tir.SBlock
        The block to be analyzed

    Returns
    -------
    traits : Optional[Tuple[List[IterTrait]]]
        The detected iter traits for axes in A, B and C. None if the block
        does not match the pattern.

    """

    if len(block.reads) != 2 or len(block.writes) != 1:
        return None

    def get_access_axes(region: list[Range]) -> list[Var]:
        axes: list[Var] = list()
        for r in region:
            if not _is_one(r.extent):
                raise ValueError("Expect elemwise block access")
            new_axes = set(analysis.undefined_vars(r.min))
            for ax in new_axes:
                if ax not in axes:
                    axes.append(ax)
        return axes

    try:
        A_axes = get_access_axes(block.reads[0].region)
        B_axes = get_access_axes(block.reads[1].region)
        C_axes = get_access_axes(block.writes[0].region)
    except ValueError:
        return None

    traits: dict[Var, IterTrait] = {}
    for iter_var in block.iter_vars:
        var = iter_var.var
        kind: IterKind
        if _is_one(iter_var.dom.extent):
            kind = IterKind.kIter_T
        elif iter_var.iter_type == iter_var.DataPar:
            if var in A_axes and var in B_axes and var in C_axes:
                kind = IterKind.kIter_S
            elif var in A_axes and var in C_axes:
                kind = IterKind.kIter_I
            elif var in B_axes and var in C_axes:
                kind = IterKind.kIter_J
            else:
                return None
        elif iter_var.iter_type == tir.IterVar.CommReduce:
            if var in A_axes and var in B_axes and var not in C_axes:
                kind = IterKind.kIter_K
            else:
                return None
        else:
            return None
        traits[var] = IterTrait(kind, iter_var.dom.extent)

    # A Gemm-kernel requires have I, J and K axes
    gemm_traits = {IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K}
    if {x.kind for x in traits.values()}.intersection(gemm_traits) != gemm_traits:
        return None

    A_traits = [traits[var] for var in A_axes]
    B_traits = [traits[var] for var in B_axes]
    C_traits = [traits[var] for var in C_axes]
    block_traits = [traits[i.var] for i in block.iter_vars]
    return A_traits, B_traits, C_traits, block_traits


def get_index_map(block: SBlock) -> tuple[tir.IndexMap, ...] | None:
    """Get index maps for the block

    Parameters
    ----------
    block : s_tir.SBlock
        The block to be analyzed

    Returns
    -------
    index_maps : Optional[Tuple[tir.IndexMap]]
        The index maps for the block, or None if the block is not a gemm-liked kernel
    """
    traits = detect_iter_traits(block)
    if traits is None:
        return None
    A_traits, B_traits, C_traits, block_traits = traits

    A_index_map = make_iter_fusion_index_map(
        A_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_K]
    )
    B_index_map = make_iter_fusion_index_map(
        B_traits, [IterKind.kIter_S, IterKind.kIter_J, IterKind.kIter_K]
    )
    C_index_map = make_iter_fusion_index_map(
        C_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J]
    )
    matmul_index_map = make_iter_fusion_index_map(
        block_traits,
        [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K],
    )

    return (
        matmul_index_map,
        A_index_map,
        B_index_map,
        C_index_map,
    )


def is_transposed(index_map):
    initial_indices = index_map.initial_indices
    final_indices = index_map.final_indices
    if len(initial_indices) not in (2, 3) or len(final_indices) != 3:
        return None

    if len(initial_indices) == 2:
        ax1, ax2 = initial_indices
    else:
        ax0, ax1, ax2 = initial_indices
    zx0, zx1, zx2 = index_map.final_indices  # [S, J, K]

    # [.., i, j] ==> [.., i, j]
    if ax1.same_as(zx1) and ax2.same_as(zx2):
        return False
    # [.., i, j] ==> [.., j, i]
    elif ax1.same_as(zx2) and ax2.same_as(zx1):
        return True
    else:
        return None


class DequantMatmulTensorization(AdrenoScheduleRule):
    """Dequant specific schedule rule for Matmul"""

    @dataclass
    class Config:
        thread_size_x: int
        thread_size_y: int
        thread_size_z: int
        splits_m: tuple[None, int, int, int]
        splits_n: tuple[None, int, int, int]
        splits_k: tuple[None, int, int, int]
        intrin_tile_m: int
        intrin_tile_n: int
        intrin_tile_k: int
        vector_size: int
        unroll: bool
        use_textures: bool
        b_trans: bool
        load_scope_a: str
        load_scope_b: str
        store_scope_c: str
        in_dtype: str
        out_dtype: str
        use_storage_align: bool

    @staticmethod
    def get_config(block_info: SBlockInfo, tgt: Target):
        """Get Schedule Configs for the given target"""
        tiles = get_matmul_supported_profile(block_info)
        if tiles is None:
            return None

        def get_max_factor(n, factors):
            factors = sorted(factors, reverse=True)
            for factor in factors:
                if n % factor == 0:
                    return factor
            return 1

        if tgt.attrs.get("supports_qcom_cooperative_matrix_conversion", False):
            tile_m_factor = 2
            intermediate_storage_scope = "local"
        else:
            tile_m_factor = 1
            intermediate_storage_scope = "shared"

        tile_m, tile_n, tile_k = tiles
        dtypes = analysis.get_in_out_dtypes(block_info.block_stmt)
        n = block_info.block_stmt.writes[0].buffer.shape[-1]
        tile_n_factor = n // tile_n
        thread_size_x, thread_size_y, thread_size_z = (
            tile_m,
            tile_m_factor,
            get_max_factor(tile_n_factor, [1, 2]),
        )
        # Special Case where shared mem is not reused
        if intermediate_storage_scope == "shared" and dtypes[1][0] == "int32":
            thread_size_z = 1
        _, a_index_map, b_index_map, _ = get_index_map(block_info.block_stmt)
        a_trans = is_transposed(a_index_map)
        b_trans = is_transposed(b_index_map)
        # This case won't be likely
        if b_trans is None or a_trans is None:
            return None

        config = DequantMatmulTensorization.Config(
            thread_size_x=thread_size_x,
            thread_size_y=thread_size_y,
            thread_size_z=thread_size_z,
            splits_m=(None, thread_size_y, 1, tile_m),
            splits_n=(None, thread_size_z, 1, tile_n),
            splits_k=(None, 2, 1, tile_k),
            intrin_tile_m=tile_m,
            intrin_tile_n=tile_n,
            intrin_tile_k=tile_k,
            vector_size=4,
            unroll=True,
            use_textures=False,
            b_trans=b_trans,
            load_scope_a="global",
            load_scope_b=intermediate_storage_scope,
            store_scope_c=intermediate_storage_scope,
            in_dtype=dtypes[0][0],
            out_dtype=dtypes[1][0],
            use_storage_align=True,
        )

        return config

    @staticmethod
    def check_applicability(func):
        block_names = ["compute", "dequantize", "matmul"]
        sch = s_tir.Schedule(func)
        main_block = analysis.get_root_block(sch)
        child_blocks = sch.get_child_blocks(main_block)
        child_names = [sch.get(blk).name_hint for blk in child_blocks]

        return all(blk_name in child_names for blk_name in block_names)

    @staticmethod
    def is_supported(func, tgt):
        conditions = []
        conditions.append(tgt.kind.name == "vulkan")
        conditions.append(tgt.attrs.get("supports_khr_cooperative_matrix", False))
        conditions.append(DequantMatmulTensorization.check_applicability(func))

        return all(conditions)

    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> s_tir.Schedule | None:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        if not DequantMatmulTensorization.is_supported(func, target):
            return None

        from tvm.s_tir.tensor_intrin.adreno import (
            get_adreno_wmma_intrin_group,
        )  # pylint: disable=import-outside-toplevel

        sch = s_tir.Schedule(func)
        root = sch.get_sblock(name="root", func_name="main")
        blocks = sch.get_child_blocks(root)
        reduction_blocks = analysis.get_reduction_blocks(sch, blocks)

        if reduction_blocks is None or len(reduction_blocks) != 1:
            return None

        main_block = reduction_blocks[0]
        main_block_info = analysis.get_sblock_info(sch, main_block)
        if not main_block_info.is_gemm():
            return None

        config = DequantMatmulTensorization.get_config(main_block_info, target)
        if config is None:
            return None

        if not get_index_map(sch.get(main_block)):
            return None

        reduction_loops = sch.get_loops(main_block)
        if not len(reduction_loops) == 4:
            return None

        mb, ms, n, k = [sch.get(ll).extent for ll in reduction_loops]
        if not (isinstance(n, tir.IntImm) and isinstance(mb, tir.IntImm)):
            return None

        index_maps = get_index_map(sch.get(main_block))
        assert index_maps is not None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        compute_blk, dequant_blk = sch.get_sblock("compute"), sch.get_sblock("dequantize")
        quant_block = sch.cache_read(compute_blk, 0, "local")
        if len(sch.get_consumers(main_block)) > 0:
            epilogue_block = sch.get_consumers(main_block)[0]
            while len(sch.get_consumers(epilogue_block)) > 0:
                sch.compute_inline(epilogue_block)
                epilogue_block = sch.get_consumers(main_block)[0]
        else:
            epilogue_block = None

        def intrin_lp_idx(scope):
            return -2 if scope in ("global", "shared") else -1

        sch.transform_block_layout(main_block, matmul_index_map)
        sch.pad_einsum(
            main_block,
            [
                1,
                config.splits_m[1] * config.splits_m[3],
                config.splits_n[1] * config.splits_n[3],
                config.intrin_tile_k,
            ],
        )
        sch.compute_inline(compute_blk)
        pada_block = None
        if not (isinstance(ms, tir.IntImm) and ms % config.intrin_tile_m == 0):
            pada_block = sch.get_producers(main_block)[0]
            align_fc = sch.get(sch.get_loops(pada_block)[-1]).extent.value
            sch.storage_align(pada_block, buffer_index=0, axis=-2, factor=align_fc, offset=64)
            padc_block = sch.get_consumers(main_block)[0]
        else:
            padc_block = sch.reindex(main_block, ("write", 0))

        loada_block = sch.cache_read(main_block, 0, config.load_scope_a)
        loadb_block = dequant_blk

        if config.use_textures and pada_block:
            sch.transform_layout(
                pada_block,
                ("write", 0),
                lambda b, i, j: (b, i, j // config.vector_size, j % config.vector_size),
            )

        if config.b_trans:
            sch.transform_block_layout(quant_block, lambda k, j: (j, k))
            sch.transform_block_layout(dequant_blk, lambda k, j: (j, k))
            sch.transform_layout(quant_block, ("write", 0), lambda k, j: (j, k))
            sch.transform_layout(dequant_blk, ("write", 0), lambda k, j: (j, k))

        bz, m, n, k = sch.get_loops(main_block)
        mb, ty, vx, tile_m = sch.split(m, config.splits_m)
        nb, tz, vy, tile_n = sch.split(n, config.splits_n)
        ko, ks, kq, tile_k = sch.split(k, config.splits_k)
        sch.reorder(bz, mb, nb, vx, vy, ty, tz, ko, ks, kq, tile_m, tile_n, tile_k)

        sch.bind(mb, "blockIdx.x")
        sch.bind(nb, "blockIdx.y")
        sch.bind(bz, "blockIdx.z")
        sch.bind(vx, "vthread.x")
        sch.bind(vy, "vthread.y")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tz, "threadIdx.z")

        A_wmma = sch.cache_read(main_block, 0, "wmma.matrix_a")

        sch.compute_at(A_wmma, kq)
        if config.load_scope_a in ("local", "shared"):
            sch.compute_at(loada_block, kq)
        else:
            sch.compute_inline(loada_block)
        sch.compute_at(loadb_block, kq)
        sch.compute_at(quant_block, ks)
        sch.reverse_compute_at(padc_block, tz)

        C_store = sch.cache_read(padc_block, 0, config.store_scope_c)
        C_init = sch.decompose_reduction(main_block, ko)
        sch.set_scope(C_init, 0, "wmma.accumulator")
        if epilogue_block:
            sch.reverse_compute_inline(epilogue_block)

        sch.bind(sch.get_loops(quant_block)[-2], "threadIdx.x")
        sch.bind(sch.get_loops(padc_block)[-2], "threadIdx.x")
        sch.unroll(sch.get_loops(quant_block)[-1])
        sch.set_scope(dequant_blk, 0, config.load_scope_b)
        B_wmma = sch.cache_write(dequant_blk, 0, config.load_scope_b)
        sch.set_scope(B_wmma, 0, "wmma.matrix_b")

        # Load A Tensor
        if config.load_scope_a in ("local", "shared"):
            sch.bind(sch.get_loops(loada_block)[-2], "threadIdx.x")
            ux, vx = sch.split(sch.get_loops(loada_block)[-1], [None, config.vector_size])
            sch.vectorize(vx)
            sch.unroll(ux)

        # Load B Tensor
        if config.load_scope_b == "local":
            sch.bind(sch.get_loops(loadb_block)[-2], "threadIdx.x")
            sch.unroll(sch.get_loops(loadb_block)[-1])
        elif config.load_scope_b == "shared":
            dtx, dvec = sch.get_loops(loadb_block)[-2:]
            _, tx = sch.split(dtx, [None, config.intrin_tile_n])
            sch.bind(tx, "threadIdx.x")

            ux, vx = sch.split(dvec, [None, config.vector_size])
            sch.vectorize(vx)
            sch.unroll(ux)
            # schedule annotation for CoopMat B block
            dtx, dvec = sch.get_loops(B_wmma)[-2:]
            dtz, dtx = sch.split(dtx, [None, config.intrin_tile_n])
            sch.bind(dtz, "threadIdx.z")

        # WMMA Bindings
        if config.load_scope_a == "local":
            sch.bind(sch.get_loops(A_wmma)[-2], "threadIdx.x")
        if config.load_scope_b == "local":
            sch.bind(sch.get_loops(B_wmma)[-2], "threadIdx.x")
        if config.store_scope_c == "local":
            sch.bind(sch.get_loops(C_store)[-2], "threadIdx.x")

        a_idx, b_idx = intrin_lp_idx(config.load_scope_a), intrin_lp_idx(config.load_scope_b)
        c_idx = intrin_lp_idx(config.store_scope_c)

        INTRIN = get_adreno_wmma_intrin_group(
            config.intrin_tile_m,
            config.intrin_tile_n,
            config.intrin_tile_k,
            (config.load_scope_a, config.load_scope_b),
            config.store_scope_c,
            False,
            True,
            config.in_dtype,
            config.out_dtype,
        )
        sch.tensorize(sch.get_loops(C_init)[-2], INTRIN["init"])
        sch.tensorize(sch.get_loops(A_wmma)[a_idx], INTRIN["load_a"])
        sch.tensorize(sch.get_loops(B_wmma)[b_idx], INTRIN["load_b"])
        sch.tensorize(sch.get_loops(main_block)[-3], INTRIN["compute"])
        sch.tensorize(sch.get_loops(C_store)[c_idx], INTRIN["store"])

        if config.unroll:
            sch.annotate(tz, ann_key="pragma_auto_unroll_max_step", ann_val=8)
            sch.annotate(tz, ann_key="pragma_unroll_explicit", ann_val=1)
        # Schedule PadA
        if pada_block:
            by, m, n = sch.get_loops(pada_block)
            ux, tx, vx = sch.split(n, [None, 32, config.vector_size])
            b = sch.fuse(m, ux)
            bx, ty = sch.split(b, [None, config.vector_size])
            sch.bind(bx, "blockIdx.x")
            sch.bind(by, "blockIdx.y")
            sch.bind(tx, "threadIdx.x")
            sch.bind(ty, "threadIdx.y")
            sch.vectorize(vx)
            sch.set_scope(pada_block, 0, "global")

        # Schedule PadC
        ux, vx = sch.split(sch.get_loops(padc_block)[-1], [None, config.vector_size])
        sch.unroll(ux)
        # sch.vectorize(vx) # TODO: Uncomment once Codegen Bug is fixed

        return sch


class MatmulTensorization(AdrenoScheduleRule):
    """Schedule rule for Adreno cooperative matrix matmul using tensor intrinsics"""

    @dataclass
    class Config:
        thread_size_x: int
        thread_size_y: int
        thread_size_z: int
        splits_m: tuple[None, int, int]
        splits_n: tuple[None, int, int]
        splits_k: tuple[None, int]
        intrin_tile_m: int
        intrin_tile_n: int
        intrin_tile_k: int
        vector_size: int
        unroll: bool
        load_scope_a: str
        load_scope_b: str
        store_scope_c: str
        a_trans: bool
        b_trans: bool
        in_dtype: str
        out_dtype: str

    @staticmethod
    def get_config(block_info: SBlockInfo, target: Target):
        """Get Schedule Configs for the given target"""
        tiles = get_matmul_supported_profile(block_info)
        if tiles is None:
            return None

        has_qcom_extn = target.attrs.get("supports_qcom_cooperative_matrix_conversion", False)

        tile_m, tile_n, tile_k = tiles
        dtypes = analysis.get_in_out_dtypes(block_info.block_stmt)
        A_dtype, B_dtype = dtypes[0]
        (C_dtype,) = dtypes[1]

        # Get transpose information
        index_maps = get_index_map(block_info.block_stmt)
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps
        a_trans = is_transposed(a_index_map)
        b_trans = is_transposed(b_index_map)
        b_trans = not b_trans if isinstance(b_trans, bool) else b_trans

        thread_size_x, thread_size_y, thread_size_z = tile_m, 1, 1
        shape = [it.dom for it in block_info.iters]
        shape = matmul_index_map.map_shape(shape)

        # Shared Memory has limits, thus we limit number of waves
        shared_factor = 2 if C_dtype == "float16" else 1
        if all(isinstance(dim, tir.IntImm) for dim in shape[1:3]):
            _, M, N, _ = shape
            BM, BN = (int(M) + tile_m - 1) // tile_m, (int(N) + tile_n - 1) // tile_n

            def p_two(x):
                return 2 ** (x.bit_length() - 1)

            pM, pN = p_two(BM), p_two(BN)

            thread_size_y, thread_size_z = 1, 1
            if pM >= 2 and pN >= 2:
                if has_qcom_extn:
                    thread_size_y, thread_size_z = 2, 2
                else:
                    thread_size_y, thread_size_z = 1, shared_factor
            else:
                if has_qcom_extn:
                    thread_size_y = min(pM, 4)
                    thread_size_z = min(pN, 4)
                else:
                    thread_size_y = min(pM, shared_factor)
                    thread_size_z = min(pN, shared_factor)
        else:
            thread_size_y, thread_size_z = 1, 1

        return MatmulTensorization.Config(
            thread_size_x,
            thread_size_y,
            thread_size_z,
            splits_m=(None, thread_size_y, tile_m),
            splits_n=(None, thread_size_z, tile_n),
            splits_k=(None, tile_k),
            intrin_tile_m=tile_m,
            intrin_tile_n=tile_n,
            intrin_tile_k=tile_k,
            vector_size=4,
            # TODO(sanjs): unrolling causes vk_pipeline failure in some of the testcases
            #              check the root cause
            unroll=False,
            a_trans=a_trans,
            b_trans=b_trans,
            load_scope_a="global",
            load_scope_b="global",
            store_scope_c="local" if has_qcom_extn else "shared",
            in_dtype=A_dtype,
            out_dtype=C_dtype,
        )

    @staticmethod
    def check_applicability(func):
        block_names = ["matmul"]
        sch = s_tir.Schedule(func)
        main_block = analysis.get_root_block(sch)
        child_blocks = sch.get_child_blocks(main_block)
        child_names = [sch.get(blk).name_hint for blk in child_blocks]

        return all(blk_name in child_names for blk_name in block_names)

    @staticmethod
    def is_supported(func, tgt):
        conditions = []
        conditions.append(tgt.kind.name == "vulkan")
        conditions.append(tgt.attrs.get("supports_khr_cooperative_matrix", False))
        conditions.append(MatmulTensorization.check_applicability(func))

        return all(conditions)

    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> s_tir.Schedule | None:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        if not MatmulTensorization.is_supported(func, target):
            return None

        from tvm.s_tir.tensor_intrin.adreno import get_adreno_wmma_intrin_group

        sch = s_tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)
        reduction_blocks = analysis.get_reduction_blocks(sch, blocks)

        if reduction_blocks is None or len(reduction_blocks) != 1:
            return None

        main_block = reduction_blocks[0]
        main_block_info = analysis.get_sblock_info(sch, main_block)
        if not main_block_info.is_gemm():
            return None

        config = MatmulTensorization.get_config(main_block_info, target)
        if config is None:
            return None

        index_maps = get_index_map(sch.get(main_block))
        assert index_maps is not None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        A_block = sch.cache_read(main_block, 0, "global")
        B_block = sch.cache_read(main_block, 1, "global")
        sch.transform_block_layout(main_block, matmul_index_map)
        sch.pad_einsum(
            main_block,
            [
                1,
                config.thread_size_y * config.intrin_tile_m,
                config.thread_size_z * config.intrin_tile_n,
                config.intrin_tile_k,
            ],
        )

        def is_padded(block):
            return sch.get(sch.get_consumers(block)[0]) != sch.get(main_block)

        is_Apad, is_Bpad = is_padded(A_block), is_padded(B_block)
        is_Cpad = len(sch.get_consumers(main_block)) > 0

        if is_Apad:
            new_A_block = sch.get_consumers(A_block)[0]
            sch.compute_inline(A_block)
            A_block = new_A_block
        else:
            if config.b_trans is not None:
                sch.compute_inline(A_block)
                A_block = None
        if is_Bpad:
            new_B_block = sch.get_consumers(B_block)[0]
            sch.compute_inline(B_block)
            B_block = new_B_block
        else:
            if config.b_trans is not None:
                sch.compute_inline(B_block)
                B_block = None

        if A_block is not None:
            sch.transform_layout(A_block, ("write", 0), a_index_map)
            config.a_trans = False
        if B_block is not None:
            sch.transform_layout(B_block, ("write", 0), b_index_map)
            config.b_trans = True
        if is_Cpad:
            C_block = sch.get_consumers(main_block)[0]
        else:
            C_block = None

        bz, m, n, k = sch.get_loops(main_block)
        mb, ty, tile_m = sch.split(m, config.splits_m)
        nb, tz, tile_n = sch.split(n, config.splits_n)
        ko, tile_k = sch.split(k, config.splits_k)
        sch.reorder(bz, mb, nb, ty, tz, ko, tile_m, tile_n, tile_k)

        sch.bind(mb, "blockIdx.x")
        sch.bind(nb, "blockIdx.y")
        sch.bind(bz, "blockIdx.z")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tz, "threadIdx.z")

        A_wmma = sch.cache_read(main_block, 0, "wmma.matrix_a")
        B_wmma = sch.cache_read(main_block, 1, "wmma.matrix_b")
        C_store = sch.cache_write(main_block, 0, "wmma.accumulator")

        sch.compute_at(A_wmma, ko)
        sch.compute_at(B_wmma, ko)
        sch.reverse_compute_at(C_store, tz)

        C_init = sch.decompose_reduction(main_block, ko)

        def intrin_lp_idx(scope):
            return -2 if scope in ("global", "shared") else -1

        if not is_Cpad:
            config.store_scope_c = "global"
        c_idx = intrin_lp_idx(config.store_scope_c)
        sch.set_scope(C_store, 0, config.store_scope_c)

        if config.store_scope_c == "local":
            sch.bind(sch.get_loops(C_store)[-2], "threadIdx.x")

        if C_block is not None:
            sch.reverse_compute_at(C_block, tz)
            pad_tx, _ = sch.split(sch.get_loops(C_block)[-2], [config.intrin_tile_m, None])
            sch.bind(pad_tx, "threadIdx.x")

        INTRIN = get_adreno_wmma_intrin_group(
            config.intrin_tile_m,
            config.intrin_tile_n,
            config.intrin_tile_k,
            (config.load_scope_a, config.load_scope_b),
            config.store_scope_c,
            config.a_trans,
            config.b_trans,
            config.in_dtype,
            config.out_dtype,
        )
        sch.tensorize(sch.get_loops(C_init)[-2], INTRIN["init"])
        sch.tensorize(sch.get_loops(A_wmma)[-2], INTRIN["load_a"])
        sch.tensorize(sch.get_loops(B_wmma)[-2], INTRIN["load_b"])
        sch.tensorize(sch.get_loops(main_block)[-3], INTRIN["compute"])
        sch.tensorize(sch.get_loops(C_store)[c_idx], INTRIN["store"])

        if config.unroll:
            sch.annotate(tz, ann_key="pragma_auto_unroll_max_step", ann_val=8)
            sch.annotate(tz, ann_key="pragma_unroll_explicit", ann_val=1)

        if A_block:
            schedule_default(sch, A_block)
        if B_block:
            schedule_default(sch, B_block)

        return sch
