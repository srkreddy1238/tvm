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
"""A Conv2d schedule rule for Adreno GPU operators."""

from dataclasses import dataclass
from typing import Literal

from tvm import s_tir, tir
from tvm.target import Target

from .. import analysis
from ..analysis import SBlockInfo
from .base import AdrenoScheduleRule
from .utils import get_matmul_supported_profile, schedule_default, schedule_inline_blocks


class Conv2DTensorization(AdrenoScheduleRule):
    @dataclass
    class Config:
        thread_size_x: int
        thread_size_y: int
        thread_size_z: int
        splits_hw: tuple[None, int]
        splits_oc: tuple[None, int]
        splits_kc: tuple[None, int]
        intrin_tile_m: int
        intrin_tile_n: int
        intrin_tile_k: int
        vector_size: int
        unroll: bool
        ldst_scope: Literal["local", "shared"]
        in_dtype: str
        out_dtype: str

    @staticmethod
    def get_config(block_info: SBlockInfo, tgt: Target):
        """Get Schedule Configs for the given Target"""
        PROFILE = get_matmul_supported_profile(block_info)
        if PROFILE is None:
            return None

        tile_m, tile_n, tile_k = PROFILE
        thread_size_x, thread_size_y, thread_size_z = tile_m, 1, 4
        in_dtype, out_dtype = analysis.get_in_out_dtypes(block_info.block_stmt)
        scope = (
            "local"
            if tgt.attrs.get("supports_qcom_cooperative_matrix_conversion", False)
            else "shared"
        )

        config = Conv2DTensorization.Config(
            thread_size_x=thread_size_x,
            thread_size_y=thread_size_y,
            thread_size_z=thread_size_z,
            splits_hw=(None, tile_m),
            splits_oc=(None, tile_n),
            splits_kc=(None, tile_k),
            intrin_tile_m=tile_m,
            intrin_tile_n=tile_n,
            intrin_tile_k=tile_k,
            vector_size=4,
            unroll=True,
            ldst_scope=scope,
            in_dtype=in_dtype[0],
            out_dtype=out_dtype[0],
        )

        return config

    @staticmethod
    def is_supported(func: tir.PrimFunc, tgt: Target) -> bool:
        has_khr_coopmat = tgt.attrs.get("supports_khr_cooperative_matrix", False)

        block_names = ["input_imed", "weight_imed", "conv_matrix", "conv_reinterpret"]

        sch = s_tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        child_block_names = [sch.get(blk).name_hint for blk in sch.get_child_blocks(root_block)]
        return has_khr_coopmat and all(blk in child_block_names for blk in block_names)

    # pylint: disable=too-many-locals,missing-docstring
    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> s_tir.Schedule | None:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        if not Conv2DTensorization.is_supported(func, target):
            return None

        sch = s_tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        child_blocks = sch.get_child_blocks(root_block)
        convolution_blks = ["input_imed", "weight_imed", "conv_matrix", "conv_reinterpret"]
        input_blk, weight_blk, compute_blk, reinterpret_blk = [
            sch.get_sblock(name) for name in convolution_blks
        ]
        remaining_blks = [
            blk
            for blk in child_blocks
            if sch.get(blk).name_hint not in ["data_pad", *convolution_blks]
        ]

        config = Conv2DTensorization.get_config(analysis.get_sblock_info(sch, compute_blk), target)
        if config is None:
            return None

        try:
            data_pad = sch.get_sblock("data_pad")
        except IndexError:
            data_pad = None

        nn, gg, pad_out_hw, pad_gg_oc, kh, kw, kc = sch.get_loops(compute_blk)
        hwo, hwi = sch.split(pad_out_hw, config.splits_hw)
        oco, oci = sch.split(pad_gg_oc, config.splits_oc)
        kco, kci = sch.split(kc, config.splits_kc)
        sch.reorder(nn, gg, hwo, oco, kh, kw, kco, hwi, oci, kci)

        sch.bind(nn, "blockIdx.z")
        sch.bind(gg, "blockIdx.y")
        sch.bind(hwo, "blockIdx.x")
        sch.bind(oco, "threadIdx.y")

        sch.compute_at(input_blk, kco)
        sch.compute_at(weight_blk, kco)

        sch.set_scope(weight_blk, 0, config.ldst_scope)
        sch.set_scope(input_blk, 0, config.ldst_scope)

        A_wmma = sch.cache_read(compute_blk, 0, "wmma.matrix_a")
        B_wmma = sch.cache_read(compute_blk, 1, "wmma.matrix_b")
        C_store = sch.cache_write(compute_blk, 0, "wmma.accumulator")
        C_init = sch.decompose_reduction(compute_blk, kh)

        output_blk = sch.cache_write(C_store, 0, config.ldst_scope)

        sch.reverse_compute_at(C_store, oco)
        sch.reverse_compute_at(output_blk, oco)

        sch.bind(sch.get_loops(input_blk)[-2], "threadIdx.x")
        sch.bind(sch.get_loops(weight_blk)[-2], "threadIdx.x")
        sch.bind(sch.get_loops(output_blk)[-2], "threadIdx.x")

        if config.ldst_scope == "shared":
            ty, tx = sch.split(sch.get_loops(B_wmma)[-2], [None, config.intrin_tile_n])
            sch.bind(ty, "threadIdx.y")
        elif config.ldst_scope == "local":
            sch.bind(sch.get_loops(A_wmma)[-2], "threadIdx.x")
            sch.bind(sch.get_loops(B_wmma)[-2], "threadIdx.x")
            sch.bind(sch.get_loops(C_store)[-2], "threadIdx.x")
        else:
            raise ValueError("Schedule Doesn't Support Global ldst_scope")

        ux, vx = sch.split(sch.get_loops(input_blk)[-1], [None, config.vector_size])
        sch.vectorize(vx)
        ux, vx = sch.split(sch.get_loops(weight_blk)[-1], [None, config.vector_size])
        sch.vectorize(vx)
        ux, vx = sch.split(sch.get_loops(output_blk)[-1], [None, config.vector_size])

        def intrin_lp_idx(scope):
            return -2 if scope in ("shared",) else -1

        idx = intrin_lp_idx(config.ldst_scope)

        from tvm.s_tir.tensor_intrin.adreno import get_adreno_wmma_intrin_group

        INTRIN = get_adreno_wmma_intrin_group(
            config.intrin_tile_m,
            config.intrin_tile_n,
            config.intrin_tile_k,
            config.ldst_scope,
            config.ldst_scope,
            False,
            True,
            "float32",
            "float32",
        )
        sch.tensorize(sch.get_loops(A_wmma)[idx], INTRIN["load_a"])
        sch.tensorize(sch.get_loops(B_wmma)[idx], INTRIN["load_b"])
        sch.tensorize(sch.get_loops(C_store)[idx], INTRIN["store"])
        sch.tensorize(sch.get_loops(C_init)[-2], INTRIN["init"])
        sch.tensorize(sch.get_loops(compute_blk)[-3], INTRIN["compute"])

        if data_pad:
            schedule_default(sch, data_pad)

        # TODO(sanjs) : Enable Inlining Reinterpret Block Once Codegen/Compiler is Fixed
        # sch.vectorize(vx)
        # sch.reverse_compute_inline(reinterpret_blk)

        schedule_default(sch, reinterpret_blk)
        remaining_blks = schedule_inline_blocks(sch, remaining_blks)
        schedule_default(sch, remaining_blks)  # TODO(sanjs) : Remove once Fixed

        return sch


class Conv2D(AdrenoScheduleRule):
    """The schedule rule for convolution computation"""

    @staticmethod
    def schedule_conv2d(sch: s_tir.Schedule, blk: s_tir.schedule.SBlockRV):
        block = analysis.get_sblock_info(sch, blk)
        vsize = 4

        n, oc, oh, ow, ob, ic, kh, kw = sch.get_loops(blk)
        ic_size = block.iters[-3].dom

        bz, vz, tz = sch.split(oc, [None, 8, 1], preserve_unit_iters=True)
        by, vy, ty = sch.split(oh, [None, 1, 16], preserve_unit_iters=True)
        bx, vx, tx = sch.split(ow, [None, 1, 16], preserve_unit_iters=True)

        bz = sch.fuse(n, bz)
        sch.reorder(bz, by, bx, vz, vy, vx, tz, ty, tx, ob)
        sch.bind(bz, "blockIdx.z")
        sch.bind(by, "blockIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.bind(vz, "vthread.z")
        sch.bind(vy, "vthread.y")
        sch.bind(vx, "vthread.x")
        sch.bind(tz, "threadIdx.z")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        icc, icb = sch.split(ic, [None, vsize if ic_size % vsize == 0 else ic_size])
        ico, ici = sch.split(icc, [None, 1])
        kho, khi = sch.split(kh, [None, 1])
        kwo, kwi = sch.split(kw, [None, 1])
        sch.reorder(ico, kho, kwo, ici, khi, kwi, icb, ob)
        sch.vectorize(ob)

        # Data Local
        kernel_rblk = sch.cache_read(blk, 1, "local")
        sch.compute_at(kernel_rblk, icb)
        sch.vectorize(sch.get_loops(kernel_rblk)[-1])

        # Output Local
        wblk = sch.cache_write(blk, 0, "local")
        sch.reverse_compute_at(wblk, tx)
        sch.vectorize(sch.get_loops(wblk)[-1])

        # Init
        init_blk = sch.decompose_reduction(blk, tx)
        sch.vectorize(sch.get_loops(init_blk)[-1])

    # pylint: disable=too-many-locals,missing-docstring
    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> s_tir.Schedule | None:
        if not (isinstance(func, tir.PrimFunc | s_tir.Schedule)) or not self.is_target_available(
            target
        ):
            return None

        if isinstance(func, tir.PrimFunc):
            sch = s_tir.Schedule(func)
            sch.work_on("main")
        elif isinstance(func, s_tir.Schedule):
            sch = func

        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)
        reduction_blocks = list(
            filter(lambda block: analysis.get_sblock_info(sch, block).is_reduction(), blocks)
        )
        remaining_blocks = [blk for blk in blocks if blk not in reduction_blocks]

        def is_convolution(blk):
            block_info = analysis.get_sblock_info(sch, blk)
            return "conv2d_NCHWc" in block_info.name

        if len(reduction_blocks) != 1 or not is_convolution(reduction_blocks[0]):
            return None

        conv_blk = reduction_blocks[0]
        Conv2D.schedule_conv2d(sch, conv_blk)
        remaining_blocks = schedule_inline_blocks(sch, remaining_blocks)
        schedule_default(sch, remaining_blocks)
        return sch
