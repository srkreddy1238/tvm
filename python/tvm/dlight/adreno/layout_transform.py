# licensed to the apache software foundation (asf) under one
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
"Texture Layout Transform Schedules for Adreno"
from typing import List, Union, Dict

from collections import namedtuple
from tvm import arith, tir
from tvm.target import Target
from tvm.tir import Schedule
from tvm.tir.schedule import BlockRV
from tvm._ffi import get_global_func

get_blockrealize = get_global_func("tir.schedule.GetBlockRealize")

from ..base import (
    detect_dominant_read,
    normalize_prim_func,
    try_inline_contiguous_spatial,
)
from .base import AdrenoScheduleRule

# BufferIndex Types
Index = namedtuple("Index", ["sub"])  # c
RemIndex = namedtuple("RemIndex", ["sub", "div"])  # c%len
DivIndex = namedtuple("DivIndex", ["sub", "div"])  # c//len
MergeIndex = namedtuple("MulIndex", ["dom", "mul", "sub"])  # co*len + cb
BufIndex = List[Union[Index, RemIndex, DivIndex, MergeIndex, None]]


def extract_index_types(sch: tir.Schedule, buf: tir.BufferRegion) -> BufIndex:
    buf_index = []
    for expr in buf.region:
        expr = expr.min
        dim = None
        if isinstance(expr, tir.expr.Add) and isinstance(expr.b, tir.expr.Var):
            var_add = expr.b
            if (
                isinstance(expr, tir.expr.Mul)
                and isinstance(expr.a, tir.expr.Var)
                and isinstance(expr.b, tir.expr.IntImm)
            ):
                mul = expr.b
                var_mul = expr.a
                dim = MergeIndex(var_mul, mul, var_add)
        elif (
            isinstance(expr, tir.expr.FloorMod)
            and isinstance(expr.a, tir.expr.Var)
            and isinstance(expr.b, tir.expr.IntImm)
        ):
            dim = RemIndex(expr.a, expr.b)
        elif (
            isinstance(expr, tir.expr.FloorDiv)
            and isinstance(expr.a, tir.expr.Var)
            and isinstance(expr.b, tir.expr.IntImm)
        ):
            dim = DivIndex(expr.a, expr.b)
        elif isinstance(expr, tir.expr.Var):
            dim = Index(expr)
        buf_index.append(dim)
    return buf_index


def extract_vector_loop(
    sch: tir.Schedule, buf: tir.BufferRegion, var_lp: Dict[tir.Var, tir.schedule.LoopRV]
) -> tir.schedule.LoopRV:
    end_index = extract_index_types(sch, buf)[-1]
    if isinstance(end_index, DivIndex):
        return None
    end_var = getattr(end_index, "sub")
    return var_lp[end_var]


class TextureTranspose(AdrenoScheduleRule):
    """Texture Layout Transform Schedules for Adreno"""

    def apply(  # pylint: disable=too-many-locals
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        # pylint: disable=invalid-name

        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        bf_vlen, tx_vlen = (8, 4)

        sch = tir.Schedule(func)
        root_block = sch.get_block("root")
        if len(sch.get_child_blocks(root_block)) != 1:
            return None

        def is_compatible(sch: tir.Schedule, blk: tir.schedule.BlockRV):
            # Supports Transform Layouts of the Form [A, B, C, D] <-> [X//len,..., X%len] <-> [Xo*len + Xb,...,]
            block = sch.get(blk)
            bufs = [*block.reads, *block.writes]
            buf_exprs = [extract_index_types(sch, buf) for buf in bufs]
            return not any(buf_exprs[-1] is None for buf_exprs in buf_exprs)

        blk, block = sch.get_child_blocks(root_block)[0], sch.get(
            sch.get_child_blocks(root_block)[0]
        )

        if not is_compatible(sch, blk):
            return None

        read_buf, write_buf = (block.reads[0], block.writes[0])
        lps = sch.get_loops(blk)
        loops = [sch.get(lp) for lp in lps]
        iter_vars = [Var.var for Var in block.iter_vars]
        iter_values = get_blockrealize(sch, blk).iter_values
        var_lp = dict([loop.loop_var, lp] for loop, lp in zip(loops, lps))
        val_lp = dict(zip(iter_vars, [var_lp.get(val, None) for val in iter_values]))
        lpv_read, lpv_write = (
            extract_vector_loop(sch, read_buf, val_lp),
            extract_vector_loop(sch, write_buf, val_lp),
        )
        vlen_read, vlen_write = min(bf_vlen, read_buf.region[-1].extent), min(
            bf_vlen, write_buf.region[-1].extent
        )
        if lpv_read is None or lpv_write is None:
            return None

        local_cache = sch.get(lpv_read) != sch.get(lpv_write)
        block_loops = [
            lp
            for lp in lps
            if sch.get(lp) != sch.get(lpv_read) and sch.get(lp) != sch.get(lpv_write)
        ]
        vec_loops = (lpv_read, lpv_write) if local_cache else (lpv_read,)
        vlps = []
        for lp in vec_loops:
            extent = min(int(sch.get(lp).extent) & ~(int(sch.get(lp).extent) - 1), bf_vlen)
            blk_lp, vec_lp = sch.split(lp, [None, extent])
            block_loops.append(blk_lp)
            vlps.append(vec_lp)
        vec_loops = vlps
        print([sch.get(lp).loop_var for lp in [*block_loops, *vec_loops]])

        sch.reorder(*block_loops, *vec_loops)
        if local_cache:
            rblk = sch.cache_read(blk, 0, "local")
            sch.compute_at(rblk, block_loops[-1])
            print(sch.mod)
            lpv = sch.get_loops(rblk)[-1]
            sch.vectorize(lpv)
        wblk = sch.cache_write(blk, 0, "local")
        sch.reverse_compute_at(wblk, vec_loops[-2] if len(vec_loops) > 1 else block_loops[-1])
        sch.vectorize(sch.get_loops(wblk)[-1])

        b = sch.fuse(*block_loops)
        tx_extent = min(sch.get(b).extent, 256)
        bx, tx = sch.split(b, [None, tx_extent])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

        return sch

