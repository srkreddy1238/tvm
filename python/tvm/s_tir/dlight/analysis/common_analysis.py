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

# pylint: disable=missing-function-docstring, missing-class-docstring
# pylint: disable=unused-argument, unused-variable
"""Analysis on TIR blocks, loops and functions."""

from collections import namedtuple
from typing import Literal

from tvm_ffi import get_global_func

from tvm import ir, s_tir, tir
from tvm.ir import Range
from tvm.runtime import DataType
from tvm.s_tir import Schedule
from tvm.s_tir.schedule import SBlockRV
from tvm.s_tir.schedule.schedule import LoopRV
from tvm.target.target import Target
from tvm.tir import Var
from tvm.tir.analysis import undefined_vars
from tvm.tir.stmt import SBlock


class IterInfo:
    """Information about a loop/iter var."""

    kind: Literal["S", "R", "O"]
    var: tir.Var
    _dom: tir.PrimExpr
    loop_rv: s_tir.schedule.LoopRV

    def __init__(
        self,
        kind: Literal["S", "R", "O"],
        var: tir.Var,
        dom: tir.PrimExpr,
        loop_rv: s_tir.schedule.LoopRV,
    ):
        """Construct an IterInfo object."""
        self.kind = kind
        self.var = var
        self._dom = dom
        self.loop_rv = loop_rv

    @property
    def dom(self) -> int | tir.PrimExpr:
        """The iteration domain of the loop."""
        return int(self._dom) if isinstance(self._dom, tir.IntImm) else self._dom

    def __str__(self) -> str:
        return f'Iter("{self.kind}", {self.dom})'

    def __repr__(self) -> str:
        return str(self)


get_sblockrealize = get_global_func("s_tir.schedule.GetSBlockRealize")
# BufferIndex Types
Index = namedtuple("Index", ["sub"])  # c
RemIndex = namedtuple("RemIndex", ["sub", "div"])  # c%len
DivIndex = namedtuple("DivIndex", ["sub", "div"])  # c//len
MergeIndex = namedtuple("MulIndex", ["dom", "mul", "sub"])  # co*len + cb
BufIndex = list[Index | RemIndex | DivIndex | MergeIndex | None]


class BufferInfo:
    "Information about Buffer. Provides useful analysis"

    buf_region: tir.BufferRegion
    shape: tuple[int]
    assoc_lps: list[s_tir.schedule.LoopRV | None]
    assoc_lps_info: list[tir.For | None]

    def __init__(
        self,
        sch: s_tir.Schedule,
        block_rv: s_tir.schedule.SBlockRV,
        buf_region: tir.BufferRegion,
        lps: list[s_tir.schedule.LoopRV] | None,
    ):
        block = sch.get(block_rv)
        if lps is None:
            lps = sch.get_loops(block_rv)
        loops = [sch.get(lp) for lp in lps]
        iter_vars = [Var.var for Var in block.iter_vars]
        iter_values = get_sblockrealize(sch, block_rv).iter_values
        lpvar_lp = dict([loop.loop_var, lp] for loop, lp in zip(loops, lps))
        var_lp = dict(zip(iter_vars, [lpvar_lp.get(val, None) for val in iter_values]))

        def extract_index_types(buf: tir.BufferRegion) -> BufIndex:
            buf_index = []
            for expr in buf.region:
                expr = expr.min
                dim = None
                if isinstance(expr, tir.expr.Add) and isinstance(expr.b, tir.expr.Var):
                    expr_in = expr.a
                    var_add = expr.b
                    if (
                        isinstance(expr_in, tir.expr.Mul)
                        and isinstance(expr_in.a, tir.expr.Var)
                        and isinstance(expr_in.b, tir.expr.IntImm)
                    ):
                        mul = expr_in.b
                        var_mul = expr_in.a
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

        indexes = extract_index_types(buf_region)
        assoc_lps = [
            (
                var_lp.get(getattr(idx, "sub"), None)
                if not isinstance(idx, DivIndex) and idx is not None
                else None
            )
            for idx in indexes
        ]

        self.buf_region = buf_region
        self.assoc_lps = assoc_lps
        self.assoc_lps_info = [(sch.get(lp) if lp is not None else None) for lp in assoc_lps]
        self.shape = buf_region.buffer.shape

    def get_scope(self) -> str:
        return self.buf_region.buffer.scope()

    def get_vecsize(self, buf_index: int = 0, vbits: int = 128):
        if self.assoc_lps_info[-1] is None:
            return None

        vlp_extent = int(self.assoc_lps_info[-1].extent) & ~(
            int(self.assoc_lps_info[-1].extent) - 1
        )
        vbuf_extent = int(self.shape[-1]) & ~(int(self.shape[-1]) - 1)

        return min(vlp_extent, vbuf_extent, vbits // DataType(self.buf_region.buffer.dtype).bits)

    def __str__(self) -> str:
        return f"BufferInfo({self.buf_region})"

    def __repr__(self) -> str:
        return str(self)


class SBlockInfo:
    """Information about a TIR block. Provides useful analysis about the block."""

    name: str
    iters: list[IterInfo]
    block_stmt: SBlock
    block_rv: SBlockRV
    reads: list[BufferInfo]
    writes: list[BufferInfo]
    producers: list[SBlock]
    consumers: list[SBlock]

    def __init__(
        self,
        sch: s_tir.Schedule,
        block_rv: SBlockRV,
    ):
        """Construct a SBlockInfo object."""
        block_stmt = sch.get(block_rv)

        def _iter_kind(loop: tir.IterVar) -> str:
            return {tir.IterVar.DataPar: "S", tir.IterVar.CommReduce: "R"}.get(loop.iter_type, "O")

        lps = sch.get_loops(block_rv)
        iter_vars = block_stmt.iter_vars

        self.name = sch.get(block_rv).name_hint
        self.iters = [
            IterInfo(
                kind=_iter_kind(iter_var),
                var=iter_var.var,
                dom=iter_var.dom.extent,
                loop_rv=loop_rv,
            )
            for loop_rv, iter_var in zip(lps, iter_vars)
        ]
        self.block_stmt = block_stmt
        self.block_rv = block_rv
        self.read_bufs = [get_buffer_info(sch, block_rv, buf, lps) for buf in block_stmt.reads]
        self.write_bufs = [get_buffer_info(sch, block_rv, buf, lps) for buf in block_stmt.writes]
        self.producers = sch.get_producers(block_rv)
        self.consumers = sch.get_consumers(block_rv)

    def dom(self) -> list[int | tir.PrimExpr]:
        """The iteration domain of the block."""
        return [i.dom for i in self.iters]

    def dom_kind(self) -> str:
        """The iteration domain kind of the block, for example, SSSS, SSSR."""
        return "".join(i.kind for i in self.iters)

    def is_injective(self) -> bool:
        """Whether the SBlock is injective, i.e. all its iteration domains are injective."""
        return all(k == "S" for k in self.dom_kind())

    def is_elementwise(self) -> bool:
        """Whether the SBlock is elementwise, i.e. trivial mapping between read/write region"""
        if not self.is_injective() or len(self.write_bufs) != 1:
            return False

        w_region = self.write_bufs[0].buf_region.region
        for read_buf in self.read_bufs:
            r_region = read_buf.buf_region.region
            if len(r_region) != len(w_region):
                return False
            for r_var, w_var in zip(r_region, w_region):
                if not r_var == w_var:
                    return False
        return True

    def is_broadcast(self) -> bool:
        """Whether the SBlock is elementwise, i.e. trivial mapping between read/write region"""

        if not self.is_injective() or len(self.write_bufs) != 1:
            return False

        w_region = self.write_bufs[0].buf_region.region
        for read_buf in self.read_bufs:
            r_region = read_buf.buf_region.region
            for r_var in r_region:
                if r_var not in w_region:
                    return False
        return True

    def get_loops(self) -> list[s_tir.schedule.LoopRV]:
        return [iter_info.loop_rv for iter_info in self.iters]

    def is_reduction(self) -> bool:
        """Whether the SBlock is a reduction workload."""
        return all(k == "S" or k == "R" for k in self.dom_kind()) and any(
            k == "R" for k in self.dom_kind()
        )

    def is_layout_transform(self) -> bool:
        """Whether the SBlock can be considered having a Layout Transform Pattern"""
        return (
            all(k == "S" for k in self.dom_kind())
            and len(self.write_bufs) == 1
            and len(self.read_bufs) == 1
            and not self.is_elementwise()
            and not get_global_func("s_tir.schedule.HasIfThenElse")(self.block_stmt)
        )

    def is_data_pad(self) -> bool:
        """Whether the SBlock can be considered having a data pad pattern"""
        return (
            all(k == "S" for k in self.dom_kind())
            and len(self.write_bufs) == 1
            and len(self.read_bufs) == 1
            and not self.is_elementwise()
            and len(self.write_bufs[0].buf_region.region)
            == len(self.read_bufs[0].buf_region.region)
            and get_global_func("s_tir.schedule.HasIfThenElse")(self.block_stmt)
        )

    def is_convolution(self) -> bool:
        """Whether a SBlock can be considered having Convolution Pattern"""
        raise NotImplementedError

    def is_pool(self) -> bool:
        """Whether a SBlock can be considered having Pooling Pattern"""
        raise NotImplementedError

    def is_gemv(self) -> bool:
        """Whether the SBlock is a GEMV workload."""
        raise NotImplementedError

    def is_gemm(self) -> bool:
        """Whether the SBlock is a GEMM workload."""
        if len(self.read_bufs) != 2 or len(self.write_bufs) != 1:
            return False

        def get_ewise_axes(region: list[Range]) -> set[Var]:
            axes: set[Var] = set()
            for r in region:
                if not (isinstance(r.extent, tir.IntImm) and r.extent.value == 1):
                    raise ValueError("Not a Elemwise Block Access")
                axes = axes.union(set(undefined_vars(r.min)))
            return axes

        try:
            C_iter_vars = get_ewise_axes(self.write_bufs[0].buf_region.region)
            A_iter_vars = get_ewise_axes(self.read_bufs[0].buf_region.region)
            B_iter_vars = get_ewise_axes(self.read_bufs[1].buf_region.region)
        except ValueError:
            return False

        conditions = [False for _ in range(3)]
        iter_vars = [block_iter.var for block_iter in self.iters]
        for iter_var in iter_vars:
            if iter_var in A_iter_vars and iter_var in B_iter_vars and iter_var in C_iter_vars:
                continue
            elif (
                iter_var in A_iter_vars and iter_var in B_iter_vars and iter_var not in C_iter_vars
            ):
                conditions[0] = True
            elif (
                iter_var in A_iter_vars and iter_var not in B_iter_vars and iter_var in C_iter_vars
            ):
                conditions[1] = True
            elif (
                iter_var not in A_iter_vars and iter_var in B_iter_vars and iter_var in C_iter_vars
            ):
                conditions[2] = True
            else:
                return False

        return all(conditions)

    def check_op_name(self, name: str):
        raise NotImplementedError

    def __str__(self) -> str:
        return f'SBlockInfo("{self.name}", "{self.dom_kind()}", {self.dom()})'

    def __repr__(self) -> str:
        return str(self)


_normalize_prim_func = get_global_func("s_tir.schedule.NormalizePrimFunc")


def normalize_prim_func(sch: s_tir.Schedule) -> list[SBlockInfo] | None:
    """Normalize the primfunc to normal form"""
    try:
        result = _normalize_prim_func(sch)
        if result is None:
            return None
    except Exception:  # pylint: disable=broad-except
        return None

    def _iter_kind(i: tir.IterVar) -> str:
        return {
            tir.IterVar.DataPar: "S",
            tir.IterVar.CommReduce: "R",
        }.get(i.iter_type, "O")

    blocks: list[SBlockInfo] = []
    for block, loops, iters, is_reduction in zip(*result):
        blocks.append(get_sblock_info(sch, block))
    return blocks


# BufferIndex Types
Index = namedtuple("Index", ["sub"])  # c
RemIndex = namedtuple("RemIndex", ["sub", "div"])  # c%len
DivIndex = namedtuple("DivIndex", ["sub", "div"])  # c//len
MergeIndex = namedtuple("MulIndex", ["dom", "mul", "sub"])  # co*len + cb
BufIndex = list[Index | RemIndex | DivIndex | MergeIndex | None]


def get_buffer_info(
    sch: s_tir.Schedule,
    blk: SBlockRV,
    buf: tir.BufferRegion,
    lps: dict[tir.Var, LoopRV],
) -> BufferInfo:
    return BufferInfo(sch, blk, buf, lps)


def get_sblock_info(sch: Schedule, block: SBlockRV) -> SBlockInfo:
    return SBlockInfo(sch, block)


def _assert_gpu_target(target: Target):
    if "gpu" not in target.keys:
        raise ValueError(f"Expect a GPU target, but got {target}")


def get_max_threads_per_block(target: Target) -> int:
    _assert_gpu_target(target)
    max_threads_per_block = None
    for name in ["max_threads_per_block", "max_num_threads"]:
        if max_threads_per_block is None:
            max_threads_per_block = target.attrs.get(name, None)
    if max_threads_per_block is None:
        max_threads_per_block = 64
    return int(max_threads_per_block)


def get_max_shared_memory_per_block(target: Target) -> int:
    _assert_gpu_target(target)
    max_shared_memory_per_block = target.attrs.get("max_shared_memory_per_block", None)
    if max_shared_memory_per_block is None:
        raise ValueError(
            f"Cannot find `max_shared_memory_per_block` in {target}, please specify it manually"
        )
    return int(max_shared_memory_per_block)


def get_root_block(sch: Schedule, func_name: str = "main") -> SBlockRV:
    try:
        block = sch.mod[func_name].body.block
    except Exception:
        raise ValueError(
            f"The function body is expected to be the root block, but got:\n"
            f"{sch.mod[func_name].body}"
        )
    return sch.get_sblock(block.name_hint)


def get_reduction_blocks(sch, blocks) -> bool:
    # Get the main computation block
    def is_reduction(block: SBlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {tir.IterVar.CommReduce, tir.IterVar.DataPar}

    def is_spatial(block: SBlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {tir.IterVar.DataPar}

    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all([is_reduction(block) or is_spatial(block) for block in blocks]):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction(block)]
    if len(reduction_blocks) != 1:
        return None

    return reduction_blocks


def get_in_out_dtypes(block: SBlock) -> tuple[list[str], list[str]]:
    """
    Retrieve In/Out dtypes of a given block
    """
    in_dtypes = list(str(buf.buffer.dtype) for buf in block.reads)
    out_dtypes = list(str(buf.buffer.dtype) for buf in block.writes)
    return (in_dtypes, out_dtypes)


def collect_block_iter_vars_used_in_access_region(
    block: tir.SBlock, region: list[ir.Range]
) -> set[tir.Var]:
    """Collect the block iter variables used in the access region of a buffer region."""
    tir_vars = set()
    for expr in region:
        assert expr.extent == 1
        tir_vars |= collect_vars_used_in_prim_expr(expr.min)
    tir_vars &= set(iter_var.var for iter_var in block.iter_vars)
    return tir_vars


def collect_vars_used_in_prim_expr(expr: tir.PrimExpr) -> set[tir.Var]:
    """Collect the variables used in the PrimExpr."""
    tir_vars = set()

    def _collect_tir_var(expr):
        if isinstance(expr, tir.Var):
            tir_vars.add(expr)

    tir.stmt_functor.post_order_visit(expr, _collect_tir_var)
    return tir_vars


def detect_dominant_read(block: tir.SBlock) -> tir.PrimExpr:
    """Detect the dominant read indices in the block."""
    dominant_read = None
    num_read_iters = -1
    for buffer_region in block.reads:
        tir_vars = collect_block_iter_vars_used_in_access_region(block, buffer_region.region)
        if num_read_iters < len(tir_vars):
            num_read_iters = len(tir_vars)
            dominant_read = buffer_region
    assert dominant_read is not None
    (result,) = dominant_read.buffer.offset_of([e.min for e in dominant_read.region])
    return result


def is_broadcast_epilogue(
    sch: s_tir.Schedule,
    block: s_tir.schedule.SBlockRV,
    epilogue: s_tir.schedule.SBlockRV,
) -> bool:
    """Check if the epilogue block is a broadcast pattern"""
    write_buffers = {r.buffer for r in sch.get(block).writes}
    epilogue_iters = {i.var: i for i in sch.get(epilogue).iter_vars if i.dom != 1}
    for buffer_region in sch.get(epilogue).reads:
        if buffer_region.buffer not in write_buffers:
            continue
        tir_vars = collect_block_iter_vars_used_in_access_region(
            sch.get(epilogue), buffer_region.region
        )
        if len(tir_vars) < len(epilogue_iters):
            return True
    return False
