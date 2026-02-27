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

import os
import copy
import pytest
import tempfile
import numpy as np

import tvm
import tvm.testing

from tvm import (
    relax,
    IRModule,
)
from tvm.relax.transform.legalize_ops import adreno as legalize_adreno
from tvm.script import ir as I, tir as T, relax as R
from tvm.target import Target
from tvm.contrib import ndk
from tvm import DataType, s_tir
from tvm.rpc import connect_tracker

from utils import build_and_run, requires_adreno_opencl_real


@pytest.mark.skip("TODO: Disabled after rebase")
@requires_adreno_opencl_real
@pytest.mark.parametrize("target", [tvm.target.Target("qcom/adreno-opencl-texture")])
@pytest.mark.parametrize("dtype", ["int8", "float16", "int16", "float32", "int32"])
@pytest.mark.parametrize("channel_size", [64, 128])
@pytest.mark.parametrize("read_width", [1, 2, 4, 8, 16])
def test_texture_copy(target, dtype, channel_size, read_width):
    M, N, K = (256, 1024, 128)
    lanes = channel_size // DataType(dtype).bits
    if read_width > lanes:
        return

    @I.ir_module
    class TextureCopy:
        @T.prim_func
        def copy(Am: T.handle, Bm: T.handle):
            A = T.match_buffer(Am, (M, N), dtype=dtype)
            B = T.match_buffer(Bm, (M, N), dtype=dtype)
            T.func_attr({"global_symbol": "copy"})
            for li, lj in T.grid(M, N):
                with T.sblock("Copy"):
                    i, j = T.axis.remap("SS", [li, lj])
                    B[i, j] = A[i, j]

        @R.function
        def main(input: R.Tensor((M, N), dtype=dtype)) -> R.Tensor((M, N), dtype=dtype):
            cls = TextureCopy
            with R.dataflow():
                gv = R.call_tir(cls.copy, (input,), out_sinfo=R.Tensor((M, N), dtype=dtype))
                R.output(gv)
            return gv

    def schedule_texture_read(sch: s_tir.Schedule):
        sch.work_on("copy")
        B_blk = sch.get_sblock("Copy")
        Ai_block = sch.cache_read(B_blk, 0, "global.texture")
        sch.transform_layout(Ai_block, ("write", 0), lambda i, j: (i, j // lanes, j % lanes))

        def schedule_default(blk, lanes):
            i, j = sch.get_loops(blk)
            jo, jv = sch.split(j, [None, lanes])

            b = sch.fuse(i, jo)
            bx, tx = sch.split(b, [None, 256])
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")

            sch.vectorize(jv)

        schedule_default(Ai_block, lanes)
        schedule_default(B_blk, read_width)

    tex_mod = copy.deepcopy(TextureCopy)
    sch = s_tir.Schedule(tex_mod)
    schedule_texture_read(sch)
    tex_mod = sch.mod

    inputs_np = [np.random.uniform(-16, 16, size=(M, N)).astype(dtype)]
    outputs_np = build_and_run(tex_mod, inputs_np, target)

    assert len(inputs_np) == 1 and len(outputs_np) == 1
    np.testing.assert_equal(inputs_np[0], outputs_np[0])


if __name__ == "__main__":
    tvm.testing.main()
