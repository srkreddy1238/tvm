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
"""CLML integration operator tests."""

import tvm
import numpy as np
from tvm import relay
from tvm.relay.op.contrib import clml
from tvm.relay import testing
from tvm.ir import IRModule
from tvm.contrib import utils
from test_clml.infrastructure import (
    build_and_run,
    build_and_run_vm,
    verify_codegen,
    verify_clml_op_count,
)
import pytest
import os


executor_type = tvm.testing.parameter("ge", "vm")


def _build_and_run_network(remote, mod, params, input_data, target, executor_type, tvm_log=""):
    """Helper function to build and run a network."""

    outputs = []
    for clml in [True, False]:
        if executor_type == "ge":
            outputs.append(
                build_and_run(
                    remote,
                    mod,
                    params,
                    input_data,
                    target,
                    enable_clml=clml,
                    stat_file=tvm_log,
                )
            )
        else:
            outputs.append(
                build_and_run_vm(
                    remote,
                    mod,
                    params,
                    input_data,
                    target,
                    enable_clml=clml,
                    stat_file=tvm_log,
                )
            )
    return outputs


# Multiheadattention pattern with q, k, v & out_bias
def _get_mod_nomask_nobias(qshape, kshape, vshape, attn_heads_dim):
    np.random.seed(0)
    dtype = "float16"
    num_heads = qshape[2] // attn_heads_dim
    q = relay.var("q", shape=qshape[1:], dtype="float16")  # query
    k = relay.var("k", shape=kshape[1:], dtype="float16")  # key
    v = relay.var("v", shape=vshape[1:], dtype="float16")  # value

    query_data = np.random.uniform(-0.1, 0.1, qshape[1:]).astype(np.float16)
    key_data = np.random.uniform(-0.1, 0.1, kshape[1:]).astype(np.float16)
    value_data = np.random.uniform(-0.1, 0.1, vshape[1:]).astype(np.float16)

    q_wt_data = np.random.uniform(-0.1, 0.1, (num_heads * attn_heads_dim, qshape[2])).astype(
        np.float16
    )
    q_wt = relay.const(q_wt_data, dtype=dtype)
    k_wt_data = np.random.uniform(-0.1, 0.1, (num_heads * attn_heads_dim, kshape[2])).astype(
        np.float16
    )
    k_wt = relay.const(k_wt_data, dtype=dtype)
    v_wt_data = np.random.uniform(
        -0.1,
        0.1,
        (
            num_heads * attn_heads_dim,
            vshape[2],
        ),
    ).astype(np.float16)
    v_wt = relay.const(v_wt_data, dtype=dtype)
    out_wt_data = np.random.uniform(-0.1, 0.1, (num_heads * attn_heads_dim, qshape[2])).astype(
        np.float16
    )
    out_wt = relay.const(out_wt_data, dtype=dtype)
    out_bias_data = np.random.uniform(-0.1, 0.1, (qshape[2],)).astype(np.float16)
    out_bias = relay.const(out_bias_data, dtype=dtype)

    const1_value = np.random.uniform(-0.1, 0.1, ((1))).astype("float16")
    const1 = relay.const(const1_value, dtype=dtype)
    scalar_const = relay.const(1.0, dtype="float16")

    op1 = relay.nn.dense(q, q_wt, units=None, out_dtype="float16")
    op2 = relay.reshape(op1, newshape=[1, qshape[1], num_heads * attn_heads_dim])
    op3 = relay.reshape(op2, newshape=[1, -1, num_heads, attn_heads_dim])
    op4 = relay.transpose(op3, axes=[0, 2, 1, 3])
    op5 = relay.multiply(op4, const1)
    op6 = relay.nn.dense(k, k_wt, units=None, out_dtype="float16")
    op7 = relay.reshape(op6, newshape=[1, kshape[1], num_heads * attn_heads_dim])
    op8 = relay.reshape(op7, newshape=[1, -1, num_heads, attn_heads_dim])
    op9 = relay.transpose(op8, axes=[0, 2, 3, 1])
    op10 = relay.multiply(op9, const1)
    op11 = relay.reshape(op10, newshape=[-1, attn_heads_dim, kshape[1]])
    op12 = relay.reshape(op5, newshape=[-1, qshape[1], attn_heads_dim])
    op13 = relay.transpose(op11, axes=[0, 2, 1])
    op14 = relay.nn.batch_matmul(op12, op13, out_dtype="float16", transpose_b=True)
    op15 = relay.reshape(op14, newshape=[1, num_heads, qshape[1], kshape[1]])
    op16 = relay.cast(op15, dtype="float32")
    op17 = relay.nn.softmax(op16, axis=3)
    op18 = relay.reshape(op17, newshape=[-1, qshape[1], kshape[1]])
    op19 = relay.nn.dense(v, v_wt)
    op20 = relay.reshape(op19, newshape=[1, vshape[1], num_heads * attn_heads_dim])
    op21 = relay.reshape(op20, newshape=[1, -1, num_heads, attn_heads_dim])
    op22 = relay.transpose(op21, axes=[0, 2, 1, 3])
    op23 = relay.reshape(op22, newshape=[-1, vshape[1], attn_heads_dim])
    op24 = relay.cast(op18, dtype="float16")
    op25 = relay.transpose(op23, axes=[0, 2, 1])
    op26 = relay.nn.batch_matmul(op24, op25, out_dtype="float16", transpose_b=True)
    op27 = relay.reshape(op26, newshape=[1, num_heads, qshape[1], attn_heads_dim])
    op28 = relay.transpose(op27, axes=[0, 2, 1, 3])
    op29 = relay.reshape(op28, newshape=[1, -1, num_heads * attn_heads_dim])
    op30 = relay.cast(op29, dtype="float32")
    op31 = relay.reshape(op30, newshape=[-1, num_heads * attn_heads_dim])
    op32 = relay.cast(op31, dtype="float16")
    op33 = relay.nn.dense(op32, out_wt)
    op34 = relay.reshape(op33, newshape=qshape)
    op35 = relay.add(out_bias, op34)
    op36 = relay.divide(op35, scalar_const)

    input_data = {
        "q": tvm.nd.array(query_data),
        "k": tvm.nd.array(key_data),
        "v": tvm.nd.array(value_data),
    }
    params = {}
    return op36, params, input_data


# Multiheadattention pattern with q, k, v, & all bias
def _get_mod_nomask_bias(qshape, kshape, vshape):

    np.random.seed(0)
    dtype = "float16"

    q = relay.var("q", shape=qshape[1:], dtype="float16")  # query
    k = relay.var("k", shape=kshape[1:], dtype="float16")  # key
    v = relay.var("v", shape=vshape[1:], dtype="float16")  # value

    q_wt_data = np.random.uniform(-0.1, 0.1, ((qshape[2], qshape[2]))).astype("float16")
    q_wt = relay.const(q_wt_data, dtype=dtype)
    k_wt_data = np.random.uniform(-0.1, 0.1, ((kshape[2], kshape[2]))).astype("float16")
    k_wt = relay.const(k_wt_data, dtype=dtype)
    v_wt_data = np.random.uniform(-0.1, 0.1, ((vshape[2], vshape[2]))).astype("float16")
    v_wt = relay.const(v_wt_data, dtype=dtype)
    out_wt_data = np.random.uniform(-0.1, 0.1, ((qshape[2], qshape[2]))).astype("float16")
    out_wt = relay.const(out_wt_data, dtype=dtype)

    out_bias_data = np.random.uniform(-0.1, 0.1, (qshape[2],)).astype("float16")
    out_bias = relay.const(out_bias_data, dtype=dtype)
    q_bias_data = np.random.uniform(-0.1, 0.1, (qshape[2],)).astype("float16")
    q_bias = relay.const(q_bias_data, dtype=dtype)
    k_bias_data = np.random.uniform(-0.1, 0.1, (kshape[2],)).astype("float16")
    k_bias = relay.const(k_bias_data, dtype=dtype)
    v_bias_data = np.random.uniform(-0.1, 0.1, (vshape[2],)).astype("float16")
    v_bias = relay.const(v_bias_data, dtype=dtype)

    const_value = np.random.uniform(-0.1, 0.1, ((1))).astype("float16")
    const1 = relay.const(const_value, dtype=dtype)
    scalar_const = relay.const(1, dtype="float16")

    op1 = relay.nn.dense(q, q_wt, units=None, out_dtype="float16")
    op2 = relay.reshape(op1, newshape=[1, 4096, 512])
    op3 = relay.add(q_bias, op2)
    op4 = relay.reshape(op3, newshape=[1, -1, 1, 512])
    op5 = relay.transpose(op4, axes=[0, 2, 1, 3])
    op6 = relay.multiply(op5, const1)
    op7 = relay.nn.dense(k, k_wt, units=None, out_dtype="float16")
    op8 = relay.reshape(op7, newshape=[1, 4096, 512])
    op9 = relay.add(k_bias, op8)
    op10 = relay.reshape(op9, newshape=[1, -1, 1, 512])
    op11 = relay.transpose(op10, axes=[0, 2, 3, 1])
    op12 = relay.multiply(op11, const1)
    op13 = relay.reshape(op12, newshape=[-1, 512, 4096])
    op14 = relay.reshape(op6, newshape=[-1, 4096, 512])
    op15 = relay.transpose(op13, axes=[0, 2, 1])
    op16 = relay.nn.batch_matmul(op14, op15, out_dtype="float16", transpose_b=True)
    op17 = relay.reshape(op16, newshape=[1, 1, 4096, 4096])
    op18 = relay.cast(op17, dtype="float32")
    op19 = relay.nn.softmax(op18, axis=3)
    op20 = relay.reshape(op19, newshape=[-1, 4096, 4096])
    op21 = relay.nn.dense(v, v_wt, units=None, out_dtype="float16")
    op22 = relay.reshape(op21, newshape=[1, 4096, 512])
    op23 = relay.add(v_bias, op22)
    op24 = relay.reshape(op23, newshape=[1, -1, 1, 512])
    op25 = relay.transpose(op24, axes=[0, 2, 1, 3])
    op26 = relay.reshape(op25, newshape=[-1, 4096, 512])
    op27 = relay.cast(op20, dtype="float16")
    op28 = relay.transpose(op26, axes=[0, 2, 1])
    op29 = relay.nn.batch_matmul(op27, op28, out_dtype="float16", transpose_b=True)
    op30 = relay.reshape(op29, newshape=[1, 1, 4096, 512])
    op31 = relay.transpose(op30, axes=[0, 2, 1, 3])
    op32 = relay.reshape(op31, newshape=[1, -1, 512])
    op33 = relay.cast(op32, dtype="float32")
    op34 = relay.reshape(op33, newshape=[-1, 512])
    op35 = relay.cast(op34, dtype="float16")
    op36 = relay.nn.dense(op35, out_wt, units=None, out_dtype="float16")
    op37 = relay.reshape(op36, newshape=[1, 4096, 512])
    op38 = relay.add(out_bias, op37)
    op39 = relay.divide(op38, scalar_const)

    query_data = np.random.uniform(-0.1, 0.1, (4096, 512)).astype(np.float16)
    key_data = np.random.uniform(-0.1, 0.1, (4096, 512)).astype(np.float16)
    value_data = np.random.uniform(-0.1, 0.1, (4096, 512)).astype(np.float16)

    input_data = {
        "q": tvm.nd.array(query_data),
        "k": tvm.nd.array(key_data),
        "v": tvm.nd.array(value_data),
    }
    params = {}
    return op39, params, input_data


# Multiheadattention pattern with q,k,v, out bias and  attn mask
def _get_mod_mask_bias(qshape, kshape, vshape):
    np.random.seed(0)
    dtype = "float16"
    q = relay.var("q", shape=qshape[1:], dtype="float16")  # query
    k = relay.var("k", shape=kshape[1:], dtype="float16")  # key
    v = relay.var("v", shape=vshape[1:], dtype="float16")  # value

    const1 = relay.const(0.125, dtype="float16")
    scalar_const = relay.const(1.0, dtype="float16")

    mask_data = np.triu(np.ones((77, 77)), k=1).astype("float16")
    mask_data = np.expand_dims(np.expand_dims(mask_data, 0), 0)
    mask_data = np.ones((1, 1, 77, 77), dtype=np.float16)

    attn_mask = relay.const(mask_data, dtype=dtype)

    out_bias_data = np.random.uniform(-0.2, 0.2, (qshape[2])).astype("float16")
    out_bias = relay.const(out_bias_data)

    q_bias_data = np.random.uniform(-0.2, 0.2, (qshape[2])).astype("float16")
    q_bias = relay.const(q_bias_data, dtype=dtype)
    k_bias_data = np.random.uniform(-0.2, 0.2, (kshape[2])).astype("float16")
    k_bias = relay.const(k_bias_data, dtype=dtype)
    v_bias_data = np.random.uniform(-0.2, 0.2, (vshape[2])).astype("float16")
    v_bias = relay.const(v_bias_data, dtype=dtype)

    q_wt_data = np.random.uniform(-0.2, 0.2, ((qshape[2], qshape[2]))).astype("float16")
    q_wt = relay.const(q_wt_data, dtype=dtype)
    k_wt_data = np.random.uniform(-0.2, 0.2, ((kshape[2], kshape[2]))).astype("float16")
    k_wt = relay.const(k_wt_data, dtype=dtype)
    v_wt_data = np.random.uniform(-0.2, 0.2, ((vshape[2], vshape[2]))).astype("float16")
    v_wt = relay.const(v_wt_data, dtype=dtype)
    out_wt_data = np.random.uniform(-0.2, 0.2, ((qshape[2], qshape[2]))).astype("float16")
    out_wt = relay.const(out_wt_data, dtype=dtype)

    op1 = relay.nn.dense(q, q_wt, units=None, out_dtype="float16")
    op2 = relay.reshape(op1, newshape=[1, 77, 1024])
    op3 = relay.add(q_bias, op2)
    op4 = relay.multiply(op3, const1)
    op5 = relay.reshape(op4, newshape=[1, 77, 16, 64])
    op6 = relay.transpose(op5, axes=[0, 2, 1, 3])
    op7 = relay.nn.dense(k, k_wt, units=None, out_dtype="float16")
    op8 = relay.reshape(op7, newshape=[1, 77, 1024])
    op9 = relay.add(k_bias, op8)
    op10 = relay.reshape(op9, newshape=[1, -1, 16, 64])
    op11 = relay.transpose(op10, axes=[0, 2, 1, 3])
    op12 = relay.reshape(op11, newshape=[16, -1, 64])
    op13 = relay.transpose(op12, axes=[0, 2, 1])
    op14 = relay.reshape(op6, newshape=[16, -1, 64])
    op15 = relay.transpose(op13, axes=[0, 2, 1])
    op16 = relay.nn.batch_matmul(op14, op15, out_dtype="float16", transpose_b=True)
    op17 = relay.reshape(op16, newshape=[16, 77, 77])
    op18 = relay.reshape(op17, newshape=[1, 16, 77, 77])
    op19 = relay.add(op18, attn_mask)
    op20 = relay.reshape(op19, newshape=[16, 77, 77])
    op21 = relay.cast(op20, dtype="float32")
    op22 = relay.nn.softmax(op21, axis=2)
    op23 = relay.nn.dense(v, v_wt, units=None, out_dtype="float16")
    op24 = relay.reshape(op23, newshape=[1, 77, 1024])
    op25 = relay.add(v_bias, op24)
    op26 = relay.reshape(op25, newshape=[1, -1, 16, 64])
    op27 = relay.transpose(op26, axes=[0, 2, 1, 3])
    op28 = relay.reshape(op27, newshape=[16, -1, 64])
    op29 = relay.cast(op22, dtype="float16")
    op30 = relay.transpose(op28, axes=[0, 2, 1])
    op31 = relay.nn.batch_matmul(op29, op30, out_dtype="float16", transpose_b=True)
    op32 = relay.reshape(op31, newshape=[16, 77, 64])
    op33 = relay.reshape(op32, newshape=[1, 16, 77, 64])
    op34 = relay.transpose(op33, axes=[0, 2, 1, 3])
    op35 = relay.reshape(op34, newshape=[1, 77, 1024])
    op36 = relay.reshape(op35, newshape=[-1, 1024])
    op37 = relay.nn.dense(op36, out_wt, units=None, out_dtype="float16")
    op38 = relay.reshape(op37, newshape=[1, 77, 1024])
    op39 = relay.add(out_bias, op38)
    op40 = relay.divide(op39, scalar_const)

    query_data = np.random.uniform(-0.2, 0.2, (77, 1024)).astype(np.float16)
    key_data = np.random.uniform(-0.2, 0.2, (77, 1024)).astype(np.float16)
    value_data = np.random.uniform(-0.2, 0.2, (77, 1024)).astype(np.float16)

    input_data = {
        "q": tvm.nd.array(query_data),
        "k": tvm.nd.array(key_data),
        "v": tvm.nd.array(value_data),
    }
    params = {}
    return op40, params, input_data


@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.skipif(
    int(os.getenv("ADRENO_TARGET_CLML_VERSION", 3)) < 4,
    reason="Requires target device with CLML v4 or above",
)
@pytest.mark.skipif(
    int(tvm.support.libinfo().get("TVM_CLML_VERSION", 2)) < 4,
    reason="Requires compiler supporting CLML v4 or above",
)
@pytest.mark.parametrize(
    "trials",
    [
        # query,key,value, has_bias, has_attnmask, attn_heads_dim
        [(1, 4096, 320), (1, 77, 1024), (1, 77, 1024), False, False, 64],
        [(1, 4096, 320), (1, 4096, 320), (1, 4096, 320), False, False, 64],
        [(1, 1024, 640), (1, 77, 1024), (1, 77, 1024), False, False, 64],
        [(1, 256, 1280), (1, 77, 1024), (1, 77, 1024), False, False, 64],
        [(1, 64, 1280), (1, 77, 1024), (1, 77, 1024), False, False, 64],
        [(1, 64, 1280), (1, 64, 1280), (1, 64, 1280), False, False, 64],
        [(1, 4096, 512), (1, 4096, 512), (1, 4096, 512), True, False, None],
        [(1, 77, 1024), (1, 77, 1024), (1, 77, 1024), True, True, None],
    ],
)
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_mha_general(remote, dtype, target, executor_type, trials):
    def _verify(qshape, kshape, vshape, has_bias, has_attn_mask, attn_heads_dim=1):
        if (not has_bias) and (not has_attn_mask):
            mod, params, inputs = _get_mod_nomask_nobias(qshape, kshape, vshape, attn_heads_dim)
        elif has_bias and (not has_attn_mask):
            mod, params, inputs = _get_mod_nomask_bias(qshape, kshape, vshape)
        elif has_bias and has_attn_mask:
            mod, params, inputs = _get_mod_mask_bias(qshape, kshape, vshape)
        else:
            pass

        mod = tvm.IRModule.from_expr(mod)
        outputs = _build_and_run_network(remote, mod, params, inputs, target, executor_type, "")

        out_tol = 1e-1

        tvm.testing.assert_allclose(
            outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
        )
        verify_clml_op_count(remote, mod, params, target)

    _verify(trials[0], trials[1], trials[2], trials[3], trials[4], trials[5])


if __name__ == "__main__":

    tvm.testing.main()
