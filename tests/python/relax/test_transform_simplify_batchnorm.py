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

import tvm
import tvm.testing
from tvm import relax
from tvm.relax import Expr
from tvm.relax.dpl import is_op, is_const
from tvm.ir.transform import PassContext
from tvm.ir.module import IRModule
from tvm.relax.transform import function_pass
from tvm.script import relax as R
import numpy as np
from tvm import dlight as dl


def build_and_run(mod, inputs_np, params_np={}, simplify=False):
    tgt = tvm.target.Target("llvm")

    with tgt:
        mod = tvm.tir.transform.BindTarget(tvm.target.Target.current(allow_none=False))(mod)
        if simplify:
            mod = tvm.relax.transform.SimplifyBatchnorm()(mod)
            mod = tvm.relax.transform.FoldConstant()(mod)
        
        print("Mod:", mod)
        seq = tvm.transform.Sequential(
            [
                tvm.relax.transform.LegalizeOps(),
                tvm.relax.transform.AnnotateTIROpPattern(),
                tvm.relax.transform.FoldConstant(),
                tvm.relax.transform.FuseOps(),
                tvm.relax.transform.FuseTIR(),
                tvm.relax.transform.DeadCodeElimination(),
                dl.ApplyDefaultSchedule(
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                ),
                tvm.relax.transform.RewriteDataflowReshape(),
                tvm.relax.transform.ToNonDataflow(),
                tvm.relax.transform.RemovePurityChecking(),
                tvm.relax.transform.CallTIRRewrite(),
                tvm.relax.transform.StaticPlanBlockMemory(),
                tvm.relax.transform.RewriteCUDAGraph(),
                tvm.relax.transform.LowerAllocTensor(),
                tvm.relax.transform.KillAfterLastUse(),
                tvm.relax.transform.VMBuiltinLower(),
                tvm.relax.transform.VMShapeLower(),
                tvm.relax.transform.AttachGlobalSymbol(),
            ]
        )
        mod = seq(mod)

    ex = relax.build(mod, "llvm")
    dev = tvm.device("llvm", 0)
    vm = relax.VirtualMachine(ex, dev)

    params_dev = []
    for k, v in params_np.items():
        params_dev.append(tvm.nd.array(v, dev))

    f = vm["main"]
    inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]
    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    return tvm_output.numpy()

def _check_inference(mod, inputs, params):
    out = build_and_run(mod, inputs, params, True)
    ref = build_and_run(mod, inputs, params, False)
    tvm.testing.assert_allclose(ref, out, rtol=1e-5, atol=1e-5)

def test_simplify():
    inp = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    gamma = relax.Constant(tvm.nd.array(np.random.uniform(low=-1, high=1, size=(3)).astype("float32")))
    beta = relax.Constant(tvm.nd.array(np.random.uniform(low=-1, high=1, size=(3)).astype("float32")))
    mean = relax.Constant(tvm.nd.array(np.random.uniform(low=-1, high=1, size=(3)).astype("float32")))
    var = relax.Constant(tvm.nd.array(np.random.uniform(low=-1, high=1, size=(3)).astype("float32")))

    bb = relax.BlockBuilder()
    with bb.function("main", [inp]):
        gv = bb.emit(relax.op.nn.batch_norm(inp, gamma, beta, mean, var, axis=1)[0])
        bb.emit_func_output(gv)

    mod = bb.get()

    inputs = []
    inputs.append(np.random.randint(-1, 1, size=(2, 3, 28, 28)).astype("float32"))

    _check_inference(mod, inputs, {})

if __name__ == "__main__":
    tvm.testing.main()
