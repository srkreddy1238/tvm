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
"""The Relax CPU backend compilation pipeline and other passes."""

import os

import tvm
from tvm import relax

from . import _ffi_api


def library_dispatch_passes(target: tvm.target.Target):  # pylint: disable=unused-argument
    """The default library dispatch passes for CPU backend."""
    if os.getenv("CPP_COMPILER_CI", "OFF") == "ON":
        return [_ffi_api.PipelineLibrary(target)]
    else:
        return []


def legalize_passes(target: tvm.target.Target):  # pylint: disable=unused-argument
    """The default legalization passes for CPU backend."""
    if os.getenv("CPP_COMPILER_CI", "OFF") == "ON":
        return [_ffi_api.PipelineLegalize(target)]
    else:
        return [
            tvm.relax.transform.LegalizeOps(),
            tvm.relax.transform.AnnotateTIROpPattern(),
            tvm.relax.transform.FoldConstant(),
            tvm.relax.transform.FuseOps(),
            tvm.relax.transform.FuseTIR(),
        ]


def dataflow_lower_passes(target: tvm.target.Target):  # pylint: disable=unused-argument
    """The default dataflow lowering passes for CPU backend."""
    if os.getenv("CPP_COMPILER_CI", "OFF") == "ON":
        return [_ffi_api.PipelineDataflow(target)]
    else:
        return [
            relax.transform.RewriteDataflowReshape(),
            relax.transform.ToNonDataflow(),
            relax.transform.RemovePurityChecking(),
            relax.transform.CallTIRRewrite(),
        ]


def finalize_passes(target: tvm.target.Target):  # pylint: disable=unused-argument
    """The default finalization passes for CPU backend."""
    if os.getenv("CPP_COMPILER_CI", "OFF") == "ON":
        return [_ffi_api.PipelineFinalize(target)]
    else:
        return [
            relax.transform.StaticPlanBlockMemory(),
            relax.transform.LowerAllocTensor(),
            relax.transform.KillAfterLastUse(),
            relax.transform.LowerRuntimeBuiltin(),
            relax.transform.ComputePrimValue(),
            relax.transform.VMShapeLower(),
            relax.transform.AttachGlobalSymbol(),
        ]


def get_default_pipeline(target: tvm.target.Target):
    """Return the default compilation pipeline for CPU."""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext):
        with target:
            seq = tvm.transform.Sequential(
                library_dispatch_passes(target)
                + legalize_passes(target)
                + dataflow_lower_passes(target)
                + finalize_passes(target)
            )
            mod = seq(mod)
        return mod

    return _pipeline
