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
# pylint: disable=invalid-name, unused-argument, pointless-exception-statement
"""Pattern table for CLML backend"""

import tvm
from tvm import IRModule
from tvm.ir.transform import PassContext, module_pass


def clml_sdk_version():
    """Utility function to get clml version"""

    return int(tvm.support.libinfo().get("TVM_CLML_VERSION", 2))


def is_clml_runtime_enabled():
    """Check if the CLML graph runtime is present.

    Returns
    -------
    ret: bool
        True if present, False if not.
    """
    check_enabled = tvm.get_global_func("relax.op.is_openclml_runtime_enabled", True)
    if check_enabled:
        return check_enabled()
    return False


@module_pass(opt_level=0, name="OpenCLMLOffLoad")
class OpenCLMLOffLoad:
    """The pass sequence used for CLML offload"""

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        """The transform"""

        return tvm.relax.backend.adreno.transform.OpenCLMLOffLoad()(mod)


@tvm.transform.module_pass(opt_level=0, name="OpenCLMLOffLoadForLLM")
class OpenCLMLOffLoadForLLM:
    """A compiler pass that partition the graph with dequant Matmul to CLML backend offload."""

    def __init__(self, target: tvm.target.Target) -> None:
        """Initializer.
        Parameters
        ----------
        target : tvm.target.Target
            Target device.
        """
        self.target = target

    def transform_module(
        self,
        mod: IRModule,
        _ctx: tvm.transform.PassContext,
    ) -> IRModule:
        """Apply required passed to transform"""

        if clml_sdk_version() >= 5:
            mod = tvm.relax.backend.adreno.transform.OpenCLMLOffLoadForLLM()(mod)
        return mod
