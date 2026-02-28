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

import pytest
from utils import requires_adreno_vulkan, verify_results

import tvm.testing
from tvm.script import ir as I
from tvm.script import relax as R

# TODO: Add codegen tests to verify extn usage.

TARGET_SUPPORTS_EXTENSION = os.getenv("ADRENO_TARGET_COOP", "").strip().lower() == "yes"
target = tvm.target.Target(
    {
        "kind": "vulkan",
        "supports_float16": 1,
        "supports_16bit_buffer": 1,
        "supports_khr_cooperative_matrix": 1,
    }
)
ref_target = tvm.target.Target("llvm")


@requires_adreno_vulkan
@pytest.mark.skipif(not TARGET_SUPPORTS_EXTENSION, reason="Device not supported.")
def test_non_batch_fp16():
    @I.ir_module
    class Matmul:
        @R.function
        def main(
            input: R.Tensor((64, 16), "float16"), weight: R.Tensor((16, 64), "float16")
        ) -> R.Tensor((64, 64), "float16"):
            with R.dataflow():
                gv = R.matmul(input, weight, "float16")
                R.output(gv)
            return gv

    verify_results(Matmul, target, ref_target)


@requires_adreno_vulkan
@pytest.mark.skipif(not TARGET_SUPPORTS_EXTENSION, reason="Device not supported.")
def test_non_batch_fp32():
    @I.ir_module
    class Matmul:
        @R.function
        def main(
            input: R.Tensor((64, 8), "float32"), weight: R.Tensor((8, 64), "float32")
        ) -> R.Tensor((64, 64), "float32"):
            with R.dataflow():
                gv = R.matmul(input, weight, "float32")
                R.output(gv)
            return gv

    verify_results(Matmul, target, ref_target)


@requires_adreno_vulkan
@pytest.mark.skipif(not TARGET_SUPPORTS_EXTENSION, reason="Device not supported.")
def test_batch_fp16():
    @I.ir_module
    class Matmul:
        @R.function
        def main(
            input: R.Tensor((16, 64, 16), "float16"), weight: R.Tensor((16, 16, 64), "float16")
        ) -> R.Tensor((16, 64, 64), "float16"):
            with R.dataflow():
                gv = R.matmul(input, weight, "float16")
                R.output(gv)
            return gv

    verify_results(Matmul, target, ref_target)


@requires_adreno_vulkan
@pytest.mark.skipif(not TARGET_SUPPORTS_EXTENSION, reason="Device not supported.")
def test_batch_fp32():
    @I.ir_module
    class Matmul:
        @R.function
        def main(
            input: R.Tensor((16, 64, 8), "float32"), weight: R.Tensor((16, 8, 64), "float32")
        ) -> R.Tensor((16, 64, 64), "float32"):
            with R.dataflow():
                gv = R.matmul(input, weight, "float32")
                R.output(gv)
            return gv

    verify_results(Matmul, target, ref_target)


@requires_adreno_vulkan
@pytest.mark.skipif(not TARGET_SUPPORTS_EXTENSION, reason="Device not supported.")
def test_invalid_shape():
    @I.ir_module
    class Matmul:
        @R.function
        def main(
            input: R.Tensor((65, 20), "float16"), weight: R.Tensor((20, 65), "float16")
        ) -> R.Tensor((65, 65), "float16"):
            with R.dataflow():
                gv = R.matmul(input, weight, "float16")
                R.output(gv)
            return gv

    verify_results(Matmul, target, ref_target)


@requires_adreno_vulkan
@pytest.mark.skipif(not TARGET_SUPPORTS_EXTENSION, reason="Device not supported.")
def test_invalid_dtype():
    @I.ir_module
    class Matmul:
        @R.function
        def main(
            input: R.Tensor((32, 16), "float32"), weight: R.Tensor((16, 32), "float32")
        ) -> R.Tensor((32, 32), "float32"):
            with R.dataflow():
                gv = R.matmul(input, weight, "float32")
                R.output(gv)
            return gv

    verify_results(Matmul, target, ref_target)


if __name__ == "__main__":
    tvm.testing.main()
