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
# ruff: noqa: F401

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.ir.module import IRModule
from tvm.script import ir as I
from tvm.script import relax as R


def test_append_reshape_to_batchnorm():
    @I.ir_module
    class Input:
        @R.function
        def main(
            data: R.Tensor((1, 3, 224, 224), dtype="float32"),
            weight: R.Tensor((32, 3, 3, 3), dtype="float32"),
            gamma: R.Tensor((32,), dtype="float32"),
            beta: R.Tensor((32,), dtype="float32"),
            mean: R.Tensor((32,), dtype="float32"),
            variance: R.Tensor((32,), dtype="float32"),
        ) -> R.Tensor((1, 32, 224, 224), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 32, 224, 224), dtype="float32") = R.nn.conv2d(
                    data,
                    weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv1: R.Tuple(
                    R.Tensor((1, 32, 224, 224), dtype="float32"),
                    R.Tensor((32,), dtype="float32"),
                    R.Tensor((32,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv,
                    gamma,
                    beta,
                    mean,
                    variance,
                    axis=1,
                    epsilon=1.0000000000000001e-05,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                    training=False,
                )
                lv2: R.Tensor((1, 32, 224, 224), dtype="float32") = lv1[0]
                R.output(lv2)
            return lv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 3, 224, 224), dtype="float32"),
            weight: R.Tensor((32, 3, 3, 3), dtype="float32"),
            gamma: R.Tensor((32,), dtype="float32"),
            beta: R.Tensor((32,), dtype="float32"),
            mean: R.Tensor((32,), dtype="float32"),
            variance: R.Tensor((32,), dtype="float32"),
        ) -> R.Tensor((1, 32, 224, 224), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 32, 224, 224), dtype="float32") = R.nn.conv2d(
                    data,
                    weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv_1: R.Tuple(
                    R.Tensor((1, 32, 224, 224), dtype="float32"),
                    R.Tensor((32,), dtype="float32"),
                    R.Tensor((32,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv,
                    gamma,
                    beta,
                    mean,
                    variance,
                    axis=1,
                    epsilon=1.0000000000000001e-05,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                    training=False,
                )
                lv1: R.Tensor((1, 32, 224, 224), dtype="float32") = lv_1[0]
                lv2: R.Tensor((1, 32, 224, 224), dtype="float32") = R.reshape(
                    lv1, R.shape([1, 32, 224, 224])
                )
                R.output(lv2)
            return lv2

    result = tvm.relax.backend.adreno.transform.AppendReshapeToBatchnorm()(Input)

    tvm.ir.assert_structural_equal(result, Expected)


if __name__ == "__main__":
    tvm.testing.main()
