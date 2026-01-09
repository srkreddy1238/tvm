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
"""TfLite tests."""

import os
import time
import tempfile
from pathlib import Path
from packaging.version import parse

import numpy as np
import pytest
import tvm
from tvm import relax
from tvm.relax.frontend.tflite import from_tflite


def _get_tflite_model(tflite_model_path, inputs_dict):
    """Convert TFlite graph to Relax."""
    try:
        import tflite.Model
    except ImportError:
        pytest.skip("Missing Tflite support")

    with open(tflite_model_path, "rb") as f:
        tflite_model_buffer = f.read()

    try:
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buffer, 0)
    except AttributeError:
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buffer, 0)
    shape_dict = {}
    dtype_dict = {}
    for input in inputs_dict:
        input_shape, input_dtype = inputs_dict[input]
        shape_dict[input] = input_shape
        dtype_dict[input] = input_dtype

    return from_tflite(tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict)


def test_squeezenet():
    def get_model():
        inputs = {"Placeholder": ((1, 224, 224, 3), "float32")}
        model_path = os.getenv("CI_TEST_INVENTORY", "/local/mnt/workspace/CI/TVM/Inventory")
        mod = _get_tflite_model(model_path + "/squeezenet.tflite", inputs_dict=inputs)
        return mod

    mod = get_model()
    print(mod)
    tgt = tvm.target.Target("llvm")
    ex = tvm.compile(mod, tgt)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    inputs = []
    for arg in mod["main"].params:
        shape = tuple(shape_val.value for shape_val in arg.struct_info.shape.values)
        inputs.append(np.random.uniform(0, 1, size=shape).astype(arg.struct_info.dtype))

    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    print(tvm_output.shape)


if __name__ == "__main__":
    test_squeezenet()
