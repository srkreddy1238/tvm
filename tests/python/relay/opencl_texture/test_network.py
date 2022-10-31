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

import re
import tvm
import numpy as np
from tvm import relay
from tvm.relay import testing
import tvm.relay.testing.tf as tf_testing
from tvm.contrib import utils
from utils.adreno_utils import gpu_preprocess, build_run_compare, get_model
import pytest

from tvm.relay.op import register_mixed_precision_conversion

dtype = tvm.testing.parameter("float32", "float16")
#dtype = tvm.testing.parameter("float32")

def convert_to_fp16(mod, dtype):
    from tvm.ir import IRModule
    mod = IRModule.from_expr(mod)
    seq = tvm.transform.Sequential(
        [
            relay.transform.InferType(),
            relay.transform.ToMixedPrecision()
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
        return mod

@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_mobilenet_v1(target, dtype):
    mod, params, inputs, dtypes = get_model(
        "https://github.com/mlcommons/mobile_models/raw/main/v0_7/tflite/mobilenet_edgetpu_224_1.0_float.tflite",
        "mobilenet_edgetpu_224_1.0_float.tflite", "tflite")
    if dtype == "float16":
        mod = convert_to_fp16(mod["main"], dtype)
    build_run_compare(mod, params, inputs, dtypes, target, [], model_name="mobilenet", model_dtype=dtype)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl")
def test_mobilebert(target, dtype):
    mod, params, inputs, dtypes = get_model(
        "https://github.com/mlcommons/mobile_models/raw/main/v0_7/tflite/mobilebert_float_384_gpu.tflite",
        "mobilebert_float_384_gpu.tflite", "tflite")
    if dtype == "float16":
        mod = convert_to_fp16(mod["main"], dtype)
    build_run_compare(mod, params, inputs, dtypes, target, [], model_name="mobilebert", model_dtype=dtype)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_deeplab_v3(target, dtype):
    mod, params, inputs, dtypes = get_model(
        "https://github.com/mlcommons/mobile_models/raw/main/v0_7/tflite/deeplabv3_mnv2_ade20k_float.tflite",
        "deeplabv3_mnv2_ade20k_float.tflite", "tflite")
    if dtype == "float16":
        mod = convert_to_fp16(mod["main"], dtype)
    build_run_compare(mod, params, inputs, dtypes, target, [], model_name="deeplabv3", model_dtype=dtype)

@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_ssd_mobilenet(target, dtype):
    tflite_model_file = tf_testing.get_workload_official(
        "https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/models/object_detection/"
        "ssd_mobilenet_v1_coco_2018_01_28.tgz",
        "ssd_mobilenet_v1_coco_2018_01_28.tflite",
    )
    mod, params, inputs, dtypes = get_model(
        None, tflite_model_file, "tflite")
    if dtype == "float16":
        mod = convert_to_fp16(mod["main"], dtype)
    build_run_compare(mod, params, inputs, dtypes, target, [], model_name="ssd_mobilenet", model_dtype=dtype)



if __name__ == "__main__":
    tvm.testing.main()

