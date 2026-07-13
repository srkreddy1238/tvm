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
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pytest
import tensorflow as tf
import tflite.Model

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend.tflite import from_tflite


class IntermediateOutputExtractor:
    def __init__(self):
        self.step = 0
        self.outputs = {}

    def __call__(self, *args):
        result = None
        is_pre = True
        func = None

        for arg in args:
            if isinstance(arg, bool):
                is_pre = arg
            if hasattr(arg, "numpy") or hasattr(arg, "__len__"):
                if not isinstance(arg, list | tuple):
                    result = arg
            if hasattr(arg, "name_hint") or "fused" in str(arg):
                func = arg

        if not is_pre and result is not None:
            self.step += 1
            op_name = getattr(func, "name_hint", str(func))
            if hasattr(result, "numpy"):
                data = result.numpy()
                self.outputs[f"step_{self.step}_{op_name}"] = data

        return args[6] if len(args) > 6 else result


def build_and_run(mod, input_arrays, target="llvm"):
    tgt = tvm.target.Target(target, host=target)
    dev = tvm.cpu()
    dev_inputs = [tvm.runtime.tensor(inp, dev) for inp in input_arrays]
    debugger = IntermediateOutputExtractor()
    relax_pipeline = relax.pipeline.get_default_pipeline(tgt)
    tir_pipeline = tvm.tir.get_default_tir_pipeline(tgt)
    rt_mod = tvm.compile(mod, tgt, relax_pipeline=relax_pipeline, tir_pipeline=tir_pipeline)
    vm = relax.VirtualMachine(rt_mod.mod, dev, profile=True)
    vm.set_input("main", *dev_inputs)
    vm.set_instrument(debugger)
    vm.invoke_stateful("main")
    return vm.get_outputs("main").numpy()


def ref_out_tflite_runtime(input_data, model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])


def tvm_relax_output(input_data, model_path):
    with open(model_path, "rb") as f:
        model_buf = f.read()
    model = tflite.Model.Model.GetRootAsModel(model_buf, 0)
    mod = from_tflite(model)
    return build_and_run(mod, [input_data])


class EvalStrategy(Enum):
    TOP_K_OVERLAP = auto()
    ELEMENTWISE_DIFF = auto()


@dataclass
class ModelConfig:
    strategy: EvalStrategy
    topk: int = 5
    min_overlap: int = 5
    max_diff: int = 3
    note: str = ""


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "resnet18": ModelConfig(
        strategy=EvalStrategy.TOP_K_OVERLAP,
        topk=10,
        min_overlap=5,
        note="At least 5 of top-10 class indices must overlap between TFLite and Relax",
    ),
    "Resnet18": ModelConfig(
        strategy=EvalStrategy.TOP_K_OVERLAP,
        topk=10,
        min_overlap=5,
        note="At least 5 of top-10 class indices must overlap between TFLite and Relax",
    ),
    "mobilenet_v2-mobilenet-v2-w8a8": ModelConfig(
        strategy=EvalStrategy.TOP_K_OVERLAP,
        topk=10,
        min_overlap=5,
        note="At least 5 of top-10 class indices must overlap between TFLite and Relax",
    ),
    "squeezenet": ModelConfig(
        strategy=EvalStrategy.TOP_K_OVERLAP,
        topk=10,
        min_overlap=5,
        note="At least 5 of top-10 class indices must overlap between TFLite and Relax",
    ),
    "inception": ModelConfig(
        strategy=EvalStrategy.TOP_K_OVERLAP,
        topk=10,
        min_overlap=5,
        note="At least 5 of top-10 class indices must overlap between TFLite and Relax",
    ),
    "quicksrnetsmall": ModelConfig(
        strategy=EvalStrategy.ELEMENTWISE_DIFF,
        max_diff=3,
        note="Every output pixel must differ by at most 3",
    ),
    "quicksrnetmedium": ModelConfig(
        strategy=EvalStrategy.ELEMENTWISE_DIFF,
        max_diff=3,
        note="Every output pixel must differ by at most 3",
    ),
    "unet_segmentation": ModelConfig(
        strategy=EvalStrategy.ELEMENTWISE_DIFF,
        max_diff=3,
        note="Every output pixel must differ by at most 3",
    ),
}


def _get_config(model_name: str) -> ModelConfig:
    stem = os.path.splitext(os.path.basename(model_name))[0]
    if stem not in MODEL_CONFIGS:
        raise ValueError(f"No ModelConfig registered for '{stem}'. Add an entry to MODEL_CONFIGS.")
    return MODEL_CONFIGS[stem]


def _print_diff_stats(t_flat: np.ndarray, r_flat: np.ndarray) -> np.ndarray:
    abs_diff = np.abs(t_flat.astype(np.int32) - r_flat.astype(np.int32))
    return abs_diff


def _compare_top_k_overlap(
    t_flat: np.ndarray,
    r_flat: np.ndarray,
    topk: int,
    min_overlap: int,
) -> bool:
    ref_idx = np.argpartition(t_flat, -topk)[-topk:]
    ref_idx = ref_idx[np.argsort(t_flat[ref_idx])[::-1]]
    qnn_idx = np.argpartition(r_flat, -topk)[-topk:]
    qnn_idx = qnn_idx[np.argsort(r_flat[qnn_idx])[::-1]]

    ref_set = set(ref_idx.tolist())
    qnn_set = set(qnn_idx.tolist())
    overlap = ref_set & qnn_set

    if len(overlap) >= min_overlap:
        return True

    return False


def _compare_elementwise_diff(abs_diff: np.ndarray, max_diff: int) -> bool:
    violations = int(np.sum(abs_diff > max_diff))
    if violations > 0:
        return False

    return True


def compare_outputs(
    tflite_output: np.ndarray,
    relax_output: np.ndarray,
    config: ModelConfig,
) -> bool:
    if tflite_output.shape != relax_output.shape:
        return False

    t_flat = tflite_output.reshape(-1).astype(np.int32)
    r_flat = relax_output.reshape(-1).astype(np.int32)
    abs_diff = _print_diff_stats(t_flat, r_flat)

    if config.strategy == EvalStrategy.TOP_K_OVERLAP:
        return _compare_top_k_overlap(
            t_flat,
            r_flat,
            topk=config.topk,
            min_overlap=config.min_overlap,
        )

    if config.strategy == EvalStrategy.ELEMENTWISE_DIFF:
        return _compare_elementwise_diff(abs_diff, max_diff=config.max_diff)

    raise ValueError(f"Unknown strategy: {config.strategy}")


def run_model(model_path, input_data):
    config = _get_config(model_path)
    tflite_out = ref_out_tflite_runtime(input_data, model_path)
    relax_out = tvm_relax_output(input_data, model_path)

    print("\n" + "=" * 60)
    print("Input:")
    print(f"  shape : {input_data.shape}")
    print(f"  dtype : {input_data.dtype}")
    print(f"  values: {input_data}")
    print("=" * 60)
    print("TFLite Runtime Output:")
    print(f"  shape : {tflite_out.shape}")
    print(f"  dtype : {tflite_out.dtype}")
    print(f"  values: {tflite_out}")
    print("=" * 60)
    print("TVM Relax Output:")
    print(f"  shape : {relax_out.shape}")
    print(f"  dtype : {relax_out.dtype}")
    print(f"  values: {relax_out}")
    print("=" * 60 + "\n")

    return compare_outputs(tflite_out, relax_out, config)


def get_tflite_input_spec(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    return tuple(input_details["shape"]), input_details["dtype"]


def gen_input_from_tflite(model_path):
    tflite_shape, tflite_dtype = get_tflite_input_spec(model_path)

    np.random.seed(42)
    arr = np.random.randint(0, 256, size=tflite_shape).astype(tflite_dtype)

    return arr


@pytest.mark.parametrize(
    "model_name",
    [
        "/Inventory/qnn/resnet18.tflite",
        "/Inventory/qnn/squeezenet.tflite",
        "/Inventory/qnn/quicksrnetsmall.tflite",
        "/Inventory/qnn/quicksrnetmedium.tflite",
        "/Inventory/qnn/unet_segmentation.tflite",
    ],
)
def test_qnn_network(model_name):
    if not os.path.isabs(model_name):
        cv_models_path = os.environ.get("CV_MODELS_PATH", "")
        model_name = os.path.join(cv_models_path, model_name)

    if not os.path.exists(model_name):
        pytest.skip(f"Model not found at '{model_name}'.")

    input_data = gen_input_from_tflite(model_name)

    config = _get_config(model_name)

    ok = run_model(model_name, input_data)
    assert ok, f"Model {model_name} failed [{config.strategy.name}] — {config.note}"


if __name__ == "__main__":
    tvm.testing.main()
