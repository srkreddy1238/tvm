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
"""Utils for adreno compute/schedules"""

import os
import tvm
import numpy as np
from tvm import relay
from tvm import autotvm
from tvm import rpc
from tvm.contrib import utils, ndk
from tvm.relay import testing
from tvm.relay.transform import recast
from tvm.contrib import graph_runtime
import json


def get_cpu_reference(mod, params1, input_shape, inputs):
    mod_fp32 = recast(mod, "float32", "float32", ops=["nn.conv2d", "add", "nn.relu"])
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod_fp32, "llvm", params=params1)
    ctx = tvm.cpu()
    m = graph_runtime.create(graph, lib, ctx)
    if isinstance(input_shape, dict):
        for key in input_shape:
            m.set_input(key, inputs[-1])
    else:
        m.set_input("data", inputs[-1])
    m.set_input(**params)
    m.run()
    return [
        m.get_output(0).asnumpy(),
    ]


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1024,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=False,
):
    from tvm.autotvm.tuner import XGBTuner
    from tvm.autotvm.tuner import GATuner

    tmp_log_file = log_filename + ".tmp"

    for i, tsk in enumerate(reversed(tasks)):
        print("Task: ", tsk)
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    autotvm.record.pick_best(tmp_log_file, log_filename)


# build module run with opencl and cpu, compare results
def build_run_compare(
    tvm_mod,
    params1,
    input_shape,
    dtypes,
    target="llvm",
    static_mem_scopes=[],
    gpu_preprocess=None,
    stat_file="adreno-tuning.log",
    tuning=False,
):

    if "TVM_TRACKER_HOST" in os.environ and "TVM_TRACKER_PORT" in os.environ:
        rpc_tracker_host = os.environ["TVM_TRACKER_HOST"]
        rpc_tracker_port = os.environ["TVM_TRACKER_PORT"]
        run_on_host = 0
        target_host = "llvm -mtriple=arm64-linux-android"
        rpc_tracker_port = int(rpc_tracker_port)
    else:
        run_on_host = 1
        target_host = "llvm"

    if gpu_preprocess:
        tvm_mod_nchwc = gpu_preprocess(tvm_mod)
    else:
        tvm_mod_nchwc = tvm_mod

    tuning_options = {
        "log_filename": stat_file,
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15),
            runner=autotvm.RPCRunner(
                "android",
                host=os.environ["TVM_TRACKER_HOST"],
                port=int(os.environ["TVM_TRACKER_PORT"]),
                number=3,
                timeout=600,
            ),
        ),
    }
    if tuning:
        tasks = autotvm.task.extract_from_program(
            tvm_mod, target=target, target_host=target_host, params=params1
        )
        tune_tasks(tasks, **tuning_options)

    if stat_file is not None:
        with autotvm.apply_history_best(stat_file):
            with tvm.transform.PassContext(opt_level=3):
                graph, lib, params = relay.build(
                    tvm_mod_nchwc, target_host=target_host, target=target, params=params1
                )
    else:
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build(
                tvm_mod_nchwc, target_host=target_host, target=target, params=params1
            )

    lib.export_library("model.so", ndk.create_shared)
    with open("model.json", "w") as fo:
        fo.write(graph)
    with open("model.params", "wb") as fo:
        fo.write(tvm.runtime.save_param_dict(params))
    exit(0)

    # verification that storage_scope has expected textures scopes
    graph_json = json.loads(graph)
    if "storage_scope" in graph_json["attrs"]:
        assert (
            len(static_mem_scopes) == len(graph_json["attrs"]["storage_scope"][1])
            or len(static_mem_scopes) == 0
        )
    else:
        assert len(static_mem_scopes) == 0

    for i in range(0, len(static_mem_scopes)):
        assert static_mem_scopes[i] == graph_json["attrs"]["storage_scope"][1][i]

    if run_on_host:
        ctx = tvm.opencl()
        m = graph_runtime.create(graph, lib, ctx)
    else:
        rpc_key = "android"
        tracker = rpc.connect_tracker(rpc_tracker_host, rpc_tracker_port)
        remote = tracker.request(rpc_key, priority=0, session_timeout=600)
        temp = utils.tempdir()
        dso_binary = "dev_lib_cl.so"
        dso_binary_path = temp.relpath(dso_binary)
        ctx = remote.cl(0)
        lib.export_library(dso_binary_path, ndk.create_shared)
        remote.upload(dso_binary_path)
        rlib = remote.load_module(dso_binary)
        m = graph_runtime.create(graph, rlib, ctx)
    m.set_input(**params)
    inputs = []
    for key in input_shape:
        inputs.append(np.random.normal(size=input_shape[key]).astype(dtypes[key]))
        m.set_input(key, inputs[-1])
    m.run()

    time_f = m.module.time_evaluator("run", remote.cl(0), number=10)
    cost = time_f().mean
    print("%g secs/iteration\n" % cost)

    ref_outputs = get_cpu_reference(tvm_mod, params1, input_shape, inputs)
    for i, ref_output in enumerate(ref_outputs):
        tvm_output = m.get_output(i)
        output = tvm_output.asnumpy()
        # for index, x in np.ndenumerate(ref_output):
        #     if abs(output[index] - x) > 0.01:
        #         print(index, output[index], x)

        np.testing.assert_allclose(output, ref_output, rtol=1e-1, atol=1e-1)
    return graph


def gpu_preprocess(tvm_mod):
    layout_config = relay.transform.LayoutConfig()
    desired_layouts = {"nn.conv2d": ["NCHW4c", "OIHW4o"]}
    with layout_config:
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = tvm.IRModule.from_expr(tvm_mod)
            tvm_mod_nchwc = seq(mod)
            return tvm_mod_nchwc

def get_model(url, local_file, module):
    def get_tensor_type_str(tensor_type):
        """Get tensor type string representation when given TFLite tensor type"""
        try:
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if tensor_type == TensorType.INT8:
            return "int8"
        if tensor_type == TensorType.INT16:
            return "int16"
        if tensor_type == TensorType.UINT8:
            return "uint8"
        if tensor_type == TensorType.FLOAT16:
            return "float16"
        if tensor_type == TensorType.FLOAT32:
            return "float32"
        if tensor_type == TensorType.INT32:
            return "int32"
        if tensor_type == TensorType.INT64:
            return "int64"
        if tensor_type == TensorType.BOOL:
            return "bool"
        raise NotImplementedError(
            "Tensor type {} is currently not supported".format(str(tensor_type))
        )

    if url is None:
        model_path = local_file
    else:
        model_path = tvm.contrib.download.download_testdata(url, local_file, module=module)

    with open(model_path, "rb") as f:
        tflite_model_buf = f.read()

    try:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except ImportError:
        raise ImportError("The tflite package must be installed")

    # keep the same as tflite
    assert tflite_model.SubgraphsLength() == 1, "only support one subgraph (main subgraph)"
    subgraph = tflite_model.Subgraphs(0)

    # model inputs
    model_inputs = subgraph.InputsAsNumpy()
    shape_dict = {}
    dtype_dict = {}
    for model_input in model_inputs:
        model_input_name = subgraph.Tensors(model_input).Name().decode("utf-8")
        model_shape_length = subgraph.Tensors(model_input).ShapeLength()
        model_input_shape = [ subgraph.Tensors(model_input).Shape(i) for i in range(model_shape_length)]
        shape_dict[model_input_name] = model_input_shape;
        dtype_dict[model_input_name] = get_tensor_type_str(subgraph.Tensors(model_input).Type())
        print("Input:", model_input_name, ":", model_input_shape, ":", get_tensor_type_str(subgraph.Tensors(model_input).Type()))

    # model Outputs
    model_outputs = subgraph.OutputsAsNumpy()
    shape_dict_out = {}
    dtype_dict_out = {}
    for model_output in model_outputs:
        model_output_name = subgraph.Tensors(model_output).Name().decode("utf-8")
        model_shape_length = subgraph.Tensors(model_output).ShapeLength()
        model_output_shape = [ subgraph.Tensors(model_output).Shape(i) for i in range(model_shape_length)]
        shape_dict_out[model_output_name] = model_output_shape;
        dtype_dict_out[model_output_name] = get_tensor_type_str(subgraph.Tensors(model_input).Type())
        print("Output:", model_output_name, ":", model_output_shape, ":", get_tensor_type_str(subgraph.Tensors(model_output).Type()))

    mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict
    )

    layout_config = relay.transform.LayoutConfig(skip_layers=[])
    desired_layouts = {"nn.conv2d": ["NCHW", "default"]}
    seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    return mod, params, shape_dict, dtype_dict
