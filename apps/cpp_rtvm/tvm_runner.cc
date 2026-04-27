/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm_runner.cc
 * \brief TVM model runner implementation.
 */

#include "tvm_runner.h"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include "cnpy.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Get the TVM device id corresponding to device string.
 * \param device the target device in string format.
 * \return dl_device corresponding to the device string.
 */
DLDeviceType GetTVMDevice(std::string device) {
  if (!device.compare("cpu")) {
    return kDLCPU;
  } else if (!device.compare("llvm")) {
    return kDLCPU;
  } else if (!device.compare("cuda")) {
    return kDLCUDA;
  } else if (!device.compare("opencl")) {
    return kDLOpenCL;
  } else if (!device.compare("vulkan")) {
    return kDLVulkan;
  } else if (!device.compare("metal")) {
    return kDLMetal;
  } else if (!device.compare("vpi")) {
    return kDLVPI;
  } else if (!device.compare("rocm")) {
    return kDLROCM;
  } else if (!device.compare("oneapi")) {
    return kDLOneAPI;
  } else {
    LOG(FATAL) << "TVMRunner : Unsupported device :" << device;
  }
}

/*!
 * \brief Parsing numpy file to get data type of npy tensor.
 * \param fname Numpy file name.
 * \return data type of npy tensor.
 */
std::string parse_npy_dtype(std::string fname) {
  if (fname.find(".npy") == std::string::npos)
    throw std::runtime_error("parse_npy_dtype: Invalid file " + fname);

  FILE* fp = fopen(fname.c_str(), "rb");

  if (!fp) throw std::runtime_error("parse_npy_dtype: Unable to open file " + fname);

  char buffer[256];
  size_t res = fread(buffer, sizeof(char), 11, fp);
  if (res != 11) throw std::runtime_error("get_data_type: failed fread");
  std::string header = fgets(buffer, 256, fp);
  size_t loc1 = header.find("descr") + 9;
  std::string str_ws = header.substr(loc1 + 1);
  size_t loc3 = str_ws.find("'");
  std::string dtype_code = str_ws.substr(0, loc3 - 1);
  std::string dtype_size = str_ws.substr(1, loc3);
  std::string data_type;
  if (dtype_code == "f")
    data_type = "float";
  else if (dtype_code == "u")
    data_type = "uint";
  else if (dtype_code == "i")
    data_type = "int";
  data_type = data_type + std::to_string(atoi(dtype_size.c_str()) * 8);
  fclose(fp);
  return data_type;
}

// Function to trim whitespace from a string
std::string trim(const std::string& str) {
  size_t first = str.find_first_not_of(' ');
  if (first == std::string::npos) return "";
  size_t last = str.find_last_not_of(' ');
  return str.substr(first, last - first + 1);
}

// Function to parse a JSON object
std::unordered_map<std::string, std::string> parse_json(const std::string& json_str) {
  std::unordered_map<std::string, std::string> json_map;
  std::istringstream ss(json_str);
  std::string line;

  while (std::getline(ss, line, ',')) {
    size_t colon_pos = line.find(':');
    if (colon_pos != std::string::npos) {
      std::string key = trim(line.substr(0, colon_pos));
      std::string value = trim(line.substr(colon_pos + 1));
      key.erase(remove(key.begin(), key.end(), '\"'), key.end());
      value.erase(remove(value.begin(), value.end(), '\"'), value.end());
      json_map[key] = value;
    }
  }

  return json_map;
}

/*!
 * \brief Calculated the memory size for the NDArray.
 * \param NDArray object.
 * \return size of the memory.
 */
inline size_t GetMemSize(Tensor& narr) {
  size_t size = 1;
  for (tvm_index_t i = 0; i < narr->ndim; ++i) {
    size *= static_cast<size_t>(narr->shape[i]);
  }
  size *= (narr->dtype.bits * narr->dtype.lanes + 7) / 8;
  return size;
}

/*!
 * \brief Save Output Tensor to npy output file.
 * \param Tensor object.
 * \param index output index id.
 * \param fname output folder name,under that it will saving as index.npy.
 * \return 0 on success else error code.
 */
int SaveNDArrayToNpyFile(Tensor& nd_arr, int index, std::string fname) {
  auto ssize = GetMemSize(nd_arr);
  LOG(INFO) << "Output Size:" << ssize << "  bytes";

  void* data = (void*)malloc(ssize * (nd_arr->dtype.bits * nd_arr->dtype.lanes + 7) / 8);
  nd_arr.CopyToBytes(data, ssize);
  std::vector<size_t> shape;

  for (int j = 0; j < nd_arr->ndim; ++j) shape.push_back(nd_arr->shape[j]);
  if (((nd_arr->dtype.bits * nd_arr->dtype.lanes + 7) / 8) == 4) {
    cnpy::npy_save<float>(fname + "/" + std::to_string(index) + ".npy", (float*)data, shape, "w");
  } else if (((nd_arr->dtype.bits * nd_arr->dtype.lanes + 7) / 8) == 2) {
    cnpy::npy_save<uint16_t>(fname + "/" + std::to_string(index) + ".npy", (uint16_t*)data, shape,
                             "w");
  } else if (((nd_arr->dtype.bits * nd_arr->dtype.lanes + 7) / 8) == 1) {
    cnpy::npy_save<int8_t>(fname + "/" + std::to_string(index) + ".npy", (int8_t*)data, shape, "w");
  } else {
    LOG(WARNING) << "DType:" << (((nd_arr->dtype.bits * nd_arr->dtype.lanes + 7) / 8) == 2)
                 << " is not supported for npy_save";
  }
  free(data);
  return 0;
}

/*!
 * \brief Constructor for TVMRunner.
 * \param path where the tfm compiler artifacts present.
 * \param device the target device where we need to load the compiled model.
 */
TVMRunner::TVMRunner(std::string path, std::string device)
    : r_model_path(path), r_device(device), r_run_was_called(false) {
  LOG(INFO) << "TVMRunner Constructor:" << r_model_path << " Devices:" << r_device;
}

/*!
 * \brief Load Setup TVM graph runtime for given model.
 * \param 0 on success else error code.
 */
int TVMRunner::Load(void) {
  LOG(INFO) << "TVMRunner Load:" << (r_model_path).c_str();
  // Load the lib file
  auto tstart = std::chrono::high_resolution_clock::now();

  ffi::Module executable = ffi::Module::LoadFromFile((r_model_path).c_str());
  auto fload_exec = executable->GetFunction("vm_load_executable");
  TVM_FFI_ICHECK(fload_exec.has_value()) << "TVM runtime cannot find vm_load_executable";
  r_graph_handle = (*fload_exec)().cast<ffi::Module>();
  // Get ref to graph executor
  (*r_graph_handle)
      ->GetFunction("vm_initialization")
      .value()(static_cast<int>(GetTVMDevice(r_device)), 0,
               static_cast<int>(tvm::runtime::AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
               static_cast<int>(tvm::runtime::AllocatorType::kPooled));
  auto tend = std::chrono::high_resolution_clock::now();
  r_module_load_ms = static_cast<double>((tend - tstart).count()) / 1e6;

  return 0;
}

/*!
 * \brief Create model inputs NDarray from npy file.
 * \param inputfile the npy file from where we read input tensor data.
 * \param 0 on success else error code.
 */
int TVMRunner::CreateInputNDArrayFromFile(std::string inputfile) {
  LOG(INFO) << "TVMRunner::SetInput (Numpy):" << inputfile;
  for (int i = 0; i < mInfo.n_inputs; i++) {
    std::string param_name = (*r_graph_handle)
                                 ->GetFunction("get_function_param_name")
                                 .value()("main", i)
                                 .cast<std::string>();
    cnpy::NpyArray npy_arry = cnpy::npy_load(inputfile + "/" + param_name + ".npy");
    std::string dtype = parse_npy_dtype(inputfile + "/" + param_name + ".npy");
    if (inputs_.size() <= i) inputs_.resize(i + 1);
    inputs_[i] =
        Tensor::Empty(ffi::Shape(npy_arry.shape.begin(), npy_arry.shape.end()),
                      tvm::runtime::StringToDLDataType(dtype), DLDevice{GetTVMDevice(r_device), 0});
    auto ssize = GetMemSize(inputs_[i]);
    inputs_[i].CopyFromBytes(npy_arry.data<char>(), ssize);
  }
  return 0;
}

/*!
 * \brief Set the model input from the given binary buffer.
 * \param input_id input node name.
 * \param raw_input binary input buffer to copy over input NDArray.
 * \param 0 on success else error code.
 */
int TVMRunner::SetInput(int index, char* raw_input) {
  if (inputs_.size() > index) {
    auto ssize = GetMemSize(inputs_[index]);
    inputs_[index].CopyFromBytes(raw_input, ssize);
  } else {
    LOG(FATAL) << "Input NDArray not created";
  }
  (*r_graph_handle)->GetFunction("set_input_with_index").value()("main", index, inputs_[index]);
  return 0;
}

/*!
 * \brief Set the model input from given NDArray with zero copy.
 * \param 0 on success else error code.
 */
int TVMRunner::SetInput() {
  for (int i = 0; i < inputs_.size(); i++) {
    (*r_graph_handle)->GetFunction("set_input_with_index").value()("main", i, inputs_[i]);
  }
  return 0;
}

/*!
 * \brief Get the model outputs and dump them to npz file.
 * \param outputfile the npz file to where we dump the output data.
 * \param 0 on success else error code.
 */
int TVMRunner::GetOutput(std::string outputfile) {
  LOG(INFO) << "TVMRunner::GetOutput (Numpy):" << outputfile;

  // Check if the directory already exists otherwise create
  if (!std::filesystem::exists(outputfile)) std::filesystem::create_directory(outputfile);
  if (mInfo.n_outputs == -1) {
    Tensor out_arr = (*r_graph_handle)->GetFunction("get_output").value()("main").cast<Tensor>();
    SaveNDArrayToNpyFile(out_arr, 0, outputfile);
  } else {
    for (int i = 0; i < mInfo.n_outputs; i++) {
      Tensor out_arr =
          (*r_graph_handle)->GetFunction("get_output").value()("main", i).cast<Tensor>();
      SaveNDArrayToNpyFile(out_arr, i, outputfile);
    }
  }
  return 0;
}

/*!
 * \brief Get output of the model as a binary buffer.
 * \param output_id output node name to read the data.
 * \param raw_output the buffer to copy the data to.
 * \param 0 on success else error code.
 */
int TVMRunner::GetOutput(int index, char* raw_output) {
  if (mInfo.n_outputs == -1) {
    Tensor out_arr = (*r_graph_handle)->GetFunction("get_output").value()("main").cast<Tensor>();
    auto ssize = GetMemSize(out_arr);
    out_arr.CopyToBytes(raw_output, ssize);
  } else {
    Tensor out_arr =
        (*r_graph_handle)->GetFunction("get_output").value()("main", index).cast<Tensor>();
    auto ssize = GetMemSize(out_arr);
    out_arr.CopyToBytes(raw_output, ssize);
  }
  return 0;
}

/*!
 * \brief Get output of the model as a binary buffer.
 * \param index output node id to read the data.
 * \return output Tensor.
 */
Tensor TVMRunner::GetOutputNDArray(int index) {
  if (mInfo.n_outputs == -1) {
    return (*r_graph_handle)->GetFunction("get_output").value()("main").cast<Tensor>();
  } else {
    return (*r_graph_handle)->GetFunction("get_output").value()("main", index).cast<Tensor>();
  }
}

/*!
 * \brief Call one cycle of execution for the model.
 * \param 0 on success else error code.
 */
int TVMRunner::Run(void) {
  r_run_was_called = true;
  (*r_graph_handle)->GetFunction("invoke_stateful").value()("main");
  mInfo.n_outputs = (*r_graph_handle)->GetFunction("get_output_arity").value()("main").cast<int>();
  return 0;
}

/*!
 * \brief Query various metadata from the graph runtime.
 * \param 0 on success else error code.
 */
TVMMetaInfo TVMRunner::GetMetaInfo(void) {
  LOG(INFO) << "TVMRunner::GetMetaInfo";
  mInfo.n_inputs = (*r_graph_handle)->GetFunction("get_function_arity").value()("main").cast<int>();
  inputs_.resize(mInfo.n_inputs);
  for (int i = 0; i < mInfo.n_inputs; i++) {
    mInfo.param_names.push_back((*r_graph_handle)
                                    ->GetFunction("get_function_param_name")
                                    .value()("main", i)
                                    .cast<std::string>());
  }
  return mInfo;
}

/*!
 * \brief Print the meta information.
 * \param 0 on success else error code.
 */
void TVMRunner::PrintMetaInfo(void) {
  LOG(INFO) << "Meta Information:" << r_model_path;
  LOG(INFO) << "    Number of Inputs:" << mInfo.n_inputs;
  LOG(INFO) << "    Input MetaInfo:";
  for (int i = 0; i < mInfo.param_names.size(); i++) {
    LOG(INFO) << "param_names - " << mInfo.param_names[i];
  }
}

/*!
 * \brief Print stats information.
 */
void TVMRunner::PrintStats(void) {
  LOG(INFO) << "Performance Stats:" << r_model_path;
  LOG(INFO) << "Total Module Load Time     :" << r_module_load_ms << " ms";
}

}  // namespace runtime
}  // namespace tvm
