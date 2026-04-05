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
 * \brief Registration of image operators (resize2d).
 * \file image.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/topi/image/resize.h>

namespace tvm {
namespace topi {

using namespace tvm;
using namespace tvm::runtime;

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("topi.image.resize2d", [](ffi::PackedArgs args, ffi::Any* rv) {
    te::Tensor data = args[0].cast<te::Tensor>();
    ffi::Array<FloatImm> roi = args[1].cast<ffi::Array<FloatImm>>();
    ffi::Array<PrimExpr> size = args[2].cast<ffi::Array<PrimExpr>>();
    std::string layout = args[3].cast<std::string>();
    std::string method = args[4].cast<std::string>();
    std::string coord_trans = args[5].cast<std::string>();
    std::string rounding_method = args[6].cast<std::string>();
    double bicubic_alpha = args[7].cast<double>();
    int bicubic_exclude = args[8].cast<int>();
    double extrapolation_value = args[9].cast<double>();

    std::transform(method.begin(), method.end(), method.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    *rv = image::resize2d(data, roi, size, layout, method, coord_trans, rounding_method,
                          bicubic_alpha, bicubic_exclude, extrapolation_value);
  });
}

}  // namespace topi
}  // namespace tvm
