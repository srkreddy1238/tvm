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
 * \file src/relax/qnn/op/tensor/manipulate.cc
 * \brief QNN manipulate operators implementation
 */

#include "./manipulate.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace tvm {
namespace relax {
namespace qnn {

/* relax.qnn.concat (builder) */
Expr concat(Expr data, Expr input_scales, Expr input_zero_points, Expr output_scale,
            Expr output_zero_point, int axis) {
  auto attrs = tvm::ffi::make_object<ConcatAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("relax.qnn.concat");
  return Call(op, {data, input_scales, input_zero_points, output_scale, output_zero_point},
              Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.qnn.op.concat", concat);
}

TVM_REGISTER_OP("relax.qnn.concat")
    .set_attrs_type<ConcatAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Tuple of Tensors", "The input tensors to concatenate (as a Tuple).")
    .add_argument("input_scales", "Tuple",
                  "Per-input quantization scales (each element is scalar () or 1-D (C,)).")
    .add_argument("input_zero_points", "Tuple",
                  "Per-input quantization zero points (each element is scalar () or 1-D (C,)).")
    .add_argument("output_scale", "Tensor",
                  "Output scale (scalar () or 1-D (C,) along the concat axis).")
    .add_argument("output_zero_point", "Tensor",
                  "Output zero point (scalar () or 1-D (C,) along the concat axis).")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoConcat)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutConcat)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace qnn
}  // namespace relax
}  // namespace tvm
