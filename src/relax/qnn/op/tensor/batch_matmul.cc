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
 * \file src/relax/qnn/op/tensor/batch_matmul.cc
 * \brief QNN batch_matmul operators implementation
 */

#include "./batch_matmul.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace tvm {
namespace relax {
namespace qnn {

// relax.qnn.batch_matmul
Expr batch_matmul(Expr x, Expr y, Expr x_zero_point, Expr y_zero_point, Expr x_scale, Expr y_scale,
                  DataType out_dtype) {
  auto attrs = tvm::ffi::make_object<MatmulAttrs>();
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("relax.qnn.batch_matmul");
  return Call(op, {x, y, x_zero_point, y_zero_point, x_scale, y_scale}, Attrs(attrs), {});
}

/* Struct info inference for relax.qnn.batch_matmul */
StructInfo InferStructInfoBatchMatmul(const Call& call, const BlockBuilder& ctx) {
  // args[0]: x, args[1]: y, args[2]: x_zero_point, args[3]: y_zero_point,
  // args[4]: x_scale, args[5]: y_scale
  if (call->args.size() != 6) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "qnn.batch_matmul expects 6 arguments "
                        "(x, y, x_zero_point, y_zero_point, x_scale, y_scale). "
                        "However, got "
                     << call->args.size());
  }
  // Attributes check
  const auto* attrs = call->attrs.as<MatmulAttrs>();
  if (attrs == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call) << "qnn.batch_matmul is missing BatchMatmulAttrs.");
  }
  auto args = GetInputTensorStructInfo(call, ctx);

  // Dtype check
  const std::string op_name = "relax.qnn.batch_matmul";
  CheckIntegerInputDtype(call, ctx, args[0], op_name);  // x
  CheckIntegerInputDtype(call, ctx, args[1], op_name);  // y
  // Zero points must be integer types
  CheckZeroPointDtype(call, ctx, args[2], op_name, "x_zero_point");
  CheckZeroPointDtype(call, ctx, args[3], op_name, "y_zero_point");
  // Scales must be floating-point
  CheckScaleDtype(call, ctx, args[4], op_name, "x_scale");
  CheckScaleDtype(call, ctx, args[5], op_name, "y_scale");

  return tvm::relax::InferStructInfoMatmul(call, ctx);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.qnn.op.batch_matmul", batch_matmul);
}

TVM_REGISTER_OP("relax.qnn.batch_matmul")
    .set_attrs_type<MatmulAttrs>()
    .set_num_inputs(6)
    .add_argument("x", "Tensor", "First quantized input tensor.")
    .add_argument("y", "Tensor", "Second quantized input tensor.")
    .add_argument("x_zero_point", "Tensor", "The quantization zero_point of the x input tensor.")
    .add_argument("y_zero_point", "Tensor", "The quantization zero_point of the y input tensor.")
    .add_argument("x_scale", "Tensor", "The quantization scale of the x input tensor.")
    .add_argument("y_scale", "Tensor", "The quantization scale of the y input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoBatchMatmul)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace qnn
}  // namespace relax
}  // namespace tvm
