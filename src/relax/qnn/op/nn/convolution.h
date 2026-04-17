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
 * \file convolution.h
 * \brief The functions to make Relax QNN convolution operator calls.
 */
#ifndef TVM_RELAX_QNN_OP_NN_CONVOLUTION_H_
#define TVM_RELAX_QNN_OP_NN_CONVOLUTION_H_

#include <string>
#include <utility>

#include "../../../op/op_common.h"
#include "tvm/relax/qnn/attrs.h"

namespace tvm {
namespace relax {

template <typename T>
inline Expr MakeConv(Expr data, Expr weight, Expr input_zero_pt, Expr weight_zero_pt,
                     ffi::Optional<Expr> input_scale, ffi::Optional<Expr> weight_scale,
                     ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
                     ffi::Array<int64_t> dilation, int groups, ffi::String data_layout,
                     ffi::String kernel_layout, ffi::String out_layout, DataType out_dtype,
                     std::string op_name) {
  auto attrs = ffi::make_object<T>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  TVM_FFI_ICHECK((input_scale && weight_scale) || (!input_scale && !weight_scale))
      << "Can't Create Op where only one of Input Scale and Kernel Scale is Present";

  const Op& op = Op::Get(op_name);
  if (!input_scale && !weight_scale)
    return Call(op, {data, weight, input_zero_pt, weight_zero_pt}, Attrs(attrs), {});

  return Call(
      op, {data, weight, input_zero_pt, weight_zero_pt, input_scale.value(), weight_scale.value()},
      Attrs(attrs), {});
}

/*! \brief 2D convolution */
Expr conv2d(Expr data, Expr weight, Expr data_zero_point, Expr weight_zero_pt,
            ffi::Optional<Expr> data_scale, ffi::Optional<Expr> weight_scale,
            ffi::Array<int64_t> strides, ffi::Array<int64_t> padding, ffi::Array<int64_t> dilation,
            int groups, ffi::String data_layout, ffi::String kernel_layout,
            ffi::Optional<ffi::String> out_layout, DataType out_dtype);

template <typename T>
inline Expr MakeConvTranspose(Expr data, Expr weight, Expr input_zero_pt, Expr weight_zero_pt,
                              ffi::Optional<Expr> input_scale, ffi::Optional<Expr> weight_scale,
                              ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
                              ffi::Array<int64_t> output_padding, ffi::Array<int64_t> dilation,
                              int groups, ffi::String data_layout, ffi::String kernel_layout,
                              ffi::String out_layout, DataType out_dtype, std::string op_name) {
  auto attrs = ffi::make_object<T>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->output_padding = std::move(output_padding);  // Conv2DTransposeAttrs field
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);

  TVM_FFI_ICHECK((input_scale && weight_scale) || (!input_scale && !weight_scale))
      << "Can't create op where only one of input_scale and weight_scale is present";

  const Op& op = Op::Get(op_name);
  if (!input_scale && !weight_scale) {
    return Call(op, {data, weight, input_zero_pt, weight_zero_pt}, Attrs(attrs), {});
  }
  return Call(
      op, {data, weight, input_zero_pt, weight_zero_pt, input_scale.value(), weight_scale.value()},
      Attrs(attrs), {});
}

/*! \brief Quantised 2-D transposed convolution (relax.qnn.conv2d_transpose) */
Expr conv2d_transpose(Expr data, Expr weight, Expr data_zero_point, Expr weight_zero_point,
                      ffi::Optional<Expr> data_scale, ffi::Optional<Expr> weight_scale,
                      ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
                      ffi::Array<int64_t> output_padding,  // <-- extra vs conv2d
                      ffi::Array<int64_t> dilation, int groups, ffi::String data_layout,
                      ffi::String kernel_layout, ffi::Optional<ffi::String> out_layout,
                      DataType out_dtype);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_QNN_OP_NN_CONVOLUTION_H_
