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
 * \file tvm/topi/image/resize.h
 * \brief Image resize operators (resize2d).
 */
#ifndef TVM_TOPI_IMAGE_RESIZE_H_
#define TVM_TOPI_IMAGE_RESIZE_H_

#include <tvm/te/operation.h>
#include <tvm/tir/op.h>
#include <tvm/topi/tags.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace tvm {
namespace topi {
namespace image {

// nchw_pack_layout: layout starts with "NCHW", contains both 'c' and 'n'
// e.g. "NCHWc4n4" → true
inline bool NchwPackLayout(const std::string& layout) {
  return layout.size() > 4 && layout.substr(0, 4) == "NCHW" &&
         layout.find('c') != std::string::npos && layout.find('n') != std::string::npos;
}

// nchw_xc_layout: layout starts with "NCHW", contains 'c', and the middle
// characters (between "NCHW" and the trailing 'c') are all digits.
// e.g. "NCHW4c" → true
inline bool NchwXcLayout(const std::string& layout) {
  if (layout.size() <= 4 || layout.substr(0, 4) != "NCHW") return false;
  if (layout.find('c') == std::string::npos) return false;
  // The part between "NCHW" and the last character must be all digits.
  std::string mid = layout.substr(4, layout.size() - 5);
  if (mid.empty()) return false;
  for (char ch : mid) {
    if (!std::isdigit(static_cast<unsigned char>(ch))) return false;
  }
  return layout.back() == 'c';
}

// Coordinate transformation: get_inx
// Maps an output coordinate x to the corresponding input coordinate.
// Mirrors python get_inx() exactly.
inline PrimExpr GetInx(PrimExpr x, PrimExpr image_size, PrimExpr target_size,
                       const std::string& coord_trans, PrimExpr roi_start, PrimExpr roi_end) {
  // All arithmetic is done in float32, matching the Python implementation.
  PrimExpr img_f = tvm::cast(DataType::Float(32), image_size);
  PrimExpr tgt_f = tvm::cast(DataType::Float(32), target_size);
  PrimExpr x_f = tvm::cast(DataType::Float(32), x);

  PrimExpr scale = tvm::div(img_f, tgt_f);  // image_size / target_size

  if (coord_trans == "half_pixel") {
    // (x + 0.5) * scale - 0.5
    return (x_f + tvm::tir::make_const(DataType::Float(32), 0.5)) * scale -
           tvm::tir::make_const(DataType::Float(32), 0.5);

  } else if (coord_trans == "align_corners") {
    // (image_size - 1) / (target_size - 1) * x
    PrimExpr num = tvm::cast(DataType::Float(32), image_size - 1);
    PrimExpr den = tvm::cast(DataType::Float(32), target_size - 1);
    return tvm::div(num, den) * x_f;

  } else if (coord_trans == "asymmetric") {
    // scale * x
    return scale * x_f;

  } else if (coord_trans == "pytorch_half_pixel") {
    // if target_size > 1: (x + 0.5) * scale - 0.5  else: 0.0
    PrimExpr val = (x_f + tvm::tir::make_const(DataType::Float(32), 0.5)) * scale -
                   tvm::tir::make_const(DataType::Float(32), 0.5);
    return tvm::if_then_else(target_size > 1, val, tvm::tir::make_const(DataType::Float(32), 0.0));

  } else if (coord_trans == "tf_half_pixel_for_nn") {
    // (x + 0.5) * scale
    return (x_f + tvm::tir::make_const(DataType::Float(32), 0.5)) * scale;

  } else if (coord_trans == "tf_crop_and_resize") {
    // if target_size > 1:
    //   roi_start*(image_size-1) + x*(roi_end-roi_start)*(image_size-1)/(target_size-1)
    // else:
    //   0.5*(roi_start+roi_end)*(image_size-1)
    PrimExpr img_m1 = tvm::cast(DataType::Float(32), image_size - 1);
    PrimExpr tgt_m1 = tvm::cast(DataType::Float(32), target_size - 1);
    PrimExpr rs = tvm::cast(DataType::Float(32), roi_start);
    PrimExpr re = tvm::cast(DataType::Float(32), roi_end);
    PrimExpr val = rs * img_m1 + x_f * (re - rs) * img_m1 / tgt_m1;
    PrimExpr alt = tvm::tir::make_const(DataType::Float(32), 0.5) * (rs + re) * img_m1;
    return tvm::if_then_else(target_size > 1, val, alt);

  } else {
    LOG(FATAL) << "Unsupported coordinate_transformation_mode: " << coord_trans;
    return PrimExpr();
  }
}

// Nearest-neighbour rounding: get_closest_index
inline PrimExpr GetClosestIndex(PrimExpr in_x, const std::string& rounding_method) {
  PrimExpr eps = tvm::tir::make_const(DataType::Float(32), 1e-5);

  if (rounding_method == "round") {
    return tvm::cast(DataType::Int(32), tvm::round(in_x));
  } else if (rounding_method == "round_prefer_floor") {
    return tvm::cast(DataType::Int(32),
                     tvm::ceil(in_x - tvm::tir::make_const(DataType::Float(32), 0.5)));
  } else if (rounding_method == "round_prefer_ceil") {
    return tvm::cast(DataType::Int(32),
                     tvm::floor(in_x + tvm::tir::make_const(DataType::Float(32), 0.5)));
  } else if (rounding_method == "floor") {
    return tvm::cast(DataType::Int(32), tvm::floor(in_x + eps));
  } else if (rounding_method == "ceil") {
    return tvm::cast(DataType::Int(32), tvm::ceil(in_x - eps));
  } else {
    LOG(FATAL) << "Unknown rounding method: " << rounding_method;
    return PrimExpr();
  }
}

// Pixel accessor: get_2d_pixel
// Clamps y and x to [0, image_height-1] / [0, image_width-1] and reads
// data at the appropriate layout-dependent indices.
inline PrimExpr Get2dPixel(const te::Tensor& data, const std::string& layout, PrimExpr image_h,
                           PrimExpr image_w, PrimExpr n, PrimExpr c, PrimExpr y, PrimExpr x,
                           PrimExpr cc, PrimExpr ib, PrimExpr ic) {
  // Clamp coordinates to valid range.
  y = tvm::max(tvm::min(y, image_h - 1), tvm::tir::make_const(DataType::Int(32), 0));
  x = tvm::max(tvm::min(x, image_w - 1), tvm::tir::make_const(DataType::Int(32), 0));

  PrimExpr val;
  if (layout == "NHWC") {
    val = data(ffi::Array<PrimExpr>{n, y, x, c});
  } else if (layout == "NCHW") {
    val = data(ffi::Array<PrimExpr>{n, c, y, x});
  } else if (NchwPackLayout(layout)) {
    val = data(ffi::Array<PrimExpr>{n, c, y, x, ib, ic});
  } else {
    // NCHWxc
    val = data(ffi::Array<PrimExpr>{n, c, y, x, cc});
  }
  return tvm::cast(DataType::Float(32), val);
}

// Linear interpolation helpers
inline PrimExpr Lerp(PrimExpr A, PrimExpr B, PrimExpr t) {
  // A*(1-t) + B*t
  return A * (tvm::tir::make_const(DataType::Float(32), 1.0) - t) + B * t;
}

// Cubic spline weights for fractional offset t (4-tap).
inline std::array<PrimExpr, 4> CubicSplineWeights(PrimExpr t, double alpha) {
  PrimExpr a = tvm::tir::make_const(DataType::Float(32), alpha);
  PrimExpr t2 = t * t;
  PrimExpr t3 = t * t * t;
  PrimExpr w0 = a * (t3 - tvm::tir::make_const(DataType::Float(32), 2.0) * t2 + t);
  PrimExpr w1 = (a + tvm::tir::make_const(DataType::Float(32), 2.0)) * t3 -
                (tvm::tir::make_const(DataType::Float(32), 3.0) + a) * t2 +
                tvm::tir::make_const(DataType::Float(32), 1.0);
  PrimExpr w2 = -(a + tvm::tir::make_const(DataType::Float(32), 2.0)) * t3 +
                (tvm::tir::make_const(DataType::Float(32), 3.0) +
                 tvm::tir::make_const(DataType::Float(32), 2.0) * a) *
                    t2 -
                a * t;
  PrimExpr w3 = -a * t3 + a * t2;
  return {w0, w1, w2, w3};
}

// Apply 4-tap cubic kernel: sum(p[i] * w[i]).
inline PrimExpr CubicKernel(const std::array<PrimExpr, 4>& p, const std::array<PrimExpr, 4>& w) {
  return p[0] * w[0] + p[1] * w[1] + p[2] * w[2] + p[3] * w[3];
}

// _resize_2d: per-element compute body.
// NCHW/NHWC/NCHWinic/NCHWxc layouts.
inline PrimExpr Resize2dCompute(const ffi::Array<tir::Var>& indices, const te::Tensor& data,
                                const ffi::Array<FloatImm>& roi, PrimExpr image_h, PrimExpr image_w,
                                PrimExpr target_h, PrimExpr target_w, const std::string& layout,
                                const std::string& method, const std::string& coord_trans,
                                std::string rounding_method, double alpha, int exclude_outside,
                                double extrapolation_value, DataType out_dtype) {
  // Unpack indices according to layout
  PrimExpr n, c, y, x, cc, ib, ic;
  cc = ib = ic = tvm::tir::make_const(DataType::Int(32), 0);

  if (layout == "NHWC") {
    n = indices[0];
    y = indices[1];
    x = indices[2];
    c = indices[3];
    cc = PrimExpr();  // unused sentinel — we use layout string to dispatch
  } else if (layout == "NCHW") {
    n = indices[0];
    c = indices[1];
    y = indices[2];
    x = indices[3];
    cc = PrimExpr();
  } else if (NchwPackLayout(layout)) {
    n = indices[0];
    c = indices[1];
    y = indices[2];
    x = indices[3];
    ib = indices[4];
    ic = indices[5];
  } else {
    // NCHWxc
    n = indices[0];
    c = indices[1];
    y = indices[2];
    x = indices[3];
    cc = indices[4];
  }

  // ROI values (float32 PrimExpr)
  // roi = [start_h, start_w, end_h, end_w]
  PrimExpr roi_y0 =
      tvm::cast(DataType::Float(32), tvm::tir::make_const(DataType::Float(64), roi[0]->value));
  PrimExpr roi_x0 =
      tvm::cast(DataType::Float(32), tvm::tir::make_const(DataType::Float(64), roi[1]->value));
  PrimExpr roi_y1 =
      tvm::cast(DataType::Float(32), tvm::tir::make_const(DataType::Float(64), roi[2]->value));
  PrimExpr roi_x1 =
      tvm::cast(DataType::Float(32), tvm::tir::make_const(DataType::Float(64), roi[3]->value));

  // Map output (y, x) → input (in_y, in_x)
  PrimExpr in_y = GetInx(y, image_h, target_h, coord_trans, roi_y0, roi_y1);
  PrimExpr in_x = GetInx(x, image_w, target_w, coord_trans, roi_x0, roi_x1);

  // Pixel accessor helper (captures layout / cc / ib / ic)
  auto pixel = [&](PrimExpr py, PrimExpr px) -> PrimExpr {
    PrimExpr cc_arg = cc.defined() ? cc : tvm::tir::make_const(DataType::Int(32), 0);
    return Get2dPixel(data, layout, image_h, image_w, n, c, py, px, cc_arg, ib, ic);
  };

  // Interpolation
  PrimExpr value;

  if (method == "nearest_neighbor") {
    if (rounding_method.empty()) {
      rounding_method = (coord_trans == "align_corners") ? "round" : "floor";
    }
    PrimExpr yi = GetClosestIndex(in_y, rounding_method);
    PrimExpr xi = GetClosestIndex(in_x, rounding_method);
    value = pixel(yi, xi);

  } else if (method == "linear") {
    PrimExpr y_int = tvm::cast(DataType::Int(32), tvm::floor(in_y));
    PrimExpr x_int = tvm::cast(DataType::Int(32), tvm::floor(in_x));
    PrimExpr y_lerp = in_y - tvm::cast(DataType::Float(32), y_int);
    PrimExpr x_lerp = in_x - tvm::cast(DataType::Float(32), x_int);

    // 2×2 neighbourhood
    PrimExpr p[2][2];
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i) p[j][i] = pixel(y_int + j, x_int + i);

    PrimExpr top = Lerp(p[0][0], p[0][1], x_lerp);
    PrimExpr bottom = Lerp(p[1][0], p[1][1], x_lerp);
    value = Lerp(top, bottom, y_lerp);

  } else if (method == "cubic") {
    PrimExpr xint = tvm::cast(DataType::Int(32), tvm::floor(in_x));
    PrimExpr xfract = in_x - tvm::cast(DataType::Float(32), xint);
    PrimExpr yint = tvm::cast(DataType::Int(32), tvm::floor(in_y));
    PrimExpr yfract = in_y - tvm::cast(DataType::Float(32), yint);

    // 4×4 neighbourhood
    PrimExpr p[4][4];
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i) p[j][i] = pixel(yint + j - 1, xint + i - 1);

    auto wx = CubicSplineWeights(xfract, alpha);
    auto wy = CubicSplineWeights(yfract, alpha);

    if (exclude_outside) {
      // Zero out weights for samples outside the image boundary and
      // renormalise, mirroring python exclude_outside logic.
      PrimExpr zero = tvm::tir::make_const(DataType::Float(32), 0.0);
      PrimExpr sum_wx = tvm::tir::make_const(DataType::Float(32), 0.0);
      PrimExpr sum_wy = tvm::tir::make_const(DataType::Float(32), 0.0);
      for (int i = 0; i < 4; ++i) {
        PrimExpr xi = xint + i - 1;
        PrimExpr yi = yint + i - 1;
        wx[i] = tvm::if_then_else(tvm::tir::Or(xi < 0, xi >= image_w), zero, wx[i]);
        wy[i] = tvm::if_then_else(tvm::tir::Or(yi < 0, yi >= image_h), zero, wy[i]);
        sum_wx = sum_wx + wx[i];
        sum_wy = sum_wy + wy[i];
      }
      for (int i = 0; i < 4; ++i) {
        wx[i] = wx[i] / sum_wx;
        wy[i] = wy[i] / sum_wy;
      }
    }

    std::array<PrimExpr, 4> cols;
    for (int j = 0; j < 4; ++j) cols[j] = CubicKernel({p[j][0], p[j][1], p[j][2], p[j][3]}, wx);
    value = CubicKernel(cols, wy);

  } else {
    LOG(FATAL) << "Unknown resize method: " << method;
    value = tvm::tir::make_const(DataType::Float(32), 0.0);
  }

  // tf_crop_and_resize extrapolation
  if (coord_trans == "tf_crop_and_resize") {
    PrimExpr ev = tvm::tir::make_const(DataType::Float(32), extrapolation_value);
    PrimExpr zero_f = tvm::tir::make_const(DataType::Float(32), 0.0);
    PrimExpr h_m1 = tvm::cast(DataType::Float(32), image_h - 1);
    PrimExpr w_m1 = tvm::cast(DataType::Float(32), image_w - 1);
    // Nest: check in_y out-of-bounds, then in_x out-of-bounds.
    PrimExpr inner = tvm::if_then_else(in_y > h_m1, ev, value);
    PrimExpr outer = tvm::if_then_else(in_y < zero_f, ev, inner);
    inner = tvm::if_then_else(in_x > w_m1, ev, outer);
    value = tvm::if_then_else(in_x < zero_f, ev, inner);
  }

  // Cast to output dtype
  DataType result_dtype = out_dtype.is_void() ? data->dtype : out_dtype;
  return tvm::cast(result_dtype, value);
}

// resize2d
inline te::Tensor resize2d(const te::Tensor& data, const ffi::Array<FloatImm>& roi,
                           const ffi::Array<PrimExpr>& size, const std::string& layout = "NCHW",
                           const std::string& method = "linear",
                           const std::string& coordinate_transformation_mode = "half_pixel",
                           const std::string& rounding_method = "", double bicubic_alpha = -0.75,
                           int bicubic_exclude = 0, double extrapolation_value = 0.0,
                           DataType out_dtype = DataType::Void()) {
  // Derive input spatial dims and output shape
  PrimExpr in_h, in_w;
  ffi::Array<PrimExpr> output_shape;

  if (layout == "NHWC") {
    in_h = data->shape[1];
    in_w = data->shape[2];
    output_shape = {data->shape[0], size[0], size[1], data->shape[3]};
  } else if (layout == "NCHW") {
    in_h = data->shape[2];
    in_w = data->shape[3];
    output_shape = {data->shape[0], data->shape[1], size[0], size[1]};
  } else if (NchwPackLayout(layout)) {
    in_h = data->shape[2];
    in_w = data->shape[3];
    output_shape = {data->shape[0], data->shape[1], size[0],
                    size[1],        data->shape[4], data->shape[5]};
  } else if (NchwXcLayout(layout)) {
    in_h = data->shape[2];
    in_w = data->shape[3];
    output_shape = {data->shape[0], data->shape[1], size[0], size[1], data->shape[4]};
  } else {
    LOG(FATAL) << "resize2d: unsupported layout: " << layout;
  }

  return te::compute(
      output_shape,
      [&](const ffi::Array<tir::Var>& indices) -> PrimExpr {
        return Resize2dCompute(indices, data, roi, in_h, in_w, size[0], size[1], layout, method,
                               coordinate_transformation_mode, rounding_method, bicubic_alpha,
                               bicubic_exclude, extrapolation_value, out_dtype);
      },
      "resize", kInjective);
}

}  // namespace image
}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_IMAGE_RESIZE_H_
