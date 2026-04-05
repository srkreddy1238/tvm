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
 * \file tvm/relax/transform/legalize_ops/nn.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */
#include <tvm/relax/attrs/nn.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/nn.h>
#include <tvm/topi/nn/group_norm.h>
#include <tvm/topi/nn/instance_norm.h>
#include <tvm/topi/nn/layer_norm.h>
#include <tvm/topi/nn/rms_norm.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/transform.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <cmath>

#include "utils.h"

namespace tvm {
namespace relax {

// relu
ffi::Array<te::Tensor> ReluTE(const ffi::Array<ffi::Any> args) {
  auto x = args[0].cast<te::Tensor>();
  return {topi::relu<float>(x)};
}

Expr LegalizeNNRelu(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  return m_te.Make(args, FTOPIHandler(ReluTE), std::string("relu"));
}
TVM_REGISTER_OP("relax.nn.relu")
    .set_attr<FLegalize>("FLegalize", LegalizeNNRelu, TVM_LEGALIZE_CPP_LEVEL);

// leakyrelu
ffi::Array<te::Tensor> LeakyReluTE(const ffi::Array<ffi::Any> args) {
  auto x = args[0].cast<te::Tensor>();
  auto alpha = args[1].cast<double>();
  return {topi::leaky_relu(x, alpha)};
}

Expr LegalizeNNLeakyRelu(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<LeakyReluAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(attrs->alpha);
  return m_te.Make(args, FTOPIHandler(LeakyReluTE), std::string("leaky_relu"));
}
TVM_REGISTER_OP("relax.nn.leakyrelu")
    .set_attr<FLegalize>("FLegalize", LegalizeNNLeakyRelu, TVM_LEGALIZE_CPP_LEVEL);

// prelu
ffi::Array<te::Tensor> PreluTE(const ffi::Array<ffi::Any> args) {
  auto x = args[0].cast<te::Tensor>();
  auto slope = args[1].cast<te::Tensor>();
  int axis = args[2].cast<int>();

  TVM_FFI_ICHECK_EQ(slope->shape.size(), 1) << "prelu: slope must be 1-D";

  int ndim = static_cast<int>(x->shape.size());
  TVM_FFI_ICHECK(axis >= 0 && axis < ndim)
      << "prelu: axis " << axis << " out of range for input with " << ndim << " dims";

  arith::Analyzer analyzer;
  if (analyzer.CanProve(slope->shape[0] == tvm::IntImm(slope->shape[0]->dtype, 1))) {
    slope = tvm::te::compute(
        ffi::Array<tvm::PrimExpr>({x->shape[axis]}),
        [&](const ffi::Array<tvm::tir::Var>& /*c*/) {
          return slope({tvm::IntImm(slope->shape[0]->dtype, 0)});
        },
        "slope_broadcasted");
  }

  TVM_FFI_ICHECK(analyzer.CanProve(slope->shape[0] == x->shape[axis]))
      << "prelu: slope.shape[0] (" << slope->shape[0] << ") must equal x.shape[axis] ("
      << x->shape[axis] << ")";

  return {tvm::te::compute(
      x->shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        auto xval = x(indices);
        return tvm::tir::Select(xval > tvm::tir::make_const(xval.dtype(), 0), xval,
                                xval * slope(ffi::Array<tvm::PrimExpr>({indices[axis]})));
      },
      "T_prelu", topi::kBroadcast)};
}

Expr LegalizeNNPrelu(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<PReluAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(attrs->axis);
  return m_te.Make(args, FTOPIHandler(PreluTE), std::string("prelu"));
}
TVM_REGISTER_OP("relax.nn.prelu")
    .set_attr<FLegalize>("FLegalize", LegalizeNNPrelu, TVM_LEGALIZE_CPP_LEVEL);

// gelu
ffi::Array<te::Tensor> GeluTE(const ffi::Array<ffi::Any> args) {
  auto x = args[0].cast<te::Tensor>();
  auto dtype = x->dtype;

  return {tvm::te::compute(
      x->shape,
      [&](const ffi::Array<tvm::tir::Var>& i) {
        auto erf_inp = x(i) * tvm::tir::make_const(dtype, std::sqrt(0.5));

        // erf branch: float16 casts through float32, others compute directly
        tvm::PrimExpr erf_val;
        if (dtype == DataType::Float(16)) {
          erf_val = tvm::cast(dtype, tvm::erf(tvm::cast(DataType::Float(32), erf_inp)));
        } else {
          erf_val = tvm::erf(erf_inp);
        }

        return x(i) *
               (tvm::tir::make_const(dtype, 0.5) + (erf_val * tvm::tir::make_const(dtype, 0.5)));
      },
      "T_gelu", topi::kElementWise)};
}

Expr LegalizeNNGelu(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  return m_te.Make(args, FTOPIHandler(GeluTE), std::string("gelu"));
}
TVM_REGISTER_OP("relax.nn.gelu")
    .set_attr<FLegalize>("FLegalize", LegalizeNNGelu, TVM_LEGALIZE_CPP_LEVEL);

// gelu_tanh
ffi::Array<te::Tensor> GeluTanhTE(const ffi::Array<ffi::Any> args) {
  auto x = args[0].cast<te::Tensor>();
  auto dtype = x->dtype;

  return {tvm::te::compute(
      x->shape,
      [&](const ffi::Array<tvm::tir::Var>& i) {
        auto xi = x(i);
        // tanh argument: sqrt(2/pi) * x * (1 + 0.044715 * x^2)
        auto tanh_arg =
            tvm::tir::make_const(dtype, std::sqrt(2.0 / M_PI)) * xi *
            (tvm::tir::make_const(dtype, 1.0) + tvm::tir::make_const(dtype, 0.044715) * xi * xi);
        // 0.5 * x * (1 + tanh(tanh_arg))
        return tvm::tir::make_const(dtype, 0.5) * xi *
               (tvm::tir::make_const(dtype, 1.0) + tvm::tanh(tanh_arg));
      },
      "T_gelu_tanh", topi::kElementWise)};
}

Expr LegalizeNNGeluTanh(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  return m_te.Make(args, FTOPIHandler(GeluTanhTE), std::string("gelu_tanh"));
}
TVM_REGISTER_OP("relax.nn.gelu_tanh")
    .set_attr<FLegalize>("FLegalize", LegalizeNNGeluTanh, TVM_LEGALIZE_CPP_LEVEL);

// selu
ffi::Array<te::Tensor> SeluTE(const ffi::Array<ffi::Any> args) {
  auto x = args[0].cast<te::Tensor>();
  auto dtype = x->dtype;

  constexpr double kAlpha = 1.6732632423543772848170429916717;
  constexpr double kScale = 1.0507009873554804934193349852946;

  auto exp_x = topi::exp(x);

  auto positive_part = tvm::te::compute(
      x->shape,
      [&](const ffi::Array<tvm::tir::Var>& i) {
        return tvm::max(x(i), tvm::tir::make_const(dtype, 0));
      },
      "selu_pos");

  auto negative_part = tvm::te::compute(
      x->shape,
      [&](const ffi::Array<tvm::tir::Var>& i) {
        return tvm::min(
            tvm::tir::make_const(dtype, 0),
            tvm::tir::make_const(dtype, kAlpha) * (exp_x(i) - tvm::tir::make_const(dtype, 1)));
      },
      "selu_neg");

  return {tvm::te::compute(
      x->shape,
      [&](const ffi::Array<tvm::tir::Var>& i) {
        return tvm::tir::make_const(dtype, kScale) * (positive_part(i) + negative_part(i));
      },
      "selu")};
}

Expr LegalizeNNSelu(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  return m_te.Make(args, FTOPIHandler(SeluTE), std::string("selu"));
}
TVM_REGISTER_OP("relax.nn.selu")
    .set_attr<FLegalize>("FLegalize", LegalizeNNSelu, TVM_LEGALIZE_CPP_LEVEL);

// silu
ffi::Array<te::Tensor> SiluTE(const ffi::Array<ffi::Any> args) {
  auto x = args[0].cast<te::Tensor>();
  auto sig = topi::sigmoid(x);
  return {tvm::te::compute(
      x->shape, [&](const ffi::Array<tvm::tir::Var>& i) { return x(i) * sig(i); }, "silu")};
}

Expr LegalizeNNSilu(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  return m_te.Make(args, FTOPIHandler(SiluTE), std::string("silu"));
}
TVM_REGISTER_OP("relax.nn.silu")
    .set_attr<FLegalize>("FLegalize", LegalizeNNSilu, TVM_LEGALIZE_CPP_LEVEL);

// softplus
ffi::Array<te::Tensor> SoftplusTE(const ffi::Array<ffi::Any> args) {
  auto x = args[0].cast<te::Tensor>();
  auto beta = args[1].cast<double>();
  auto threshold = args[2].cast<double>();
  auto dtype = x->dtype;

  return {tvm::te::compute(
      x->shape,
      [&](const ffi::Array<tvm::tir::Var>& i) {
        auto xi = x(i);
        auto beta_x = tvm::tir::make_const(dtype, beta) * xi;
        // When beta*x > threshold use the linear approximation x,
        // otherwise compute (1/beta) * log(1 + exp(beta*x)).
        auto softplus_val = tvm::tir::make_const(dtype, 1.0 / beta) *
                            tvm::log(tvm::tir::make_const(dtype, 1.0) + tvm::exp(beta_x));
        return tvm::if_then_else(beta_x > tvm::tir::make_const(dtype, threshold), xi, softplus_val);
      },
      "softplus")};
}

Expr LegalizeNNSoftplus(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<SoftplusAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(attrs->beta);
  args.push_back(attrs->threshold);
  return m_te.Make(args, FTOPIHandler(SoftplusTE), std::string("softplus"));
}
TVM_REGISTER_OP("relax.nn.softplus")
    .set_attr<FLegalize>("FLegalize", LegalizeNNSoftplus, TVM_LEGALIZE_CPP_LEVEL);

// softmax
ffi::Array<te::Tensor> SoftmaxTE(const ffi::Array<ffi::Any> args) {
  auto x = args[0].cast<te::Tensor>();
  int axis = args[1].cast<int>();

  int ndim = static_cast<int>(x->shape.size());
  if (axis < 0) axis += ndim;
  TVM_FFI_ICHECK(axis >= 0 && axis < ndim)
      << "softmax: axis " << axis << " out of range for input with " << ndim << " dims";

  // reduced_shape: shape with the axis dimension removed
  ffi::Array<tvm::PrimExpr> reduced_shape;
  for (int i = 0; i < ndim; ++i) {
    if (i != axis) reduced_shape.push_back(x->shape[i]);
  }

  // k1, k2: independent reduce axes over shape[axis]
  auto k1 = tvm::te::reduce_axis(tvm::Range(0, x->shape[axis]), "k");
  auto k2 = tvm::te::reduce_axis(tvm::Range(0, x->shape[axis]), "k");

  auto insert_reduce_index = [&](const ffi::Array<tvm::tir::Var>& indices,
                                 const tvm::tir::IterVar& reduce_var) -> ffi::Array<tvm::PrimExpr> {
    ffi::Array<tvm::PrimExpr> full;
    int ri = 0;
    for (int i = 0; i < ndim; ++i) {
      if (i == axis) {
        full.push_back(reduce_var);
      } else {
        full.push_back(indices[ri++]);
      }
    }
    return full;
  };

  auto get_non_reduce_indices =
      [&](const ffi::Array<tvm::tir::Var>& indices) -> ffi::Array<tvm::PrimExpr> {
    ffi::Array<tvm::PrimExpr> non_reduce;
    for (int i = 0; i < ndim; ++i) {
      if (i != axis) non_reduce.push_back(indices[i]);
    }
    return non_reduce;
  };

  // max_elem
  auto max_elem = tvm::te::compute(
      reduced_shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        return tvm::max(x(insert_reduce_index(indices, k1)), ffi::Array<tvm::tir::IterVar>(1, k1));
      },
      "T_softmax_maxelem");

  // exp
  auto exp = tvm::te::compute(
      x->shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        return tvm::exp(x(indices) - max_elem(get_non_reduce_indices(indices)));
      },
      "T_softmax_exp");

  // expsum
  auto expsum = tvm::te::compute(
      reduced_shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        return tvm::sum(exp(insert_reduce_index(indices, k2)),
                        ffi::Array<tvm::tir::IterVar>(1, k2));
      },
      "T_softmax_expsum");

  // output
  ffi::Map<ffi::String, ffi::Any> attrs;
  attrs.Set("axis", axis);
  return {tvm::te::compute(
      x->shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        return exp(indices) / expsum(get_non_reduce_indices(indices));
      },
      "T_softmax_norm", "softmax_output", attrs)};
}

Expr LegalizeNNSoftmax(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<SoftmaxAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(attrs->axis);
  return m_te.Make(args, FTOPIHandler(SoftmaxTE), std::string("softmax"));
}
TVM_REGISTER_OP("relax.nn.softmax")
    .set_attr<FLegalize>("FLegalize", LegalizeNNSoftmax, TVM_LEGALIZE_CPP_LEVEL);

// log_softmax
ffi::Array<te::Tensor> LogSoftmaxTE(const ffi::Array<ffi::Any> args) {
  auto x = args[0].cast<te::Tensor>();
  int axis = args[1].cast<int>();

  int ndim = static_cast<int>(x->shape.size());
  if (axis < 0) axis += ndim;
  TVM_FFI_ICHECK(axis >= 0 && axis < ndim)
      << "log_softmax: axis " << axis << " out of range for input with " << ndim << " dims";

  // reduced_shape: shape with the axis dimension removed
  ffi::Array<tvm::PrimExpr> reduced_shape;
  for (int i = 0; i < ndim; ++i) {
    if (i != axis) reduced_shape.push_back(x->shape[i]);
  }

  // k1, k2: independent reduce axes over shape[axis]
  auto k1 = tvm::te::reduce_axis(tvm::Range(0, x->shape[axis]), "k");
  auto k2 = tvm::te::reduce_axis(tvm::Range(0, x->shape[axis]), "k");

  auto insert_reduce_index = [&](const ffi::Array<tvm::tir::Var>& indices,
                                 const tvm::tir::IterVar& reduce_var) -> ffi::Array<tvm::PrimExpr> {
    ffi::Array<tvm::PrimExpr> full;
    int ri = 0;
    for (int i = 0; i < ndim; ++i) {
      if (i == axis) {
        full.push_back(reduce_var);
      } else {
        full.push_back(indices[ri++]);
      }
    }
    return full;
  };

  auto get_non_reduce_indices =
      [&](const ffi::Array<tvm::tir::Var>& indices) -> ffi::Array<tvm::PrimExpr> {
    ffi::Array<tvm::PrimExpr> non_reduce;
    for (int i = 0; i < ndim; ++i) {
      if (i != axis) non_reduce.push_back(indices[i]);
    }
    return non_reduce;
  };

  auto max_elem = tvm::te::compute(
      reduced_shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        return tvm::max(x(insert_reduce_index(indices, k1)), ffi::Array<tvm::tir::IterVar>(1, k1));
      },
      "T_softmax_maxelem");

  auto expsum = tvm::te::compute(
      reduced_shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        ffi::Array<tvm::PrimExpr> max_idx;
        for (auto v : indices) max_idx.push_back(v);
        return tvm::sum(tvm::exp(x(insert_reduce_index(indices, k2)) - max_elem(max_idx)),
                        ffi::Array<tvm::tir::IterVar>(1, k2));
      },
      "T_softmax_expsum");

  ffi::Map<ffi::String, ffi::Any> attrs;
  attrs.Set("axis", axis);
  return {tvm::te::compute(
      x->shape,
      [&](const ffi::Array<tvm::tir::Var>& indices) {
        auto non_reduce = get_non_reduce_indices(indices);
        return x(indices) - max_elem(non_reduce) - tvm::log(expsum(non_reduce));
      },
      "T_log_softmax_norm", "log_softmax_output", attrs)};
}

Expr LegalizeNNLogSoftmax(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<SoftmaxAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(attrs->axis);
  return m_te.Make(args, FTOPIHandler(LogSoftmaxTE), std::string("log_softmax"));
}
TVM_REGISTER_OP("relax.nn.log_softmax")
    .set_attr<FLegalize>("FLegalize", LegalizeNNLogSoftmax, TVM_LEGALIZE_CPP_LEVEL);

// cross_entropy_with_logits
ffi::Array<te::Tensor> CrossEntropyWithLogitsTE(const ffi::Array<ffi::Any> args) {
  auto x = args[0].cast<te::Tensor>();
  auto y = args[1].cast<te::Tensor>();

  auto xy = tvm::te::compute(
      x->shape, [&](const ffi::Array<tvm::tir::Var>& i) { return x(i) * y(i); },
      "cross_entropy_xy");

  auto sum_xy = topi::sum(xy, ffi::Array<Integer>(nullptr), /*keepdims=*/false);

  auto neg_sum = tvm::te::compute(
      sum_xy->shape, [&](const ffi::Array<tvm::tir::Var>& i) { return -sum_xy(i); },
      "cross_entropy_neg");

  if (x->shape.size() > 1) {
    return {tvm::te::compute(
        neg_sum->shape,
        [&](const ffi::Array<tvm::tir::Var>& i) {
          return neg_sum(i) / tvm::cast(x->dtype, x->shape[0]);
        },
        "cross_entropy_with_logits")};
  }
  return {neg_sum};
}

Expr LegalizeNNCrossEntropyWithLogits(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  return m_te.Make(args, FTOPIHandler(CrossEntropyWithLogitsTE),
                   std::string("cross_entropy_with_logits"));
}
TVM_REGISTER_OP("relax.nn.cross_entropy_with_logits")
    .set_attr<FLegalize>("FLegalize", LegalizeNNCrossEntropyWithLogits, TVM_LEGALIZE_CPP_LEVEL);

// layer_norm
ffi::Array<te::Tensor> LayerNormTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto gamma = args[1].cast<te::Tensor>();
  auto beta = args[2].cast<te::Tensor>();
  auto axis = args[3].cast<ffi::Array<Integer>>();
  auto epsilon = args[4].cast<double>();
  return {topi::nn::layer_norm(data, gamma, beta, axis, epsilon)};
}

Expr LegalizeNNLayerNorm(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<LayerNormAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(call->args[2]);
  args.push_back(attrs->axes);
  args.push_back(attrs->epsilon);
  return m_te.Make(args, FTOPIHandler(LayerNormTE), std::string("layer_norm"));
}
TVM_REGISTER_OP("relax.nn.layer_norm")
    .set_attr<FLegalize>("FLegalize", LegalizeNNLayerNorm, TVM_LEGALIZE_CPP_LEVEL);

// group_norm
ffi::Array<te::Tensor> GroupNormTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto gamma = args[1].cast<te::Tensor>();
  auto beta = args[2].cast<te::Tensor>();
  auto num_groups = args[3].cast<int>();
  auto channel_axis = args[4].cast<int>();
  auto axes = args[5].cast<ffi::Array<Integer>>();
  auto epsilon = args[6].cast<double>();
  return {topi::nn::group_norm(data, gamma, beta, num_groups, channel_axis, axes, epsilon)};
}

Expr LegalizeNNGroupNorm(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<GroupNormAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(call->args[2]);
  args.push_back(attrs->num_groups);
  args.push_back(attrs->channel_axis);
  args.push_back(attrs->axes);
  args.push_back(attrs->epsilon);
  return m_te.Make(args, FTOPIHandler(GroupNormTE), std::string("group_norm"));
}
TVM_REGISTER_OP("relax.nn.group_norm")
    .set_attr<FLegalize>("FLegalize", LegalizeNNGroupNorm, TVM_LEGALIZE_CPP_LEVEL);

// instance_norm
ffi::Array<te::Tensor> InstanceNormTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto gamma = args[1].cast<te::Tensor>();
  auto beta = args[2].cast<te::Tensor>();
  auto channel_axis = args[3].cast<int>();
  auto axis = args[4].cast<ffi::Array<Integer>>();
  auto epsilon = args[5].cast<double>();
  return {topi::nn::instance_norm(data, gamma, beta, channel_axis, axis, epsilon)};
}

Expr LegalizeNNInstanceNorm(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<InstanceNormAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(call->args[2]);
  args.push_back(attrs->channel_axis);
  args.push_back(attrs->axes);
  args.push_back(attrs->epsilon);
  return m_te.Make(args, FTOPIHandler(InstanceNormTE), std::string("instance_norm"));
}
TVM_REGISTER_OP("relax.nn.instance_norm")
    .set_attr<FLegalize>("FLegalize", LegalizeNNInstanceNorm, TVM_LEGALIZE_CPP_LEVEL);

// rms_norm
ffi::Array<te::Tensor> RmsNormTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto weight = args[1].cast<te::Tensor>();
  auto axis = args[2].cast<ffi::Array<Integer>>();
  auto epsilon = args[3].cast<double>();
  return {topi::nn::rms_norm(data, weight, axis, epsilon)};
}

Expr LegalizeNNRmsNorm(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<RMSNormAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(attrs->axes);
  args.push_back(attrs->epsilon);
  return m_te.Make(args, FTOPIHandler(RmsNormTE), std::string("rms_norm"));
}
TVM_REGISTER_OP("relax.nn.rms_norm")
    .set_attr<FLegalize>("FLegalize", LegalizeNNRmsNorm, TVM_LEGALIZE_CPP_LEVEL);

// nll_loss
ffi::Array<te::Tensor> NllLossTE(const ffi::Array<ffi::Any> args) {
  auto predictions = args[0].cast<te::Tensor>();
  auto targets = args[1].cast<te::Tensor>();
  auto weights = args[2].cast<te::Tensor>();
  auto reduction = std::string(args[3].cast<ffi::String>());
  auto ignore_index = args[4].cast<int>();
  return {topi::nll_loss(predictions, targets, weights, reduction, ignore_index)};
}

ffi::Array<te::Tensor> NllLossNoWeightTE(const ffi::Array<ffi::Any> args) {
  auto predictions = args[0].cast<te::Tensor>();
  auto targets = args[1].cast<te::Tensor>();
  auto reduction = std::string(args[2].cast<ffi::String>());
  auto ignore_index = args[3].cast<int>();

  auto num_classes =
      (predictions->shape.size() > 1) ? predictions->shape[1] : predictions->shape[0];
  auto weights =
      topi::full({num_classes}, predictions->dtype, tvm::tir::make_const(predictions->dtype, 1.0));

  return {topi::nll_loss(predictions, targets, weights, reduction, ignore_index)};
}

Expr LegalizeNNNllLoss(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<NLLLossAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;

  if (call->args.size() == 2) {
    args.push_back(call->args[0]);
    args.push_back(call->args[1]);
    args.push_back(attrs->reduction);
    args.push_back(attrs->ignore_index);
    return m_te.Make(args, FTOPIHandler(NllLossNoWeightTE), std::string("nll_loss_without_weight"));
  }

  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(call->args[2]);
  args.push_back(attrs->reduction);
  args.push_back(attrs->ignore_index);
  return m_te.Make(args, FTOPIHandler(NllLossTE), std::string("nll_loss"));
}
TVM_REGISTER_OP("relax.nn.nll_loss")
    .set_attr<FLegalize>("FLegalize", LegalizeNNNllLoss, TVM_LEGALIZE_CPP_LEVEL);

// batch_flatten
ffi::Array<te::Tensor> BatchFlattenTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto new_shape = args[1].cast<ffi::Array<tvm::PrimExpr>>();
  return {topi::reshape(data, new_shape)};
}

Expr LegalizeNNBatchFlatten(const BlockBuilder& bb, const Call& call) {
  auto sinfo = GetStructInfo(call).as<TensorStructInfoNode>();
  if (!sinfo || !sinfo->shape.defined()) {
    return call;
  }
  auto shape_values = sinfo->shape.as<ShapeExprNode>()->values;

  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(shape_values);
  return m_te.Make(args, FTOPIHandler(BatchFlattenTE), std::string("reshape"));
}
TVM_REGISTER_OP("relax.nn.batch_flatten")
    .set_attr<FLegalize>("FLegalize", LegalizeNNBatchFlatten, TVM_LEGALIZE_CPP_LEVEL);

// dropout
Expr LegalizeNNDropout(const BlockBuilder& bb, const Call& call) {
  LOG(INFO) << "Dropout is handled by frontend translator at this moment and is not legalized.";
  return call;
}
TVM_REGISTER_OP("relax.nn.dropout")
    .set_attr<FLegalize>("FLegalize", LegalizeNNDropout, TVM_LEGALIZE_CPP_LEVEL);

// attention / attention_bias / attention_var_len
Expr LegalizeNNAttentionVarLen(const BlockBuilder& bb, const Call& call) {
  LOG(FATAL) << "Legalization of attention_var_len op is not supported yet.";
  return call;
}
TVM_REGISTER_OP("relax.nn.attention_var_len")
    .set_attr<FLegalize>("FLegalize", LegalizeNNAttentionVarLen, TVM_LEGALIZE_CPP_LEVEL);

// batch_norm
ffi::Array<te::Tensor> BatchNormTE(const ffi::Array<ffi::Any> args) {
  auto data = args[0].cast<te::Tensor>();
  auto gamma = args[1].cast<te::Tensor>();
  auto beta = args[2].cast<te::Tensor>();
  auto moving_mean = args[3].cast<te::Tensor>();
  auto moving_var = args[4].cast<te::Tensor>();
  int axis = args[5].cast<int>();
  double epsilon = args[6].cast<double>();
  bool center = args[7].cast<bool>();
  bool scale = args[8].cast<bool>();
  bool training = args[9].cast<bool>();
  double momentum = args[10].cast<double>();

  int ndim = static_cast<int>(data->shape.size());
  if (axis < 0) axis += ndim;

  ffi::Array<tvm::PrimExpr> broadcast_shape;
  for (int i = 0; i < ndim; ++i) {
    broadcast_shape.push_back(i == axis ? data->shape[axis]
                                        : tvm::IntImm(data->shape[0]->dtype, 1));
  }

  ffi::Array<Integer> reduce_axes_arr;
  for (int i = 0; i < ndim; ++i) {
    if (i != axis) reduce_axes_arr.push_back(Integer(i));
  }

  tvm::PrimExpr shape_prod = tvm::IntImm(data->shape[0]->dtype, 1);
  for (int i = 0; i < ndim; ++i) {
    if (i != axis) shape_prod = shape_prod * data->shape[i];
  }
  shape_prod = tvm::cast(data->dtype, shape_prod);

  te::Tensor out;
  te::Tensor ret_mean;  // returned as output[1]
  te::Tensor ret_var;   // returned as output[2]

  if (training) {
    // data_mean: shape [C]  (sum over reduce_axes / shape_prod)
    auto sum_data = topi::sum(data, ffi::Optional<ffi::Array<Integer>>(reduce_axes_arr),
                              /*keepdims=*/false, /*atleast1d=*/true);
    auto data_mean = topi::divide(sum_data, shape_prod);

    // data_mean_rs: broadcast shape
    auto data_mean_rs = topi::reshape(data_mean, broadcast_shape);

    // centred: data - data_mean_rs  (broadcast)
    auto centred = topi::subtract(data, data_mean_rs);

    // data_var: sum(centred^2, reduce_axes) / shape_prod
    auto centred_sq = topi::multiply(centred, centred);
    auto sum_sq = topi::sum(centred_sq, ffi::Optional<ffi::Array<Integer>>(reduce_axes_arr),
                            /*keepdims=*/false, /*atleast1d=*/true);
    auto data_var = topi::divide(sum_sq, shape_prod);

    // data_var_rs: broadcast shape
    auto data_var_rs = topi::reshape(data_var, broadcast_shape);

    // out = centred / sqrt(data_var_rs + epsilon)
    auto var_eps = topi::add(data_var_rs, tvm::tir::make_const(data->dtype, epsilon));
    auto std_dev = topi::sqrt(var_eps);
    out = topi::divide(centred, std_dev);

    if (scale) {
      auto gamma_rs = topi::reshape(gamma, broadcast_shape);
      out = topi::multiply(out, gamma_rs);
    }
    if (center) {
      auto beta_rs = topi::reshape(beta, broadcast_shape);
      out = topi::add(out, beta_rs);
    }

    // new_moving_mean = (1 - momentum) * moving_mean + momentum * data_mean
    auto one_minus_mom = tvm::tir::make_const(data->dtype, 1.0 - momentum);
    auto mom = tvm::tir::make_const(data->dtype, momentum);
    ret_mean =
        topi::add(topi::multiply(one_minus_mom, moving_mean), topi::multiply(mom, data_mean));
    ret_var = topi::add(topi::multiply(one_minus_mom, moving_var), topi::multiply(mom, data_var));
  } else {
    // Inference path
    auto moving_mean_rs = topi::reshape(moving_mean, broadcast_shape);
    auto moving_var_rs = topi::reshape(moving_var, broadcast_shape);

    auto centred = topi::subtract(data, moving_mean_rs);
    auto var_eps = topi::add(moving_var_rs, tvm::tir::make_const(data->dtype, epsilon));
    auto std_dev = topi::sqrt(var_eps);
    out = topi::divide(centred, std_dev);

    if (scale) {
      auto gamma_rs = topi::reshape(gamma, broadcast_shape);
      out = topi::multiply(out, gamma_rs);
    }
    if (center) {
      auto beta_rs = topi::reshape(beta, broadcast_shape);
      out = topi::add(out, beta_rs);
    }

    // Moving mean and var aren't updated during inference.
    // Multiply by 1 to avoid placeholder reuse.
    auto one = tvm::tir::make_const(data->dtype, 1);
    ret_mean = topi::multiply(moving_mean, one);
    ret_var = topi::multiply(moving_var, one);
  }

  return {out, ret_mean, ret_var};
}

Expr LegalizeNNBatchNorm(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<BatchNormAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);  // data
  args.push_back(call->args[1]);  // gamma
  args.push_back(call->args[2]);  // beta
  args.push_back(call->args[3]);  // moving_mean
  args.push_back(call->args[4]);  // moving_var
  args.push_back(attrs->axis);
  args.push_back(attrs->epsilon);
  args.push_back(attrs->center);
  args.push_back(attrs->scale);
  args.push_back(attrs->training);
  args.push_back(attrs->momentum);
  return m_te.Make(args, FTOPIHandler(BatchNormTE), std::string("batch_norm"));
}
TVM_REGISTER_OP("relax.nn.batch_norm")
    .set_attr<FLegalize>("FLegalize", LegalizeNNBatchNorm, TVM_LEGALIZE_CPP_LEVEL);

}  // namespace relax
}  // namespace tvm
