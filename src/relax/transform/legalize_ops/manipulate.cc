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
 * \file tvm/relax/transform/legalize_ops/manipulate.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 */
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/topi/scatter.h>
#include <tvm/topi/transform.h>
#include <tvm/topi/utils.h>

#include "utils.h"

namespace tvm {
namespace relax {

#define TVM_LEGALIZE_MANIPULATE_RESHAPE(OpName, TopiHandler, PrimFuncName, CollapseSumLike) \
  Expr MAKE_NAME(LegalizeStatistical, OpName)(const BlockBuilder& bb, const Call& call) {   \
    auto m_te = MakeCallTE(bb, call);                                                       \
    tvm::ffi::Array<tvm::ffi::Any> args;                                                    \
    relax::Expr tgt_shape;                                                                  \
    if (CollapseSumLike) {                                                                  \
      tgt_shape = GetStructInfo(call->args[1]).as<TensorStructInfoNode>()->shape.value();   \
    } else {                                                                                \
      tgt_shape = call->args[1];                                                            \
    }                                                                                       \
    if (tgt_shape.as<VarNode>()) {                                                          \
      tgt_shape = bb->LookupBinding(tgt_shape.as<Var>().value()).value();                   \
      TVM_FFI_ICHECK_NOTNULL(tgt_shape.as<ShapeExprNode>());                                \
    }                                                                                       \
    args.push_back(call->args[0]);                                                          \
    args.push_back(tgt_shape);                                                              \
    auto call_ret = m_te.Make(args, ffi::String(#TopiHandler), std::string(#PrimFuncName)); \
    return call_ret;                                                                        \
  }                                                                                         \
  TVM_REGISTER_OP("relax." #OpName)                                                         \
      .set_attr<FLegalize>("FLegalize", MAKE_NAME(LegalizeStatistical, OpName),             \
                           TVM_LEGALIZE_CPP_LEVEL);

TVM_LEGALIZE_MANIPULATE_RESHAPE(broadcast_to, topi.broadcast_to, broadcast_to, false);
TVM_LEGALIZE_MANIPULATE_RESHAPE(reshape, topi.reshape, reshape, false);
TVM_LEGALIZE_MANIPULATE_RESHAPE(collapse_sum_like, topi.collapse_sum, collapse_sum, true);
TVM_LEGALIZE_MANIPULATE_RESHAPE(collapse_sum_to, topi.collapse_sum, collapse_sum, false);

// Concat
Expr LegalizeConcat(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  relax::Expr t = call->args[0];
  size_t n_fields = GetStructInfo(t).as<TupleStructInfoNode>()->fields.size();
  while (t.as<VarNode>()) {
    auto binding = bb->LookupBinding(Downcast<Var>(t));
    if (!(binding.as<TupleNode>() || binding.as<VarNode>())) {
      break;
    }
    t = binding.value();
  }
  tvm::ffi::Array<tvm::ffi::Any> args;

  if (auto tpl = t.as<TupleNode>()) {
    args.push_back(tpl->fields);
  } else if (t.as<VarNode>()) {
    tvm::ffi::Array<Expr> t_fields;
    for (size_t i = 0; i < n_fields; ++i) {
      t_fields.push_back(bb->Emit(TupleGetItem(t, i)));
    }
    args.push_back(t_fields);
  } else {
    LOG(FATAL) << "Expected Tuple or Var as argument for concat, but received " << t;
  }
  const auto* attrs = call->attrs.as<ConcatAttrs>();
  if (attrs->axis.has_value()) {
    args.push_back(attrs->axis.value());
  } else {
    args.push_back(0);
  }
  auto call_ret = m_te.Make(args, ffi::String("topi.concatenate"), std::string("concatenate"));
  return call_ret;
}
TVM_REGISTER_OP("relax.concat")
    .set_attr<FLegalize>("FLegalize", LegalizeConcat, TVM_LEGALIZE_CPP_LEVEL);

// ExpandDims
Expr LegalizeExpandDims(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<ExpandDimsAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(attrs->axis);

  std::function<ffi::Array<te::Tensor>(const ffi::Array<ffi::Any>)> _te_expand_dims =
      [&](const ffi::Array<ffi::Any> args) -> ffi::Array<te::Tensor> {
    te::Tensor data = args[0].cast<te::Tensor>();
    auto axis = tvm::topi::ArrayOrInt(args[1]).value();
    ffi::Array<te::Tensor> ret;

    tvm::OpAttrMap<FInferStructInfo> op_map_infer_struct_info_ =
        Op::GetAttrMap<FInferStructInfo>("FInferStructInfo");

    auto data_relax =
        relax::Var("data", relax::TensorStructInfo(relax::ShapeExpr(data->shape), data->dtype));
    auto* op_ptr = call->op.as<OpNode>();
    Op op = ffi::GetRef<Op>(op_ptr);
    TVM_FFI_ICHECK(op_map_infer_struct_info_.count(op))
        << " Cannot find the FInferStructInfo attribute registered to op: " << op->name;
    ffi::ObjectPtr<relax::ExpandDimsAttrs> t_attrs = ffi::make_object<relax::ExpandDimsAttrs>();
    t_attrs->axis = axis;

    auto t_call =
        relax::Call(tvm::Op::Get("relax.expand_dims"), {data_relax}, tvm::Attrs(t_attrs), {});
    auto out_sinfo = op_map_infer_struct_info_[op](t_call, bb);
    auto out_shape = out_sinfo.as<TensorStructInfoNode>()->shape.as<ShapeExprNode>()->values;
    auto output_ndim = out_shape.size();

    ffi::Array<int64_t> data_dims;
    for (size_t i = 0; i < output_ndim; ++i) {
      bool found1 = true;
      bool found2 = true;
      for (size_t j = 0; j < axis.size(); ++j) {
        if (axis[j] == i) {
          found1 = false;
        }
        if (axis[j] == (i - output_ndim)) {
          found2 = false;
        }
      }
      if (found1 && found2) {
        data_dims.push_back(i);
      }
    }

    return {tvm::te::compute(
        out_shape,
        [&](const ffi::Array<tvm::tir::Var>& indices) {
          ffi::Array<PrimExpr> idx;
          for (size_t i = 0; i < data_dims.size(); ++i) {
            idx.push_back(indices[data_dims[i]]);
          }
          return data(idx);
        },
        "expand_dims")};
  };
  auto call_ret = m_te.Make(args, FTOPIHandler(_te_expand_dims), std::string("expand_dims"));
  return call_ret;
}
TVM_REGISTER_OP("relax.expand_dims")
    .set_attr<FLegalize>("FLegalize", LegalizeExpandDims, TVM_LEGALIZE_CPP_LEVEL);

// Flatten
Expr LegalizeFlatten(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(
      GetStructInfo(call).as<TensorStructInfoNode>()->shape.value().as<ShapeExprNode>()->values);
  auto call_ret = m_te.Make(args, ffi::String("topi.reshape"), std::string("reshape"));
  return call_ret;
}
TVM_REGISTER_OP("relax.flatten")
    .set_attr<FLegalize>("FLegalize", LegalizeFlatten, TVM_LEGALIZE_CPP_LEVEL);

// PermuteDims
Expr LegalizePermuteDims(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<PermuteDimsAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(attrs->axes);
  auto call_ret = m_te.Make(args, ffi::String("topi.transpose"), std::string("transpose"));
  return call_ret;
}
TVM_REGISTER_OP("relax.permute_dims")
    .set_attr<FLegalize>("FLegalize", LegalizePermuteDims, TVM_LEGALIZE_CPP_LEVEL);

// Split
Expr LegalizeSplit(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<SplitAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);

  std::function<ffi::Array<te::Tensor>(const ffi::Array<ffi::Any>)> _te_split =
      [&](const ffi::Array<ffi::Any> args) -> ffi::Array<te::Tensor> {
    te::Tensor data = args[0].cast<te::Tensor>();
    if (auto splits = attrs->indices_or_sections.as<tvm::IntImmNode>()) {
      return tvm::topi::split_n_sections(data, splits->value, attrs->axis);
    } else if (auto indices = attrs->indices_or_sections.as<ffi::Array<PrimExpr>>()) {
      return tvm::topi::split_indices_array(data, indices.value(), attrs->axis);
    } else {
      LOG(FATAL) << "Unexpected argument for split:" << attrs->indices_or_sections;
      return {};
    }
  };
  auto call_ret = m_te.Make(args, FTOPIHandler(_te_split), std::string("split"));
  return call_ret;
}
TVM_REGISTER_OP("relax.split")
    .set_attr<FLegalize>("FLegalize", LegalizeSplit, TVM_LEGALIZE_CPP_LEVEL);

// Squeeze
Expr LegalizeSqueeze(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<SqueezeAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(attrs->axis);
  auto call_ret = m_te.Make(args, ffi::String("topi.squeeze"), std::string("squeeze"));
  return call_ret;
}
TVM_REGISTER_OP("relax.squeeze")
    .set_attr<FLegalize>("FLegalize", LegalizeSqueeze, TVM_LEGALIZE_CPP_LEVEL);

// Stack
Expr LegalizeStack(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  relax::Expr t = call->args[0];
  size_t n_fields = GetStructInfo(t).as<TupleStructInfoNode>()->fields.size();
  while (t.as<VarNode>()) {
    auto binding = bb->LookupBinding(Downcast<Var>(t));
    if (!(binding.as<TupleNode>() || binding.as<VarNode>())) {
      break;
    }
    t = binding.value();
  }
  tvm::ffi::Array<tvm::ffi::Any> args;

  if (auto tpl = t.as<TupleNode>()) {
    args.push_back(tpl->fields);
  } else if (t.as<VarNode>()) {
    tvm::ffi::Array<Expr> t_fields;
    for (size_t i = 0; i < n_fields; ++i) {
      t_fields.push_back(bb->Emit(TupleGetItem(t, i)));
    }
    args.push_back(t_fields);
  } else {
    LOG(FATAL) << "Expected Tuple or Var as argument for stack, but received " << t;
  }
  const auto* attrs = call->attrs.as<StackAttrs>();
  if (attrs->axis.has_value()) {
    args.push_back(attrs->axis.value()->value);
  } else {
    args.push_back(0);
  }
  auto call_ret = m_te.Make(args, ffi::String("topi.stack"), std::string("stack"));
  return call_ret;
}
TVM_REGISTER_OP("relax.stack")
    .set_attr<FLegalize>("FLegalize", LegalizeStack, TVM_LEGALIZE_CPP_LEVEL);

// Repeat
Expr LegalizeRepeat(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<RepeatAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);

  std::function<ffi::Array<te::Tensor>(const ffi::Array<ffi::Any>)> _te_repeat =
      [&](const ffi::Array<ffi::Any> args) -> ffi::Array<te::Tensor> {
    te::Tensor data = args[0].cast<te::Tensor>();
    if (attrs->axis.has_value()) {
      return {tvm::topi::repeat(data, attrs->repeats, attrs->axis.value())};
    } else {
      auto shape_prod = tvm::tir::make_const(data->shape[0]->dtype, 1);
      for (size_t i = 0; i < data->shape.size(); ++i) {
        shape_prod = shape_prod * data->shape[i];
      }
      return {tvm::topi::repeat(tvm::topi::reshape(data, ffi::Array<PrimExpr>({shape_prod})),
                                attrs->repeats, 0)};
    }
  };

  auto call_ret = m_te.Make(args, FTOPIHandler(_te_repeat), std::string("repeat"));
  return call_ret;
}
TVM_REGISTER_OP("relax.repeat")
    .set_attr<FLegalize>("FLegalize", LegalizeRepeat, TVM_LEGALIZE_CPP_LEVEL);

// Tile
Expr LegalizeTile(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<TileAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(attrs->repeats);
  auto call_ret = m_te.Make(args, ffi::String("topi.tile"), std::string("tile"));
  return call_ret;
}
TVM_REGISTER_OP("relax.tile")
    .set_attr<FLegalize>("FLegalize", LegalizeTile, TVM_LEGALIZE_CPP_LEVEL);

// Flip
Expr LegalizeFlip(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<FlipAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(attrs->axis->value);
  auto call_ret = m_te.Make(args, ffi::String("topi.flip"), std::string("flip"));
  return call_ret;
}
TVM_REGISTER_OP("relax.flip")
    .set_attr<FLegalize>("FLegalize", LegalizeFlip, TVM_LEGALIZE_CPP_LEVEL);

// Gather Elements
Expr LegalizeGatherElements(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<GatherElementsAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(attrs->axis->value);
  args.push_back(call->args[1]);
  auto call_ret = m_te.Make(args, ffi::String("topi.gather"), std::string("gather"));
  return call_ret;
}
TVM_REGISTER_OP("relax.gather_elements")
    .set_attr<FLegalize>("FLegalize", LegalizeGatherElements, TVM_LEGALIZE_CPP_LEVEL);

// Gather ND
Expr LegalizeGatherND(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<GatherNDAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(call->args[1]);

  std::function<ffi::Array<te::Tensor>(const ffi::Array<ffi::Any>)> _te_gather_nd =
      [&](const ffi::Array<ffi::Any> args) -> ffi::Array<te::Tensor> {
    te::Tensor data = args[0].cast<te::Tensor>();
    te::Tensor indices = args[1].cast<te::Tensor>();
    int64_t indices_ndim = indices->shape.size();
    ffi::Array<Integer> axes;
    axes.push_back(indices_ndim - 1);
    for (int64_t i = 0; i < (indices_ndim - 1); ++i) {
      axes.push_back(i);
    }
    return {
        tvm::topi::gather_nd(data, tvm::topi::transpose(indices, axes), attrs->batch_dims->value)};
  };

  auto call_ret = m_te.Make(args, FTOPIHandler(_te_gather_nd), std::string("gather_nd"));
  return call_ret;
}
TVM_REGISTER_OP("relax.gather_nd")
    .set_attr<FLegalize>("FLegalize", LegalizeGatherND, TVM_LEGALIZE_CPP_LEVEL);

// IndexTensor
Expr LegalizeIndexTensor(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  relax::Expr t = call->args[1];
  size_t n_fields = GetStructInfo(t).as<TupleStructInfoNode>()->fields.size();
  tvm::ffi::Array<Expr> t_fields;
  for (size_t i = 0; i < n_fields; ++i) {
    t_fields.push_back(bb->Emit(TupleGetItem(t, i)));
  }
  args.push_back(t_fields);
  auto call_ret = m_te.Make(args, ffi::String("topi.index_tensor"), std::string("index_tensor"));
  return call_ret;
}
TVM_REGISTER_OP("relax.index_tensor")
    .set_attr<FLegalize>("FLegalize", LegalizeIndexTensor, TVM_LEGALIZE_CPP_LEVEL);

// IndexPut
Expr LegalizeIndexPut(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<IndexPutAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);

  relax::Expr t = call->args[1];
  tvm::ffi::Array<Expr> indices_list;
  if (GetStructInfo(t).as<TupleStructInfoNode>()) {
    size_t n_fields = GetStructInfo(t).as<TupleStructInfoNode>()->fields.size();
    for (size_t i = 0; i < n_fields; ++i) {
      indices_list.push_back(bb->Emit(TupleGetItem(t, i)));
    }
  } else {
    indices_list.push_back(t);
  }

  args.push_back(indices_list);
  args.push_back(call->args[2]);
  args.push_back(attrs->accumulate);
  auto call_ret = m_te.Make(args, ffi::String("topi.index_put"), std::string("index_put"));
  return call_ret;
}
TVM_REGISTER_OP("relax.index_put")
    .set_attr<FLegalize>("FLegalize", LegalizeIndexPut, TVM_LEGALIZE_CPP_LEVEL);

// Meshgrid
Expr LegalizeMeshgrid(const BlockBuilder& bb, const Call& call) {
  auto m_te = MakeCallTE(bb, call);
  relax::Expr t = call->args[0];
  size_t n_fields = GetStructInfo(t).as<TupleStructInfoNode>()->fields.size();
  while (t.as<VarNode>()) {
    auto binding = bb->LookupBinding(Downcast<Var>(t));
    if (!(binding.as<TupleNode>() || binding.as<VarNode>())) {
      break;
    }
    t = binding.value();
  }
  tvm::ffi::Array<tvm::ffi::Any> args;

  if (auto tpl = t.as<TupleNode>()) {
    args.push_back(tpl->fields);
  } else if (t.as<VarNode>()) {
    tvm::ffi::Array<Expr> t_fields;
    for (size_t i = 0; i < n_fields; ++i) {
      t_fields.push_back(bb->Emit(TupleGetItem(t, i)));
    }
    args.push_back(t_fields);
  } else {
    LOG(FATAL) << "Expected Tuple or Var as argument for meshgrid, but received " << t;
  }

  const auto* attrs = call->attrs.as<MeshgridAttrs>();
  if (attrs->indexing.has_value()) {
    args.push_back(attrs->indexing.value());
  } else {
    args.push_back("ij");
  }
  auto call_ret = m_te.Make(args, ffi::String("topi.meshgrid"), std::string("meshgrid"));
  return call_ret;
}
TVM_REGISTER_OP("relax.meshgrid")
    .set_attr<FLegalize>("FLegalize", LegalizeMeshgrid, TVM_LEGALIZE_CPP_LEVEL);

// ScatterElements
Expr LegalizeScatterElements(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<ScatterElementsAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);
  args.push_back(call->args[1]);
  args.push_back(call->args[2]);

  std::function<ffi::Array<te::Tensor>(const ffi::Array<ffi::Any>)> _te_scatter_elements =
      [&](const ffi::Array<ffi::Any> te_args) -> ffi::Array<te::Tensor> {
    te::Tensor data = te_args[0].cast<te::Tensor>();
    te::Tensor indices = te_args[1].cast<te::Tensor>();
    te::Tensor updates = te_args[2].cast<te::Tensor>();
    int axis = static_cast<int>(attrs->axis.as<IntImmNode>()->value);
    return {tvm::topi::scatter_elements(data, indices, updates, axis, attrs->reduction)};
  };

  auto call_ret =
      m_te.Make(args, FTOPIHandler(_te_scatter_elements), std::string("scatter_elements"));
  return call_ret;
}
TVM_REGISTER_OP("relax.scatter_elements")
    .set_attr<FLegalize>("FLegalize", LegalizeScatterElements, TVM_LEGALIZE_CPP_LEVEL);

// Scatter ND
// The Relax op passes indices in (..., M) layout; topi::scatter_nd expects
// (M, ...).  We transpose here before calling the native C++ implementation.
Expr LegalizeScatterND(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<ScatterNDAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);  // data
  args.push_back(call->args[1]);  // indices (transposed inside handler)
  args.push_back(call->args[2]);  // updates

  std::function<ffi::Array<te::Tensor>(const ffi::Array<ffi::Any>)> _te_scatter_nd =
      [&](const ffi::Array<ffi::Any> te_args) -> ffi::Array<te::Tensor> {
    te::Tensor data = te_args[0].cast<te::Tensor>();
    te::Tensor indices = te_args[1].cast<te::Tensor>();
    te::Tensor updates = te_args[2].cast<te::Tensor>();
    // Transpose indices: move last axis to front (Python: axes[-1:] + axes[:-1])
    int64_t indices_ndim = static_cast<int64_t>(indices->shape.size());
    ffi::Array<Integer> axes;
    axes.push_back(indices_ndim - 1);
    for (int64_t i = 0; i < indices_ndim - 1; ++i) axes.push_back(i);
    te::Tensor indices_t = tvm::topi::transpose(indices, axes);
    return {tvm::topi::scatter_nd(data, indices_t, updates, attrs->reduction)};
  };

  auto call_ret = m_te.Make(args, FTOPIHandler(_te_scatter_nd), std::string("scatter_nd"));
  return call_ret;
}
TVM_REGISTER_OP("relax.scatter_nd")
    .set_attr<FLegalize>("FLegalize", LegalizeScatterND, TVM_LEGALIZE_CPP_LEVEL);

// SliceScatter
Expr LegalizeSliceScatter(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<SliceScatterAttrs>();
  auto m_te = MakeCallTE(bb, call);
  tvm::ffi::Array<tvm::ffi::Any> args;
  args.push_back(call->args[0]);  // input
  args.push_back(call->args[1]);  // src

  std::function<ffi::Array<te::Tensor>(const ffi::Array<ffi::Any>)> _te_slice_scatter =
      [&](const ffi::Array<ffi::Any> te_args) -> ffi::Array<te::Tensor> {
    te::Tensor input = te_args[0].cast<te::Tensor>();
    te::Tensor src = te_args[1].cast<te::Tensor>();
    // start / end / step are PrimValue scalars passed as PrimExpr constants
    int start = static_cast<int>(call->args[2].as<PrimValueNode>()->value.as<IntImmNode>()->value);
    int end = static_cast<int>(call->args[3].as<PrimValueNode>()->value.as<IntImmNode>()->value);
    int step = static_cast<int>(call->args[4].as<PrimValueNode>()->value.as<IntImmNode>()->value);
    int axis = attrs->axis;
    return {tvm::topi::slice_scatter(input, src, start, end, step, axis)};
  };

  return m_te.Make(args, FTOPIHandler(_te_slice_scatter), std::string("slice_scatter"));
}
TVM_REGISTER_OP("relax.slice_scatter")
    .set_attr<FLegalize>("FLegalize", LegalizeSliceScatter, TVM_LEGALIZE_CPP_LEVEL);

// OneHot
Expr LegalizeOneHot(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<OneHotAttrs>();
  auto m_te = MakeCallTE(bb, call);
  auto indices = call->args[0];
  auto on_value = call->args[1];
  auto off_value = call->args[2];
  TVM_FFI_ICHECK_NOTNULL(on_value.as<relax::PrimValueNode>());
  TVM_FFI_ICHECK_NOTNULL(off_value.as<relax::PrimValueNode>());

  auto on_value_prim = on_value.as<relax::PrimValueNode>()->value;
  auto off_value_prim = off_value.as<relax::PrimValueNode>()->value;

  TVM_FFI_ICHECK(on_value_prim->dtype == off_value_prim->dtype)
      << "OneHot on and off values should have same dtype";

  tvm::ffi::Array<tvm::ffi::Any> args = {indices,      on_value_prim, off_value_prim,
                                         attrs->depth, attrs->axis,   on_value_prim->dtype};

  auto call_ret = m_te.Make(args, ffi::String("topi.one_hot"), std::string("one_hot"));
  return call_ret;
}
TVM_REGISTER_OP("relax.one_hot")
    .set_attr<FLegalize>("FLegalize", LegalizeOneHot, TVM_LEGALIZE_CPP_LEVEL);

// LayoutTransform
Expr LegalizeLayoutTransform(const BlockBuilder& bb, const Call& call) {
  const auto* attrs = call->attrs.as<LayoutTransformAttrs>();
  auto m_te = MakeCallTE(bb, call);
  auto index_map = attrs->index_map;
  PrimValue pad_value;
  if (attrs->pad_value.has_value()) {
    pad_value = attrs->pad_value.value();
  } else {
    if (GetStructInfo(call->args[0]).as<TensorStructInfoNode>()->dtype.is_float()) {
      pad_value = PrimValue(PrimExpr(0.0f));
    } else {
      pad_value = PrimValue(PrimExpr(0));
    }
  }
  std::function<ffi::Array<PrimExpr>(const ffi::Array<tir::Var>)> _pad_func =
      [&](const ffi::Array<tir::Var> args) -> ffi::Array<PrimExpr> { return {pad_value->value}; };
  auto pad_value_index_map =
      tvm::tir::IndexMap::FromFunc(index_map->final_indices.size(), _pad_func);

  auto axis_separators = attrs->axis_separators;
  auto input_axis_separators = attrs->input_axis_separators;

  std::string primfunc_name = "te_layout_transform";
  ffi::Array<Range> initial_ranges;
  for (auto dim :
       GetStructInfo(call->args[0]).as<TensorStructInfoNode>()->shape.as<ShapeExprNode>()->values) {
    initial_ranges.push_back(Range(PrimExpr(0), dim));
  }

  arith::Analyzer analyzer;
  auto [map, padding_predicate] = index_map.NonSurjectiveInverse(initial_ranges, &analyzer);
  if (!padding_predicate.as<tvm::IntImmNode>()) {
    primfunc_name += "_with_pad";
  }
  if (axis_separators.has_value() && (axis_separators.value().size() != 0)) {
    primfunc_name += "_axis_separator";
  }

  tvm::ffi::Array<tvm::ffi::Any> args = {call->args[0]};
  std::function<ffi::Array<te::Tensor>(const ffi::Array<ffi::Any>)> _te_layout_transform =
      [&](const ffi::Array<ffi::Any> args) -> ffi::Array<te::Tensor> {
    te::Tensor data = args[0].cast<te::Tensor>();
    return {tvm::te::compute(
        data->shape, [&](const ffi::Array<tvm::tir::Var>& indices) { return data(indices); },
        primfunc_name)};
  };

  auto [tir_func, call_tir_args, output_sinfo, tir_vars] =
      m_te.GenCallTirInputs(args, FTOPIHandler(_te_layout_transform), primfunc_name);

  // Schedule
  tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> func_map;
  func_map.Set(GlobalVar("main"), tir_func);
  auto sch = s_tir::Schedule::Traced(IRModule(func_map), -1, 0,
                                     s_tir::ScheduleErrorRenderLevel::kDetail, true);
  auto s_block = sch->GetSBlock(primfunc_name, std::nullopt);
  auto block_obj = sch->Get(s_block);
  auto block_name = block_obj->name_hint;
  auto buffer_obj = block_obj->writes[0]->buffer;

  sch->TransformLayout(s_block, 0, tvm::s_tir::BufferIndexType::kWrite, index_map,
                       pad_value_index_map, false);
  if (axis_separators.has_value()) {
    sch->SetAxisSeparator(s_block, 0, tvm::s_tir::BufferIndexType::kWrite, axis_separators.value());
  }
  if (input_axis_separators.has_value()) {
    sch->SetAxisSeparator(s_block, 0, tvm::s_tir::BufferIndexType::kRead,
                          input_axis_separators.value());
  }

  // Build call
  auto sch_func = Downcast<tir::PrimFunc>(sch->mod()->Lookup("main"));
  tir_func = tvm::WithoutAttr(sch_func, tvm::attr::kGlobalSymbol);
  auto gvar = bb->AddFunction(tir_func, primfunc_name);
  static const Op& call_tir_op_ = Op::Get("relax.call_tir");
  ffi::Array<Expr> call_args = {gvar, Tuple(call_tir_args)};
  if (tir_vars.size() > 0) {
    call_args.push_back(ShapeExpr(tir_vars));
  }
  auto output_shape = index_map->MapShape(
      GetStructInfo(call->args[0]).as<TensorStructInfoNode>()->shape.as<ShapeExprNode>()->values,
      &analyzer);
  auto out_sinfo = TensorStructInfo(ShapeExpr(output_shape),
                                    GetStructInfo(call->args[0]).as<TensorStructInfoNode>()->dtype);

  return Call(call_tir_op_, call_args, tvm::Attrs(), {out_sinfo});
}
TVM_REGISTER_OP("relax.layout_transform")
    .set_attr<FLegalize>("FLegalize", LegalizeLayoutTransform, TVM_LEGALIZE_CPP_LEVEL);

}  // namespace relax
}  // namespace tvm
