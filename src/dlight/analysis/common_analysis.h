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

#ifndef TVM_DLIGHT_ANALYSIS_COMMON_ANALYSIS_H_
#define TVM_DLIGHT_ANALYSIS_COMMON_ANALYSIS_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/schedule/schedule.h>

namespace tvm {
namespace dlight {

class DlightIterInfoNode : public runtime::Object {
 public:
  tvm::tir::IterVar iter_;
  s_tir::LoopRV loop_rv_;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DlightIterInfoNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dl.DlightIterInfo", DlightIterInfoNode, runtime::Object);
};

class DlightIterInfo : public runtime::ObjectRef {
 public:
  /*\brief  Constructor */
  DlightIterInfo(tvm::tir::IterVar iter, s_tir::LoopRV loop_rv);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DlightIterInfo, runtime::ObjectRef,
                                                DlightIterInfoNode);
};

class DlightSBlockInfoNode : public runtime::Object {
 public:
  ffi::String name_;
  ffi::Array<DlightIterInfo> iters_;
  s_tir::SBlockRV block_rv_;
  bool reduction_block_;

  ffi::String DomKind(void) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DlightSBlockInfoNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dl.DlightSBlockInfo", DlightSBlockInfoNode, runtime::Object);
};

class DlightSBlockInfo : public runtime::ObjectRef {
 public:
  /*\brief  Constructor */
  DlightSBlockInfo(ffi::String name, ffi::Array<DlightIterInfo> iters, s_tir::SBlockRV block_rv,
                   bool reduction_block);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DlightSBlockInfo, runtime::ObjectRef,
                                                DlightSBlockInfoNode);
};

ffi::Array<DlightSBlockInfo> DlightNormalizePrimFunc(const s_tir::Schedule& sch);

}  // namespace dlight
}  // namespace tvm

#endif  // TVM_DLIGHT_ANALYSIS_COMMON_ANALYSIS_H_
