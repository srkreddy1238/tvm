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
#include <tvm/tir/stmt.h>

#include <string>
#include <vector>

namespace tvm {
namespace dlight {

/*!
 * \brief Information about one loop / iter-var of a block.
 *
 * Fields:
 *   kind_     – 'S' (DataPar), 'R' (CommReduce), or 'O' (other)
 *   var_      – the TIR Var of the iter-var
 *   dom_      – the extent of the iteration domain (PrimExpr)
 *   loop_rv_  – the LoopRV of the surrounding for-loop
 */
class DlightIterInfoNode : public runtime::Object {
 public:
  char kind_;              // 'S', 'R', or 'O'
  tir::Var var_;           // iter-var's TIR variable
  PrimExpr dom_;           // extent of the iteration domain
  s_tir::LoopRV loop_rv_;  // surrounding loop

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DlightIterInfoNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dl.DlightIterInfo", DlightIterInfoNode, runtime::Object);
};

class DlightIterInfo : public runtime::ObjectRef {
 public:
  /*!
   * \brief Construct a DlightIterInfo.
   *
   * \param kind    Iteration kind: 'S' (spatial), 'R' (reduction), 'O' (other).
   * \param var     The TIR variable of the iter-var.
   * \param dom     The extent of the iteration domain.
   * \param loop_rv The surrounding for-loop.
   */
  DlightIterInfo(char kind, tir::Var var, PrimExpr dom, s_tir::LoopRV loop_rv);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DlightIterInfo, runtime::ObjectRef,
                                                DlightIterInfoNode);
};

/*!
 * \brief Per-buffer analysis information.
 *
 * For each dimension of the buffer's access region records which LoopRV
 * (if any) drives that dimension, enabling identification of the vector
 * loop and computation of the vectorisation width.
 */
class DlightBufferInfoNode : public runtime::Object {
 public:
  /*! \brief The buffer region this info describes. */
  tir::BufferRegion buf_region_;
  /*!
   * \brief For each dimension of buf_region_->region, the LoopRV whose
   *        loop-var drives that dimension, or nullopt when the dimension
   *        is not driven by a single loop variable.
   */
  std::vector<ffi::Optional<s_tir::LoopRV>> assoc_lps_;

  /*! \brief Return the storage scope of the underlying buffer. */
  std::string GetScope() const;

  /*!
   * \brief Compute the vectorisation width for this buffer.
   *
   * \param vbits  Target vector register width in bits (default 128).
   * \return The vectorisation width, or 1 if it cannot be determined.
   */
  int GetVecSize(int vbits = 128) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DlightBufferInfoNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dl.DlightBufferInfo", DlightBufferInfoNode, runtime::Object);
};

class DlightBufferInfo : public runtime::ObjectRef {
 public:
  /*!
   * \brief Construct a DlightBufferInfo for one buffer region of a block.
   *
   * \param sch        The schedule that owns the block.
   * \param block_rv   The block whose buffer region is being analysed.
   * \param buf_region The specific buffer region (read or write).
   * \param loops      The loops surrounding the block (from sch->GetLoops).
   */
  DlightBufferInfo(const s_tir::Schedule& sch, const s_tir::SBlockRV& block_rv,
                   const tir::BufferRegion& buf_region, const ffi::Array<s_tir::LoopRV>& loops);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DlightBufferInfo, runtime::ObjectRef,
                                                DlightBufferInfoNode);
};

/*!
 * \brief Information about a TIR block.
 *
 * Use GetSBlockInfo(sch, block_rv) to construct.
 *
 * Fields:
 *   name_       – block name hint
 *   iters_      – one DlightIterInfo per iter-var (zipped with GetLoops)
 *   block_rv_   – the SBlockRV handle
 *   read_bufs_  – DlightBufferInfo for each read buffer
 *   write_bufs_ – DlightBufferInfo for each write buffer
 *   producers_  – producer SBlockRVs
 *   consumers_  – consumer SBlockRVs
 */
class DlightSBlockInfoNode : public runtime::Object {
 public:
  ffi::String name_;
  ffi::Array<DlightIterInfo> iters_;
  s_tir::SBlockRV block_rv_;
  ffi::Array<DlightBufferInfo> read_bufs_;
  ffi::Array<DlightBufferInfo> write_bufs_;
  ffi::Array<s_tir::SBlockRV> producers_;
  ffi::Array<s_tir::SBlockRV> consumers_;

  /*! \brief Returns the iteration domain kind string, e.g. "SSSR". */
  ffi::String DomKind() const;

  /*! \brief Returns true when all iteration domains are spatial ('S'). */
  bool IsInjective() const;

  /*! \brief Returns true when the block has at least one reduction iter and no opaque iters. */
  bool IsReduction() const;

  /*!
   * \brief Returns true when the block is elementwise:
   *        injective, exactly one write buffer, and read/write regions match.
   */
  bool IsElementwise() const;

  /*!
   * \brief Returns true when the block matches the data-pad pattern:
   *        injective, one read and one write buffer of equal rank, with an
   *        IfThenElse node in the body.
   *
   * \param sch  The schedule owning this block.
   */
  bool IsDataPad(const s_tir::Schedule& sch) const;

  /*!
   * \brief Returns true when the block is a layout transform:
   *        injective, one read and one write buffer, not elementwise,
   *        and no IfThenElse node in the body.
   *
   * \param sch  The schedule owning this block.
   */
  bool IsLayoutTransform(const s_tir::Schedule& sch) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DlightSBlockInfoNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("dl.DlightSBlockInfo", DlightSBlockInfoNode, runtime::Object);
};

class DlightSBlockInfo : public runtime::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DlightSBlockInfo, runtime::ObjectRef,
                                             DlightSBlockInfoNode);
};

/*!
 * \brief Construct a DlightSBlockInfo for a single block.
 *
 * Uses sch->GetLoops(block_rv) to obtain surrounding loops and zips them
 * with block_stmt->iter_vars.  Does not transform the schedule.
 *
 * \param sch       The schedule that owns the block.
 * \param block_rv  The block to analyse.
 * \return          A fully populated DlightSBlockInfo.
 */
DlightSBlockInfo GetSBlockInfo(const s_tir::Schedule& sch, const s_tir::SBlockRV& block_rv);

/*!
 * \brief Build DlightBufferInfo objects for all read buffers of a block.
 *
 * \param sch       The schedule that owns the block.
 * \param block_rv  The block to analyse.
 */
ffi::Array<DlightBufferInfo> GetReadBufferInfos(const s_tir::Schedule& sch,
                                                const s_tir::SBlockRV& block_rv);

/*!
 * \brief Build DlightBufferInfo objects for all write buffers of a block.
 *
 * \param sch       The schedule that owns the block.
 * \param block_rv  The block to analyse.
 */
ffi::Array<DlightBufferInfo> GetWriteBufferInfos(const s_tir::Schedule& sch,
                                                 const s_tir::SBlockRV& block_rv);

/*!
 * \brief Return the input and output dtypes of a block as string vectors.
 *
 * \param block  The TIR block to inspect.
 * \return       A pair (in_dtypes, out_dtypes).
 */
std::pair<std::vector<std::string>, std::vector<std::string>> GetInOutDtypes(
    const tir::SBlock& block);

/*!
 * \brief Normalize the prim func and return DlightSBlockInfo for every leaf block.
 *
 * Calls s_tir.schedule.NormalizePrimFunc which transforms the schedule
 * (collapses unit-extent loops).  Do not call this before schedule primitives
 * that depend on the original loop count.
 *
 * \param sch  The schedule to normalize.
 * \return     Block info array, empty if normalization fails.
 */
ffi::Array<DlightSBlockInfo> DlightNormalizePrimFunc(const s_tir::Schedule& sch);

}  // namespace dlight
}  // namespace tvm

#endif  // TVM_DLIGHT_ANALYSIS_COMMON_ANALYSIS_H_
