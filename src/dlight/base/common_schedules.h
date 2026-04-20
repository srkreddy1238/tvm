
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
 * \file src/dlight/base/common_schedules.h
 * \brief Common shhedules
 */

#ifndef TVM_DLIGHT_BASE_COMMON_SCHEDULES_H_
#define TVM_DLIGHT_BASE_COMMON_SCHEDULES_H_

#include <tvm/dlight/dlight_rule.h>
#include <tvm/ffi/reflection/registry.h>

#include "../analysis/common_analysis.h"

namespace tvm {
namespace dlight {

/*!
 * \brief Try to inline all blocks in \p blk_info, removing successfully
 *        inlined entries from the list.
 *
 * \param sch       The TIR schedule.
 * \param blk_info  In/out: block-info list; inlined blocks are removed.
 */
void DlightTryInline(const s_tir::Schedule& sch, ffi::Array<DlightSBlockInfo>* blk_info);

/*!
 * \brief Try to inline contiguous runs of spatial (injective) blocks,
 *        leaving non-injective blocks in place.
 *
 * \param sch       The TIR schedule.
 * \param blk_info  In/out: block-info list; inlined blocks are removed.
 */
void DlightTryInlineContiguousSpatial(const s_tir::Schedule& sch,
                                      ffi::Array<DlightSBlockInfo>* blk_info);

}  // namespace dlight
}  // namespace tvm

#endif  // TVM_DLIGHT_BASE_COMMON_SCHEDULES_H_
