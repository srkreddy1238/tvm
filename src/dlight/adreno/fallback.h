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
 * \file src/dlight/adreno/fallback.h
 * \brief Internal helpers shared between Adreno dlight rules.
 *
 * Exposes AdrenoScheduleDefault and AdrenoScheduleInlineBlocks so that
 * convolution.cc and matmul.cc can reuse the scheduling logic from fallback.cc
 * without duplicating it.
 */

#ifndef TVM_DLIGHT_ADRENO_FALLBACK_H_
#define TVM_DLIGHT_ADRENO_FALLBACK_H_

#include <tvm/s_tir/schedule/schedule.h>

#include "../analysis/common_analysis.h"

namespace tvm {
namespace dlight {

/*!
 * \brief Apply the default vectorised Adreno schedule to a single block.
 *
 * Defined in fallback.cc.
 */
void AdrenoScheduleDefault(const s_tir::Schedule& sch, const s_tir::SBlockRV& blk);

/*!
 * \brief Auto-inline injective non-data-pad blocks.
 *
 * Defined in fallback.cc.
 *
 * \param sch       The schedule to transform.
 * \param blk_infos In/out: block-info list; successfully inlined blocks are removed.
 */
void AdrenoScheduleInlineBlocks(const s_tir::Schedule& sch,
                                ffi::Array<DlightSBlockInfo>* blk_infos);

}  // namespace dlight
}  // namespace tvm

#endif  // TVM_DLIGHT_ADRENO_FALLBACK_H_
