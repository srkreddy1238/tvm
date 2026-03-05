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
 * \file tvm/relax/backend/adreno/pipeline.h
 * \brief Adreno GPU relax pipeline
 */
#ifndef TVM_RELAX_BACKEND_ADRENO_PIPELINE_H_
#define TVM_RELAX_BACKEND_ADRENO_PIPELINE_H_

#include <tvm/relax/backend/pipeline.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {
namespace backend {
namespace adreno {

using Pass = tvm::transform::Pass;
using PassContext = tvm::transform::PassContext;
using tvm::transform::CreateModulePass;

/*
 * Adreno Relax Pipelines resolved as
 *
 * tvm::relax::backend::AdrenoRelaxPipelineAll
 * tvm::relax::backend::AdrenoRelaxPipelineLibrary
 * tvm::relax::backend::AdrenoRelaxPipelineLegalize
 * tvm::relax::backend::AdrenoRelaxPipelineDataflow
 * tvm::relax::backend::AdrenoRelaxPipelineFInalize
 *
 */

TVM_RELAX_BACKEND_PIPELINE_DECL(AdrenoRelaxPipeline);

}  // namespace adreno
}  // namespace backend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BACKEND_ADRENO_PIPELINE_H_
