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
 * \file src/relax/transform/attach_external_modules.cc
 * \brief Compile and attach nn.ExternModules to an IRModule's
 *        "external_mods" attribute.
 *
 * For each nn.ExternModule in the provided list the pass calls
 * ExternModuleNode::Load() to obtain a runtime::Module, then appends it
 * to the IRModule's "external_mods" attribute (creating the attribute if
 * it does not yet exist).
 */

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/relax/transform.h>

// ExternModuleNode lives in the nn frontend source tree.
// We include it via its src-relative path; the build system adds src/ to
// the include search path for internal (non-installed) targets.
#include "../frontend/nn/extern.h"

namespace tvm {
namespace relax {
namespace transform {

Pass AttachExternModules(ffi::Array<runtime::ObjectRef> extern_modules) {
  auto pass_func = [extern_modules](IRModule mod, PassContext /* pc */) {
    // Retrieve the existing "external_mods" attribute, defaulting to empty.
    ffi::Array<ffi::Module> external_mods =
        mod->GetAttr<ffi::Array<ffi::Module>>("external_mods")
            .value_or(ffi::Array<ffi::Module>());

    // Compile / load each ExternModule and append the result.
    for (const runtime::ObjectRef& obj : extern_modules) {
      TVM_FFI_ICHECK(obj.defined())
          << "AttachExternModules: encountered a null ExternModule in the list";
      auto ext = Downcast<frontend::nn::ExternModule>(obj);
      external_mods.push_back(ext->Load());
    }

    mod = WithAttr(mod, "external_mods", external_mods);
    return mod;
  };
  return CreateModulePass(pass_func, 0, "AttachExternModules", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.AttachExternModules", AttachExternModules);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
