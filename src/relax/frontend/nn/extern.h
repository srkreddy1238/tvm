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
 * \file src/relax/frontend/nn/extern.h
 * \brief Native C++ Object definitions for nn.ExternModule, nn.ObjectModule,
 *        and nn.SourceModule.
 *
 * Design
 * ------
 * ExternModuleNode
 *   - Holds a Map<String, ffi::Function> of symbol-name → shape/dtype
 *     inference callables.
 *   - Provides GetItem(func_name) which returns a wrapped callable that,
 *     when invoked with nn.Tensor / scalar arguments, emits a
 *     call_dps_packed node into the current BlockBuilder and returns the
 *     result as an nn.Tensor.
 *   - Load() is a pure-virtual-equivalent: subclasses override it.
 *   - LoadFromPath(path) is the shared helper that calls
 *     runtime::load_static_library.
 *
 * ObjectModuleNode  (subclass of ExternModuleNode)
 *   - Adds a filepath field.
 *   - Load() calls LoadFromPath(filepath).
 *
 * SourceModuleNode  (subclass of ExternModuleNode)
 *   - Adds source_code, compile_options, compiler, source_suffix,
 *     output_suffix fields.
 *   - Compile(output_path) writes source to a temp file and invokes
 *     tvm.contrib.cc.create_shared via FFI.
 *   - Load() calls Compile then LoadFromPath.
 *   - Static helpers TvmHome(), GetIncludes(), GetCompileOptions() are
 *     exposed as FFI globals so Python can call them directly.
 *
 * Python side
 * -----------
 * Each class is decorated with @tvm_ffi.register_object and constructed
 * via __init_handle_by_constructor__.  All field access and method calls
 * go through the FFI.  The Python wrappers in extern.py become thin
 * shells that delegate entirely to C++.
 */

#ifndef TVM_RELAX_FRONTEND_NN_EXTERN_H_
#define TVM_RELAX_FRONTEND_NN_EXTERN_H_

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/object.h>

#include <string>

#include "core.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ===========================================================================
// ExternModuleNode
// ===========================================================================

/*!
 * \brief Base class for external modules that can be linked into a compiled
 *        TVM IRModule.
 *
 * Holds a map from symbol name to its shape/dtype inference callable.
 * GetItem(func_name) returns a wrapped ffi::Function that, when called with
 * nn.Tensor / scalar arguments, emits a call_dps_packed node and returns the
 * result tensor.
 */
class ExternModuleNode : public runtime::Object {
 public:
  /*! \brief Map from symbol name to shape/dtype inference callable. */
  ffi::Map<ffi::String, ffi::Function> symbols;

  explicit ExternModuleNode(ffi::Map<ffi::String, ffi::Function> symbols)
      : symbols(std::move(symbols)) {}

  /*!
   * \brief Return a wrapped callable for the given symbol.
   *
   * The returned ffi::Function accepts the same arguments as the
   * shape/dtype inference function (nn.Tensor, int, float, str, PrimExpr,
   * or nested tuples/lists thereof) and emits a call_dps_packed node into
   * the current BlockBuilder, returning the result as an nn.Tensor.
   *
   * \param func_name  Symbol name registered in `symbols`.
   * \return           Wrapped callable.
   */
  ffi::Function GetItem(ffi::String func_name) const;

  /*!
   * \brief Load the external module into a TVM runtime::Module.
   *
   * Subclasses must override this.  The base implementation throws.
   */
  virtual ffi::Module Load() const;

  /*!
   * \brief Shared helper: load a pre-compiled .o / .obj file as a static
   *        library, registering the symbols listed in `symbols`.
   *
   * \param path  Absolute path to the compiled object file.
   * \return      The loaded ffi::Module.
   */
  ffi::Module LoadFromPath(ffi::String path) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExternModuleNode>()
        .def(refl::init<ffi::Map<ffi::String, ffi::Function>>())
        .def_ro("symbols", &ExternModuleNode::symbols)
        .def("_cpp_getitem", &ExternModuleNode::GetItem,
             "Return a wrapped callable for the given symbol.")
        .def("_load",
             static_cast<ffi::Module (ExternModuleNode::*)() const>(&ExternModuleNode::Load),
             "Load the external module.");
  }

  static constexpr bool _type_mutable = false;
  // Reserve child slots for ObjectModuleNode and SourceModuleNode
  static constexpr uint32_t _type_child_slots = 2;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.frontend.nn.ExternModule", ExternModuleNode, runtime::Object);
};

class ExternModule : public runtime::ObjectRef {
 public:
  explicit ExternModule(ffi::Map<ffi::String, ffi::Function> symbols);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExternModule, runtime::ObjectRef, ExternModuleNode);
};

// ===========================================================================
// ObjectModuleNode
// ===========================================================================

/*!
 * \brief An ExternModule backed by a pre-compiled object (.o / .obj) file.
 *
 * The file is loaded directly via runtime::load_static_library without any
 * compilation step.
 */
class ObjectModuleNode : public ExternModuleNode {
 public:
  /*! \brief Absolute path to the pre-compiled object file. */
  ffi::String filepath;

  ObjectModuleNode(ffi::Map<ffi::String, ffi::Function> symbols, ffi::String filepath)
      : ExternModuleNode(std::move(symbols)), filepath(std::move(filepath)) {}

  /*!
   * \brief Load the object file as a static library.
   * \return The loaded ffi::Module.
   */
  ffi::Module Load() const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ObjectModuleNode>()
        .def(refl::init<ffi::Map<ffi::String, ffi::Function>, ffi::String>())
        .def_ro("symbols", &ExternModuleNode::symbols)
        .def_ro("filepath", &ObjectModuleNode::filepath)
        .def("_cpp_getitem", &ExternModuleNode::GetItem)
        .def("_load",
             static_cast<ffi::Module (ObjectModuleNode::*)() const>(&ObjectModuleNode::Load));
  }

  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.ObjectModule", ObjectModuleNode,
                                    ExternModuleNode);
};

class ObjectModule : public runtime::ObjectRef {
 public:
  explicit ObjectModule(ffi::Map<ffi::String, ffi::Function> symbols, ffi::String filepath);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ObjectModule, runtime::ObjectRef, ObjectModuleNode);
};

// ===========================================================================
// SourceModuleNode
// ===========================================================================

/*!
 * \brief An ExternModule that compiles C++/CUDA source code on the fly.
 *
 * Compilation is performed by calling tvm.contrib.cc.create_shared via the
 * FFI, so the full Python toolchain (ccache detection, platform-specific
 * flags, etc.) is reused without duplicating it in C++.
 *
 * Static helpers TvmHome / GetIncludes / GetCompileOptions are exposed as
 * FFI globals so they can be called from both C++ and Python.
 */
class SourceModuleNode : public ExternModuleNode {
 public:
  /*! \brief The source code string (already read from file if a path was given). */
  ffi::String source_code;
  /*! \brief Compilation flags (e.g. ["-I", "/path/to/include", "-O3", ...]). */
  ffi::Array<ffi::String> compile_options;
  /*! \brief Optional compiler override (e.g. "clang++"). Null = use default. */
  ffi::Optional<ffi::String> compiler;
  /*! \brief Input file suffix: ".cpp" or ".cu". */
  ffi::String source_suffix;
  /*! \brief Output file suffix: ".o", ".obj", or ".wasm". */
  ffi::String output_suffix;

  SourceModuleNode(ffi::Map<ffi::String, ffi::Function> symbols, ffi::String source_code,
                   ffi::Array<ffi::String> compile_options, ffi::Optional<ffi::String> compiler,
                   ffi::String source_suffix, ffi::String output_suffix)
      : ExternModuleNode(std::move(symbols)),
        source_code(std::move(source_code)),
        compile_options(std::move(compile_options)),
        compiler(std::move(compiler)),
        source_suffix(std::move(source_suffix)),
        output_suffix(std::move(output_suffix)) {}

  /*!
   * \brief Compile the source code to an object file at output_path.
   *
   * Writes source_code to a temporary file, then calls
   * tvm.contrib.cc.create_shared via FFI.
   *
   * \param output_path  Destination path for the compiled object file.
   */
  void Compile(ffi::String output_path) const;

  /*!
   * \brief Compile to a temporary file and load as a static library.
   * \return The loaded ffi::Module.
   */
  ffi::Module Load() const override;

  // ---- Static helpers (also registered as FFI globals) -------------------

  /*!
   * \brief Find TVM's home directory.
   *
   * Checks the TVM_HOME environment variable first; if unset, walks up from
   * the tvm Python package location until a directory containing both
   * "include" and "3rdparty" sub-directories is found.
   *
   * \return Absolute path to the TVM home directory.
   */
  static ffi::String TvmHome();

  /*!
   * \brief Return the default include paths for compiling TVM-aware kernels.
   *
   * Always includes:
   *   - <tvm_home>/include
   *   - <tvm_home>/3rdparty/tvm-ffi/include
   *   - <tvm_home>/3rdparty/tvm-ffi/3rdparty/dlpack/include
   *
   * Optionally appends extra paths from tvm_pkg (relative to
   * <tvm_home>/3rdparty/).
   *
   * \param tvm_pkg  Optional list of relative paths under 3rdparty/.
   * \return         List of absolute include directory paths.
   */
  static ffi::Array<ffi::String> GetIncludes(ffi::Optional<ffi::Array<ffi::String>> tvm_pkg);

  /*!
   * \brief Return the default compile options for the given source format.
   *
   * Includes the default include flags from GetIncludes() plus:
   *   - For "cpp": ["-c", "-O3", "-std=c++17"]
   *   - For "cu":  ["-c", "-O3", "-std=c++17", "-Xcompiler=-fPIC"]
   *
   * \param source_format  "cpp" or "cu".
   * \param tvm_pkg        Optional extra 3rdparty packages to include.
   * \return               List of compilation flags.
   */
  static ffi::Array<ffi::String> GetCompileOptions(
      ffi::String source_format, ffi::Optional<ffi::Array<ffi::String>> tvm_pkg);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SourceModuleNode>()
        .def(refl::init<ffi::Map<ffi::String, ffi::Function>, ffi::String,
                        ffi::Array<ffi::String>, ffi::Optional<ffi::String>, ffi::String,
                        ffi::String>())
        .def_ro("symbols", &ExternModuleNode::symbols)
        .def_ro("source_code", &SourceModuleNode::source_code)
        .def_ro("compile_options", &SourceModuleNode::compile_options)
        .def_ro("compiler", &SourceModuleNode::compiler)
        .def_ro("source_suffix", &SourceModuleNode::source_suffix)
        .def_ro("output_suffix", &SourceModuleNode::output_suffix)
        .def("_cpp_getitem", &ExternModuleNode::GetItem)
        .def("_compile", &SourceModuleNode::Compile)
        .def("_load",
             static_cast<ffi::Module (SourceModuleNode::*)() const>(&SourceModuleNode::Load));
  }

  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.frontend.nn.SourceModule", SourceModuleNode,
                                    ExternModuleNode);
};

class SourceModule : public runtime::ObjectRef {
 public:
  explicit SourceModule(ffi::Map<ffi::String, ffi::Function> symbols, ffi::String source_code,
                        ffi::Array<ffi::String> compile_options,
                        ffi::Optional<ffi::String> compiler, ffi::String source_suffix,
                        ffi::String output_suffix);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SourceModule, runtime::ObjectRef, SourceModuleNode);
};

/*!
 * \brief Factory for SourceModule: handles source_format detection,
 *        source_code path resolution, and default compile_options.
 *
 * Mirrors the Python SourceModule.__init__ logic:
 *   - Detects input/output suffixes from source_format / output_format.
 *   - Reads source_code from a file path if a valid path is given.
 *   - Falls back to GetCompileOptions(source_format) when compile_options
 *     is not provided.
 *
 * \param symbols         Symbol → inference-callable map.
 * \param source_code     Source code string or path to source file.
 * \param source_format   "cpp" or "cu".
 * \param compile_options Optional override for compile flags.
 * \param compiler        Optional compiler override.
 * \param output_format   "obj" (default) or "wasm".
 * \return                Fully constructed SourceModule.
 */
SourceModule MakeSourceModule(ffi::Map<ffi::String, ffi::Function> symbols,
                              ffi::String source_code, ffi::String source_format,
                              ffi::Optional<ffi::Array<ffi::String>> compile_options,
                              ffi::Optional<ffi::String> compiler, ffi::String output_format);

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_FRONTEND_NN_EXTERN_H_
