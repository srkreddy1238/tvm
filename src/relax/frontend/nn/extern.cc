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
 * \file src/relax/frontend/nn/extern.cc
 * \brief C++ implementation of nn.ExternModule, nn.ObjectModule, and
 *        nn.SourceModule.
 *
 * Implementation notes
 * --------------------
 * GetItem / argument conversion
 *   The returned callable converts each argument to a relax Expr following
 *   the same rules as the Python ExternModule.__getitem__ helper:
 *     nn.Tensor / NNTensor  -> tensor._expr (relax.Var)
 *     int64                 -> rx.PrimValue(tir.IntImm("int64", v))
 *     float64               -> rx.PrimValue(tir.FloatImm("float64", v))
 *     String                -> rx.StringImm(s)
 *     tir.PrimExpr          -> rx.PrimValue(expr)
 *     Array / Tuple         -> rx.Tuple([convert(e) for e in ...])
 *   The shape/dtype inference callable is invoked with the original
 *   arguments to obtain the output StructInfo, then call_dps_packed is
 *   emitted and the result is wrapped via WrapNested.
 *
 * TvmHome / GetIncludes / GetCompileOptions
 *   These are pure C++ implementations that mirror the Python statics on
 *   SourceModule.  They are also registered as FFI globals so Python can
 *   call them directly (e.g. SourceModule.tvm_home() delegates to the FFI).
 *
 * Compile / Load
 *   Compilation is delegated to tvm.contrib.cc.create_shared via the FFI
 *   global "tvm.contrib.cc.create_shared".  This reuses the Python
 *   toolchain (ccache detection, platform-specific linker flags, etc.)
 *   without duplicating it in C++.
 */

#include "extern.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/expr.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {

// ===========================================================================
// Internal helpers
// ===========================================================================

namespace {

/*!
 * \brief Convert a single argument to a relax::Expr for use in
 *        call_dps_packed.
 *
 * Mirrors Python ExternModule.__getitem__._convert():
 *   nn.Tensor / NNTensor  -> tensor.expr (relax.Var)
 *   int64                 -> PrimValue(IntImm("int64", v))
 *   float64               -> PrimValue(FloatImm("float64", v))
 *   String                -> StringImm(s)
 *   tir.PrimExpr          -> PrimValue(expr)
 *   Array<Any>            -> Tuple([convert(e) for e in arr])
 */
Expr ConvertArg(const ffi::Any& arg) {
  // nn.Tensor (NNTensor / NNParameter)
  if (auto opt = arg.try_cast<NNTensor>()) {
    return opt.value()->expr;
  }
  if (auto opt = arg.try_cast<NNParameter>()) {
    return opt.value()->expr;
  }
  // int64
  if (auto opt = arg.try_cast<int64_t>()) {
    return PrimValue(IntImm(DataType::Int(64), opt.value()));
  }
  // double / float64
  if (auto opt = arg.try_cast<double>()) {
    return PrimValue(FloatImm(DataType::Float(64), opt.value()));
  }
  // String -> StringImm
  if (auto opt = arg.try_cast<ffi::String>()) {
    return StringImm(std::string(opt.value()));
  }
  // tir::PrimExpr -> PrimValue
  if (auto opt = arg.try_cast<PrimExpr>()) {
    return PrimValue(opt.value());
  }
  // Array<Any> -> Tuple
  if (auto opt = arg.try_cast<ffi::Array<ffi::Any>>()) {
    ffi::Array<Expr> fields;
    for (const ffi::Any& elem : opt.value()) {
      fields.push_back(ConvertArg(elem));
    }
    return relax::Tuple(fields);
  }
  TVM_FFI_THROW(TypeError) << "ExternModule: unsupported argument type: " << arg.GetTypeKey();
  TVM_FFI_UNREACHABLE();
}

/*!
 * \brief Convert a variadic list of ffi::Any arguments to a relax::Tuple
 *        (or a single Expr if there is exactly one argument).
 *
 * Used to build the `args` operand of call_dps_packed.
 */
Expr ConvertArgs(const ffi::Array<ffi::Any>& args) {
  if (args.size() == 1) {
    return ConvertArg(args[0]);
  }
  ffi::Array<Expr> fields;
  for (const ffi::Any& a : args) {
    fields.push_back(ConvertArg(a));
  }
  return relax::Tuple(fields);
}

/*!
 * \brief Check whether a filesystem path exists and is a directory.
 */
bool IsDir(const std::string& path) {
  std::error_code ec;
  return std::filesystem::is_directory(path, ec);
}

/*!
 * \brief Return the absolute, canonical form of a path.
 */
std::string AbsPath(const std::string& path) {
  std::error_code ec;
  auto p = std::filesystem::canonical(path, ec);
  if (ec) return path;
  return p.string();
}

/*!
 * \brief Join two path components.
 */
std::string PathJoin(const std::string& base, const std::string& rel) {
  return (std::filesystem::path(base) / rel).string();
}

}  // anonymous namespace

// ===========================================================================
// ExternModuleNode
// ===========================================================================

ExternModule::ExternModule(ffi::Map<ffi::String, ffi::Function> symbols) {
  data_ = ffi::make_object<ExternModuleNode>(std::move(symbols));
}

ffi::Function ExternModuleNode::GetItem(ffi::String func_name) const {
  auto it = symbols.find(func_name);
  TVM_FFI_ICHECK(it != symbols.end())
      << "ExternModule: symbol '" << func_name << "' not found in symbols map";

  ffi::Function inference_fn = (*it).second;
  std::string fname = std::string(func_name);

  // Return a closure that:
  //   1. Calls inference_fn(*input_args) to get the output StructInfo.
  //   2. Converts input_args to relax Exprs.
  //   3. Emits call_dps_packed(func_name, rx_inputs, rx_outputs_sinfo).
  //   4. Wraps the result via WrapNested.
  return ffi::Function([inference_fn, fname](ffi::PackedArgs args, ffi::Any* rv) {
    // Collect all arguments into an Array<Any>
    ffi::Array<ffi::Any> input_args;
    for (int i = 0; i < args.size(); ++i) {
      input_args.push_back(args[i]);
    }

    // 1. Run shape/dtype inference to get output StructInfo.
    //    inference_fn returns an NNTensor (placeholder) whose struct_info
    //    describes the output.
    ffi::Any infer_result;
    inference_fn.CallPacked(args, &infer_result);

    // Extract StructInfo from the inference result.
    // The inference function returns an NNTensor placeholder; we need its
    // struct_info.  We call ConvertArg on it to get the relax Expr, then
    // read its struct_info.
    Expr dummy_expr = ConvertArg(infer_result);
    StructInfo out_sinfo = GetStructInfo(dummy_expr);

    // 2. Convert input arguments to a relax Tuple / single Expr.
    Expr rx_inputs = ConvertArgs(input_args);

    // 3. Emit call_dps_packed.
    //    call_dps_packed(func_name, args, out_sinfo) is the DPS calling
    //    convention: the output tensor is pre-allocated and passed as the
    //    last argument.
    static const Op& call_dps_op = Op::Get("relax.call_dps_packed");
    Expr call_expr = Call(call_dps_op, {ExternFunc(fname), rx_inputs}, tvm::Attrs(), {out_sinfo});

    // 4. Wrap via WrapNested (emits into current BlockBuilder).
    *rv = WrapNested(call_expr, fname);
  });
}

ffi::Module ExternModuleNode::Load() const {
  TVM_FFI_THROW(NotImplementedError) << "ExternModule.load() is not implemented in the base class. "
                                     << "Use ObjectModule or SourceModule instead.";
  TVM_FFI_UNREACHABLE();
}

ffi::Module ExternModuleNode::LoadFromPath(ffi::String path) const {
  // Collect symbol names
  ffi::Array<ffi::String> func_names;
  for (const auto& kv : symbols) {
    func_names.push_back(kv.first);
  }
  // Call runtime.ModuleLoadStaticLibrary via FFI (registered in src/runtime/static_library.cc)
  auto load_fn = ffi::Function::GetGlobal("runtime.ModuleLoadStaticLibrary");
  TVM_FFI_ICHECK(load_fn.has_value()) << "runtime.load_static_library not found";
  ffi::Any result = load_fn.value()(path, func_names);
  return result.cast<ffi::Module>();
}

// ===========================================================================
// ObjectModuleNode
// ===========================================================================

ObjectModule::ObjectModule(ffi::Map<ffi::String, ffi::Function> symbols, ffi::String filepath) {
  data_ = ffi::make_object<ObjectModuleNode>(std::move(symbols), std::move(filepath));
}

ffi::Module ObjectModuleNode::Load() const { return LoadFromPath(filepath); }

// ===========================================================================
// SourceModuleNode – static helpers
// ===========================================================================

/*static*/ ffi::String SourceModuleNode::TvmHome() {
  // 1. Check TVM_HOME environment variable.
  const char* env_home = std::getenv("TVM_HOME");
  std::string tvm_path;

  if (env_home && std::strlen(env_home) > 0) {
    tvm_path = env_home;
    TVM_FFI_ICHECK(IsDir(tvm_path))
        << "Using environment variable TVM_HOME, but directory not found: " << tvm_path;
    tvm_path = AbsPath(tvm_path);
  } else {
    // 2. Ask Python for the tvm package location via FFI.
    //    We call the Python-registered global "relax.frontend.nn.GetTvmPackagePath"
    //    which returns the directory of the tvm Python package.
    auto get_pkg_path_fn = ffi::Function::GetGlobal("relax.frontend.nn.GetTvmPackagePath");
    TVM_FFI_ICHECK(get_pkg_path_fn.has_value())
        << "relax.frontend.nn.GetTvmPackagePath not registered";
    ffi::Any pkg_path_any = get_pkg_path_fn.value()();
    tvm_path = std::string(pkg_path_any.cast<ffi::String>());
    TVM_FFI_ICHECK(IsDir(tvm_path)) << "tvm package path is not a directory: " << tvm_path;
    tvm_path = AbsPath(tvm_path);
  }

  // 3. Walk up until we find a directory with both "include" and "3rdparty".
  std::filesystem::path p(tvm_path);
  while (true) {
    bool has_include = IsDir(PathJoin(p.string(), "include"));
    bool has_3rdparty = IsDir(PathJoin(p.string(), "3rdparty"));
    if (has_include && has_3rdparty) {
      return ffi::String(AbsPath(p.string()));
    }
    std::filesystem::path parent = p.parent_path();
    if (parent == p) {
      TVM_FFI_THROW(ValueError)
          << "Cannot detect TVM directory. "
          << "Please explicitly specify it by setting the TVM_HOME environment variable, "
          << "and make sure it contains 'include' and '3rdparty' as direct sub-directories.";
      TVM_FFI_UNREACHABLE();
    }
    p = parent;
  }
}

/*static*/ ffi::Array<ffi::String> SourceModuleNode::GetIncludes(
    ffi::Optional<ffi::Array<ffi::String>> tvm_pkg) {
  std::string tvm_home = std::string(TvmHome());

  std::vector<std::string> paths = {
      PathJoin(tvm_home, "include"),
      PathJoin(tvm_home, "3rdparty/tvm-ffi/include"),
      PathJoin(tvm_home, "3rdparty/tvm-ffi/3rdparty/dlpack/include"),
  };

  if (tvm_pkg.has_value()) {
    for (const ffi::String& rel : tvm_pkg.value()) {
      paths.push_back(PathJoin(PathJoin(tvm_home, "3rdparty"), std::string(rel)));
    }
  }

  ffi::Array<ffi::String> result;
  for (const std::string& path : paths) {
    TVM_FFI_ICHECK(IsDir(path)) << "Include path not found or not a directory: " << path;
    result.push_back(ffi::String(path));
  }
  return result;
}

/*static*/ ffi::Array<ffi::String> SourceModuleNode::GetCompileOptions(
    ffi::String source_format, ffi::Optional<ffi::Array<ffi::String>> tvm_pkg) {
  ffi::Array<ffi::String> includes = GetIncludes(tvm_pkg);

  ffi::Array<ffi::String> result;
  // Add -I flags for each include path
  for (const ffi::String& inc : includes) {
    result.push_back(ffi::String("-I"));
    result.push_back(inc);
  }

  std::string fmt = std::string(source_format);
  if (fmt == "cpp") {
    result.push_back(ffi::String("-c"));
    result.push_back(ffi::String("-O3"));
    result.push_back(ffi::String("-std=c++17"));
  } else if (fmt == "cu") {
    result.push_back(ffi::String("-c"));
    result.push_back(ffi::String("-O3"));
    result.push_back(ffi::String("-std=c++17"));
    result.push_back(ffi::String("-Xcompiler=-fPIC"));
  } else {
    TVM_FFI_THROW(ValueError) << "SourceModule.get_compile_options: invalid source_format '" << fmt
                              << "'. Expected 'cpp' or 'cu'.";
    TVM_FFI_UNREACHABLE();
  }
  return result;
}

// ===========================================================================
// SourceModuleNode – Compile / Load
// ===========================================================================

void SourceModuleNode::Compile(ffi::String output_path) const {
  // Delegate to tvm.contrib.cc.create_shared via the FFI global registered
  // by Python as "tvm.contrib.cc.create_shared".
  //
  // We write source_code to a temporary file, then call create_shared with
  // the same arguments as the Python SourceModule.compile() method.
  auto create_shared_opt = ffi::Function::GetGlobal("tvm.contrib.cc.create_shared");
  TVM_FFI_ICHECK(create_shared_opt.has_value())
      << "tvm.contrib.cc.create_shared not registered. "
      << "Import tvm.relax.frontend.nn.extern to register Python helpers.";
  ffi::Function create_shared_fn = create_shared_opt.value();

  // Build a temporary directory path using std::filesystem.
  std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() /
      ("tvm_nn_extern_" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())));
  std::error_code ec;
  std::filesystem::create_directories(tmp_dir, ec);
  TVM_FFI_ICHECK(!ec) << "Failed to create temporary directory: " << tmp_dir.string()
                      << " error: " << ec.message();

  // Write source code to temp file
  std::string source_filename = "main" + std::string(source_suffix);
  std::filesystem::path source_path = tmp_dir / source_filename;
  {
    std::ofstream ofs(source_path.string());
    TVM_FFI_ICHECK(ofs.is_open()) << "Failed to open source file for writing: "
                                  << source_path.string();
    ofs << std::string(source_code);
  }

  std::string object_filename = "main" + std::string(output_suffix);
  std::filesystem::path object_path = tmp_dir / object_filename;

  // Build options array as ffi::Array<ffi::String>
  ffi::Array<ffi::String> opts = compile_options;

  // Detect ccache availability via FFI helper registered by Python
  ffi::Optional<ffi::Map<ffi::String, ffi::String>> ccache_env = std::nullopt;
  auto ccache_opt = ffi::Function::GetGlobal("tvm.contrib.cc.has_ccache");
  if (ccache_opt.has_value()) {
    ffi::Any has_ccache = ccache_opt.value()();
    if (has_ccache.cast<bool>()) {
      ffi::Map<ffi::String, ffi::String> env;
      env.Set("CCACHE_COMPILERCHECK", "content");
      env.Set("CCACHE_NOHASHDIR", "1");
      ccache_env = env;
    }
  }

  // Call tvm.contrib.cc.create_shared(output, objects, options, cc, cwd, ccache_env)
  create_shared_fn(
      /*output=*/ffi::String(object_filename),
      /*objects=*/ffi::Array<ffi::String>{ffi::String(source_filename)},
      /*options=*/opts,
      /*cc=*/compiler.has_value() ? ffi::Any(compiler.value()) : ffi::Any(nullptr),
      /*cwd=*/ffi::String(tmp_dir.string()),
      /*ccache_env=*/ccache_env.has_value() ? ffi::Any(ccache_env.value()) : ffi::Any(nullptr));

  // Move compiled object to output_path
  std::filesystem::rename(object_path, std::filesystem::path(std::string(output_path)), ec);
  if (ec) {
    // rename may fail across filesystems; fall back to copy + remove
    std::filesystem::copy_file(object_path, std::filesystem::path(std::string(output_path)),
                               std::filesystem::copy_options::overwrite_existing, ec);
    TVM_FFI_ICHECK(!ec) << "Failed to move compiled object to " << std::string(output_path) << ": "
                        << ec.message();
    std::filesystem::remove(object_path, ec);
  }

  // Clean up temp directory
  std::filesystem::remove_all(tmp_dir, ec);
}

ffi::Module SourceModuleNode::Load() const {
  // Create a unique temporary output path
  std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() /
      ("tvm_nn_extern_load_" +
       std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())));
  std::error_code ec;
  std::filesystem::create_directories(tmp_dir, ec);
  TVM_FFI_ICHECK(!ec) << "Failed to create temporary directory: " << tmp_dir.string();

  std::filesystem::path output_path = tmp_dir / ("main" + std::string(output_suffix));
  Compile(ffi::String(output_path.string()));
  ffi::Module mod = LoadFromPath(ffi::String(output_path.string()));

  // Clean up
  std::filesystem::remove_all(tmp_dir, ec);
  return mod;
}

// ===========================================================================
// SourceModule constructor + MakeSourceModule factory
// ===========================================================================

SourceModule::SourceModule(ffi::Map<ffi::String, ffi::Function> symbols, ffi::String source_code,
                           ffi::Array<ffi::String> compile_options,
                           ffi::Optional<ffi::String> compiler, ffi::String source_suffix,
                           ffi::String output_suffix) {
  data_ = ffi::make_object<SourceModuleNode>(std::move(symbols), std::move(source_code),
                                             std::move(compile_options), std::move(compiler),
                                             std::move(source_suffix), std::move(output_suffix));
}

/*!
 * \brief Detect the input file suffix from source_format.
 */
static ffi::String DetectInputSuffix(const std::string& source_format) {
  if (source_format == "cpp") return ffi::String(".cpp");
  if (source_format == "cu") return ffi::String(".cu");
  TVM_FFI_THROW(ValueError) << "MakeSourceModule: invalid source_format '" << source_format
                            << "'. Expected 'cpp' or 'cu'.";
  TVM_FFI_UNREACHABLE();
}

/*!
 * \brief Detect the output file suffix from output_format.
 *
 * Platform detection is delegated to the Python FFI helper
 * "relax.frontend.nn.DetectOutputSuffix" which wraps tvm.contrib.cc's
 * _is_linux_like / _is_windows_like checks.
 */
static ffi::String DetectOutputSuffix(const std::string& output_format) {
  if (output_format == "wasm") return ffi::String(".wasm");
  if (output_format == "obj") {
    // Delegate platform detection to Python
    auto detect_opt = ffi::Function::GetGlobal("relax.frontend.nn.DetectOutputSuffix");
    TVM_FFI_ICHECK(detect_opt.has_value())
        << "relax.frontend.nn.DetectOutputSuffix not registered. "
        << "Import tvm.relax.frontend.nn.extern to register Python helpers.";
    ffi::Any result = detect_opt.value()(ffi::String(output_format));
    return result.cast<ffi::String>();
  }
  TVM_FFI_THROW(ValueError) << "MakeSourceModule: invalid output_format '" << output_format
                            << "'. Expected 'obj' or 'wasm'.";
  TVM_FFI_UNREACHABLE();
}

/*!
 * \brief Attempt to read source_code as a file path; return the file
 *        contents if it is a valid file, otherwise return source_code as-is.
 *
 * Mirrors Python SourceModule.__init__._detect_source_code().
 */
static ffi::String DetectSourceCode(const std::string& source_code) {
  // Try to interpret source_code as a filesystem path
  std::error_code ec;
  std::filesystem::path p(source_code);
  bool is_file = std::filesystem::is_regular_file(p, ec);
  if (!ec && is_file) {
    std::ifstream ifs(p.string());
    if (ifs.is_open()) {
      std::string contents((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
      return ffi::String(contents);
    }
  }
  // Not a file path (or unreadable): treat as raw source code
  return ffi::String(source_code);
}

SourceModule MakeSourceModule(ffi::Map<ffi::String, ffi::Function> symbols, ffi::String source_code,
                              ffi::String source_format,
                              ffi::Optional<ffi::Array<ffi::String>> compile_options,
                              ffi::Optional<ffi::String> compiler, ffi::String output_format) {
  std::string fmt = std::string(source_format);
  std::string out_fmt = std::string(output_format);

  ffi::String src_suffix = DetectInputSuffix(fmt);
  ffi::String out_suffix = DetectOutputSuffix(out_fmt);
  ffi::String src_code = DetectSourceCode(std::string(source_code));

  ffi::Array<ffi::String> opts;
  if (compile_options.has_value()) {
    opts = compile_options.value();
  } else {
    opts = SourceModuleNode::GetCompileOptions(source_format, std::nullopt);
  }

  return SourceModule(std::move(symbols), std::move(src_code), std::move(opts), std::move(compiler),
                      std::move(src_suffix), std::move(out_suffix));
}

// ===========================================================================
// FFI registration
// ===========================================================================

TVM_FFI_STATIC_INIT_BLOCK() {
  ExternModuleNode::RegisterReflection();
  ObjectModuleNode::RegisterReflection();
  SourceModuleNode::RegisterReflection();

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      // ---- ExternModule factory ------------------------------------------
      .def("relax.frontend.nn.ExternModule",
           [](ffi::Map<ffi::String, ffi::Function> symbols) {
             return ExternModule(std::move(symbols));
           })

      // ---- ObjectModule factory ------------------------------------------
      .def("relax.frontend.nn.MakeObjectModule",
           [](ffi::Map<ffi::String, ffi::Function> symbols, ffi::String filepath) {
             return ObjectModule(std::move(symbols), std::move(filepath));
           })

      // ---- SourceModule factory ------------------------------------------
      .def("relax.frontend.nn.MakeSourceModule",
           [](ffi::Map<ffi::String, ffi::Function> symbols, ffi::String source_code,
              ffi::String source_format, ffi::Optional<ffi::Array<ffi::String>> compile_options,
              ffi::Optional<ffi::String> compiler, ffi::String output_format) {
             return MakeSourceModule(std::move(symbols), std::move(source_code),
                                     std::move(source_format), std::move(compile_options),
                                     std::move(compiler), std::move(output_format));
           })

      // ---- SourceModule static helpers (also callable from Python) -------
      .def("relax.frontend.nn.SourceModule_TvmHome", []() { return SourceModuleNode::TvmHome(); })

      .def("relax.frontend.nn.SourceModule_GetIncludes",
           [](ffi::Optional<ffi::Array<ffi::String>> tvm_pkg) {
             return SourceModuleNode::GetIncludes(std::move(tvm_pkg));
           })

      .def("relax.frontend.nn.SourceModule_GetCompileOptions",
           [](ffi::String source_format, ffi::Optional<ffi::Array<ffi::String>> tvm_pkg) {
             return SourceModuleNode::GetCompileOptions(std::move(source_format),
                                                        std::move(tvm_pkg));
           });
}

}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
