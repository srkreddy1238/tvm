# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""External modules to be linked into the exported IRModule.

All three classes (ExternModule, ObjectModule, SourceModule) are backed by
native C++ objects registered under the "relax.frontend.nn" FFI namespace.
Python holds only the object handle; all logic lives in
src/relax/frontend/nn/extern.cc.

Two small Python-only FFI helpers are registered here so that the C++
implementation can call back into Python for platform-specific details:

  relax.frontend.nn.GetTvmPackagePath
      Returns the directory of the installed ``tvm`` Python package.
      Used by SourceModuleNode::TvmHome() when TVM_HOME is not set.

  relax.frontend.nn.DetectOutputSuffix
      Returns the platform-appropriate object-file suffix (".o" or ".obj")
      by delegating to tvm.contrib.cc._is_linux_like / _is_windows_like.
      Used by MakeSourceModule in C++.

  tvm.contrib.cc.create_shared
      Registered by tvm.contrib.cc so that SourceModuleNode::Compile() can
      invoke the Python compiler driver without duplicating it in C++.

  tvm.contrib.cc.has_ccache
      Registered here; returns True when ccache is available on PATH.
"""

import sys
from collections.abc import Callable
from pathlib import Path

import tvm_ffi

from tvm.contrib import cc as _cc
from tvm.runtime import Module

from . import _ffi_api

# ---------------------------------------------------------------------------
# Python-side FFI helpers required by the C++ implementation
# ---------------------------------------------------------------------------


def _register_python_helpers() -> None:
    """Register Python callbacks that the C++ extern.cc implementation needs."""
    import shutil  # pylint: disable=import-outside-toplevel

    import tvm  # pylint: disable=import-outside-toplevel

    @tvm_ffi.register_global_func("relax.frontend.nn.GetTvmPackagePath", override=True)
    def _get_tvm_package_path() -> str:  # pylint: disable=unused-variable
        """Return the directory of the installed tvm Python package."""
        return str(Path(tvm.__file__).parent)

    @tvm_ffi.register_global_func("relax.frontend.nn.DetectOutputSuffix", override=True)
    def _detect_output_suffix(output_format: str) -> str:  # pylint: disable=unused-variable
        """Return the platform-appropriate object-file suffix."""
        if output_format == "obj":
            if _cc._is_linux_like():  # pylint: disable=protected-access
                return ".o"
            if _cc._is_windows_like():  # pylint: disable=protected-access
                return ".obj"
            raise ValueError(f"Unsupported platform: {sys.platform}")
        if output_format == "wasm":
            return ".wasm"
        raise ValueError(f"Invalid output format: {output_format}")

    @tvm_ffi.register_global_func("tvm.contrib.cc.create_shared", override=True)
    def _create_shared(  # pylint: disable=unused-variable
        output, objects, options, cc, cwd, ccache_env
    ) -> None:
        """Thin bridge so C++ can call tvm.contrib.cc.create_shared."""
        _cc.create_shared(
            output=str(output),
            objects=[str(o) for o in objects],
            options=list(options) if options is not None else None,
            cc=str(cc) if cc is not None else None,
            cwd=str(cwd) if cwd is not None else None,
            ccache_env=dict(ccache_env) if ccache_env is not None else None,
        )

    @tvm_ffi.register_global_func("tvm.contrib.cc.has_ccache", override=True)
    def _has_ccache() -> bool:  # pylint: disable=unused-variable
        """Return True when ccache is available on PATH."""
        return shutil.which("ccache") is not None


_register_python_helpers()

# ---------------------------------------------------------------------------
# ExternModule  -  native C++ object
# ---------------------------------------------------------------------------


@tvm_ffi.register_object("relax.frontend.nn.ExternModule")
class ExternModule(tvm_ffi.Object):
    """The abstract base class for external modules.

    External modules are designed to help incorporate user-provided
    handcrafted kernels into the exported TVM IRModule.  All logic lives
    in the native C++ ExternModuleNode (src/relax/frontend/nn/extern.cc).

    Parameters
    ----------
    symbols : dict[str, Callable]
        Maps each symbol name in the external object file to its
        shape/dtype inference function.
    """

    def __init__(self, symbols: dict[str, Callable]) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ExternModule,
            symbols,
        )

    def __getitem__(self, func_name: str) -> Callable:
        """Return a wrapped callable for the given symbol.

        The callable accepts the same arguments as the shape/dtype inference
        function and emits a ``call_dps_packed`` node into the current
        BlockBuilder, returning the result as an ``nn.Tensor``.
        """
        cpp_fn = self._cpp_getitem(func_name)  # ffi::Function from C++

        def _call(*input_args):
            return cpp_fn(*input_args)

        return _call

    def load(self) -> Module:
        """Load the external module into a TVM runtime module."""
        return self._load()


# ---------------------------------------------------------------------------
# ObjectModule  -  native C++ object
# ---------------------------------------------------------------------------


@tvm_ffi.register_object("relax.frontend.nn.ObjectModule")
class ObjectModule(ExternModule):
    """An ``nn.ExternModule`` backed by a pre-compiled object (.o / .obj) file.

    The file is loaded directly via ``runtime.load_static_library`` without
    any compilation step.  All logic lives in the native C++
    ObjectModuleNode (src/relax/frontend/nn/extern.cc).

    Parameters
    ----------
    symbols : dict[str, Callable]
        Maps each symbol name to its shape/dtype inference function.

    filepath : Path | str
        Path to the pre-compiled object file.  Must be an existing regular
        file.
    """

    def __init__(
        self,
        symbols: dict[str, Callable],
        filepath: Path | str,
    ) -> None:
        filepath = Path(filepath)
        if not filepath.is_file():
            raise ValueError(f"Not a file: {filepath!s}")
        self.__init_handle_by_constructor__(
            _ffi_api.MakeObjectModule,
            symbols,
            str(filepath.resolve()),
        )


# ---------------------------------------------------------------------------
# SourceModule  -  native C++ object
# ---------------------------------------------------------------------------


@tvm_ffi.register_object("relax.frontend.nn.SourceModule")
class SourceModule(ExternModule):
    """An ``nn.ExternModule`` that compiles C++/CUDA source code on the fly.

    All compilation logic lives in the native C++ SourceModuleNode
    (src/relax/frontend/nn/extern.cc).  The static helpers ``tvm_home``,
    ``get_includes``, and ``get_compile_options`` delegate to C++ FFI
    globals so the same logic is available from both C++ and Python.

    **Shape/dtype inference.** The ``nn.ExternModule`` system requires users
    to provide additional information to work, namely, ``symbols``.  It is a
    dictionary that maps each symbol in the external object file to its
    shape/dtype inference function.  Consider a case where function
    ``my_func`` accepts two tensors, ``a`` of shape ``(x, y, 1)``, and ``b``
    of shape ``(y, z, 5)``, and produces a tensor ``c`` of shape
    ``(x, y, z, 9)``, the shape/dtype inference function should look like:

    .. code-block:: python

        def shape_dtype_inference(a, b):
            x, y, _ = a.shape
            _, z, _ = b.shape
            return nn.Tensor.placeholder((x, y, z, 9), dtype="float32")

    and the ``symbols`` dictionary should be provided as:

    .. code-block:: python

        symbols={
            "my_func": shape_dtype_inference,
        }

    **Calling convention.** All external modules follow
    "destination-passing-style" (DPS) calling convention, which means the
    returned tensors are pre-allocated by the system already and passed in
    as an argument of the external function.

    To expose the symbol, ``TVM_FFI_DLL_EXPORT_TYPED_FUNC(symbol, function)``
    is guaranteed available:

    .. code-block:: C++

        // those headers are guaranteed to be available
        #include <dlpack/dlpack.h>
        #include <tvm/runtime/data_type.h>
        #include <tvm/ffi/function.h>

        namespace {
        int _my_func_impl(DLTensor* a, DLTensor* b, DLTensor* c) {
            // `a` and `b` are inputs, and `c` is the output
        }
        }
        TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_func, _my_func_impl);

    **A compiler pass ``AttachExternModules``.** It is introduced to attach a
    list of ``nn.ExternModule``\s into an IRModule at any stage of the
    compilation pipeline, and attach the compiled external modules as
    ``runtime.Module``\s into IRModule's ``external_mods`` attribute.

    **Caveats.** It is required to call ``nn.add_extern`` to register
    external modules exactly once during ``export_tvm``.  Each symbol should
    be registered exactly once to avoid potential conflicts.

    Parameters
    ----------
    symbols : dict[str, Callable]
        Maps each symbol name to its shape/dtype inference function.

    source_code : str | Path
        Source code string or path to a source file.

    source_format : str
        ``"cpp"`` or ``"cu"``.

    compile_options : list[str] | None
        Compilation flags.  Defaults to ``get_compile_options(source_format)``.

    compiler : str | None
        Compiler executable.  ``None`` uses the platform default.

    output_format : str
        ``"obj"`` (default) or ``"wasm"``.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        symbols: dict[str, Callable],
        source_code: str | Path,
        source_format: str,  # "cpp", "cu"
        compile_options: list[str] | None = None,
        compiler: str | None = None,
        output_format: str = "obj",  # "obj", "wasm"
    ) -> None:
        # Convert Path to string so C++ can handle it uniformly
        if isinstance(source_code, Path):
            source_code = str(source_code)
        self.__init_handle_by_constructor__(
            _ffi_api.MakeSourceModule,
            symbols,
            str(source_code),
            str(source_format),
            list(compile_options) if compile_options is not None else None,
            str(compiler) if compiler is not None else None,
            str(output_format),
        )

    def compile(self, output_path: Path | str) -> None:
        """Compile the source code and write the object file to *output_path*."""
        self._compile(str(output_path))

    # ---- Static helpers (delegate to C++ FFI globals) ----------------------

    @staticmethod
    def tvm_home() -> Path:
        """Find TVM's home directory.

        If the ``TVM_HOME`` environment variable is set, use it.  Otherwise,
        walk up from the ``tvm`` Python package location until a directory
        containing both ``include`` and ``3rdparty`` sub-directories is found.

        Returns
        -------
        tvm_home : pathlib.Path
            The TVM home directory, guaranteed to contain ``include`` and
            ``3rdparty`` as direct sub-directories.
        """
        return Path(str(_ffi_api.SourceModule_TvmHome()))

    @staticmethod
    def get_includes(tvm_pkg: list[str] | None = None) -> list[Path]:
        """Return the default include paths for compiling TVM-aware kernels.

        Always includes TVM, tvm-ffi, and DLPack headers.  With *tvm_pkg*
        provided, also includes the specified packages under
        ``tvm_home/3rdparty``.

        Parameters
        ----------
        tvm_pkg : list[str] | None
            Relative paths under ``tvm_home/3rdparty`` to include.

        Returns
        -------
        includes : list[pathlib.Path]
            Absolute include directory paths.
        """
        raw = _ffi_api.SourceModule_GetIncludes(list(tvm_pkg) if tvm_pkg else None)
        return [Path(str(p)) for p in raw]

    @staticmethod
    def get_compile_options(
        source_format: str,
        tvm_pkg: list[str] | None = None,
    ) -> list[str]:
        """Return the default compile options for the given source format.

        Includes the default include flags from :meth:`get_includes` plus
        ``-O3`` and ``-std=c++17``.

        Parameters
        ----------
        source_format : str
            ``"cpp"`` or ``"cu"``.

        tvm_pkg : list[str] | None
            Extra 3rdparty packages to include.

        Returns
        -------
        compile_options : list[str]
            Compilation flags.
        """
        raw = _ffi_api.SourceModule_GetCompileOptions(
            source_format, list(tvm_pkg) if tvm_pkg else None
        )
        return [str(s) for s in raw]
