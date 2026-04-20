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
"""Compilation specifications for nn.Module export.

All six spec types are now native C++ objects:

  Int        (relax.frontend.nn.spec.Int)      - no fields
  Tensor     (relax.frontend.nn.spec.Tensor)   - shape, dtype
  Tuple      (relax.frontend.nn.spec.Tuple)    - name, elements, is_tuple
  MethodSpec (relax.frontend.nn.spec.MethodSpec)
    forward:     ffi::Function(Map<String,Any>) -> Any
    arg_names:   Array<String>
    arg_specs:   Array<Any>
    param_mode:  String
    effect_mode: String
  ModuleSpec (relax.frontend.nn.spec.ModuleSpec)
    method_names: Array<String>
    method_specs: Array<Any>
    named_params: Map<String, NNParameter>

  Object - stays Python: object_type is a Python class used as a constructor.

The forward function in MethodSpec receives Map<String, Any> where each
value is an nn.Tensor (TensorNode).  The implementation casts ffi::Any to
NNTensor as needed.  No inspect.signature is required.

MethodSpec.from_raw() and ModuleSpec.from_raw() are Python helpers that
build the C++ objects from the user-facing dict API.  They use
inspect.signature only to discover arg_names; the forward function they
register is a Python closure that calls the original method.
"""

import inspect
import typing

import tvm_ffi

if typing.TYPE_CHECKING:
    from .core import Module as _nn_module_class

# ---------------------------------------------------------------------------
# Type aliases (unchanged public API)
# ---------------------------------------------------------------------------
ArgSpecType = typing.Union["Int", "Tensor"]
MethodSpecType = typing.Union["MethodSpec", dict[str, ArgSpecType]]
ModuleSpecType = typing.Union["ModuleSpec", dict[str, MethodSpecType]]
SpecAny = typing.Union["Object", "Int", "Tensor", "Tuple"]

# ---------------------------------------------------------------------------
# All native C++ spec classes inherit tvm_ffi.Object and use self.__ffi_init__
# to construct the underlying C++ object.  register_object sets __c_ffi_init__
# on the class from the "__ffi_init__" type method registered by refl::init<>()
# in C++, and __ffi_init__ calls __init_handle_by_constructor__(__c_ffi_init__).
# ---------------------------------------------------------------------------


# ===========================================================================
# Int  -  native C++ object
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.spec.Int")
class Int(tvm_ffi.Object):
    """Spec for a scalar integer input (becomes a tir.Var in the IR)."""

    def __init__(self) -> None:
        self.__ffi_init__()

    def __repr__(self) -> str:
        return "int"


# ===========================================================================
# Tensor  -  native C++ object
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.spec.Tensor")
class Tensor(tvm_ffi.Object):
    """Spec for a tensor input: static ndim/dtype, symbolic or static shapes."""

    def __init__(self, shape: typing.Sequence[int | str], dtype: str) -> None:
        self.__ffi_init__(list(shape), dtype)

    def __repr__(self) -> str:
        # shape and dtype are C++ fields exposed directly by register_object
        return f"Tensor({list(self.shape)}, '{self.dtype}')"


# ===========================================================================
# Tuple  -  native C++ object
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.spec.Tuple")
class Tuple(tvm_ffi.Object):
    """Spec for a tuple or list input containing nested specs."""

    def __init__(self, name: str, elements: "list[SpecAny] | tuple[SpecAny, ...]") -> None:
        assert isinstance(elements, list | tuple)
        is_tuple = isinstance(elements, tuple)
        self.__ffi_init__(name, list(elements), is_tuple)

    @property
    def elements(self) -> "list[SpecAny] | tuple[SpecAny, ...]":
        # name, elements, is_tuple are C++ fields exposed directly by register_object
        raw = list(self.elements)
        return tuple(raw) if self.is_tuple else raw


# ===========================================================================
# Object  -  stays Python
# object_type is a Python class used as a constructor in exporter.py.
# ===========================================================================


class Object:
    """Spec for a non-tensor opaque frontend object (e.g. KVCache)."""

    object_type: type

    def __init__(self, object_type: type) -> None:
        self.object_type = object_type

    def __repr__(self) -> str:
        return "object"


# ===========================================================================
# MethodSpec  -  native C++ object
#
# The forward function is stored as ffi::Function(Map<String,Any>) -> Any.
# Python wraps the original method in a closure that:
#   1. Receives Map<String, Any> from C++.
#   2. Extracts each nn.Tensor by name.
#   3. Calls the original Python method.
#   4. Returns the result as-is (Tensor or tuple of Tensors).
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.spec.MethodSpec")
class MethodSpec(tvm_ffi.Object):
    """Spec for a single compiled method.

    The forward function receives Map<String, Any> where each value is an
    nn.Tensor.  arg_names is provided explicitly; no inspect.signature needed
    at the C++ level.
    """

    def __init__(
        self,
        method: typing.Callable,
        arg_names: list[str],
        arg_specs: list[ArgSpecType],
        param_mode: str,
        effect_mode: str,
    ) -> None:
        if param_mode not in ("plain", "packed", "none"):
            raise ValueError(f"Invalid param_mode: {param_mode!r}")
        if effect_mode not in ("plain", "packed", "none"):
            raise ValueError(f"Invalid effect_mode: {effect_mode!r}")

        def _forward(named_args):
            args = [named_args[name] for name in arg_names]
            return method(*args)

        self.__ffi_init__(_forward, arg_names, arg_specs, param_mode, effect_mode)

    # ---- read-only properties from C++ fields ----------------------------
    # arg_names, arg_specs, param_mode, effect_mode are exposed directly
    # as attributes by register_object; no Python property wrappers needed.

    def _repr(self, name: str) -> str:
        args = ", ".join(f"{n}: {s}" for n, s in zip(self.arg_names, self.arg_specs))
        return f"{name}({args})"

    def __repr__(self) -> str:
        return self._repr("MethodSpec")

    @staticmethod
    def from_raw(spec: MethodSpecType, method: typing.Callable) -> "MethodSpec":
        """Build a MethodSpec from a raw dict.

        inspect.signature is used here (Python side) only to discover
        arg_names.  The C++ MethodSpecNode never calls inspect.
        """
        if isinstance(spec, MethodSpec):
            return spec

        config: dict[str, typing.Any] = spec.pop("$", {})  # type: ignore[union-attr]
        param_mode = config.get("param_mode", "plain")
        effect_mode = config.get("effect_mode", "plain")

        sig = inspect.signature(method)
        arg_names = list(sig.parameters.keys())
        arg_specs: list[ArgSpecType] = []

        def _convert(arg_spec, arg_name: str) -> ArgSpecType:
            if arg_spec is Int or arg_spec is int:
                return Int()
            if isinstance(arg_spec, str) and arg_spec == "int":
                return Int()
            if isinstance(arg_spec, Int | Tensor | Object):
                return arg_spec
            if isinstance(arg_spec, list | tuple | Tuple):
                elems = (
                    list(arg_spec) if not isinstance(arg_spec, Tuple) else list(arg_spec.elements)
                )
                converted = (
                    tuple(_convert(e, f"{arg_name}_{i}") for i, e in enumerate(elems))
                    if isinstance(arg_spec, tuple)
                    else [_convert(e, f"{arg_name}_{i}") for i, e in enumerate(elems)]
                )
                return Tuple(arg_name, converted)
            raise TypeError(f"Invalid spec for argument {arg_name!r}: {arg_spec!r}")

        for arg_name in arg_names:
            if arg_name in spec:  # type: ignore[operator]
                arg_specs.append(_convert(spec[arg_name], arg_name))  # type: ignore[index]

        return MethodSpec(
            method, arg_names, arg_specs, param_mode=param_mode, effect_mode=effect_mode
        )

    @staticmethod
    def from_torch(args: list[typing.Any], method: typing.Callable) -> "MethodSpec":
        """Build a MethodSpec from a list of example torch tensors."""
        from .torch import _method_spec_from_torch  # pylint: disable=import-outside-toplevel

        return _method_spec_from_torch(args, method)


# ===========================================================================
# ModuleSpec  -  native C++ object
#
# named_params is pre-collected from Module.named_parameters() so the C++
# Exporter never needs to call back into Python for parameter discovery.
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.spec.ModuleSpec")
class ModuleSpec(tvm_ffi.Object):
    """Spec for a complete nn.Module compilation."""

    def __init__(
        self,
        module: "_nn_module_class",
        method_names: list[str],
        method_specs: list[MethodSpec],
    ) -> None:
        from .core import Parameter, _attribute_finder  # pylint: disable=import-outside-toplevel

        named_params = {
            name: param
            for name, param in _attribute_finder(
                module, prefix="", condition_yield=lambda x: isinstance(x, Parameter)
            )
        }
        self.__ffi_init__(method_names, method_specs, named_params)
        # Keep a Python-side reference to the module so that exporter.py
        # can call _attribute_finder on it (e.g. to discover Effect instances).
        # The C++ ModuleSpec only stores named_params, not the module itself.
        self._module = module

    # method_names, method_specs, named_params are C++ fields exposed
    # directly as attributes by register_object; no property wrappers needed.

    @staticmethod
    def from_raw(spec: ModuleSpecType, module: "_nn_module_class") -> "ModuleSpec":
        """Build a ModuleSpec from a raw dict or return as-is."""
        if isinstance(spec, ModuleSpec):
            return spec
        method_names = list(spec.keys())  # type: ignore[union-attr]
        method_specs: list[MethodSpec] = []
        for method_name in method_names:
            method_spec = spec[method_name]  # type: ignore[index]
            if not isinstance(method_spec, MethodSpec):
                method_spec = MethodSpec.from_raw(method_spec, getattr(module, method_name))
            method_specs.append(method_spec)
        return ModuleSpec(module, method_names, method_specs)

    def __repr__(self) -> str:
        lines = "\n".join(
            "  " + ms._repr(name) for name, ms in zip(self.method_names, self.method_specs)
        )
        return f"ModuleSpec:\n{lines}"
