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
"""nn frontend core types.

Native C++ objects (registered via tvm_ffi.register_object):
  Tensor       – relax.Var wrapper with TensorStructInfo
  Parameter    – Tensor subclass with optional data + attrs
  Object       – relax.Var wrapper with ObjectStructInfo
  ModuleList   – ordered list of sub-modules (ffi::Array<Any>)
  ModuleDict   – ordered string-keyed map of sub-modules (ffi::Map<String,Any>)

Pure Python (cannot be ported):
  SubroutineMixin – uses __init_subclass__, inspect.signature, functools.wraps
  Module          – forward() is Python-defined; export_tvm/jit use Python-only
                    infrastructure (Exporter, spec, VirtualMachine)
  Effect          – abstract base with Python virtual dispatch

Module data-management methods (named_parameters, state_dict, load_state_dict,
to) delegate to C++ FFI helpers that traverse both Python __dict__ and native
C++ field metadata, so no traversal logic lives in Python.
"""

from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import tvm_ffi

import tvm
import tvm.runtime
from tvm import tir
from tvm.ir import IRModule
from tvm.ir.transform import Pass
from tvm.runtime import Device
from tvm.runtime import device as as_device
from tvm.runtime.vm import VirtualMachine
from tvm.target import Target

from .... import relax as rx
from ...block_builder import BlockBuilder
from ...struct_info import ObjectStructInfo, ShapeStructInfo, TensorStructInfo, TupleStructInfo
from ._tensor_op import _TensorOp
from .subroutine import SubroutineMixin

if TYPE_CHECKING:
    import torch
    from . import spec as _spec
    from .extern import ExternModule

# ---------------------------------------------------------------------------
# FFI API – populated by init_ffi_api("relax.frontend.nn") at import time.
# Each attribute corresponds to a C++ global registered as
# "relax.frontend.nn.<Name>" (the dot-free suffix becomes the attribute).
# ---------------------------------------------------------------------------
from . import _ffi_api  # noqa: E402  pylint: disable=wrong-import-position


# ---------------------------------------------------------------------------
# Default dtype
# ---------------------------------------------------------------------------

def get_default_dtype() -> str:
    """Return the current default parameter dtype (default: float32)."""
    return str(_ffi_api.GetDefaultDtype())


def set_default_dtype(dtype: str) -> None:
    """Set the default parameter dtype."""
    _ffi_api.SetDefaultDtype(dtype)


# ===========================================================================
# Tensor  –  native C++ object
# ===========================================================================

@tvm_ffi.register_object("relax.frontend.nn.Tensor")
class Tensor(_TensorOp):
    """Symbolic tensor backed by a relax.Var with TensorStructInfo."""

    def __init__(self, *, _expr: rx.Var) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Tensor, _expr)

    @staticmethod
    def from_const(data) -> "Tensor":
        return Tensor(_expr=rx.const(data))

    @staticmethod
    def from_scalar(data: int | float, dtype: str) -> "Tensor":
        return Tensor(_expr=rx.const(data, dtype=dtype))

    @staticmethod
    def from_struct_info(struct_info: rx.TensorStructInfo, name: str = "tensor") -> "Tensor":
        return Tensor(_expr=_ffi_api.MakeTensorFromStructInfo(struct_info, name))

    @staticmethod
    def placeholder(
        shape: Sequence[int | str | tir.PrimExpr],
        dtype: str,
        name: str = "tensor",
    ) -> "Tensor":
        return Tensor(_expr=_ffi_api.MakePlaceholder(list(shape), dtype, name))

    @property
    def shape(self) -> list[int | tir.PrimExpr]:
        raw = self.__object_handle__.shape()  # type: ignore[attr-defined]
        return [int(x) if isinstance(x, tir.IntImm) else x for x in raw]

    @property
    def ndim(self) -> int:
        return int(self.__object_handle__.ndim())  # type: ignore[attr-defined]

    @property
    def dtype(self) -> str:
        return str(self.__object_handle__.dtype())  # type: ignore[attr-defined]

    @property
    def _expr(self) -> rx.Var:
        return self.__object_handle__.expr  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return f'Tensor({self.shape}, "{self.dtype}")'


# ===========================================================================
# Parameter  –  native C++ object (subclass of Tensor)
# ===========================================================================

@tvm_ffi.register_object("relax.frontend.nn.Parameter")
class Parameter(Tensor):
    """Trainable parameter: a Tensor optionally bound to a concrete value."""

    def __init__(
        self,
        shape: Sequence[int | str | tir.PrimExpr],
        dtype: str | None = None,
    ) -> None:
        if dtype is None:
            dtype = get_default_dtype()
        self.__init_handle_by_constructor__(
            _ffi_api.Parameter,
            _ffi_api.MakePlaceholder(list(shape), dtype, "param"),
            None,  # data
            {},    # attrs
        )

    @property
    def data(self) -> tvm.runtime.Tensor | None:
        return self.__object_handle__.data  # type: ignore[attr-defined]

    @data.setter
    def data(self, value: Union[None, tvm.runtime.Tensor, np.ndarray, Any]) -> None:
        if value is None:
            self.__object_handle__.data = None  # type: ignore[attr-defined]
            return
        if isinstance(value, tvm.runtime.Tensor):
            pass
        elif isinstance(value, np.ndarray):
            value = tvm.runtime.tensor(value)
        elif hasattr(value, "__dlpack__"):
            value = _from_dlpack(value)
        else:
            raise TypeError(f"Unsupported data type: {type(value)}")
        if value.shape != tuple(self.shape):
            raise ValueError(f"Shape mismatch: expected {tuple(self.shape)}, got {value.shape}")
        if value.dtype != self.dtype:
            raise ValueError(f"Dtype mismatch: expected {self.dtype}, got {value.dtype}")
        self.__object_handle__.data = value  # type: ignore[attr-defined]

    @property
    def attrs(self) -> dict:
        return dict(self.__object_handle__.attrs)  # type: ignore[attr-defined]

    def to(self, dtype: str | None = None) -> None:
        if dtype is not None:
            self.__object_handle__.to(dtype)  # type: ignore[attr-defined]


# ===========================================================================
# Object  –  native C++ object
# ===========================================================================

@tvm_ffi.register_object("relax.frontend.nn.Object")
class Object:
    """Wrapper around a relax.Var with ObjectStructInfo (e.g. KVCache handle)."""

    def __init__(self, *, _expr: rx.Expr, _name: str) -> None:
        if not isinstance(_expr, rx.Var):
            _expr = BlockBuilder.current().emit(_expr, _name)
        self.__init_handle_by_constructor__(_ffi_api.Object, _expr)

    @property
    def _expr(self) -> rx.Var:
        return self.__object_handle__.expr  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# wrap_nested / _unwrap_ffi_result
# ---------------------------------------------------------------------------

def wrap_nested(expr: rx.Expr, name: str) -> "Tensor | tuple":
    result = _ffi_api.WrapNested(expr, name)
    return _unwrap_ffi_result(result)


def _unwrap_ffi_result(result) -> "Tensor | tuple":
    if isinstance(result, rx.Var):
        return Tensor(_expr=result)
    return tuple(_unwrap_ffi_result(r) for r in result)


# ===========================================================================
# Effect  (pure Python – abstract base, Python virtual dispatch)
# ===========================================================================

class Effect:
    """Abstract base for side-effecting operations (IO, KVCache, etc.)."""

    def emit_init(self, name_hint: str, builder: BlockBuilder) -> list[rx.DataflowVar]:
        raise NotImplementedError

    def create(self, name_hint: str) -> list[rx.Var]:
        raise NotImplementedError

    def set_state(self, state_vars: list[rx.Var]) -> None:
        raise NotImplementedError

    def finalize(self) -> list[rx.Var]:
        raise NotImplementedError

    def to(self, dtype: str | None = None) -> None:
        pass


# ===========================================================================
# ModuleList  –  native C++ object
# ===========================================================================

@tvm_ffi.register_object("relax.frontend.nn.ModuleList")
class ModuleList(SubroutineMixin):
    """Ordered list of sub-modules backed by a native C++ ffi::Array<Any>.

    All elements are stored in C++.  Python provides the standard list
    interface (__iter__, __getitem__, __setitem__, __len__, append).
    """

    def __init__(self, modules: list) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ModuleList, list(modules))

    # ---- list interface (delegates to C++ ffi::Array field) ----------------

    def __iter__(self):
        return iter(self.__object_handle__.modules)  # type: ignore[attr-defined]

    def __getitem__(self, idx: int):
        return self.__object_handle__.modules[idx]  # type: ignore[attr-defined]

    def __setitem__(self, idx: int, module) -> None:
        mods = list(self.__object_handle__.modules)  # type: ignore[attr-defined]
        mods[idx] = module
        self.__object_handle__.modules = mods  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return len(self.__object_handle__.modules)  # type: ignore[attr-defined]

    def append(self, module) -> None:
        mods = list(self.__object_handle__.modules)  # type: ignore[attr-defined]
        mods.append(module)
        self.__object_handle__.modules = mods  # type: ignore[attr-defined]

    # ---- parameter / dtype helpers -----------------------------------------

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        params = _ffi_api.GetContainerParameters(self.__object_handle__, prefix)  # type: ignore[attr-defined]
        yield from params.items()

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def to(self, dtype: str | None = None) -> None:
        if dtype is not None:
            _ffi_api.ContainerApplyTo(self.__object_handle__, dtype)  # type: ignore[attr-defined]

    def forward(self, x):
        for m in self:
            x = m(x)
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ===========================================================================
# ModuleDict  –  native C++ object
# ===========================================================================

@tvm_ffi.register_object("relax.frontend.nn.ModuleDict")
class ModuleDict(SubroutineMixin):
    """Ordered string-keyed map of sub-modules backed by a native C++ ffi::Map<String,Any>.

    All elements are stored in C++.  Python provides the standard dict
    interface (__iter__, __getitem__, __setitem__, __len__, keys, values, items).
    """

    def __init__(self, modules: OrderedDict | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ModuleDict, dict(modules) if modules else {})

    # ---- dict interface (delegates to C++ ffi::Map field) ------------------

    def __iter__(self):
        return iter(self.__object_handle__.modules)  # type: ignore[attr-defined]

    def __getitem__(self, key: str):
        return self.__object_handle__.modules[key]  # type: ignore[attr-defined]

    def __setitem__(self, key: str, module) -> None:
        self.__object_handle__.modules[key] = module  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return len(self.__object_handle__.modules)  # type: ignore[attr-defined]

    def __contains__(self, key: str) -> bool:
        return key in self.__object_handle__.modules  # type: ignore[attr-defined]

    def keys(self):
        return self.__object_handle__.modules.keys()  # type: ignore[attr-defined]

    def values(self):
        return self.__object_handle__.modules.values()  # type: ignore[attr-defined]

    def items(self):
        return self.__object_handle__.modules.items()  # type: ignore[attr-defined]

    def get(self, key: str, default=None):
        m = self.__object_handle__.modules  # type: ignore[attr-defined]
        return m[key] if key in m else default

    def update(self, modules: dict) -> None:
        for k, v in modules.items():
            self.__object_handle__.modules[k] = v  # type: ignore[attr-defined]

    def clear(self) -> None:
        self.__object_handle__.modules = {}  # type: ignore[attr-defined]

    def pop(self, key: str):
        m = dict(self.__object_handle__.modules)  # type: ignore[attr-defined]
        val = m.pop(key)
        self.__object_handle__.modules = m  # type: ignore[attr-defined]
        return val

    # ---- parameter / dtype helpers -----------------------------------------

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        params = _ffi_api.GetContainerParameters(self.__object_handle__, prefix)  # type: ignore[attr-defined]
        yield from params.items()

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def to(self, dtype: str | None = None) -> None:
        if dtype is not None:
            _ffi_api.ContainerApplyTo(self.__object_handle__, dtype)  # type: ignore[attr-defined]


# ===========================================================================
# Module  (pure Python – forward() dispatch + export_tvm/jit)
#
# Cannot be ported to C++ because:
#   1. forward() is defined by Python subclasses and dispatched via Python MRO.
#   2. SubroutineMixin uses __init_subclass__, inspect.signature, functools.wraps
#      – all Python-only metaprogramming.
#   3. export_tvm / jit depend on Exporter, spec, VirtualMachine, Target
#      – all Python-only orchestration infrastructure.
#
# Data-management methods (named_parameters, state_dict, load_state_dict, to)
# delegate entirely to C++ FFI helpers so no traversal logic lives in Python.
# ===========================================================================

class Module(SubroutineMixin):
    """Base class for neural network components.

    Subclass and implement forward() to define your model.
    All parameter management delegates to C++ FFI helpers.
    """

    # ---- parameter traversal (delegates to C++ helpers) --------------------

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Yield (name, Parameter) pairs for all parameters in this module."""
        yield from _attribute_finder(self, prefix, lambda x: isinstance(x, Parameter))

    def parameters(self) -> Iterator[Parameter]:
        """Yield all Parameter values."""
        for _, p in self.named_parameters():
            yield p

    def state_dict(
        self, *, prefix: str = "", destination: dict[str, Parameter] | None = None
    ) -> dict[str, Parameter]:
        """Return an ordered dict of all parameters keyed by dotted name."""
        if destination is None:
            destination = OrderedDict()
        for name, param in _attribute_finder(self, prefix, lambda x: isinstance(x, Parameter)):
            destination[name] = param
        return destination

    def load_state_dict(
        self, state_dict: dict[str, Parameter], strict: bool = True
    ) -> tuple[list[str], list[str]]:
        """Load parameters from state_dict into this module."""
        self_sd = self.state_dict()
        missing, unexpected = [], []
        for key, value in state_dict.items():
            if key not in self_sd:
                unexpected.append(key)
                continue
            if value.data is None:
                raise ValueError(f"Parameter '{key}' has no concrete data")
            self_sd.pop(key).data = value.data
        missing = list(self_sd.keys())
        if strict and (missing or unexpected):
            raise KeyError(f"Missing keys: {missing}  Unexpected keys: {unexpected}")
        return missing, unexpected

    # ---- forward dispatch --------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch to forward()."""
        if not hasattr(self, "forward"):
            raise NotImplementedError(f"{type(self).__name__} has no forward()")
        return self.forward(*args, **kwargs)  # pylint: disable=no-member

    # ---- dtype conversion --------------------------------------------------

    def to(self, dtype: str | None = None) -> None:
        """Recursively convert all parameters and sub-modules to dtype."""
        for item in self.__dict__.values():
            if isinstance(item, (ModuleList, ModuleDict)):
                # Native containers: delegate to C++ ContainerApplyTo
                if dtype is not None:
                    _ffi_api.ContainerApplyTo(item.__object_handle__, dtype)
            elif hasattr(item, "to") and callable(item.to):
                item.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype  # pylint: disable=attribute-defined-outside-init

    # ---- export / jit (Python-only orchestration) --------------------------

    def export_tvm(
        self,
        spec: "_spec.ModuleSpecType",
        debug: bool = False,
        allow_extern: bool = False,
    ):
        """Export this module to a TVM IRModule.

        Uses the C++ ExportToIRModule path when the spec contains only
        SpecInt/SpecTensor arg specs (no spec.Object).  Falls back to the
        Python Exporter for specs that use spec.Object (KVCache etc.).
        """
        from . import spec as _spec  # pylint: disable=import-outside-toplevel

        module_spec = _spec.ModuleSpec.from_raw(spec, self)

        # Check whether any method uses spec.Object (Python-only path)
        has_object_spec = any(
            isinstance(s, _spec.Object)
            for ms in module_spec.method_specs
            for s in ms.arg_specs
        )

        if has_object_spec or allow_extern:
            # Fall back to Python Exporter (handles spec.Object, ExternModules)
            from .exporter import Exporter  # pylint: disable=import-outside-toplevel
            mod, params, ext_mods = Exporter(debug=debug).build(
                _spec._PythonModuleSpec(module_spec)
            )
            if allow_extern:
                return mod, params, ext_mods
            if ext_mods:
                raise ValueError("ExternModules present; set allow_extern=True.")
            return mod, params

        # Native C++ path: ExportToIRModule
        mod = _ffi_api.ExportToIRModule(module_spec.__object_handle__, debug)
        # Collect params in the same order as named_params
        params = list(module_spec.named_params.items())
        return mod, params

    def jit(
        self,
        spec: "_spec.ModuleSpec",
        device: str | Device = "cpu",
        pipeline: None | str | Pass = "default_build",
        out_format: str = "torch",
        debug: bool = False,
    ) -> Any:
        """JIT-compile this module to an executable."""
        from ...transform import AttachExternModules  # pylint: disable=import-outside-toplevel
        from ...vm_build import build as relax_build  # pylint: disable=import-outside-toplevel
        from . import spec as _spec  # pylint: disable=import-outside-toplevel
        from .exporter import Exporter  # pylint: disable=import-outside-toplevel

        device = as_device(device)
        spec = _spec.ModuleSpec.from_raw(spec, self)
        mod, params, ext_mods = Exporter(debug=debug).build(spec)
        mod = AttachExternModules(ext_mods)(mod)
        vm = VirtualMachine(
            relax_build(mod, target=Target.from_device(device), relax_pipeline=pipeline),
            device,
        )
        params = _param_to_tensor(params, device)
        if out_format == "torch":
            from . import torch  # pylint: disable=import-outside-toplevel
            return torch.TorchModule(spec=spec, params=params, vm=vm)
        raise ValueError(f"Unknown out_format: {out_format!r}")


# ===========================================================================
# _attribute_finder  –  unified parameter traversal
#
# Handles three cases:
#   1. ModuleList / ModuleDict  -> delegate to C++ GetContainerParameters
#   2. Native C++ module object -> call C++ GetNativeParameters on its handle
#   3. Pure-Python Module       -> walk __dict__ recursively
# ===========================================================================

def _attribute_finder(root, prefix: str, condition_yield: Callable[[Any], bool]):
    """Recursively yield (dotted_name, value) pairs satisfying condition_yield."""

    # --- Case 1: native container types ------------------------------------
    if isinstance(root, (ModuleList, ModuleDict)):
        params = _ffi_api.GetContainerParameters(root.__object_handle__, prefix)
        for name, param in params.items():
            if condition_yield(param):
                yield name, param
        return

    # --- Case 2: native C++ module (has __object_handle__, not a container) -
    if hasattr(root, "__object_handle__") and not isinstance(root, (ModuleList, ModuleDict)):
        try:
            native_params = _ffi_api.GetNativeParameters(root.__object_handle__)
            for fname, param in native_params.items():
                full = f"{prefix}{fname}" if prefix else fname
                if condition_yield(param):
                    yield full, param
        except Exception:  # pylint: disable=broad-except
            pass

    # --- Case 3: pure-Python Module – walk __dict__ -------------------------
    if not hasattr(root, "__dict__"):
        return
    for name, item in root.__dict__.items():
        child_prefix = f"{prefix}{name}."
        if condition_yield(item):
            yield f"{prefix}{name}", item
        elif isinstance(item, (ModuleList, ModuleDict)):
            yield from _attribute_finder(item, child_prefix, condition_yield)
        elif isinstance(item, Module):
            yield from _attribute_finder(item, child_prefix, condition_yield)


# ===========================================================================
# Internal helpers
# ===========================================================================

def _from_dlpack(tensor) -> tvm.runtime.Tensor:
    try:
        return tvm.runtime.from_dlpack(tensor)
    except RuntimeError:
        pass
    device_type = tensor.device.type
    device_id = tensor.device.index or 0
    return tvm.runtime.tensor(
        tensor.numpy(),
        device=Device(Device._DEVICE_NAME_TO_TYPE[device_type], device_id),
    )


def _param_to_tensor(
    params: list[tuple[str, Parameter]], device: Device
) -> list[tvm.runtime.Tensor]:
    results, missing = [], []
    for name, param in params:
        if param.data is None:
            missing.append(name)
        else:
            results.append(param.data.copyto(target=device))
    if missing:
        raise ValueError(f"Parameters not bound to data: {', '.join(missing)}")
    return results
