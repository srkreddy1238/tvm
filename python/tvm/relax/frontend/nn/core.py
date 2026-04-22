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
  Tensor       - relax.Var wrapper with TensorStructInfo
  Parameter    - Tensor subclass with optional data + attrs
  Object       - relax.Var wrapper with ObjectStructInfo
  ModuleList   - ordered list of sub-modules (ffi::Array<Any>)
  ModuleDict   - ordered string-keyed map of sub-modules (ffi::Map<String,Any>)

Pure Python (cannot be ported):
  SubroutineMixin - uses __init_subclass__, inspect.signature, functools.wraps
  Module          - forward() is Python-defined; export_tvm/jit use Python-only
                    infrastructure (Exporter, spec, VirtualMachine)
  Effect          - abstract base with Python virtual dispatch

Module data-management methods (named_parameters, state_dict, load_state_dict,
to) delegate to C++ FFI helpers that traverse both Python __dict__ and native
C++ field metadata, so no traversal logic lives in Python.
"""

from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import tvm_ffi

import tvm
import tvm.runtime
from tvm import tir
from tvm.ir.transform import Pass
from tvm.runtime import Device
from tvm.runtime import device as as_device
from tvm.runtime.vm import VirtualMachine
from tvm.target import Target

from .... import relax as rx
from ...block_builder import BlockBuilder
from ...struct_info import TensorStructInfo
from ._tensor_op import _TensorOp
from .subroutine import SubroutineMixin

if TYPE_CHECKING:
    from . import spec as _spec

# ---------------------------------------------------------------------------
# FFI API - populated by init_ffi_api("relax.frontend.nn") at import time.
# Each attribute corresponds to a C++ global registered as
# "relax.frontend.nn.<Name>" (the dot-free suffix becomes the attribute).
# ---------------------------------------------------------------------------
from . import _ffi_api

# ---------------------------------------------------------------------------
# Default dtype
# ---------------------------------------------------------------------------


def get_default_dtype() -> str:
    """Return the current default parameter dtype (default: float32)."""
    return str(_ffi_api.GetDefaultDtype())


def set_default_dtype(dtype: str) -> None:
    """Set the default parameter dtype."""
    _ffi_api.SetDefaultDtype(dtype)


# ---------------------------------------------------------------------------
# _ConstTensor  -  lightweight Python wrapper for inline constants
#
# from_const / from_scalar return this instead of a full Tensor so that
# the Constant expr is passed inline to op calls (e.g. R.full fill_value)
# without being emitted as a separate dataflow binding.
# ---------------------------------------------------------------------------


class _ConstTensor(_TensorOp):
    """Python-only Tensor wrapper for relax.Constant values.

    Holds the Constant expr directly so it can be passed inline to ops
    without being emitted as a standalone dataflow binding.
    """

    def __init__(self, const_expr: rx.Constant) -> None:
        self._const_expr = const_expr

    @property
    def _expr(self) -> rx.Constant:
        return self._const_expr

    @property
    def shape(self) -> list[int | tir.PrimExpr]:
        sinfo = self._const_expr.struct_info_
        if sinfo is None or not isinstance(sinfo, TensorStructInfo):
            return []
        if sinfo.shape is None:
            return []
        shape_sinfo = sinfo.shape.struct_info_
        if shape_sinfo is None or not hasattr(shape_sinfo, "values") or shape_sinfo.values is None:
            return []
        return [int(x) if isinstance(x, tir.IntImm) else x for x in shape_sinfo.values]

    @property
    def ndim(self) -> int:
        sinfo = self._const_expr.struct_info_
        if sinfo is None or not isinstance(sinfo, TensorStructInfo):
            return 0
        return sinfo.ndim

    @property
    def dtype(self) -> str:
        sinfo = self._const_expr.struct_info_
        if sinfo is None or not isinstance(sinfo, TensorStructInfo):
            return ""
        return str(sinfo.dtype)

    def __repr__(self) -> str:
        return f'ConstTensor({self.shape}, "{self.dtype}")'


# ===========================================================================
# Tensor  -  native C++ object
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Tensor")
class Tensor(_TensorOp):
    """Symbolic tensor backed by a relax.Var with TensorStructInfo."""

    def __init__(self, *, _expr: rx.Var) -> None:
        self.__ffi_init__(_expr)

    @staticmethod
    def from_const(data) -> "Tensor":
        return _ConstTensor(rx.const(data))

    @staticmethod
    def from_scalar(data: int | float, dtype: str) -> "Tensor":
        return _ConstTensor(rx.const(data, dtype=dtype))

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
        sinfo = self._expr.struct_info_
        if sinfo is None or not isinstance(sinfo, TensorStructInfo):
            return []
        if sinfo.shape is None:
            return []
        shape_sinfo = sinfo.shape.struct_info_
        if shape_sinfo is None or not hasattr(shape_sinfo, "values") or shape_sinfo.values is None:
            return []
        return [int(x) if isinstance(x, tir.IntImm) else x for x in shape_sinfo.values]

    @property
    def ndim(self) -> int:
        sinfo = self._expr.struct_info_
        if sinfo is None or not isinstance(sinfo, TensorStructInfo):
            return -1
        return sinfo.ndim

    @property
    def dtype(self) -> str:
        sinfo = self._expr.struct_info_
        if sinfo is None or not isinstance(sinfo, TensorStructInfo):
            return ""
        return str(sinfo.dtype)

    @property
    def _expr(self) -> rx.Var:
        return tvm_ffi.Object.__getattribute__(self, "expr")  # C++ field

    def __repr__(self) -> str:
        return f'Tensor({self.shape}, "{self.dtype}")'


# ===========================================================================
# Parameter  -  native C++ object (subclass of Tensor)
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
        self.__ffi_init__(
            _ffi_api.MakePlaceholder(list(shape), dtype, "param"),
            None,  # data
            {},  # attrs
        )

    @property
    def data(self) -> tvm.runtime.Tensor | None:
        # Read C++ 'data' field directly via its FieldGetter (bypasses this property).
        return _PARAMETER_DATA_GETTER(self)

    @data.setter
    def data(self, value: None | tvm.runtime.Tensor | np.ndarray | Any) -> None:
        if value is None:
            _PARAMETER_DATA_SETTER(self, None)
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
        _PARAMETER_DATA_SETTER(self, value)

    @property
    def attrs(self) -> dict:
        return dict(_PARAMETER_ATTRS_GETTER(self))

    def to(self, dtype: str | None = None) -> None:
        if dtype is not None:
            _PARAMETER_TO_METHOD(self, dtype)


# ---------------------------------------------------------------------------
# Lazy C++ field/method accessors for Parameter.
# These are populated on first use after register_object has run, so the
# C++ descriptors are available.  We cannot use self.field_name inside the
# class body because our @property definitions shadow the C++ descriptors.
# ---------------------------------------------------------------------------


def _get_parameter_type_info():
    return tvm_ffi.core._type_cls_to_type_info(Parameter)


def _PARAMETER_DATA_GETTER(self):
    ti = _get_parameter_type_info()
    if ti:
        for f in ti.fields:
            if f.name == "data":
                return f.getter(self)
    return None


def _PARAMETER_DATA_SETTER(self, value):
    ti = _get_parameter_type_info()
    if ti:
        for f in ti.fields:
            if f.name == "data":
                f.setter(self, value)
                return


def _PARAMETER_ATTRS_GETTER(self):
    ti = _get_parameter_type_info()
    if ti:
        for f in ti.fields:
            if f.name == "attrs":
                return f.getter(self)
    return {}


def _PARAMETER_TO_METHOD(self, dtype):
    ti = _get_parameter_type_info()
    if ti:
        for m in ti.methods:
            if m.name == "to":
                m.func(self, dtype)
                return


# ===========================================================================
# Object  -  native C++ object
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Object")
class Object(tvm_ffi.Object):
    """Wrapper around a relax.Var with ObjectStructInfo (e.g. KVCache handle)."""

    def __init__(self, *, _expr: rx.Expr, _name: str) -> None:
        if not isinstance(_expr, rx.Var):
            _expr = BlockBuilder.current().emit(_expr, _name)
        self.__ffi_init__(_expr)

    @property
    def _expr(self) -> rx.Var:
        return tvm_ffi.Object.__getattribute__(self, "expr")  # C++ field


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
# Effect  (pure Python - abstract base, Python virtual dispatch)
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
# ModuleList  -  native C++ object
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.ModuleList")
class ModuleList(tvm_ffi.Object, SubroutineMixin):
    """Ordered list of sub-modules backed by a native C++ ffi::Array<Any>.

    All elements are stored in C++.  Python provides the standard list
    interface (__iter__, __getitem__, __setitem__, __len__, append).
    """

    def __init__(self, modules: list) -> None:
        self.__ffi_init__(list(modules))

    # ---- list interface (delegates to C++ ffi::Array field) ----------------

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, idx: int):
        return self.modules[idx]

    def __setitem__(self, idx: int, module) -> None:
        mods = list(self.modules)
        mods[idx] = module
        self.modules = mods

    def __len__(self) -> int:
        return len(self.modules)

    def append(self, module) -> None:
        mods = list(self.modules)
        mods.append(module)
        self.modules = mods

    # ---- parameter / dtype helpers -----------------------------------------

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        params = _ffi_api.GetContainerParameters(self, prefix)
        yield from params.items()

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def to(self, dtype: str | None = None) -> None:
        if dtype is not None:
            _ffi_api.ContainerApplyTo(self, dtype)

    def forward(self, x):
        for m in self:
            x = m(x)
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ===========================================================================
# ModuleDict  -  native C++ object
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.ModuleDict")
class ModuleDict(tvm_ffi.Object, SubroutineMixin):
    """Ordered string-keyed map of sub-modules backed by a native C++ ffi::Map<String,Any>.

    All elements are stored in C++.  Python provides the standard dict
    interface (__iter__, __getitem__, __setitem__, __len__, keys, values, items).
    """

    def __init__(self, modules: OrderedDict | None = None) -> None:
        self.__ffi_init__(dict(modules) if modules else {})

    # ---- dict interface (delegates to C++ ffi::Map field) ------------------

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, key: str):
        return self.modules[key]

    def __setitem__(self, key: str, module) -> None:
        self.modules[key] = module

    def __len__(self) -> int:
        return len(self.modules)

    def __contains__(self, key: str) -> bool:
        return key in self.modules

    def keys(self):
        return self.modules.keys()

    def values(self):
        return self.modules.values()

    def items(self):
        return self.modules.items()

    def get(self, key: str, default=None):
        m = self.modules
        return m[key] if key in m else default

    def update(self, modules: dict) -> None:
        for k, v in modules.items():
            self.modules[k] = v

    def clear(self) -> None:
        self.modules = {}

    def pop(self, key: str):
        m = dict(self.modules)
        val = m.pop(key)
        self.modules = m
        return val

    # ---- parameter / dtype helpers -----------------------------------------

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        params = _ffi_api.GetContainerParameters(self, prefix)
        yield from params.items()

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def to(self, dtype: str | None = None) -> None:
        if dtype is not None:
            _ffi_api.ContainerApplyTo(self, dtype)


# ===========================================================================
# Module  (pure Python - forward() dispatch + export_tvm/jit)
#
# Cannot be ported to C++ because:
#   1. forward() is defined by Python subclasses and dispatched via Python MRO.
#   2. SubroutineMixin uses __init_subclass__, inspect.signature, functools.wraps
#      - all Python-only metaprogramming.
#   3. export_tvm / jit depend on Exporter, spec, VirtualMachine, Target
#      - all Python-only orchestration infrastructure.
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
        if dtype is not None:
            # Delegate to C++ helper which handles Parameters, ModuleLists,
            # ModuleDicts, native C++ modules, and nested dicts uniformly.
            _ffi_api.PythonModuleApplyTo(self.__dict__, dtype)
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

        Delegates to the C++ Exporter, which uses the fast ExportToIRModule
        path when the spec contains only SpecInt/SpecTensor arg specs, and
        falls back to the Python path for specs that use spec.Object.

        External modules (nn.ExternModule) registered via nn.add_extern during
        forward() are compiled and attached via the C++ AttachExternModules
        pass when allow_extern=True.
        """
        from . import spec as _spec  # pylint: disable=import-outside-toplevel
        from .exporter import Exporter  # pylint: disable=import-outside-toplevel

        module_spec = _spec.ModuleSpec.from_raw(spec, self)
        mod, params, ext_mods = Exporter(debug=debug).build(module_spec)

        if allow_extern:
            return mod, params, ext_mods
        if ext_mods:
            raise ValueError("ExternModules present; set allow_extern=True.")
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
# _attribute_finder  -  unified parameter traversal
#
# Handles three cases:
#   1. ModuleList / ModuleDict  -> delegate to C++ GetContainerParameters
#   2. Native C++ module object -> call C++ GetNativeParameters on its handle
#   3. Pure-Python Module       -> walk __dict__ recursively
# ===========================================================================


def _attribute_finder(root, prefix: str, condition_yield: Callable[[Any], bool]):
    """Recursively yield (dotted_name, value) pairs satisfying condition_yield."""

    # --- Case 1: native container types ------------------------------------
    if isinstance(root, ModuleList | ModuleDict):
        params = _ffi_api.GetContainerParameters(root, prefix)
        for name, param in params.items():
            if condition_yield(param):
                yield name, param
        return

    # --- Case 2: native C++ module (not a container) -----------------------
    if isinstance(root, tvm_ffi.Object) and not isinstance(root, ModuleList | ModuleDict):
        try:
            native_params = _ffi_api.GetNativeParameters(root)
            for fname, param in native_params.items():
                full = f"{prefix}{fname}" if prefix else fname
                if condition_yield(param):
                    yield full, param
        except Exception:  # pylint: disable=broad-except
            pass

    # --- Case 3: pure-Python Module - walk __dict__ -------------------------
    if not hasattr(root, "__dict__"):
        return
    for name, item in root.__dict__.items():
        child_prefix = f"{prefix}{name}."
        if condition_yield(item):
            yield f"{prefix}{name}", item
        elif isinstance(item, ModuleList | ModuleDict):
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
