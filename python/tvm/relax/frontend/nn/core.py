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
"""Core types for the nn module frontend.

Native C++ objects (registered via ``tvm_ffi.register_object``):

  Tensor
      Wraps a ``relax.Var`` with ``TensorStructInfo``.  Exposes ``shape``,
      ``ndim``, and ``dtype`` as read-only properties and inherits all
      operator overloads from ``_TensorOp``.

  Parameter
      Extends ``Tensor`` with an optional concrete ``data`` buffer
      (``tvm.runtime.Tensor``) and a string-keyed ``attrs`` map for
      quantisation metadata and similar annotations.

  Object
      Wraps a ``relax.Var`` with ``ObjectStructInfo``.  Used for opaque
      runtime handles such as KVCache.

  ModuleList
      Ordered list of sub-modules backed by a native C++ ``ffi::Array<Any>``.
      Pure-Python items are kept in a Python-side ``_py_items`` list with
      ``None`` sentinels in the C++ array.

  ModuleDict
      Ordered string-keyed map of sub-modules backed by a native C++
      ``ffi::Map<String,Any>``.  Pure-Python items are kept in a Python-side
      ``_py_modules`` ``OrderedDict``.

Pure Python (cannot be ported to C++):

  SubroutineMixin
      Uses ``__init_subclass__``, ``inspect.signature``, and
      ``functools.wraps`` — all Python-only metaprogramming.

  Module
      ``forward()`` is defined by Python subclasses and dispatched via the
      Python MRO.  ``export_tvm`` / ``jit`` depend on ``Exporter``, ``spec``,
      and ``VirtualMachine`` — all Python-only orchestration.

  Effect
      Abstract base with Python virtual dispatch (``emit_init``, ``create``,
      ``set_state``, ``finalize``).

Data-management methods (``named_parameters``, ``state_dict``,
``load_state_dict``, ``to``) delegate to C++ FFI helpers that traverse both
Python ``__dict__`` and native C++ field metadata, so no traversal logic
lives in Python.
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
    """Return the current thread-local default parameter dtype.

    Returns
    -------
    dtype : str
        The current default dtype string, e.g. ``"float32"``.
    """
    return str(_ffi_api.GetDefaultDtype())


def set_default_dtype(dtype: str) -> None:
    """Set the thread-local default parameter dtype.

    Parameters
    ----------
    dtype : str
        New default dtype string, e.g. ``"float16"``.
    """
    _ffi_api.SetDefaultDtype(dtype)


# ---------------------------------------------------------------------------
# _ConstTensor  -  lightweight Python wrapper for inline constants
#
# from_const / from_scalar return this instead of a full Tensor so that
# the Constant expr is passed inline to op calls (e.g. R.full fill_value)
# without being emitted as a separate dataflow binding.
# ---------------------------------------------------------------------------


class _ConstTensor(_TensorOp):
    """Python-only ``Tensor`` wrapper for ``relax.Constant`` values.

    Holds the ``Constant`` expression directly so it can be passed inline
    to op calls (e.g. ``R.full`` fill_value) without being emitted as a
    separate dataflow binding.  Created by :meth:`Tensor.from_const` and
    :meth:`Tensor.from_scalar`.
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
        return f'Tensor({self.shape}, "{self.dtype}")'


# ===========================================================================
# Tensor  -  native C++ object
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.Tensor")
class Tensor(_TensorOp):
    """Symbolic tensor backed by a ``relax.Var`` with ``TensorStructInfo``.

    All operator overloads (``+``, ``-``, ``*``, ``/``, ``@``, indexing,
    etc.) are inherited from :class:`_TensorOp`.  The underlying relax
    expression is accessible via the ``_expr`` property.
    """

    def __init__(self, *, _expr: rx.Var) -> None:
        self.__ffi_init__(_expr)

    @staticmethod
    def from_const(data) -> "Tensor":
        """Create a constant ``Tensor`` from a numpy-compatible array.

        The constant is held inline and not emitted as a dataflow binding.

        Parameters
        ----------
        data :
            Any value accepted by ``relax.const`` (numpy array, scalar, etc.).

        Returns
        -------
        tensor : Tensor
            A :class:`_ConstTensor` wrapping the constant expression.
        """
        return _ConstTensor(rx.const(data))

    @staticmethod
    def from_scalar(data: int | float, dtype: str) -> "Tensor":
        """Create a scalar constant ``Tensor``.

        Parameters
        ----------
        data : int | float
            The scalar value.
        dtype : str
            Data type string, e.g. ``"float32"``.

        Returns
        -------
        tensor : Tensor
            A :class:`_ConstTensor` wrapping the scalar constant.
        """
        return _ConstTensor(rx.const(data, dtype=dtype))

    @staticmethod
    def from_struct_info(struct_info: rx.TensorStructInfo, name: str = "tensor") -> "Tensor":
        """Create a placeholder ``Tensor`` from an existing ``TensorStructInfo``.

        Parameters
        ----------
        struct_info : relax.TensorStructInfo
            The struct info to use for the new ``relax.Var``.
        name : str
            Name hint for the created ``Var``.

        Returns
        -------
        tensor : Tensor
            A new ``Tensor`` owning the created ``Var``.
        """
        return Tensor(_expr=_ffi_api.MakeTensorFromStructInfo(struct_info, name))

    @staticmethod
    def placeholder(
        shape: Sequence[int | str | tir.PrimExpr],
        dtype: str,
        name: str = "tensor",
    ) -> "Tensor":
        """Create an unbound placeholder ``Tensor``.

        Parameters
        ----------
        shape : Sequence[int | str | tir.PrimExpr]
            Shape specification.  Each element may be an ``int`` (static
            dimension), a ``str`` (symbolic variable name), or a
            ``tir.PrimExpr``.
        dtype : str
            Data type string, e.g. ``"float32"``.
        name : str
            Name hint for the created ``relax.Var``.

        Returns
        -------
        tensor : Tensor
            A new ``Tensor`` owning the placeholder ``Var``.
        """
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
    """Trainable parameter: a ``Tensor`` optionally bound to a concrete value.

    The ``data`` property holds the bound ``tvm.runtime.Tensor`` when the
    parameter has been loaded from a checkpoint; it is ``None`` for unbound
    parameters.  The ``attrs`` property exposes the C++ annotation map for
    quantisation metadata and similar user-defined annotations.
    """

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
# Fix Parameter.__c_ffi_init__
#
# register_object calls _add_class_attrs which sets __c_ffi_init__ only when
# `not hasattr(cls, name)`.  Since Parameter inherits __c_ffi_init__ from
# Tensor, the check passes and the 3-arg Parameter constructor is never
# installed.  Force-set it here directly from Parameter's own TypeInfo.
# ---------------------------------------------------------------------------
def _fix_parameter_c_ffi_init() -> None:
    ti = tvm_ffi.core._type_cls_to_type_info(Parameter)
    if ti is None:
        return
    for method in ti.methods:
        if method.name == "__ffi_init__":
            Parameter.__c_ffi_init__ = method.as_callable(Parameter)  # type: ignore[attr-defined]
            return


_fix_parameter_c_ffi_init()


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
    """Wrapper around a ``relax.Var`` with ``ObjectStructInfo``.

    Used for opaque runtime handles such as KVCache objects that are passed
    through the IR without shape or dtype information.
    """

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
    """Emit *expr* into the current ``BlockBuilder`` and wrap the result.

    For a ``TensorStructInfo`` result, emits a single binding and returns a
    :class:`Tensor`.  For a ``TupleStructInfo`` result, emits the tuple and
    then emits a ``TupleGetItem`` for each element, returning a ``tuple`` of
    :class:`Tensor` objects.

    Parameters
    ----------
    expr : relax.Expr
        The expression to emit.
    name : str
        Name hint for the emitted binding.

    Returns
    -------
    result : Tensor | tuple[Tensor, ...]
        A single :class:`Tensor` for scalar outputs, or a ``tuple`` of
        :class:`Tensor` objects for tuple outputs.
    """
    result = _ffi_api.WrapNested(expr, name)
    return _unwrap_ffi_result(result)


def _unwrap_ffi_result(result) -> "Tensor | tuple":
    """Recursively convert a C++ FFI result to ``Tensor`` or ``tuple``.

    Parameters
    ----------
    result :
        A ``relax.Var`` (single tensor) or a TVM ``Array`` (tuple of results).

    Returns
    -------
    wrapped : Tensor | tuple[Tensor, ...]
        The wrapped result.
    """
    if isinstance(result, rx.Var):
        return Tensor(_expr=result)
    return tuple(_unwrap_ffi_result(r) for r in result)


# ===========================================================================
# Effect  (pure Python - abstract base, Python virtual dispatch)
# ===========================================================================


class Effect:
    """Abstract base class for side-effecting state objects.

    Concrete subclasses (:class:`~tvm.relax.frontend.nn.modules.IOEffect`,
    :class:`~tvm.relax.frontend.nn.modules.KVCache`) implement the four
    protocol methods used by the exporter to manage effect state across
    function boundaries.

    The exporter detects effects via ``isinstance(x, Effect)`` and calls
    the protocol methods directly.
    """

    def emit_init(self, name_hint: str, builder: BlockBuilder) -> list[rx.DataflowVar]:
        """Emit the initialisation expression into *builder*.

        Called once per export to create the initial effect state objects.

        Parameters
        ----------
        name_hint : str
            Name hint for the emitted binding.
        builder : BlockBuilder
            The active ``BlockBuilder``.

        Returns
        -------
        state_vars : list[relax.DataflowVar]
            The emitted initial state variables.
        """
        raise NotImplementedError

    def create(self, name_hint: str) -> list[rx.Var]:
        """Create placeholder state ``Var``\s and store them internally.

        Called by the exporter to allocate function-argument ``Var``\s for
        the effect state before the dataflow block is opened.

        Parameters
        ----------
        name_hint : str
            Name hint for the created ``Var``\s.

        Returns
        -------
        state_vars : list[relax.Var]
            Newly created placeholder ``Var``\s.
        """
        raise NotImplementedError

    def set_state(self, state_vars: list[rx.Var]) -> None:
        """Restore internal state from previously created ``Var``\s.

        Parameters
        ----------
        state_vars : list[relax.Var]
            ``Var``\s produced by a prior call to :meth:`create`.
        """
        raise NotImplementedError

    def finalize(self) -> list[rx.Var]:
        """Return the current state ``Var``\s and clear internal state.

        Called after the dataflow block is closed to collect the final
        effect outputs for the function return value.

        Returns
        -------
        state_vars : list[relax.Var]
            The current state ``Var``\s.
        """
        raise NotImplementedError

    def to(self, dtype: str | None = None) -> None:
        """Recursively convert effect state to *dtype* (no-op by default).

        Parameters
        ----------
        dtype : str | None
            Target dtype string, e.g. ``"float16"``.
        """
        pass


# ===========================================================================
# ModuleList  -  native C++ object
# ===========================================================================


@tvm_ffi.register_object("relax.frontend.nn.ModuleList")
class ModuleList(tvm_ffi.Object, SubroutineMixin):
    """Ordered list of sub-modules backed by a native C++ ffi::Array<Any>.

    Native C++ module objects (tvm_ffi.Object subclasses) are stored in the
    C++ ``modules`` field.  Pure-Python objects (nn.Module, nn.Effect, etc.)
    cannot survive the FFI round-trip and are kept in a Python-side
    ``_py_items`` list instead, using sentinel ``None`` values in the C++
    array to mark their positions.  All public list methods merge both stores
    transparently by logical index.

    ``_py_items`` is lazily initialised on first access so that objects
    reconstructed directly from a C++ FFI handle (bypassing ``__init__``)
    still work correctly.
    """

    def __init__(self, modules: list) -> None:
        ffi_slots = []
        py_items = []
        for item in modules:
            # ModuleList and ModuleDict carry Python-side state (_py_items_data /
            # _py_modules_data) that does NOT survive an FFI round-trip.  Store
            # them in the Python overlay even though they are tvm_ffi.Objects.
            if isinstance(item, tvm_ffi.Object) and not isinstance(item, ModuleList | ModuleDict):
                ffi_slots.append(item)
                py_items.append(None)
            else:
                ffi_slots.append(None)
                py_items.append(item)
        self.__ffi_init__(ffi_slots)
        object.__setattr__(self, "_py_items_data", py_items)

    @property
    def _py_items(self) -> list:
        try:
            return object.__getattribute__(self, "_py_items_data")
        except AttributeError:
            py = [None] * len(self.modules)
            object.__setattr__(self, "_py_items_data", py)
            return py

    @_py_items.setter
    def _py_items(self, value: list) -> None:
        object.__setattr__(self, "_py_items_data", value)

    def _get(self, idx: int):
        py = self._py_items
        if py[idx] is not None:
            return py[idx]
        return self.modules[idx]

    def _set(self, idx: int, module) -> None:
        py = self._py_items
        if isinstance(module, tvm_ffi.Object) and not isinstance(module, ModuleList | ModuleDict):
            mods = list(self.modules)
            mods[idx] = module
            self.modules = mods
            py[idx] = None
        else:
            mods = list(self.modules)
            mods[idx] = None
            self.modules = mods
            py[idx] = module

    def __iter__(self):
        return (self._get(i) for i in range(len(self._py_items)))

    def __getitem__(self, idx: int):
        return self._get(idx)

    def __setitem__(self, idx: int, module) -> None:
        self._set(idx, module)

    def __len__(self) -> int:
        return len(self._py_items)

    def append(self, module) -> None:
        py = self._py_items
        if isinstance(module, tvm_ffi.Object) and not isinstance(module, ModuleList | ModuleDict):
            mods = list(self.modules)
            mods.append(module)
            self.modules = mods
            py.append(None)
        else:
            mods = list(self.modules)
            mods.append(None)
            self.modules = mods
            py.append(module)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        params = _ffi_api.GetContainerParameters(self, prefix)
        yield from params.items()
        for i, item in enumerate(self._py_items):
            if item is not None:
                child_prefix = f"{prefix}{i}." if prefix else f"{i}."
                yield from _attribute_finder(item, child_prefix, lambda x: isinstance(x, Parameter))

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def to(self, dtype: str | None = None) -> None:
        if dtype is not None:
            _ffi_api.ContainerApplyTo(self, dtype)
            for item in self._py_items:
                if item is not None and hasattr(item, "to"):
                    item.to(dtype=dtype)

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

    Native C++ module objects (tvm_ffi.Object subclasses) are stored in the
    C++ ``modules`` field.  Pure-Python objects (nn.Module, nn.Effect, etc.)
    cannot survive the FFI round-trip and are kept in a Python-side
    ``_py_modules`` OrderedDict instead.  All public dict methods merge both
    stores transparently, preserving insertion order.

    ``_py_modules`` is lazily initialised on first access so that objects
    reconstructed directly from a C++ FFI handle (bypassing ``__init__``)
    still work correctly.
    """

    def __init__(self, modules: OrderedDict | None = None) -> None:
        ffi_mods = {}
        py_mods = OrderedDict()
        if modules:
            for k, v in modules.items():
                # ModuleList and ModuleDict carry Python-side state that does NOT
                # survive an FFI round-trip.  Store them in the Python overlay.
                if isinstance(v, tvm_ffi.Object) and not isinstance(v, ModuleList | ModuleDict):
                    ffi_mods[k] = v
                else:
                    py_mods[k] = v
        self.__ffi_init__(ffi_mods)
        object.__setattr__(self, "_py_modules_data", py_mods)

    @property
    def _py_modules(self) -> OrderedDict:
        try:
            return object.__getattribute__(self, "_py_modules_data")
        except AttributeError:
            py = OrderedDict()
            object.__setattr__(self, "_py_modules_data", py)
            return py

    @_py_modules.setter
    def _py_modules(self, value: OrderedDict) -> None:
        object.__setattr__(self, "_py_modules_data", value)

    def _all_items(self):
        for k, v in self.modules.items():
            yield k, v
        for k, v in self._py_modules.items():
            yield k, v

    def __iter__(self):
        return (k for k, _ in self._all_items())

    def __getitem__(self, key: str):
        py = self._py_modules
        if key in py:
            return py[key]
        return self.modules[key]

    def __setitem__(self, key: str, module) -> None:
        if isinstance(module, tvm_ffi.Object) and not isinstance(module, ModuleList | ModuleDict):
            self._py_modules.pop(key, None)
            mods = dict(self.modules)
            mods[key] = module
            self.modules = mods
        else:
            mods = dict(self.modules)
            if key in mods:
                del mods[key]
                self.modules = mods
            self._py_modules[key] = module

    def __len__(self) -> int:
        return len(self.modules) + len(self._py_modules)

    def __contains__(self, key: str) -> bool:
        return key in self._py_modules or key in self.modules

    def keys(self):
        return list(k for k, _ in self._all_items())

    def values(self):
        return list(v for _, v in self._all_items())

    def items(self):
        return list(self._all_items())

    def get(self, key: str, default=None):
        if key in self._py_modules:
            return self._py_modules[key]
        m = self.modules
        return m[key] if key in m else default

    def update(self, modules: dict) -> None:
        for k, v in modules.items():
            self[k] = v

    def clear(self) -> None:
        self.modules = {}
        self._py_modules.clear()

    def pop(self, key: str):
        if key in self._py_modules:
            return self._py_modules.pop(key)
        m = dict(self.modules)
        val = m.pop(key)
        self.modules = m
        return val

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        params = _ffi_api.GetContainerParameters(self, prefix)
        yield from params.items()
        for key, mod in self._py_modules.items():
            child_prefix = f"{prefix}{key}." if prefix else f"{key}."
            yield from _attribute_finder(mod, child_prefix, lambda x: isinstance(x, Parameter))

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def to(self, dtype: str | None = None) -> None:
        if dtype is not None:
            _ffi_api.ContainerApplyTo(self, dtype)
            for mod in self._py_modules.values():
                if hasattr(mod, "to"):
                    mod.to(dtype=dtype)


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
        """Yield ``(dotted_name, Parameter)`` pairs for all parameters.

        Recursively walks ``__dict__``, descending into :class:`Module`,
        :class:`ModuleList`, and :class:`ModuleDict` children.

        Parameters
        ----------
        prefix : str
            Dotted prefix prepended to each name.  Pass ``""`` at the root.

        Yields
        ------
        name : str
            Dotted parameter name relative to this module.
        param : Parameter
            The parameter object.
        """
        yield from _attribute_finder(self, prefix, lambda x: isinstance(x, Parameter))

    def parameters(self) -> Iterator[Parameter]:
        """Yield all :class:`Parameter` values in this module tree.

        Yields
        ------
        param : Parameter
        """
        for _, p in self.named_parameters():
            yield p

    def state_dict(
        self, *, prefix: str = "", destination: dict[str, Parameter] | None = None
    ) -> dict[str, Parameter]:
        """Return an ordered dict of all parameters keyed by dotted name.

        Parameters
        ----------
        prefix : str
            Dotted prefix prepended to each name.
        destination : dict[str, Parameter] | None
            If provided, parameters are inserted into this dict in-place.

        Returns
        -------
        state_dict : dict[str, Parameter]
            Ordered mapping from dotted name to :class:`Parameter`.
        """
        if destination is None:
            destination = OrderedDict()
        for name, param in _attribute_finder(self, prefix, lambda x: isinstance(x, Parameter)):
            destination[name] = param
        return destination

    def load_state_dict(
        self, state_dict: dict[str, Parameter], strict: bool = True
    ) -> tuple[list[str], list[str]]:
        """Load parameters from *state_dict* into this module.

        Parameters
        ----------
        state_dict : dict[str, Parameter]
            Mapping of dotted name to :class:`Parameter` with bound data.
        strict : bool
            If ``True``, raise :exc:`KeyError` when any key is missing or
            unexpected.

        Returns
        -------
        missing_keys : list[str]
            Keys present in this module but absent from *state_dict*.
        unexpected_keys : list[str]
            Keys present in *state_dict* but absent from this module.
        """
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
        """Dispatch to :meth:`forward`.

        Parameters
        ----------
        *args, **kwargs :
            Forwarded verbatim to :meth:`forward`.

        Returns
        -------
        result : Any
            The return value of :meth:`forward`.
        """
        if not hasattr(self, "forward"):
            raise NotImplementedError(f"{type(self).__name__} has no forward()")
        return self.forward(*args, **kwargs)  # pylint: disable=no-member

    # ---- dtype conversion --------------------------------------------------

    def to(self, dtype: str | None = None) -> None:
        """Recursively convert all parameters and sub-modules to *dtype*.

        Delegates to the C++ helper ``PythonModuleApplyTo`` which handles
        :class:`Parameter`, :class:`ModuleList`, :class:`ModuleDict`, native
        C++ modules, and nested ``__dict__`` entries uniformly.

        Parameters
        ----------
        dtype : str | None
            Target dtype string, e.g. ``"float16"``.
        """
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
        """Export this module to a TVM ``IRModule``.

        Uses the fast C++ ``ExportToIRModule`` path when the spec contains
        only :class:`~spec.Int` / :class:`~spec.Tensor` arg specs, and falls
        back to the Python path for specs that use :class:`~spec.Object`.

        External modules (:class:`~extern.ExternModule`) registered via
        :func:`~exporter.add_extern` during ``forward()`` are compiled and
        attached via the C++ ``AttachExternModules`` pass when
        ``allow_extern=True``.

        Parameters
        ----------
        spec : ModuleSpecType
            Compilation specification (dict or :class:`~spec.ModuleSpec`).
        debug : bool
            If ``True``, add an ``IOEffect`` token to every method signature
            to enable :func:`~op.debug_func` / :func:`~op.print_`.
        allow_extern : bool
            If ``True``, return the list of external modules as a third
            return value.  If ``False`` and external modules are present,
            raise :exc:`ValueError`.

        Returns
        -------
        mod : tvm.ir.IRModule
            The compiled Relax IR module.
        params : list[tuple[str, Parameter]]
            Named parameters in the order they appear in the IR.
        extern_mods : list[ExternModule]
            Only returned when ``allow_extern=True``.
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
        """JIT-compile this module to a ready-to-run executable.

        Parameters
        ----------
        spec : ModuleSpec
            Compilation specification.
        device : str | Device
            Execution device, e.g. ``"cpu"`` or ``"cuda:0"``.
        pipeline : str | Pass | None
            Relax compilation pipeline name or a custom pass.
            Defaults to ``"default_build"``.
        out_format : str
            Output wrapper format.  Currently only ``"torch"`` is supported,
            which returns a :class:`~torch.TorchModule`.
        debug : bool
            If ``True``, add an ``IOEffect`` token to every method signature.

        Returns
        -------
        module : TorchModule
            A callable wrapper that accepts ``torch.Tensor`` inputs and
            returns ``torch.Tensor`` outputs.
        """
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
    """Recursively yield ``(dotted_name, value)`` pairs satisfying *condition_yield*.

    Handles three cases:

    1. :class:`ModuleList` / :class:`ModuleDict` — delegates to the
       container's own ``named_parameters()``, which covers both the C++
       store and the Python overlay (``_py_items`` / ``_py_modules``).
    2. Native C++ module (not a container) — calls
       ``_ffi_api.GetNativeParameters`` on the object handle.
    3. Pure-Python :class:`Module` — walks ``__dict__`` recursively.

    Parameters
    ----------
    root :
        The root object to traverse.
    prefix : str
        Dotted prefix prepended to each yielded name.
    condition_yield : Callable[[Any], bool]
        Predicate; only items for which this returns ``True`` are yielded.

    Yields
    ------
    name : str
        Dotted name of the matching item.
    value :
        The matching item.
    """

    # --- Case 1: native container types ------------------------------------
    # Delegate to the container's own named_parameters(), which covers both
    # the C++ store and the Python overlay (_py_items / _py_modules).
    if isinstance(root, ModuleList | ModuleDict):
        for name, param in root.named_parameters(prefix=prefix):
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
