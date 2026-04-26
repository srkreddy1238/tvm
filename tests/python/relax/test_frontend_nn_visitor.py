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
"""Tests for the nn.Mutator visitor / mutator infrastructure.

Each test exercises a distinct aspect of the Mutator traversal:

  test_mutator_naming_basic        – dotted names are built correctly for a
                                     four-level nested Module hierarchy.
  test_mutator_naming_moduledict   – dotted names through ModuleDict → ModuleList.
  test_mutator_naming_modulelist   – dotted names through ModuleList → ModuleList.
  test_mutator_module              – visit_module can replace a sub-module.
  test_mutator_moduledict          – visit_module can replace an entry in a
                                     ModuleDict.
  test_mutator_modulelist          – visit_module can replace an entry in a
                                     ModuleList.
  test_mutator_effect              – visit_effect can replace an Effect child.
  test_mutator_param               – visit_param can replace a Parameter child.
  test_mutator_recursively         – visit_param recurses into nested Modules.
  test_mutator_default_visit_param – the default visit_param (no override)
                                     returns the node unchanged.
  test_mutator_default_visit_module – the default visit_module returns the
                                      node unchanged.
  test_mutator_noop_on_non_module  – visit() on a plain Python object that is
                                     not an nn.Module returns it unchanged.
  test_mutator_modulelist_nested_moduledict – ModuleList containing ModuleDicts
                                             is traversed correctly.
  test_mutator_param_dtype_upgrade – visit_param upgrades every float16
                                     parameter to float32 across a mixed tree.
"""

from typing import Any

import pytest
import tvm
from tvm.relax.frontend import nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Recorder(nn.Mutator):
    """Records every (name, node) pair seen by visit_param."""

    def __init__(self):
        self.seen: list[tuple[str, nn.Parameter]] = []

    def visit_param(self, name: str, node: nn.Parameter) -> Any:
        self.seen.append((name, node))
        return node


# ---------------------------------------------------------------------------
# Naming tests
# ---------------------------------------------------------------------------


def test_mutator_naming_basic():
    """Dotted names are built correctly for a four-level nested Module."""

    class Module0(nn.Module):
        def __init__(self):
            self.param0 = nn.Parameter((32, 128), "float64")

    class Module1(nn.Module):
        def __init__(self):
            self.mod0 = Module0()
            self.param1 = nn.Parameter((32, 128), "float32")

    class Module2(nn.Module):
        def __init__(self):
            self.mod1 = Module1()
            self.param2 = nn.Parameter((32, 128), "float16")

    class Module3(nn.Module):
        def __init__(self):
            self.mod2 = Module2()
            self.param3 = nn.Parameter((32, 128), "float8")

    expected = {
        "float8": "mod3.param3",
        "float16": "mod3.mod2.param2",
        "float32": "mod3.mod2.mod1.param1",
        "float64": "mod3.mod2.mod1.mod0.param0",
    }

    class CheckMutator(nn.Mutator):
        def visit_param(self, name: str, node: nn.Parameter) -> Any:
            assert name == expected[node.dtype], (
                f"dtype={node.dtype}: expected name {expected[node.dtype]!r}, got {name!r}"
            )
            return node

    CheckMutator().visit("mod3", Module3())


def test_mutator_naming_moduledict():
    """Dotted names are built correctly through ModuleDict → ModuleList."""

    class Leaf(nn.Module):
        def __init__(self, dtype):
            self.param = nn.Parameter((32, 128), dtype)

    expected = {
        "float64": "mod_dict.k0.0.param",
        "float32": "mod_dict.k0.1.param",
        "float16": "mod_dict.k1.0.param",
        "float8": "mod_dict.k1.1.param",
    }

    class CheckMutator(nn.Mutator):
        def visit_param(self, name: str, node: nn.Parameter) -> Any:
            assert name == expected[node.dtype], (
                f"dtype={node.dtype}: expected {expected[node.dtype]!r}, got {name!r}"
            )
            return node

    mod_dict = nn.ModuleDict(
        {
            "k0": nn.ModuleList([Leaf("float64"), Leaf("float32")]),
            "k1": nn.ModuleList([Leaf("float16"), Leaf("float8")]),
        }
    )
    CheckMutator().visit("mod_dict", mod_dict)


def test_mutator_naming_modulelist():
    """Dotted names are built correctly through ModuleList → ModuleList."""

    class Leaf(nn.Module):
        def __init__(self, dtype):
            self.param = nn.Parameter((32, 128), dtype)

    expected = {
        "float64": "mod_list.0.0.param",
        "float32": "mod_list.0.1.param",
        "float16": "mod_list.1.0.param",
        "float8": "mod_list.1.1.param",
    }

    class CheckMutator(nn.Mutator):
        def visit_param(self, name: str, node: nn.Parameter) -> Any:
            assert name == expected[node.dtype], (
                f"dtype={node.dtype}: expected {expected[node.dtype]!r}, got {name!r}"
            )
            return node

    mod_list = nn.ModuleList(
        [
            nn.ModuleList([Leaf("float64"), Leaf("float32")]),
            nn.ModuleList([Leaf("float16"), Leaf("float8")]),
        ]
    )
    CheckMutator().visit("mod_list", mod_list)


# ---------------------------------------------------------------------------
# Mutation tests — Module
# ---------------------------------------------------------------------------


def test_mutator_module():
    """visit_module can replace a sub-module with a different type."""

    class Sub1(nn.Module):
        pass

    class Sub2(nn.Module):
        pass

    class Parent(nn.Module):
        def __init__(self):
            self.mod = Sub1()

    class ReplaceSub1(nn.Mutator):
        def visit_module(self, name: str, node: nn.Module) -> Any:
            return Sub2() if isinstance(node, Sub1) else node

    parent = Parent()
    assert isinstance(parent.mod, Sub1)
    parent = ReplaceSub1().visit("", parent)
    assert isinstance(parent.mod, Sub2)


def test_mutator_moduledict():
    """visit_module can replace an entry inside a ModuleDict."""

    class M1(nn.Module):
        pass

    class M2(nn.Module):
        pass

    class M3(nn.Module):
        pass

    class ReplaceM3(nn.Mutator):
        def visit_module(self, name: str, node: nn.Module) -> Any:
            return M1() if isinstance(node, M3) else node

    md = nn.ModuleDict({"k0": M1(), "k1": M2(), "k2": M3()})
    assert isinstance(md["k0"], M1)
    assert isinstance(md["k1"], M2)
    assert isinstance(md["k2"], M3)

    md = ReplaceM3().visit("", md)

    assert isinstance(md["k0"], M1)
    assert isinstance(md["k1"], M2)
    assert isinstance(md["k2"], M1)  # replaced


def test_mutator_modulelist():
    """visit_module can replace an entry inside a ModuleList."""

    class M1(nn.Module):
        pass

    class M2(nn.Module):
        pass

    class M3(nn.Module):
        pass

    class ReplaceM3(nn.Mutator):
        def visit_module(self, name: str, node: nn.Module) -> Any:
            return M1() if isinstance(node, M3) else node

    ml = nn.ModuleList([M1(), M2(), M3()])
    assert isinstance(ml[0], M1)
    assert isinstance(ml[1], M2)
    assert isinstance(ml[2], M3)

    ml = ReplaceM3().visit("", ml)

    assert isinstance(ml[0], M1)
    assert isinstance(ml[1], M2)
    assert isinstance(ml[2], M1)  # replaced


# ---------------------------------------------------------------------------
# Mutation tests — Effect
# ---------------------------------------------------------------------------


def test_mutator_effect():
    """visit_effect can replace an Effect child."""

    class Eff1(nn.Effect):
        def emit_init(self, name_hint, builder):
            return []

        def create(self, name_hint):
            return []

        def set_state(self, state_vars):
            pass

        def finalize(self):
            return []

    class Eff2(nn.Effect):
        def emit_init(self, name_hint, builder):
            return []

        def create(self, name_hint):
            return []

        def set_state(self, state_vars):
            pass

        def finalize(self):
            return []

    class Parent(nn.Module):
        def __init__(self):
            self.effect = Eff1()

    class ReplaceEff1(nn.Mutator):
        def visit_effect(self, name: str, node: nn.Effect) -> Any:
            return Eff2() if isinstance(node, Eff1) else node

    parent = Parent()
    assert isinstance(parent.effect, Eff1)
    parent = ReplaceEff1().visit("", parent)
    assert isinstance(parent.effect, Eff2)


# ---------------------------------------------------------------------------
# Mutation tests — Parameter
# ---------------------------------------------------------------------------


def test_mutator_param():
    """visit_param can replace a Parameter with a different dtype."""

    class Parent(nn.Module):
        def __init__(self):
            self.weight = nn.Parameter((128, 64), "float16")

    class UpcastParam(nn.Mutator):
        def visit_param(self, name: str, node: nn.Parameter) -> Any:
            if node.dtype == "float16":
                return nn.Parameter(node.shape, "float32")
            return node

    parent = Parent()
    assert parent.weight.dtype == "float16"
    parent = UpcastParam().visit("", parent)
    assert parent.weight.dtype == "float32"


def test_mutator_recursively():
    """visit_param recurses into nested Modules."""

    class Sub(nn.Module):
        def __init__(self):
            self.weight = nn.Parameter((128, 64), "float16")

    class Parent(nn.Module):
        def __init__(self):
            self.mod = Sub()

    class UpcastParam(nn.Mutator):
        def visit_param(self, name: str, node: nn.Parameter) -> Any:
            if node.dtype == "float16":
                return nn.Parameter(node.shape, "float32")
            return node

    parent = Parent()
    assert parent.mod.weight.dtype == "float16"
    parent = UpcastParam().visit("", parent)
    assert parent.mod.weight.dtype == "float32"


# ---------------------------------------------------------------------------
# Default-behaviour tests
# ---------------------------------------------------------------------------


def test_mutator_default_visit_param():
    """The base Mutator.visit_param returns the node unchanged."""

    class Parent(nn.Module):
        def __init__(self):
            self.weight = nn.Parameter((4, 8), "float32")

    parent = Parent()
    original_dtype = parent.weight.dtype
    nn.Mutator().visit("", parent)
    assert parent.weight.dtype == original_dtype


def test_mutator_default_visit_module():
    """The base Mutator.visit_module returns the node unchanged."""

    class Sub(nn.Module):
        pass

    class Parent(nn.Module):
        def __init__(self):
            self.sub = Sub()

    parent = Parent()
    result = nn.Mutator().visit("", parent)
    assert isinstance(result.sub, Sub)


def test_mutator_noop_on_non_module():
    """visit() on a plain Python int returns it unchanged."""
    result = nn.Mutator().visit("x", 42)
    assert result == 42


# ---------------------------------------------------------------------------
# Composite / integration tests
# ---------------------------------------------------------------------------


def test_mutator_modulelist_nested_moduledict():
    """ModuleList containing ModuleDicts is traversed correctly."""

    class Leaf(nn.Module):
        def __init__(self, dtype):
            self.param = nn.Parameter((4,), dtype)

    recorder = _Recorder()
    ml = nn.ModuleList(
        [
            nn.ModuleDict({"a": Leaf("float32"), "b": Leaf("float16")}),
            nn.ModuleDict({"c": Leaf("float64")}),
        ]
    )
    recorder.visit("root", ml)

    seen_names = {name for name, _ in recorder.seen}
    assert "root.0.a.param" in seen_names
    assert "root.0.b.param" in seen_names
    assert "root.1.c.param" in seen_names


def test_mutator_param_dtype_upgrade():
    """visit_param upgrades every float16 parameter to float32 across a mixed tree."""

    class Leaf(nn.Module):
        def __init__(self, dtype):
            self.w = nn.Parameter((8,), dtype)

    class Root(nn.Module):
        def __init__(self):
            self.a = Leaf("float16")
            self.b = Leaf("float32")
            self.c = nn.ModuleList([Leaf("float16"), Leaf("float64")])
            self.d = nn.ModuleDict({"x": Leaf("float16"), "y": Leaf("float32")})

    class Upgrade(nn.Mutator):
        def visit_param(self, name: str, node: nn.Parameter) -> Any:
            if node.dtype == "float16":
                return nn.Parameter(node.shape, "float32")
            return node

    root = Root()
    root = Upgrade().visit("", root)

    assert root.a.w.dtype == "float32"
    assert root.b.w.dtype == "float32"
    assert root.c[0].w.dtype == "float32"
    assert root.c[1].w.dtype == "float64"  # unchanged
    assert root.d["x"].w.dtype == "float32"
    assert root.d["y"].w.dtype == "float32"


if __name__ == "__main__":
    tvm.testing.main()
