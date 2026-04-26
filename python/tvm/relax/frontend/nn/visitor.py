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
"""The visitor and mutator infra for nn.Module."""

from typing import Any

from . import core as nn


class Mutator:
    """The mutator for nn.Module transform. Users can override the ``visit_*``
    methods to apply transforms in different structures, or override ``visit``
    to change the traversal strategy entirely."""

    def visit_module(self, name: str, node: nn.Module) -> Any:
        """Visit an nn.Module node.

        Parameters
        ----------
        name : str
            Dotted path of this node within its parent.
        node : nn.Module
            The module node to visit.

        Returns
        -------
        ret_node : Any
            The (possibly replaced) node.
        """
        return self.visit(name, node)

    def visit_effect(self, name: str, node: nn.Effect) -> Any:
        """Visit an nn.Effect node.

        Parameters
        ----------
        name : str
            Dotted path of this node within its parent.
        node : nn.Effect
            The effect node to visit.

        Returns
        -------
        ret_node : Any
            The (possibly replaced) node.
        """
        return self.visit(name, node)

    def visit_param(self, name: str, node: nn.Parameter) -> Any:
        """Visit an nn.Parameter node.

        Parameters
        ----------
        name : str
            Dotted path of this node within its parent.
        node : nn.Parameter
            The parameter to visit.

        Returns
        -------
        ret_node : Any
            The (possibly replaced) parameter.
        """
        return self.visit(name, node)

    def visit_moduledict(self, name: str, node: nn.ModuleDict) -> Any:
        """Visit an nn.ModuleDict node.

        Parameters
        ----------
        name : str
            Dotted path of this node within its parent.
        node : nn.ModuleDict
            The ModuleDict to visit.

        Returns
        -------
        ret_node : Any
            The (possibly replaced) node.
        """
        return self.visit(name, node)

    def visit_modulelist(self, name: str, node: nn.ModuleList) -> Any:
        """Visit an nn.ModuleList node.

        Parameters
        ----------
        name : str
            Dotted path of this node within its parent.
        node : nn.ModuleList
            The ModuleList to visit.

        Returns
        -------
        ret_node : Any
            The (possibly replaced) node.
        """
        return self.visit(name, node)

    def visit(self, name: str, node: Any) -> Any:
        """Dispatch driver: recurse into the module tree and call the
        appropriate ``visit_*`` method for each child.

        Parameters
        ----------
        name : str
            Dotted path of *node* within its parent (pass ``""`` at root).
        node : Any
            The node to visit.

        Returns
        -------
        ret_node : Any
            The (possibly mutated) node.
        """

        def _get_child_name(parent: str, child: str) -> str:
            if parent == "":
                return child
            return f"{parent}.{child}"

        _MODULEDICT_KEY = "relax.frontend.nn.ModuleDict"
        _MODULELIST_KEY = "relax.frontend.nn.ModuleList"

        def _type_key(value: Any) -> str | None:
            get_key = getattr(value, "GetTypeKey", None)
            if get_key is not None:
                return get_key()
            return None

        def _dispatch(child_name: str, value: Any) -> Any:
            key = _type_key(value)
            if key == _MODULEDICT_KEY or isinstance(value, nn.ModuleDict):
                return self.visit_moduledict(child_name, value)
            if key == _MODULELIST_KEY or isinstance(value, nn.ModuleList):
                return self.visit_modulelist(child_name, value)
            if isinstance(value, nn.Effect):
                return self.visit_effect(child_name, value)
            if isinstance(value, nn.Parameter):
                return self.visit_param(child_name, value)
            if isinstance(value, nn.Module):
                return self.visit_module(child_name, value)
            return value

        if isinstance(node, nn.ModuleList):
            for i in range(len(node)):
                child_name = _get_child_name(name, str(i))
                old = node[i]
                new = _dispatch(child_name, old)
                if new is not old:
                    node[i] = new
        elif isinstance(node, nn.ModuleDict):
            for k, v in node.items():
                child_name = _get_child_name(name, k)
                new = _dispatch(child_name, v)
                if new is not v:
                    node[k] = new
        elif isinstance(node, nn.Module):
            for key, value in node.__dict__.items():
                child_name = _get_child_name(name, key)
                new = _dispatch(child_name, value)
                if new is not value:
                    setattr(node, key, new)
        # Non-module values (int, str, etc.) are returned unchanged.
        return node
