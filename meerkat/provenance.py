from __future__ import annotations

import warnings
import weakref
from copy import copy
from functools import wraps
from inspect import getcallargs
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import meerkat as mk
from meerkat.errors import ExperimentalWarning

_provenance_enabled = False


def set_provenance(enabled=True):
    global _provenance_enabled
    _provenance_enabled = enabled


class provenance:
    def __init__(self, enabled: bool = True):
        """Context manager for enabling provenance capture in Meerkat.

        Example:
        ```python
        import meerkat as mk
        with mk.provenance():
            df = mk.DataFrame.from_batch({...})
        ```
        """
        self.prev = None
        self.enabled = enabled

    def __enter__(self):
        global _provenance_enabled
        self.prev = _provenance_enabled
        _provenance_enabled = self.enabled

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        global _provenance_enabled
        _provenance_enabled = self.prev


def is_provenance_enabled():
    return _provenance_enabled


class ProvenanceMixin:
    def __init__(self, *args, **kwargs):
        super(ProvenanceMixin, self).__init__(*args, **kwargs)
        self._init_node()

    def _init_node(self):
        self._node = ProvenanceObjNode(self)

    @property
    def node(self):
        return self._node

    def get_provenance(
        self, include_columns: bool = False, last_parent_only: bool = False
    ):
        if self._node is None:
            return None
        nodes, edges = self._node.get_provenance(last_parent_only=last_parent_only)
        if include_columns:
            return nodes, edges
        else:
            # filter out edges and nodes
            nodes = [
                node
                for node in nodes
                if (
                    (
                        isinstance(node, ProvenanceOpNode)
                        and any(
                            [
                                issubclass(node.type, mk.DataFrame)
                                for node, key in node.children
                            ]
                        )
                    )
                    or issubclass(node.type, mk.DataFrame)
                )
            ]
            edges = [
                edge for edge in edges if all([(node in nodes) for node in edge[:2]])
            ]
            return nodes, edges


class ProvenanceNode:
    def add_child(self, node: ProvenanceNode, key: Tuple):
        self._children.append((node, key))
        with provenance(enabled=False):
            self.cache_repr()

    def add_parent(self, node: ProvenanceNode, key: Tuple):
        self._parents.append((node, key))
        with provenance(enabled=False):
            self.cache_repr()

    @property
    def parents(self):
        return self._parents

    @property
    def last_parent(self):
        if len(self._parents) > 0:
            return self._parents[-1]
        else:
            return None

    @property
    def children(self):
        return self._children

    def get_provenance(self, last_parent_only: bool = False):
        edges = set()
        nodes = set()
        for parent, key in (
            self.parents[-1:]
            if (last_parent_only and self.__class__ == ProvenanceObjNode)
            else self.parents
        ):
            prev_nodes, prev_edges = parent.get_provenance(
                last_parent_only=last_parent_only
            )
            edges |= prev_edges
            edges.add((parent, self, key))
            nodes |= prev_nodes

        nodes.add(self)
        return nodes, edges

    def cache_repr(self):
        self._cached_repr = self._repr() + "(deleted)"

    def _repr(self):
        raise NotImplementedError()

    def __repr__(self):
        if self.ref() is None:
            return self._cached_repr
        else:
            return self._repr()

    def __getstate__(self):
        state = copy(self.__dict__)
        state["ref"] = None
        return state

    def __setstate__(self, state: dict):
        state["ref"] = lambda: None
        self.__dict__.update(state)


class ProvenanceObjNode(ProvenanceNode):
    def __init__(self, obj: ProvenanceMixin):
        self.ref = weakref.ref(obj)
        self.type = type(obj)
        self._parents: List[Tuple(ProvenanceOpNode, tuple)] = []
        self._children: List[Tuple(ProvenanceOpNode, tuple)] = []

    def _repr(self):
        return f"ObjNode({str(self.ref())})"


class ProvenanceOpNode(ProvenanceNode):
    def __init__(
        self, fn: callable, inputs: dict, outputs: object, captured_args: dict
    ):
        self.ref = weakref.ref(fn)
        self.type = type(fn)
        self.name = fn.__qualname__
        self.module = fn.__module__
        self.captured_args = captured_args

        self._parents: List[Tuple(ProvenanceObjNode, tuple)] = []
        self._children: List[Tuple(ProvenanceObjNode, tuple)] = []

        for key, inp in inputs.items():
            self.add_parent(inp.node, key)
            inp.node.add_child(self, key)

        input_ids = {id(v) for v in inputs.values()}
        for key, output in outputs.items():
            # only include outputs that weren't passed in
            if id(output) not in input_ids:
                self.add_child(output.node, key)
                output.node.add_parent(self, key)

    def _repr(self):
        return f"OpNode({self.module}.{self.name})"


def capture_provenance(capture_args: Sequence[str] = None):
    capture_args = [] if capture_args is None else capture_args

    def _provenance(fn: callable):
        @wraps(fn)
        def _wrapper(*args, **kwargs):
            if not is_provenance_enabled():
                return fn(*args, **kwargs)
            args_dict = getcallargs(fn, *args, **kwargs)
            if "kwargs" in args_dict:
                args_dict.update(args_dict.pop("kwargs"))

            captured_args = {arg: args_dict[arg] for arg in capture_args}
            out = fn(*args, **kwargs)

            # collect instances of ProvenanceMixin nested in the output
            inputs = get_nested_objs(args_dict)
            outputs = get_nested_objs(out)
            ProvenanceOpNode(
                fn=fn, inputs=inputs, outputs=outputs, captured_args=captured_args
            )
            return out

        return _wrapper

    return _provenance


def get_nested_objs(data):
    """Recursively get DataFrames and Columns from nested collections."""
    objs = {}
    _get_nested_objs(objs, (), data)
    return objs


def _get_nested_objs(objs: Dict, key: Tuple[str], data: object):
    if isinstance(data, Sequence) and not isinstance(data, str):
        for idx, item in enumerate(data):
            _get_nested_objs(objs, key=(*key, idx), data=item)

    elif isinstance(data, Mapping):
        for curr_key, item in data.items():
            _get_nested_objs(objs, key=(*key, curr_key), data=item)
    elif isinstance(data, mk.DataFrame):
        objs[key] = data
        for curr_key, item in data.items():
            _get_nested_objs(objs, key=(*key, curr_key), data=item)
    elif isinstance(data, mk.Column):
        objs[key] = data


def visualize_provenance(
    obj: Union[ProvenanceObjNode, ProvenanceOpNode],
    show_columns: bool = False,
    last_parent_only: bool = False,
):

    warnings.warn(  # pragma: no cover
        ExperimentalWarning(
            "The function `meerkat.provenance.visualize_provenance` is experimental and"
            " has limited test coverage. Proceed with caution."
        )
    )
    try:  # pragma: no cover
        import cyjupyter
    except ImportError:  # pragma: no cover
        raise ImportError(
            "`visualize_provenance` requires the `cyjupyter` dependency."
            "See https://github.com/cytoscape/cytoscape-jupyter-widget"
        )

    nodes, edges = obj.get_provenance(  # pragma: no cover
        include_columns=show_columns, last_parent_only=last_parent_only
    )
    cy_nodes = [  # pragma: no cover
        {
            "data": {
                "id": id(node),
                "name": str(node),
                "type": "obj" if isinstance(node, ProvenanceObjNode) else "op",
            }
        }
        for node in nodes
    ]
    cy_edges = [  # pragma: no cover
        {
            "data": {
                "source": id(edge[0]),
                "target": id(edge[1]),
                "key": edge[2],
            }
        }
        for edge in edges
    ]

    cy_data = {"elements": {"nodes": cy_nodes, "edges": cy_edges}}  # pragma: no cover

    style = [  # pragma: no cover
        {
            "selector": "node",
            "css": {
                "content": "data(name)",
                "background-color": "#fc8d62",
                "border-color": "#252525",
                "border-opacity": 1.0,
                "border-width": 3,
            },
        },
        {
            "selector": "node[type = 'op']",
            "css": {
                "shape": "barrel",
                "background-color": "#8da0cb",
            },
        },
        {
            # need to double index to access metadata (degree etc.)
            "selector": "node[[degree = 0]]",
            "css": {
                "visibility": "hidden",
            },
        },
        {
            "selector": "edge",
            "css": {
                "content": "data(key)",
                "line-color": "#252525",
                "mid-target-arrow-color": "#8da0cb",
                "mid-target-arrow-shape": "triangle",
                "arrow-scale": 2.5,
                "text-margin-x": 10,
                "text-margin-y": 10,
            },
        },
    ]
    return cyjupyter.Cytoscape(  # pragma: no cover
        data=cy_data, visual_style=style, layout_name="breadthfirst"
    )
