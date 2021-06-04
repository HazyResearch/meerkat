from __future__ import annotations

import weakref
from functools import wraps
from inspect import getcallargs
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import mosaic as ms

provenance_enabled = False


class provenance:
    def __init__(
        self,
    ):
        """Context manager for enabling provenance capture in Mosaic.
        Example:
        ```python
        import mosaic as ms
        with ms.provenance():
            dp = ms.DataPanel.from_batch({...})
        ```
        """
        self.prev = True

    def __enter__(self):
        global provenance_enabled
        self.prev = provenance_enabled
        provenance_enabled = True

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        global provenance_enabled
        provenance_enabled = self.prev


class ProvenanceMixin:
    def __init__(self, *args, **kwargs):
        super(ProvenanceMixin, self).__init__(*args, **kwargs)
        self._node = ProvenanceObjNode(self)

    @property
    def node(self):
        return self._node

    def get_provenance(self):
        if self._node is None:
            return None
        return self._node.get_provenance()


class ProvenanceNode:
    def add_child(self, node: ProvenanceNode, key: Tuple):
        self._children.append((node, key))

    def add_parent(self, node: ProvenanceNode, key: Tuple):
        self._parents.append((node, key))

    @property
    def parents(self):
        return self._parents

    @property
    def children(self):
        return self._children

    def get_provenance(self):
        edges = []
        nodes = set()
        for parent, key in self._parents:
            prev_nodes, prev_edges = parent.get_provenance()
            edges.extend(prev_edges)
            edges.append({"parent": parent, "child": self, "key": key})
            nodes |= prev_nodes
            nodes.add(self)
        return nodes, edges


class ProvenanceObjNode(ProvenanceNode):
    def __init__(self, obj: ProvenanceMixin):
        self.obj = weakref.ref(obj)
        self.type = type(obj)
        self._parents: List[Tuple(ProvenanceOpNode, tuple)] = []
        self._children: List[Tuple(ProvenanceOpNode, tuple)] = []

    def __repr__(self):
        return str(self.obj())


class ProvenanceOpNode(ProvenanceNode):
    def __init__(self, fn: callable, inputs: dict, outputs: object, meta: dict):
        # import pdb
        # pdb.set_trace()
        self.fn = weakref.ref(fn)
        self.name = fn.__qualname__
        self.module = fn.__module__
        self.meta = meta

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

    def __repr__(self):
        return f"{self.module}.{self.name}"


def capture_provenance(capture_args: Sequence[str] = None):
    capture_args = [] if capture_args is None else capture_args

    def _provenance(fn: callable):
        @wraps(fn)
        def _wrapper(*args, **kwargs):
            if not provenance_enabled:
                return fn(*args, **kwargs)
            args_dict = getcallargs(fn, *args, **kwargs)
            if "kwargs" in args_dict:
                args_dict.update(args_dict.pop("kwargs"))

            metadata = {arg: args_dict[arg] for arg in capture_args}
            out = fn(*args, **kwargs)

            # collect instances of ProvenanceMixin nested in the output
            inputs = get_nested_objs(args_dict)
            outputs = get_nested_objs(out)
            ProvenanceOpNode(fn=fn, inputs=inputs, outputs=outputs, meta=metadata)
            return out

        return _wrapper

    return _provenance


def get_nested_objs(data):
    """Recursively get DataPanels and Columns from nested collections."""
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
    elif isinstance(data, ms.DataPanel):
        objs[key] = data
        for curr_key, item in data.items():
            _get_nested_objs(objs, key=(*key, curr_key), data=item)

    elif isinstance(data, ms.AbstractColumn):
        objs[key] = data


def visualize_provenance(
    obj: Union[ProvenanceObjNode, ProvenanceOpNode], show_columns: bool = False
):
    try:
        import cyjupyter
    except ImportError:
        raise ImportError(
            "`visualize_provenance` requires the `cyjupyter` dependency."
            "See https://github.com/cytoscape/cytoscape-jupyter-widget"
        )

    nodes, edges = obj.get_provenance()
    cy_nodes = [
        {
            "data": {
                "id": id(node),
                "name": str(node),
                "type": "obj" if isinstance(node, ProvenanceObjNode) else "op",
            }
        }
        for node in nodes
        if (
            show_columns
            or isinstance(node, ProvenanceOpNode)
            or issubclass(node.type, ms.DataPanel)
        )
    ]
    cy_edges = [
        {
            "data": {
                "source": id(edge["parent"]),
                "target": id(edge["child"]),
                "key": edge["key"],
            }
        }
        for edge in edges
    ]

    cy_data = {"elements": {"nodes": cy_nodes, "edges": cy_edges}}

    style = [
        {
            "selector": "node",
            "css": {
                "content": "data(name)",
                "border-color": "rgb(256,256,256)",
                "border-opacity": 1.0,
                "border-width": 2,
            },
        },
        {
            "selector": "node[type = 'op']",
            "css": {
                "shape": "rectangle",
                "background-color": "#f53e37",
                "width": 20,
                "height": 20,
            },
        },
        {
            "selector": "node[[degree = 0]]",
            "css": {
                "visibility": "hidden",
            },
        },
        {
            "selector": "edge",
            "css": {
                "content": "data(key)",
                "mid-target-arrow-color": "#f53e37",
                "mid-target-arrow-shape": "triangle",
                "text-margin-x": 10,
                "text-margin-y": 10,
            },
        },
    ]
    return cyjupyter.Cytoscape(
        data=cy_data, visual_style=style, layout_name="breadthfirst"
    )
