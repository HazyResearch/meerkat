from __future__ import annotations

import weakref
from functools import wraps
from inspect import getcallargs
from typing import TYPE_CHECKING, List, Sequence, Union

if TYPE_CHECKING:
    from mosaic import AbstractCell, AbstractColumn, DataPanel
from mosaic.tools.utils import nested_map


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


class ProvenanceObjNode:
    def __init__(self, obj: Union[DataPanel, AbstractColumn, AbstractCell]):
        self.obj = weakref.ref(obj)
        self.name = str(obj)
        self._parents: List[ProvenanceOpNode] = []
        self._children: List[ProvenanceOpNode] = []

    def add_child(self, op: ProvenanceOpNode):
        self._children.append(op)

    def add_parent(self, op: ProvenanceOpNode):
        self._parents.append(op)

    def get_provenance(self):
        edges = []
        for parent in self._parents:
            edges.extend(parent.get_provenance())
            edges.append((parent, self))
        return edges

    def __repr__(self):
        return str(self.obj())


class ProvenanceOpNode:
    def __init__(self, fn: callable, inputs: dict, outputs: object, meta: dict):
        # import pdb
        # pdb.set_trace()
        self.fn = weakref.ref(fn)
        self.name = fn.__qualname__
        self.module = fn.__module__
        self.meta = meta

        self.inputs = {}
        for name, inp in inputs.items():
            if isinstance(inp, ProvenanceMixin) and hasattr(inp, "node"):
                self.inputs[name] = inp.node
                inp.node.add_child(self)

        self.outputs = []
        for output in outputs:
            if isinstance(output, ProvenanceMixin) and hasattr(output, "node"):
                self.outputs.append(output.node)
                output.node.add_parent(self)

    def __repr__(self):
        return f"OpNode({self.module}.{self.name})"

    def get_provenance(self):
        edges = []
        for key, inp in self.inputs.items():
            edges.extend(inp.get_provenance())
            edges.append((inp, self))
        return edges


def capture_provenance(capture_args: Sequence[str] = None):
    capture_args = [] if capture_args is None else capture_args

    def _provenance(fn: callable):
        @wraps(fn)
        def _wrapper(*args, **kwargs):

            args_dict = getcallargs(fn, *args, **kwargs)

            if "kwargs" in args_dict:
                args_dict.update(args_dict.pop("kwargs"))

            metadata = {arg: args_dict[arg] for arg in capture_args}
            out = fn(*args, **kwargs)

            # collect instances of ProvenanceMixin nested in the output
            outputs = []

            def collect_outputs(output):
                if isinstance(output, ProvenanceMixin):
                    outputs.append(output)

            nested_map(collect_outputs, out)

            ProvenanceOpNode(fn=fn, inputs=args_dict, outputs=outputs, meta=metadata)
            return out

        return _wrapper

    return _provenance
