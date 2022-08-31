from abc import ABC
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Union

from pydantic import BaseModel

from meerkat.mixins.identifiable import IdentifiableMixin

if TYPE_CHECKING:
    from meerkat.datapanel import DataPanel


class NodeMixin:
    def __init__(self):
        self.children: Set[Operation] = set()

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)

    def has_children(self):
        return len(self.children) > 0


Primitive = Union[int, str, float, bool]
Storeable = Union[Primitive, List[Primitive], Dict[Primitive, Primitive]]


class BoxConfig(BaseModel):
    box_id: str
    type: str = "DataPanel"


class Box(IdentifiableMixin, NodeMixin):
    identifiable_group: str = "boxes"

    def __init__(self, obj):
        super().__init__()
        self.obj = obj

    @property
    def config(self):
        return BoxConfig(box_id=self.id, type="DataPanel")


class Modification(BaseModel, ABC):
    id: str

    @property
    def node(self):
        raise NotImplementedError()


class BoxModification(Modification):
    scope: List[str]
    type: str = "box"

    @property
    def node(self):
        from meerkat.state import state

        return state.identifiables.get(group="boxes", id=self.id)


class StoreModification(Modification):
    id: str
    value: Storeable
    type: str = "store"

    @property
    def node(self):
        from meerkat.state import state

        return state.identifiables.get(group="stores", id=self.id)


class PivotConfig(BoxConfig):
    pass


class Pivot(Box):
    pass


class DerivedConfig(BoxConfig):
    pass


class Derived(Box):
    pass


class StoreConfig(BaseModel):
    store_id: str
    value: Any
    has_children: bool


class Store(IdentifiableMixin, NodeMixin):
    identifiable_group: str = "stores"

    def __init__(self, value: Any):
        super().__init__()
        self.value = value

    @property
    def config(self):
        return StoreConfig(
            store_id=self.id,
            value=self.value,
            has_children=self.has_children(),
        )


def make_store(value: Union[str, Storeable]):
    if isinstance(value, Store):
        return value
    return Store(value)


class Operation(NodeMixin):
    def __init__(
        self, fn: Callable, args: List[Box], kwargs: Dict[str, Box], result: Derived
    ):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.result = result

    def __call__(self):
        unpacked_args, unpacked_kwargs, _, _ = _unpack_boxes_and_stores(
            *self.args, **self.kwargs
        )
        result = self.fn(*unpacked_args, **unpacked_kwargs)
        # TODO: result may be multiple
        self.result.obj = result

        from meerkat.datapanel import DataPanel

        if isinstance(result, DataPanel):
            return BoxModification(id=self.result.id, scope=result.columns)

        return BoxModification(id=self.result.id, scope=[])


def _topological_sort(root_nodes: List[NodeMixin]) -> List[NodeMixin]:
    """
    Perform a topological sort on a graph.

    Args:
        nodes: A graph represented as a dictionary mapping nodes to lists of its
            parents.
    """
    # get a mapping from node to the children of each node
    parents = defaultdict(set)
    nodes = set()
    while root_nodes:
        node = root_nodes.pop(0)
        for child in node.children:
            parents[child].add(node)
            nodes.add(node)
            root_nodes.append(child)

    current = [
        node for node in nodes if not parents[node]
    ]  # get a set of all the nodes without an incoming edge

    while current:
        node = current.pop(0)
        yield node

        for child in node.children:
            parents[child].remove(node)
            if not parents[child]:
                current.append(child)


def trigger(modifications: List[Modification]) -> List[Modification]:
    """Trigger the computation graph of an interface based on a list of
    modifications.

    Return:
        List[Modification]: The list of modifications that resulted from running the
            computation graph.

    """
    # build a graph rooted at the stores and boxes in the modifications list
    root_nodes = [mod.node for mod in modifications]

    order = [
        node for node in _topological_sort(root_nodes) if isinstance(node, Operation)
    ]

    new_modifications = [op() for op in order]
    return modifications + new_modifications


def _unpack_boxes_and_stores(*args, **kwargs):
    # TODO(Sabri): this should be nested
    boxes = []
    stores = []
    unpacked_args = []
    for arg in args:
        if isinstance(arg, Box):
            boxes.append(arg)
            unpacked_args.append(arg.obj)
        elif isinstance(arg, Store):
            stores.append(arg)
            unpacked_args.append(arg.value)
        else:
            unpacked_args.append(arg)

    unpacked_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, Box):
            boxes.append(v)
            unpacked_kwargs[k] = v.obj
        elif isinstance(v, Store):
            stores.append(v)
            unpacked_kwargs[k] = v.value
        else:
            unpacked_kwargs[k] = v

    return unpacked_args, unpacked_kwargs, boxes, stores


def interface_op(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # TODO(Sabri): this should be nested
        unpacked_args, unpacked_kwargs, boxes, stores = _unpack_boxes_and_stores(
            *args, **kwargs
        )

        result = fn(*unpacked_args, **unpacked_kwargs)

        if len(boxes) > 0 or len(stores) > 0:
            derived = Derived(result)
            # this `fn` explicitly shouldn't be the wrapped version!
            op = Operation(fn=fn, args=args, kwargs=kwargs, result=derived)
            for input_box in boxes:
                input_box.children.add(op)
            for input_store in stores:
                input_store.children.add(op)
            op.children.add(derived)
            # TODO(Sabri): this should be nested
            return derived

        return result

    return wrapper


@interface_op
def head(dp: "DataPanel", n: int = 5):
    new_dp = dp.head(n)
    import numpy as np

    new_dp["head_column"] = np.zeros(len(new_dp))
    return new_dp
