from abc import ABC
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from functools import partial, wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Set,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel

from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.ops.sliceby.sliceby import SliceBy
from meerkat.tools.utils import nested_apply

if TYPE_CHECKING:
    from meerkat.datapanel import DataPanel
    from meerkat.ops.sliceby.sliceby import SliceByCards


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
Storeable = Union[
    None, 
    Primitive,
    List[Primitive],
    Dict[Primitive, Primitive],
    Dict[Primitive, List[Primitive]],
    List[Dict[Primitive, Primitive]],
]


class BoxConfig(BaseModel):
    box_id: str
    type: str = "DataPanel"
    is_store: bool = True


T = TypeVar("T", "DataPanel", "SliceBy")


class Box(IdentifiableMixin, NodeMixin, Generic[T]):
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


class Pivot(Box, Generic[T]):
    pass


class DerivedConfig(BoxConfig):
    pass


class Derived(Box):
    pass


class StoreConfig(BaseModel):
    store_id: str
    value: Any
    has_children: bool
    is_store: bool = True


class Store(IdentifiableMixin, NodeMixin, Generic[T]):
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


def make_box(value: Union[any, Box]):
    if isinstance(value, Box):
        return value
    return Pivot(value)


class Operation(NodeMixin):
    def __init__(
        self, fn: Callable, args: List[Box], kwargs: Dict[str, Box], result: Derived
    ):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.result = result
        nested_apply(self.result, self.children.add)

    def __call__(self):
        unpacked_args, unpacked_kwargs, _, _ = _unpack_boxes_and_stores(
            *self.args, **self.kwargs
        )
        update = self.fn(*unpacked_args, **unpacked_kwargs)

        modifications = []
        _update_result(self.result, update, modifications=modifications)

        return modifications


def _update_result(result: object, update: object, modifications: List[Modification]):
    from meerkat.datapanel import DataPanel

    if isinstance(result, list):
        return [_update_result(r, u, modifications) for r, u in zip(result, update)]
    elif isinstance(result, tuple):
        return tuple(
            _update_result(r, u, modifications) for r, u in zip(result, update)
        )
    elif isinstance(result, dict):
        return {
            k: _update_result(v, update[k], modifications) for k, v in result.items()
        }
    elif isinstance(result, Box):
        result.obj = update
        if isinstance(result.obj, DataPanel):
            modifications.append(
                BoxModification(id=result.id, scope=result.obj.columns)
            )
        return result
    elif isinstance(result, Store):
        result.value = update
        modifications.append(StoreModification(id=result.id, value=update))
        return result
    else:
        return update


def _topological_sort(root_nodes: List[NodeMixin]) -> List[NodeMixin]:
    """Perform a topological sort on a graph.

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

    new_modifications = [mod for op in order for mod in op()]
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


def _pack_boxes_and_stores(obj):
    from meerkat.datapanel import DataPanel
    from meerkat.ops.sliceby.sliceby import SliceBy

    if isinstance(obj, (DataPanel, SliceBy)):
        return Derived(obj)

    # TODO(Sabri): we should think more deeply about how to handle nested outputs
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return Store(obj)
    return obj


def interface_op(fn: Callable = None, nested_return: bool = True) -> Callable:
    """
    Functions decorated with this will create nodes in the operation graph.

    Args:
        fn: The function to decorate.
        nested_return: Whether the function returns an object (e.g. List, Dict) with
            a nested structure. If True, a `Store` or `Derived` will be created for
            every element in the nested structure. If False, a single `Store` or
            `Derived` wrapping the entire object will be created. For example, if the
            function returns two DataPanels in a tuple, then `nested_return` should be
            `True`. However, if the functions returns a variable length list of ints,
            then `nested_return` should likely be `False`.
    """
    if fn is None:
        # need to make passing args to the args optional
        # note: all of the args passed to the decorator MUST be optional
        return partial(interface_op, nested_return=nested_return)

    def _interface_op(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # TODO(Sabri): this should be nested
            unpacked_args, unpacked_kwargs, boxes, stores = _unpack_boxes_and_stores(
                *args, **kwargs
            )

            result = fn(*unpacked_args, **unpacked_kwargs)

            if len(boxes) > 0 or len(stores) > 0:
                from meerkat.datapanel import DataPanel
                from meerkat.ops.sliceby.sliceby import SliceBy

                if nested_return:
                    derived = nested_apply(result, fn=_pack_boxes_and_stores)
                elif isinstance(result, (DataPanel, SliceBy)):
                    derived = Derived(result)
                else:
                    derived = Store(result)

                op = Operation(fn=fn, args=args, kwargs=kwargs, result=derived)

                for input_box in boxes:
                    input_box.children.add(op)
                for input_store in stores:
                    input_store.children.add(op)

                return derived

            return result

        return wrapper

    return _interface_op(fn)


@interface_op
def head(dp: "DataPanel", n: int = 5):
    new_dp = dp.head(n)
    import numpy as np

    new_dp["head_column"] = np.zeros(len(new_dp))
    return new_dp
