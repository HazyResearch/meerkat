from abc import ABC
from collections import defaultdict
from functools import partial, wraps
from typing import Any, Callable, Dict, Generic, List, Set, TypeVar, Union

from pydantic import BaseModel, StrictBool, StrictFloat, StrictInt, StrictStr
from tqdm import tqdm

from meerkat.dataframe import DataFrame
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.ops.sliceby.sliceby import SliceBy
from meerkat.tools.utils import nested_apply


class NodeMixin:
    """
    Class for defining nodes in a graph.

    Add this mixin to any class whose objects should be nodes
    in a graph.

    This mixin is used in Box, Store and Operation to make
    them part of a computation graph.
    """

    def __init__(self):
        # The children of this node
        self.children: Set[Operation] = set()

    def __hash__(self):
        """Hash is based on the id of the node."""
        return hash(id(self))

    def __eq__(self, other):
        """Two nodes are equal if they have the same id."""
        return id(self) == id(other)

    def has_children(self):
        """Returns True if this node has children."""
        return len(self.children) > 0


def _topological_sort(root_nodes: List[NodeMixin]) -> List[NodeMixin]:
    """
    Perform a topological sort on a graph.

    Args:
        root_nodes (List[NodeMixin]): The root nodes of the graph.

    Returns:
        List[NodeMixin]: The topologically sorted nodes.
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


Primitive = Union[StrictInt, StrictStr, StrictFloat, StrictBool]
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
    type: str = "DataFrame"
    is_store: bool = True


T = TypeVar("T", "DataFrame", "SliceBy")


class PivotConfig(BoxConfig):
    pass


class Box(IdentifiableMixin, NodeMixin, Generic[T]):
    identifiable_group: str = "boxes"

    def __init__(self, obj):
        super().__init__()
        self.obj = obj

    @property
    def config(self):
        return BoxConfig(box_id=self.id, type="DataFrame")

    def __getattr__(self, name):
        return getattr(self.obj, name)

    def __getitem__(self, key):
        return self.obj[key]

    def __repr__(self):
        return f"Box({self.obj})"


class Pivot(Box, Generic[T]):
    def __repr__(self):
        return f"Pivot({self.obj})"


class DerivedConfig(BoxConfig):
    pass


class Derived(Box):
    def __repr__(self):
        return f"Derived({self.obj})"


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


def make_store(value: Union[str, Storeable]) -> Store:
    """
    Make a Store.

    If value is a Store, return it. Otherwise, return a
    new Store that wraps value.

    Args:
        value (Union[str, Storeable]): The value to wrap.

    Returns:
        Store: The Store wrapping value.
    """
    return value if isinstance(value, Store) else Store(value)


def make_box(value: Union[any, Box]) -> Box:
    """
    Make a Box.

    If value is a Box, return it. Otherwise, return a
    new Box that wraps value.

    Args:
        value (Union[any, Box]): The value to wrap.

    Returns:
        Box: The Box wrapping value.
    """
    # TODO(karan): why is this calling Pivot(value)?
    return value if isinstance(value, Box) else Pivot(value)


class Modification(BaseModel, ABC):
    """
    Base class for modifications.

    Modifications are used to track changes to Box and Store nodes
    in the graph.

    Attributes:
        id (str): The id of the Box or Store.
    """

    id: str

    @property
    def node(self):
        """The Box or Store node that this modification is for."""
        raise NotImplementedError()


class BoxModification(Modification):
    scope: List[str]
    type: str = "box"

    @property
    def node(self) -> Box:
        from meerkat.state import state

        return state.identifiables.get(group="boxes", id=self.id)


class StoreModification(Modification):
    value: Storeable
    type: str = "store"

    @property
    def node(self) -> Store:
        from meerkat.state import state

        return state.identifiables.get(group="stores", id=self.id)


class Operation(NodeMixin):
    def __init__(
        self,
        fn: Callable,
        args: List[Box],
        kwargs: Dict[str, Box],
        result: Derived,
        on=None,  # TODO: add support for on
    ):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.result = result
        self.on = on
        nested_apply(self.result, self.children.add)

    def __call__(self) -> List[Modification]:
        """
        Execute the operation. Unpack the arguments and keyword arguments
        and call the function. Then, update the result Box with the result
        and return a list of modifications.

        These modifications describe the delta changes made to the result Box,
        and are used to update the state of the GUI.
        """
        unpacked_args, unpacked_kwargs, _, _ = _unpack_boxes_and_stores(
            *self.args, **self.kwargs
        )
        update = self.fn(*unpacked_args, **unpacked_kwargs)

        modifications = []
        _update_result(self.result, update, modifications=modifications)

        return modifications


def _update_result(
    result: Union[list, tuple, dict, Box, Store, Primitive],
    update: Union[list, tuple, dict, Box, Store, Primitive],
    modifications: List[Modification],
) -> Union[list, tuple, dict, Box, Store, Primitive]:
    """
    Update the result object with the update object. This recursive
    function will perform a nested update to the result with the update.
    This function will also update the modifications list
    with the changes made to the result object.

    Args:
        result: The result object to update.
        update: The update object to use to update the result.
        modifications: The list of modifications to update.

    Returns:
        The updated result object.
    """

    if isinstance(result, list):
        # Recursively update each element of the list
        return [_update_result(r, u, modifications) for r, u in zip(result, update)]
    elif isinstance(result, tuple):
        # Recursively update each element of the tuple
        return tuple(
            _update_result(r, u, modifications) for r, u in zip(result, update)
        )
    elif isinstance(result, dict):
        # Recursively update each element of the dict
        return {
            k: _update_result(v, update[k], modifications) for k, v in result.items()
        }
    elif isinstance(result, Box):
        # If the result is a Box, then we need to update the Box's object
        # and return a BoxModification
        result.obj = update
        if isinstance(result.obj, DataFrame):
            modifications.append(
                BoxModification(id=result.id, scope=result.obj.columns)
            )
        return result
    elif isinstance(result, Store):
        # If the result is a Store, then we need to update the Store's value
        # and return a StoreModification
        result.value = update
        modifications.append(StoreModification(id=result.id, value=update))
        return result
    else:
        # If the result is not a Box or Store, then it is a primitive type
        # and we can just return the update
        return update


def trigger(modifications: List[Modification]) -> List[Modification]:
    """
    Trigger the computation graph of an interface based on a list of
    modifications.

    Return:
        List[Modification]: The list of modifications that resulted from running the
            computation graph.
    """
    # build a graph rooted at the stores and boxes in the modifications list
    root_nodes = [mod.node for mod in modifications]

    # Sort the nodes in topological order, and keep the Operation nodes
    order = [
        node for node in _topological_sort(root_nodes) if isinstance(node, Operation)
    ]

    print(f"trigged pipeline: {'->'.join([node.fn.__name__ for node in order])}")
    new_modifications = []
    with tqdm(total=len(order)) as pbar:
        for op in order:
            pbar.set_postfix_str(f"Running {op.fn.__name__}")
            mods = op()
            new_modifications.extend(mods)
            pbar.update(1)

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


def _nested_apply(obj: object, fn: callable, return_type: type = None):
    if return_type is Store or return_type is Box:
        return fn(obj, return_type=return_type)

    if isinstance(obj, list):
        if return_type is not None:
            assert return_type.__origin__ is list
            return_type = return_type.__args__[0]
        return [_nested_apply(v, fn=fn, return_type=return_type) for v in obj]
    elif isinstance(obj, tuple):
        if return_type is not None:
            assert return_type.__origin__ is tuple
            return_type = return_type.__args__[0]
        return tuple(_nested_apply(v, fn=fn, return_type=return_type) for v in obj)
    elif isinstance(obj, dict):
        if return_type is not None:
            assert return_type.__origin__ is dict
            return_type = return_type.__args__[1]
        return {
            k: _nested_apply(v, fn=fn, return_type=return_type) for k, v in obj.items()
        }
    else:
        return fn(obj, return_type=return_type)


def _pack_boxes_and_stores(obj, return_type: type = None):
    if return_type is Store:
        return Store(obj)
    elif return_type is Derived:
        return Derived(obj)

    if isinstance(obj, (DataFrame, SliceBy)):
        return Derived(obj)

    # TODO(Sabri): we should think more deeply about how to handle nested outputs
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return Store(obj)
    return obj


def _add_op_as_child(op: Operation, *boxes_and_stores: Union[Box, Store]):
    """
    Add the operation as a child of the boxes and stores.
    """
    for box_or_store in boxes_and_stores:
        if isinstance(box_or_store, (Box, Store)):
            box_or_store.children.add(op)


def interface_op(
    fn: Callable = None,
    nested_return: bool = True,
    return_type: type = None,
    first_call: Any = None,
    on: Union[Box, Store, List[Box], List[Store]] = None,
    also_on: Union[Box, Store, List[Box], List[Store]] = None,
) -> Callable:
    """
    Decorator that is used to mark a function as an interface operation.
    Functions decorated with this will create nodes in the operation graph, which
    are executed whenever their inputs are modified.

    Args:
        fn: The function to decorate.
        nested_return: Whether the function returns an object (e.g. List, Dict) with
            a nested structure. If True, a `Store` or `Derived` will be created for
            every element in the nested structure. If False, a single `Store` or
            `Derived` wrapping the entire object will be created. For example, if the
            function returns two DataFrames in a tuple, then `nested_return` should be
            `True`. However, if the functions returns a variable length list of ints,
            then `nested_return` should likely be `False`.
        return_type: The type of the return value.
        first_call: Return value for the first call to the function. This is useful for
            time consuming operations (e.g. image generation) that shouldn't trigger
            when the script is first run, and wait until an interaction with the GUI
            happens.

            Ideally, pass in a return value here that looks like the return value of the
            function. For example, if the function returns a DataFrame with columns `id`
            and `image`, then pass in an empty DataFrame with the same columns.

            You can also pass in a function that returns the first call value. This
            function should take the same arguments as the function being decorated
            (or should absorb arguments with `*args, **kwargs`).
        on: A Box or Store, or a list of Boxes or Stores. When these are modified, the
            function will be called. *This will prevent the function from being
            triggered when its inputs are modified.*
        also_on: A Box or Store, or a list of Boxes or Stores. When these are modified,
            the function will be called. *The function will continue to be
            triggered when its inputs are modified.*

    Returns:
        A decorated function that creates an operation node in the operation graph.
    """
    # Assert that only one of `on` and `also_on` is specified, if any.
    assert not (
        on is not None and also_on is not None
    ), "Must specify only one of `on` and `also_on` but not both. \
        Use `on` to prevent the decorated function from being called when its \
        arguments are modified (and only pay attention to the objects passed \
        into `on`), and use `also_on` to trigger the function when its arguments \
        are modified (and additionally when the objects passed into `also_on` \
        are modified)."

    if fn is None:
        # need to make passing args to the args optional
        # note: all of the args passed to the decorator MUST be optional
        return partial(
            interface_op,
            nested_return=nested_return,
            return_type=return_type,
            first_call=first_call,
            on=on,
            also_on=also_on,
        )

    def _interface_op(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            """
            This `wrapper` function is only run once. It creates a node in the
            operation graph and returns a `Box` object that wraps the output of the
            function.

            Subsequent calls to the function will be handled by the graph.
            """
            # TODO(karan): have to make `on` and `also_on` nonlocal otherwise
            # it throws an UnboundLocalError. But why doesn't this happen for
            # `first_call`?!
            nonlocal on, also_on

            # TODO(Sabri): this should be nested
            unpacked_args, unpacked_kwargs, boxes, stores = _unpack_boxes_and_stores(
                *args, **kwargs
            )

            if first_call is not None:
                # For expensive functions, the user can specify a first call value
                # that allows us to setup the Operation without running the function
                if isinstance(first_call, Callable):
                    result = first_call(*unpacked_args, **unpacked_kwargs)
                else:
                    result = first_call
            else:
                # By default, run the function to produce a result
                # Call the function on the unpacked args and kwargs
                result = fn(*unpacked_args, **unpacked_kwargs)

            # Setup an Operation node if any of the
            # args or kwargs were boxes or stores
            if len(boxes) > 0 or len(stores) > 0:

                # The result should be placed inside a Store or Derived
                # (or a nested object) containing Stores and Deriveds.
                # Then we can update the contents of this result when the
                # function is called again.
                if nested_return:
                    derived = _nested_apply(
                        result, fn=_pack_boxes_and_stores, return_type=return_type
                    )
                elif isinstance(result, (DataFrame, SliceBy)):
                    derived = Derived(result)
                else:
                    derived = Store(result)

                # Create the Operation node
                op = Operation(fn=fn, args=args, kwargs=kwargs, result=derived)

                if on is None:
                    # Add this Operation node as a child of all of the boxes and stores
                    # regardless of the value of `also_on`
                    _add_op_as_child(op, *boxes, *stores)
                else:
                    # Add this Operation node as a child of the boxes and stores
                    # passed into `on`
                    # TODO(Sabri): this should be nested
                    if isinstance(on, (Box, Store)):
                        on = [on]
                    _, _, boxes, stores = _unpack_boxes_and_stores(*on)
                    _add_op_as_child(op, *boxes, *stores)

                if also_on is not None:
                    # Add this Operation node as a child of the boxes and stores
                    # passed into `also_on`
                    # TODO(Sabri): this should be nested
                    if isinstance(also_on, (Box, Store)):
                        also_on = [also_on]
                    _, _, boxes, stores = _unpack_boxes_and_stores(*also_on)
                    _add_op_as_child(op, *boxes, *stores)

                return derived

            return result

        return wrapper

    return _interface_op(fn)


@interface_op
def head(df: "DataFrame", n: int = 5):
    new_df = df.head(n)
    import numpy as np

    new_df["head_column"] = np.zeros(len(new_df))
    return new_df
