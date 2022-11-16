import inspect
from functools import partial, wraps
from typing import Any, Callable, Dict, Generic, List, Union

from pydantic import BaseModel
from tqdm import tqdm
from wrapt import ObjectProxy

from meerkat.dataframe import DataFrame
from meerkat.interactive.modification import (
    Modification,
    ReferenceModification,
    StoreModification,
)
from meerkat.interactive.node import NodeMixin, _topological_sort
from meerkat.interactive.types import Primitive, Storeable, T
from meerkat.mixins.identifiable import IdentifiableMixin


# A stack that manages if reactive mode is enabled
# The stack is reveresed so that the top of the stack is
# the last index in the list.
_IS_REACTIVE = []


def is_reactive():
    return len(_IS_REACTIVE) > 0 and _IS_REACTIVE[-1]


def reactify(fn, **kwargs):
    """Reactify a function when reactive mode is enabled.

    To enable reactive mode, the function/method should be called
    in the :class:`mk.gui.react` context.

    Args:
        fn: The function to reactify.
        **kwargs: Keyword arguments to pass to :func:`interface_op`.
            Cannot include `force_reactify` or `nested_return` arguments.

    Returns:
        Callable: The reactified function.
    """
    # nested_return is False because any operations on the outputs of the
    # function should recursively generate Stores / References.
    # For example, if fn returns a list. The reactified fn will return a Store(list).
    # Then, Store(list)[0] should also return a Store.
    # TODO (arjun): These if this assumption holds.
    if is_reactive():
        fn = interface_op(fn=fn, force_reactify=True, nested_return=False, **kwargs)

    return fn


def reactify_decorator(fn=None, **interface_op_kwargs):
    """A decorator for reactifying functions and methods."""
    if fn is None:
        return partial(reactify_decorator, **interface_op_kwargs)

    def _decorator(fn):
        @wraps(fn)
        def _wrapper(*args, **kwargs):
            return reactify(fn, **interface_op_kwargs)(*args, **kwargs)

        return _wrapper

    return _decorator(fn)


class react:
    def __init__(self, reactive: bool = True):
        self._reactive = reactive

    def __enter__(self):
        _IS_REACTIVE.append(self._reactive)
        return self

    def __exit__(self, type, value, traceback):
        _IS_REACTIVE.pop(-1)


class no_react(react):
    def __init__(self):
        super().__init__(reactive=False)


class StoreConfig(BaseModel):
    store_id: str
    value: Any
    has_children: bool
    is_store: bool = True


# ObjectProxy must be the last base class
class Store(IdentifiableMixin, NodeMixin, Generic[T], ObjectProxy):

    _self_identifiable_group: str = "stores"

    def __init__(self, wrapped: T):
        super().__init__(wrapped=wrapped)

    @property
    def config(self):
        return StoreConfig(
            store_id=self.id,
            value=self.__wrapped__,
            has_children=self.has_children(),
        )

    def set(self, new_value: T):
        """Set the value of the store."""
        mod = StoreModification(id=self.id, value=new_value)
        self.__wrapped__ = new_value
        mod.add_to_queue()

    def __repr__(self) -> str:
        return f"Store({self.__wrapped__})"

    @property
    def detail(self):
        return f"Store({self.__wrapped__}) has id {self.id} and node id {self.node_id}"

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.__wrapped__, name)

        # Only executing functions/methods should make the output reactifiable.
        # TODO: See if we want accessing attributes/properties to be reactifiable.
        # The reason they are not reactifiable now is that it is not clear what
        # storing the attributes as a state would give us.
        return reactify(attr) if callable(attr) else attr

    # @reactify_decorator()
    def __add__(self, other):
        # TODO (arjun): This should not fail with karan's changes.
        return super().__add__(other)


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


class Operation(NodeMixin):
    def __init__(
        self,
        fn: Callable,
        args: List[Any],
        kwargs: Dict[str, Any],
        result: Any,
    ):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.result = result

    def __call__(self) -> List[Modification]:
        """
        Execute the operation. Unpack the arguments and keyword arguments
        and call the function. Then, update the result Reference with the result
        and return a list of modifications.

        These modifications describe the delta changes made to the result Reference,
        and are used to update the state of the GUI.
        """
        update = self.fn(*self.args, **self.kwargs)

        modifications = []
        self.result = _update_result(self.result, update, modifications=modifications)

        return modifications


def _update_result(
    result: Union[list, tuple, dict, Store, Primitive],
    update: Union[list, tuple, dict, Store, Primitive],
    modifications: List[Modification],
) -> Union[list, tuple, dict, Store, Primitive]:
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

    if isinstance(result, DataFrame):
        # Detach the result object from the Node
        inode = result.detach_inode()

        # Attach the inode to the update object
        update.attach_to_inode(inode)

        # Create modifications
        modifications.append(ReferenceModification(id=inode.id, scope=update.columns))

        return update

    elif isinstance(result, Store):
        # If the result is a Store, then we need to update the Store's value
        # and return a StoreModification
        # TODO(karan): now checking if the value is the same
        # This is assuming that all values put into Stores have an __eq__ method
        # defined that can be used to check if the value has changed.
        if isinstance(result, (str, int, float, bool, type(None), tuple)):
            # We can just check if the value is the same
            if result != update:
                result.set(update)
                modifications.append(
                    StoreModification(id=result.inode.id, value=update)
                )
        else:
            # We can't just check if the value is the same if the Store contains
            # a list, dict or object, since they are mutable (and it would just
            # return True).
            result.set(update)
            modifications.append(StoreModification(id=result.inode.id, value=update))
        return result
    elif isinstance(result, list):
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
    else:
        # If the result is not a Reference or Store, then it is a primitive type
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
    # build a graph rooted at the stores and refs in the modifications list
    root_nodes = [mod.node for mod in modifications]

    # Sort the nodes in topological order, and keep the Operation nodes
    order = [
        node.obj
        for node in _topological_sort(root_nodes)
        if isinstance(node.obj, Operation)
    ]

    print(f"trigged pipeline: {'->'.join([node.fn.__name__ for node in order])}")
    new_modifications = []
    with tqdm(total=len(order)) as pbar:
        # Go through all the operations in order: run them and add their modifications
        # to the new_modifications list
        for op in order:
            pbar.set_postfix_str(f"Running {op.fn.__name__}")
            mods = op()
            # TODO: check this
            mods = [mod for mod in mods if not isinstance(mod, StoreModification)]
            new_modifications.extend(mods)
            pbar.update(1)
    return modifications + new_modifications


def _get_nodeables(*args, **kwargs):
    nodeables = []
    for arg in args:
        if isinstance(arg, NodeMixin):
            nodeables.append(arg)
        elif isinstance(arg, list) or isinstance(arg, tuple):
            nodeables.extend(_get_nodeables(*arg))
        elif isinstance(arg, dict):
            nodeables.extend(_get_nodeables(**arg))

    for _, v in kwargs.items():
        if isinstance(v, NodeMixin):
            nodeables.append(v)
        elif isinstance(v, list) or isinstance(v, tuple):
            nodeables.extend(_get_nodeables(*v))
        elif isinstance(v, dict):
            nodeables.extend(_get_nodeables(**v))
    return nodeables


def _wrap_outputs(obj, return_type: type = None):
    if isinstance(obj, NodeMixin):
        return obj
    return Store(obj)


def _add_op_as_child(
    op: Operation,
    *nodeables: NodeMixin,
    triggers: bool = True,
):
    """
    Add the operation as a child of the nodeables.

    Args:
        op: The operation to add as a child.
        nodeables: The nodeables to add the operation as a child.
        triggers: Whether the operation is triggered by changes in the
            nodeables.
    """
    for nodeable in nodeables:
        assert isinstance(nodeable, NodeMixin)
        # Make a node for this nodeable if it doesn't have one
        if not nodeable.has_inode():
            inode_id = None if not isinstance(nodeable, Store) else nodeable.id
            nodeable.attach_to_inode(nodeable.create_inode(inode_id=inode_id))

        # Add the operation as a child of the nodeable
        nodeable.inode.add_child(op.inode, triggers=triggers)


def _nested_apply(obj: object, fn: callable):
    if isinstance(obj, Store) or isinstance(obj, NodeMixin):
        return fn(obj)

    if isinstance(obj, list):
        return [_nested_apply(v, fn=fn) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_nested_apply(v, fn=fn) for v in obj)
    elif isinstance(obj, dict):
        return {k: _nested_apply(v, fn=fn) for k, v in obj.items()}
    else:
        raise ValueError(f"Unexpected type {type(obj)}.")


def interface_op(
    fn: Callable = None,
    nested_return: bool = True,
    return_type: type = None,
    force_reactify: bool = False,
) -> Callable:
    """
    Decorator that is used to mark a function as an interface operation.
    Functions decorated with this will create nodes in the operation graph, which
    are executed whenever their inputs are modified.

    A basic example that adds two numbers:
    .. code-block:: python

        @interface_op
        def add(a: int, b: int) -> int:
            return a + b

        a = Store(1)
        b = Store(2)
        c = add(a, b)

    When either `a` or `b` is modified, the `add` function will be called again
    with the new values of `a` and `b`.

    A more complex example that concatenates two mk.DataFrame objects:
    .. code-block:: python

        @interface_op
        def concat(df1: mk.DataFrame, df2: mk.DataFrame) -> mk.DataFrame:
            return mk.concat([df1, df2])

        df1 = mk.DataFrame(...)
        df2 = mk.DataFrame(...)
        df3 = concat(df1, df2)

    Args:
        fn: The function to decorate.
        nested_return: Whether the function returns an object (e.g. List, Dict) with
            a nested structure. If True, a `Store` or `Reference` will be created for
            every element in the nested structure. If False, a single `Store` or
            `Reference` wrapping the entire object will be created. For example, if the
            function returns two DataFrames in a tuple, then `nested_return` should be
            `True`. However, if the functions returns a variable length list of ints,
            then `nested_return` should likely be `False`.
        return_type: The type of the return value.
        force_reactify: Whether to force the function to be reactified. This is useful
            for returning outputs that are always reactified.

    Returns:
        A decorated function that creates an operation node in the operation graph.
    """
    if fn is None:
        # need to make passing args to the args optional
        # note: all of the args passed to the decorator MUST be optional
        return partial(
            interface_op,
            nested_return=nested_return,
            return_type=return_type,
        )

    def _interface_op(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            """
            This `wrapper` function is only run once. It creates a node in the
            operation graph and returns a `Reference` object that wraps the
            output of the function.

            Subsequent calls to the function will be handled by the graph.
            """
            # Get all the NodeMixin objects from the args and kwargs
            # These objects will be parents of the Operation node
            # that is created for this function
            nodeables = _get_nodeables(*args, **kwargs)

            # Check if fn is a bound method
            if hasattr(fn, "__self__") and fn.__self__ is not None:
                if isinstance(fn.__self__, NodeMixin):
                    nodeables.append(fn.__self__)

            # Call the function on the args and kwargs
            result = fn(*args, **kwargs)

            # Setup an Operation node if any of the
            # args or kwargs were nodeables or we're forcing reactivity
            if (len(nodeables) > 0) or force_reactify:
                # The result should be placed inside a Store
                # (or a nested object) containing Stores if it isn't
                # already a NodeMixin ("nodeable") object.
                # Then we can update the contents of this result when the
                # function is called again.
                if nested_return:
                    result = _nested_apply(result, fn=_wrap_outputs)
                elif isinstance(result, NodeMixin):
                    result = result
                else:
                    result = Store(result)

                # Create the Operation node
                op = Operation(fn=fn, args=args, kwargs=kwargs, result=result)

                # For normal functions
                # Make a node for the operation if it doesn't have one
                if not op.has_inode():
                    op.attach_to_inode(op.create_inode())

                # Add this Operation node as a child of all of the nodeables
                _add_op_as_child(op, *nodeables, triggers=True)

                # Attach the Operation node to its children
                def _foo(nodeable: NodeMixin):
                    if not nodeable.has_inode():
                        inode_id = (
                            None if not isinstance(nodeable, Store) else nodeable.id
                        )
                        nodeable.attach_to_inode(
                            nodeable.create_inode(inode_id=inode_id)
                        )

                    op.inode.add_child(nodeable.inode)

                _nested_apply(result, _foo)

            return result

        return wrapper

    return _interface_op(fn)


# A stack that manages if reactive mode is enabled
# The stack is reveresed so that the top of the stack is
# the last index in the list.
_IS_REACTIVE = []


def is_reactive():
    return len(_IS_REACTIVE) > 0 and _IS_REACTIVE[-1]


def reactify(fn, **kwargs):
    """Reactify a function when reactive mode is enabled.

    To enable reactive mode, the function/method should be called
    in the :class:`mk.gui.react` context.

    Args:
        fn: The function to reactify.
        **kwargs: Keyword arguments to pass to :func:`interface_op`.
            Cannot include `force_reactify` or `nested_return` arguments.

    Returns:
        Callable: The reactified function.
    """
    # nested_return is False because any operations on the outputs of the
    # function should recursively generate Stores / References.
    # For example, if fn returns a list. The reactified fn will return a Store(list).
    # Then, Store(list)[0] should also return a Store.
    # TODO (arjun): These if this assumption holds.
    if is_reactive():
        fn = interface_op(fn=fn, force_reactify=True, nested_return=False, **kwargs)

    return fn


class react:
    def __init__(self, reactive: bool = True):
        self._reactive = reactive

    def __enter__(self):
        _IS_REACTIVE.append(self._reactive)
        return self

    def __exit__(self, type, value, traceback):
        _IS_REACTIVE.pop(-1)


class no_react(react):
    def __init__(self):
        super().__init__(reactive=False)
