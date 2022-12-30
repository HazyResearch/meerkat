from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from meerkat.interactive.graph.operation import Operation
from meerkat.interactive.graph.utils import (
    _get_nodeables,
    _replace_nodeables_with_nodes,
)
from meerkat.interactive.node import NodeMixin

__all__ = ["react", "reactive", "is_reactive", "get_reactive_kwargs"]


def reactive(
    fn: Callable = None,
    nested_return: bool = False,
) -> Callable:
    """Decorator that is used to mark a function as an interface operation.

    Functions decorated with this will create nodes in the operation graph,
    which are executed whenever their inputs are modified.

    TODO: Remove nested_return argument. With the addition of __iter__ and __next__
    to mk.gui.Store, we no longer need to support nested return values.

    A basic example that adds two numbers:

    .. code-block:: python

        @reactive
        def add(a: int, b: int) -> int:
            return a + b

        a = Store(1)
        b = Store(2)
        c = add(a, b)

    When either `a` or `b` is modified, the `add` function will be called again
    with the new values of `a` and `b`.

    A more complex example that concatenates two mk.DataFrame objects:

    .. code-block:: python

        @reactive
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

    Returns:
        A decorated function that creates an operation node in the operation graph.
    """

    if fn is None:
        # need to make passing args to the args optional
        # note: all of the args passed to the decorator MUST be optional
        return partial(
            reactive,
            nested_return=nested_return,
        )

    def _reactive(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            """This `wrapper` function is only run once. It creates a node in
            the operation graph and returns a `Reference` object that wraps the
            output of the function.

            Subsequent calls to the function will be handled by the
            graph.
            """
            from meerkat.interactive.graph.store import Store

            # nested_return is False because any operations on the outputs of the
            # function should recursively generate Stores / References.
            # For example, if fn returns a list. The reactified fn will return
            # a Store(list).
            # Then, Store(list)[0] should also return a Store.
            # TODO (arjun): These if this assumption holds.
            nonlocal nested_return
            nonlocal fn

            # Check if fn is a bound method (i.e. an instance method).
            # If so, we need to functionalize the method (i.e. make the method
            # into a function).
            # First argument in *args must be the instance.
            # We assume that the type of the instance will not change.
            def _fn_outer_wrapper(_fn):
                @wraps(_fn)
                def _fn_wrapper(*args, **kwargs):
                    return _fn(*args, **kwargs)

                return _fn_wrapper

            if hasattr(fn, "__self__") and fn.__self__ is not None:
                args = (fn.__self__, *args)
                # The method bound to the class.
                fn_class = getattr(fn.__self__.__class__, fn.__name__)
                fn = _fn_outer_wrapper(fn_class)

            # Call the function on the args and kwargs
            with no_react():
                result = fn(*args, **kwargs)

            if not is_reactive():
                # If we are not in a reactive context, then we don't need to create
                # any nodes in the graph.
                # `fn` should be run as normal.
                return result

            # Now we're in a reactive context i.e. is_reactive() == True

            # Get all the NodeMixin objects from the args and kwargs
            # These objects will be parents of the Operation node
            # that is created for this function
            nodeables = _get_nodeables(*args, **kwargs)

            # By default, nested return is True when the output is a tuple.
            if nested_return is None:
                nested_return = isinstance(result, tuple)

            # Wrap the Result in NodeMixin objects
            if nested_return:
                result = _nested_apply(result, fn=_wrap_outputs)
            elif isinstance(result, NodeMixin):
                result = result
            else:
                result = Store(result)

            # Setup an Operation node if any of the args or kwargs
            # were nodeables
            op = None

            # Create Nodes for each NodeMixin object
            _create_nodes_for_nodeables(*nodeables)
            args = _replace_nodeables_with_nodes(args)
            kwargs = _replace_nodeables_with_nodes(kwargs)

            # Create the Operation node
            op = Operation(fn=fn, args=args, kwargs=kwargs, result=result)

            # For normal functions
            # Make a node for the operation if it doesn't have one
            if not op.has_inode():
                op.attach_to_inode(op.create_inode())

            # Add this Operation node as a child of all of the nodeables
            _add_op_as_child(op, *nodeables, triggers=True)

            # Attach the Operation node to its children (if it is not None)
            def _foo(nodeable: NodeMixin):
                # FIXME: make sure they are not returning a nodeable that
                # is already in the dag. May be related to checking that the graph
                # is acyclic.
                if not nodeable.has_inode():
                    inode_id = None if not isinstance(nodeable, Store) else nodeable.id
                    nodeable.attach_to_inode(nodeable.create_inode(inode_id=inode_id))

                if op is not None:
                    op.inode.add_child(nodeable.inode)

            _nested_apply(result, _foo)

            return result

        return wrapper

    return _reactive(fn)


# A stack that manages if reactive mode is enabled
# The stack is reveresed so that the top of the stack is
# the last index in the list.
class _ReactiveState:
    def __init__(self, *, reactive: bool, nested_return: Optional[bool]) -> None:
        self.reactive = reactive
        self.kwargs = dict(nested_return=nested_return)

    def __bool__(self):
        return self.reactive

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"reactive={self.reactive}, "
            f"nested_return={self.kwargs['nested_return']}"
            ")"
        )


_IS_REACTIVE: List[_ReactiveState] = []


def is_reactive() -> bool:
    """Whether the code is in reactive context.

    Returns:
        bool: True if the code is in a reactive context.
    """
    return len(_IS_REACTIVE) > 0 and bool(_IS_REACTIVE[-1])


def get_reactive_kwargs() -> Dict[str, Any]:
    """Get the kwargs for the current reactive context.

    Returns:
        Dict[str, Any]: The kwargs for the current reactive context.
    """
    return _IS_REACTIVE[-1].kwargs


# Used for annotating decorator usage of 'react'.
# Adapted from PyTorch:
# https://mypy.readthedocs.io/en/latest/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


class react:
    """Context-manager that is used control if code is an interface operation.

    Code-blocks in this context manager will create nodes
    in the operation graph, which are executed whenever their inputs
    are modified.

    A basic example that adds two numbers:

    .. code-block:: python

        a = Store(1)
        b = Store(2)
        with react():
            c = a + b

    When either `a` or `b` is modified, the code block will re-execute
    with the new values of `a` and `b`.

    Also functions as a decorator. (Make sure to instantiate with parenthesis.):

    .. code-block:: python

            @react()
            def add(a: int, b: int) -> int:
                return a + b

            a = Store(1)
            b = Store(2)
            c = add(a, b)

    A more complex example that concatenates two mk.DataFrame objects:

    .. code-block:: python

        @react()
        def concat(df1: mk.DataFrame, df2: mk.DataFrame) -> mk.DataFrame:
            return mk.concat([df1, df2])

        df1 = mk.DataFrame(...)
        df2 = mk.DataFrame(...)
        df3 = concat(df1, df2)

    Args:
        reactive: The function to decorate.
        nested_return: Whether the function returns an object (e.g. List, Dict) with
            a nested structure. If True, a `Store` or `Reference` will be created for
            every element in the nested structure. If False, a single `Store` or
            `Reference` wrapping the entire object will be created. For example, if the
            function returns two DataFrames in a tuple, then `nested_return` should be
            `True`. However, if the functions returns a variable length list of ints,
            then `nested_return` should likely be `False`.

    Returns:
        A decorated function that creates an operation node in the operation graph.
    """

    def __init__(self, reactive: bool = True, *, nested_return: bool = False):
        self._reactive = reactive
        self._nested_return = nested_return

    def __call__(self, func):
        @wraps(func)
        def decorate_context(*args, **kwargs):
            with self.clone():
                return reactive(func, nested_return=self._nested_return)(
                    *args, **kwargs
                )

        return cast(F, decorate_context)

    def __enter__(self):
        _IS_REACTIVE.append(
            _ReactiveState(reactive=self._reactive, nested_return=self._nested_return)
        )
        return self

    def __exit__(self, type, value, traceback):
        _IS_REACTIVE.pop(-1)

    def clone(self):
        return self.__class__(
            reactive=self._reactive, nested_return=self._nested_return
        )


class no_react(react):
    def __init__(self, nested_return: bool = False):
        super().__init__(reactive=False, nested_return=nested_return)


def _nested_apply(obj: object, fn: callable):
    from meerkat.interactive.graph.store import Store

    def _internal(_obj: object, depth: int = 0):
        if isinstance(_obj, Store) or isinstance(_obj, NodeMixin):
            return fn(_obj)
        if isinstance(_obj, list):
            return [_internal(v, depth=depth + 1) for v in _obj]
        elif isinstance(_obj, tuple):
            return tuple(_internal(v, depth=depth + 1) for v in _obj)
        elif isinstance(_obj, dict):
            return {k: _internal(v, depth=depth + 1) for k, v in _obj.items()}
        elif _obj is None:
            return None
        elif depth > 0:
            # We want to call the function on the object (including primitives) when we
            # have recursed into it at least once.
            return fn(_obj)
        else:
            raise ValueError(f"Unexpected type {type(_obj)}.")

    return _internal(obj)


def _add_op_as_child(
    op: Operation,
    *nodeables: NodeMixin,
    triggers: bool = True,
):
    """Add the operation as a child of the nodeables.

    Args:
        op: The operation to add as a child.
        nodeables: The nodeables to add the operation as a child.
        triggers: Whether the operation is triggered by changes in the
            nodeables.
    """
    for nodeable in nodeables:
        # Add the operation as a child of the nodeable
        nodeable.inode.add_child(op.inode, triggers=triggers)


def _wrap_outputs(obj):
    from meerkat.interactive.graph.store import Store

    if isinstance(obj, NodeMixin):
        return obj
    return Store(obj)


def _create_nodes_for_nodeables(*nodeables: NodeMixin):
    from meerkat.interactive.graph.store import Store

    for nodeable in nodeables:
        assert isinstance(nodeable, NodeMixin)
        # Make a node for this nodeable if it doesn't have one
        if not nodeable.has_inode():
            inode_id = None if not isinstance(nodeable, Store) else nodeable.id
            nodeable.attach_to_inode(nodeable.create_inode(inode_id=inode_id))
