import inspect
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from meerkat.interactive.graph.operation import (
    Operation,
    _check_fn_has_leading_self_arg,
)
from meerkat.interactive.graph.utils import (
    _get_nodeables,
    _replace_nodeables_with_nodes,
)
from meerkat.interactive.node import NodeMixin

__all__ = ["_react", "_reactive", "is_reactive", "get_reactive_kwargs"]


def _reactive(
    fn: Callable = None,
    nested_return: bool = None,
    skip_fn: Callable[..., bool] = None,
) -> Callable:
    """Internal decorator that is used to mark a function as reactive.
    This is only meant for internal use, and users should use the
    :func:`react` decorator instead.

    Functions decorated with this will create nodes in the operation graph,
    which are executed whenever their inputs are modified.

    A basic example that adds two numbers:

    .. code-block:: python

        @_reactive
        def add(a: int, b: int) -> int:
            return a + b

        a = Store(1)
        b = Store(2)
        c = add(a, b)

    When either `a` or `b` is modified, the `add` function will be called again
    with the new values of `a` and `b`.

    A more complex example that concatenates two mk.DataFrame objects:

    .. code-block:: python

        @_reactive
        def concat(df1: mk.DataFrame, df2: mk.DataFrame) -> mk.DataFrame:
            return mk.concat([df1, df2])

        df1 = mk.DataFrame(...)
        df2 = mk.DataFrame(...)
        df3 = concat(df1, df2)

    Args:
        fn: See :func:`react`.
        nested_return: See :func:`react`.
        skip_fn: See :func:`react`.

    Returns:
        See :func:`react`.
    """
    # TODO: Remove nested_return argument. With the addition of __iter__ and __next__
    # to mk.gui.Store, we no longer need to support nested return values.
    # This will require looking through current use of reactive and patching them.
    if fn is None:
        # need to make passing args to the args optional
        # note: all of the args passed to the decorator MUST be optional
        return partial(_reactive, nested_return=nested_return, skip_fn=skip_fn)

    def __reactive(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            """This `wrapper` function is only run once. It creates a node in
            the operation graph and returns a `Reference` object that wraps the
            output of the function.

            Subsequent calls to the function will be handled by the
            graph.
            """
            from meerkat.interactive.graph.store import (
                Store,
                _unpack_stores_from_object,
            )

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

            # Unpack the stores from the args and kwargs
            # unpacked_args, unpacked_kwargs, _ = _unpack_stores(*args, **kwargs)
            unpacked_args, _ = _unpack_stores_from_object(list(args))
            unpacked_kwargs, _ = _unpack_stores_from_object(kwargs)

            if hasattr(fn, "__self__") and fn.__self__ is not None:
                args = (fn.__self__, *args)

                # Unpack the stores from the args and kwargs because
                # args has changed!
                # TODO: make this all nicer
                # unpacked_args, unpacked_kwargs, _ = _unpack_stores(*args, **kwargs)
                unpacked_args, _ = _unpack_stores_from_object(list(args))
                unpacked_kwargs, _ = _unpack_stores_from_object(kwargs)

                # The method bound to the class.
                fn_class = getattr(fn.__self__.__class__, fn.__name__)
                fn = _fn_outer_wrapper(fn_class)

                # If `fn` is an instance method, then the first argument in `args`
                # is the instance. We should **not** unpack the `self` argument
                # if it is a Store.
                if isinstance(args[0], Store):
                    unpacked_args[0] = args[0]
            elif _check_fn_has_leading_self_arg(fn):
                # If `fn` is an instance method, then the first argument in `args`
                # is the instance. We should **not** unpack the `self` argument
                # if it is a Store.
                if isinstance(args[0], Store):
                    unpacked_args[0] = args[0]

            # Call the function on the args and kwargs
            with no_react():
                result = fn(*unpacked_args, **unpacked_kwargs)
                # result = fn(*args, **kwargs)

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
            op = Operation(
                fn=fn, args=args, kwargs=kwargs, result=result, skip_fn=skip_fn
            )

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

    return __reactive(fn)


def react(
    input: Union[Callable, Any],
    nested_return: bool = None,
    skip_fn: Callable[..., bool] = None,
) -> Union[Callable, Any]:
    """Make a function or object reactive. Use as a decorator, or call with a
    function or object.

    If the input is a function, then the function will be added to a
    computational graph, and will be re-run when any of its inputs change.

    If the input is an object, then the object will become reactive: all of its
    methods and properties will become reactive. It will be returned as a
    `Store` object.

    Args:
        input: A function or object to make reactive.
        nested_return: Whether the function returns a nested tuple of objects.
            If `None`, then the function will be assumed to return a nested tuple
            if the return value is a tuple.
        skip_fn: A function that takes the same arguments as the function being
            wrapped, and returns a boolean. If the function returns `True`, then
            the function will not be added to the graph.

    Returns:
        A reactive function or object.

    Examples:

    Use `react` on primitive types:

        >>> x: Store = react(1)
        >>> # x is now a `Store` object
        >>> y = x + 1
        >>> # y will be a `Store` object, and will re-run when `x` changes

    Use `react` on complex types:

        >>> x: Store = react([1, 2, 3])
        >>> y = x[0] + 1
        >>> # y will be a `Store` object, and will re-run when `x` changes

    Use `react` on instances of classes:

        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2, 3]})
        >>> x: Store = react(df)
        >>> y = x.head()
        >>> # y will be a `Store` object, and will re-run when `x` changes

        >>> class Foo:
        ...     def __init__(self, x):
        ...         self.x = x
        ...     def __call__(self):
        ...         return self.x + 1
        >>> f = Foo(1)
        >>> x: Store = react(f)
        >>> y = x()
        >>> # y will be a `Store` object, and will re-run when `x` changes

    Use `react` on built-in functions:

        >>> print = react(print)
        >>> x: Store = react(1)
        >>> print(x)
        >>> # The print statement will re-run when `x` changes


    Use `react` as a decorator:

        >>> @react
        ... def f(x: int):
        ...     return x + 1
        >>> x: Store = react(1)
        >>> y = f(x)
        >>> # y will be a `Store` object, and will re-run when `x` changes
    """
    from meerkat.dataframe import DataFrame
    from meerkat.interactive.graph.store import Store

    # We setup an `if` condition that catches:
    # - primitive types (int, float, str, etc.)
    # - complex types (e.g. `list`, `dict`)
    # - instances of classes (e.g. `pd.DataFrame`), regardless of whether they
    #   are callable or not.
    #
    # The following are allowed to pass through to `_reactive`:
    # - user-defined functions (standard functions, lambdas, etc.)
    # - built-in functions, such as `len`, `sum`, etc.
    # - methods of classes (e.g. `pd.DataFrame.head`)

    if isinstance(input, DataFrame):
        # TODO: Set some property of the DataFrame to indicate that it is reactive.
        pass

    if callable(input):
        if inspect.isclass(input):
            # Allow anything that inspect.isclass() returns True for.
            # For example, given a class `Test`, function `foo`
            #   inspect.isclass(Test)           True        is a class
            #   inspect.isclass(map)            True        has class implementation
            #   inspect.isclass(range)          True        has class implementation
            #   inspect.isclass(foo)            False       is a function
            #   inspect.isclass(Test())         False       is an instance of a class
            #   inspect.isclass(print)          False       is a built-in function
            #   inspect.isclass(open)           False       is a built-in function
            #   inspect.isclass(len)            False       is a built-in function
            pass

        elif inspect.isfunction(input):
            # This is a standard function, lambda, etc.
            # We let this pass through.
            pass

        elif inspect.isbuiltin(input):
            # This is a built-in function, such as `len`, `sum`, etc.
            # We let this pass through.
            pass

        elif inspect.ismethod(input):
            # If we're here, we have a method of a class.
            # We let this pass through.
            pass

        elif hasattr(input, "__self__") and input.__self__ is not None:
            pass

        else:
            # If we're here, we (likely) have an instance of a class that has
            # a `__call__` method. Wrap this in a `Store` and return it.
            return Store(input)
    else:
        # Here, wrap whatever we get in a `Store` and return it.
        # This will include:
        # - instances of primitive types: int, float, complex, str, bytes,
        #   bool, type(None)
        # - instances of complex types: list, tuple, dict, set, frozenset, etc.
        # - instances of classes (e.g. `pd.DataFrame`) that are not callable
        return Store(input)

    return _reactive(input, nested_return, skip_fn)


# A stack that manages if reactive mode is enabled
# The stack is reveresed so that the top of the stack is
# the last index in the list.
class _ReactiveState:
    def __init__(
        self,
        *,
        reactive: bool,
        nested_return: Optional[bool],
        skip_fn: Optional[Callable],
    ) -> None:
        self.reactive = reactive
        self.kwargs = dict(nested_return=nested_return, skip_fn=skip_fn)

    def __bool__(self):
        return self.reactive

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"reactive={self.reactive}, "
            f"nested_return={self.kwargs['nested_return']}, "
            f"skip_fn={self.kwargs['skip_fn']}"
            ")"
        )


_IS_REACTIVE: List[_ReactiveState] = []


def is_reactive() -> bool:
    """Whether the code is in reactive context.

    Returns:
        bool: True if the code is in a reactive context.
    """
    # By default, we should assume we are in a reactive context.
    # This will allow functions that are decorated with `reactive` to
    # add nodes to the graph.
    if len(_IS_REACTIVE) == 0:
        return True

    # TODO: we need to check this since users are only allowed the use
    # of the `no_react` context manager. Therefore, everything is reactive
    # by default, *unless the user has explicitly disabled it*.
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


class _react:
    """Context-manager that is used control if code is an interface operation.

    Code-blocks in this context manager will create nodes
    in the operation graph, which are executed whenever their inputs
    are modified.

    A basic example that adds two numbers:

    .. code-block:: python

        a = Store(1)
        b = Store(2)
        with _react():
            c = a + b

    When either `a` or `b` is modified, the code block will re-execute
    with the new values of `a` and `b`.

    Also functions as a decorator. (Make sure to instantiate with parenthesis.):

    .. code-block:: python

            @_react()
            def add(a: int, b: int) -> int:
                return a + b

            a = Store(1)
            b = Store(2)
            c = add(a, b)

    A more complex example that concatenates two mk.DataFrame objects:

    .. code-block:: python

        @_react()
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
        skip_fn: A function that returns True if the operation should be skipped.
            This is useful for functions that are expensive to run.
            The function should take in parameters starting with "old_" and "new_" and
            ending with the name of the parameter. Instance methods cannot be used as
            skip functions.

            .. code-block:: python

                def skip_fn(old_a: int, new_a: int, old_b: int, new_b: int) -> bool:
                    # Addition is commutative, so we can skip the operation
                    # if the values of a and b are the switched.
                    return old_a == new_b and old_b == new_a

                @react(skip_fn=skip_fn)
                def add(a: int, b: int) -> int:
                    return a + b

    Returns:
        A decorated function that creates an operation node in the operation graph.
    """

    def __init__(
        self,
        reactive: bool = True,
        *,
        nested_return: bool = None,
        skip_fn: Callable = None,
    ):
        self._reactive = reactive
        self._nested_return = nested_return
        self._skip_fn = skip_fn

    def __call__(self, func):
        @wraps(func)
        def decorate_context(*args, **kwargs):
            with self.clone():
                return _reactive(
                    func, nested_return=self._nested_return, skip_fn=self._skip_fn
                )(*args, **kwargs)

        return cast(F, decorate_context)

    def __enter__(self):
        _IS_REACTIVE.append(
            _ReactiveState(
                reactive=self._reactive,
                nested_return=self._nested_return,
                skip_fn=self._skip_fn,
            )
        )
        return self

    def __exit__(self, type, value, traceback):
        _IS_REACTIVE.pop(-1)

    def clone(self):
        return self.__class__(
            reactive=self._reactive,
            nested_return=self._nested_return,
            skip_fn=self._skip_fn,
        )


class no_react(_react):
    def __init__(self, nested_return: bool = None):
        super().__init__(reactive=False, nested_return=nested_return)


def _nested_apply(obj: object, fn: Callable):
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
