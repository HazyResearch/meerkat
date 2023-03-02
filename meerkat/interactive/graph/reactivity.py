import types
from functools import partial, wraps
from typing import Callable, Iterator

from meerkat.interactive.graph.marking import is_unmarked_context, unmarked
from meerkat.interactive.graph.operation import (
    Operation,
    _check_fn_has_leading_self_arg,
)
from meerkat.interactive.graph.utils import (
    _get_nodeables,
    _replace_nodeables_with_nodes,
)
from meerkat.interactive.node import NodeMixin
from meerkat.mixins.reactifiable import MarkableMixin

__all__ = ["reactive", "reactive", "is_unmarked_context"]

_REACTIVE_FN = "reactive"


def isclassmethod(method):
    """
    StackOverflow: https://stackoverflow.com/a/19228282
    """
    bound_to = getattr(method, "__self__", None)
    if not isinstance(bound_to, type):
        # must be bound to a class
        return False
    name = method.__name__
    for cls in bound_to.__mro__:
        descriptor = vars(cls).get(name)
        if descriptor is not None:
            return isinstance(descriptor, classmethod)
    return False


def reactive(
    fn: Callable = None,
    nested_return: bool = False,
    skip_fn: Callable[..., bool] = None,
) -> Callable:
    """Internal decorator that is used to mark a function as reactive.
    This is only meant for internal use, and users should use the
    :func:`react` decorator instead.

    Functions decorated with this will create nodes in the operation graph,
    which are executed whenever their inputs are modified.

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
        fn: See :func:`react`.
        nested_return: See :func:`react`.
        skip_fn: See :func:`react`.

    Returns:
        See :func:`react`.
    """
    # TODO: Remove nested_return argument. With the addition of __iter__ and __next__
    # to mk.Store, we no longer need to support nested return values.
    # This will require looking through current use of reactive and patching them.
    if fn is None:
        # need to make passing args to the args optional
        # note: all of the args passed to the decorator MUST be optional
        return partial(reactive, nested_return=nested_return, skip_fn=skip_fn)

    # Built-in functions cannot be wrapped in reactive.
    # They have to be converted to a lambda function first and then run.
    if isinstance(fn, types.BuiltinFunctionType):
        raise ValueError(
            "Cannot wrap built-in function in reactive. "
            "Please convert to lambda function first:\n"
            "    >>> reactive(lambda x: {}(x))".format(fn.__name__)
        )

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
                _IteratorStore,
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

            _is_unmarked_context = is_unmarked_context()

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
            unpacked_args, _ = _unpack_stores_from_object(list(args))
            unpacked_kwargs, _ = _unpack_stores_from_object(kwargs)

            _force_no_react = False
            if hasattr(fn, "__self__") and fn.__self__ is not None:
                if isclassmethod(fn):
                    # If the function is a classmethod, then it will always be
                    # bound to the class when we grab it later in this block,
                    # and we don't need to unpack the first argument.
                    args = args
                else:
                    args = (fn.__self__, *args)

                # Unpack the stores from the args and kwargs because
                # args has changed!
                # TODO: make this all nicer
                unpacked_args, _ = _unpack_stores_from_object(list(args))
                unpacked_kwargs, _ = _unpack_stores_from_object(kwargs)

                # The method bound to the class.
                try:
                    fn_class = getattr(fn.__self__.__class__, fn.__name__)
                except AttributeError:
                    fn_class = getattr(fn.__self__.mro()[0], fn.__name__)
                fn = _fn_outer_wrapper(fn_class)

                # If `fn` is an instance method, then the first argument in `args`
                # is the instance. We should **not** unpack the `self` argument
                # if it is a Store.
                if args and isinstance(args[0], Store):
                    unpacked_args[0] = args[0]
            elif _check_fn_has_leading_self_arg(fn):
                # If the object is a MarkableMixin and fn has a leading self arg,
                # (i.e. fn(self, ...)), then we need to check if the function
                # should be added to the graph.
                # If the object is a MarkableMixin, the fn will be added
                # to the graph only when the object is marked (i.e. `obj.marked`).
                # This is required for magic methods for MarkableMixin instances
                # because shorthand accessors (e.g. x[0] for x.__getitem__(0)) do not
                # use the __getattribute__ method.
                # TODO: When the function is an instance method, should
                #       instance.marked determine if the function is reactive?
                # obj = args[0]
                # if isinstance(obj, MarkableMixin):
                #     with unmarked():
                #         is_obj_reactive = obj.marked
                #     _force_no_react = not is_obj_reactive

                # If `fn` is an instance method, then the first argument in `args`
                # is the instance. We should **not** unpack the `self` argument
                # if it is a Store.
                if isinstance(args[0], Store):
                    unpacked_args[0] = args[0]

            # We need to check the arguments to see if they are reactive.
            # If any of the inputs into fn are reactive, we need to add fn
            # to the graph.
            with unmarked():
                any_inputs_marked = _any_inputs_marked(*args, **kwargs)

            # Call the function on the args and kwargs
            with unmarked():
                result = fn(*unpacked_args, **unpacked_kwargs)

            # TODO: Check if result is equal to one of the inputs.
            # If it is, we need to copy it.

            if _is_unmarked_context or _force_no_react or not any_inputs_marked:
                # If we are in an unmarked context, then we don't need to create
                # any nodes in the graph.
                # `fn` should be run as normal.
                return result

            # Now we're in a reactive context i.e. is_reactive() == True

            # Get all the NodeMixin objects from the args and kwargs.
            # These objects will be parents of the Operation node
            # that is created for this function.
            nodeables = _get_nodeables(*args, **kwargs)

            # Wrap the Result in NodeMixin objects
            if nested_return:
                result = _nested_apply(result, fn=_wrap_outputs)
            elif isinstance(result, NodeMixin):
                result = result
            elif isinstance(result, Iterator):
                result = _IteratorStore(result)
            else:
                result = Store(result)

            # If the object is a ReactifiableMixin, we should turn
            # reactivity on.
            # TODO: This should be done in a nested way.
            if isinstance(result, MarkableMixin):
                result._self_marked = True

            with unmarked():
                # Setup an Operation node if any of the args or kwargs
                # were nodeables
                op = None

                # Create Nodes for each NodeMixin object
                _create_nodes_for_nodeables(*nodeables)
                args = _replace_nodeables_with_nodes(args)
                kwargs = _replace_nodeables_with_nodes(kwargs)

                # Create the Operation node
                op = Operation(
                    fn=fn,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    skip_fn=skip_fn,
                )

                # For normal functions
                # Make a node for the operation if it doesn't have one
                if not op.has_inode():
                    op.attach_to_inode(op.create_inode())

                # Add this Operation node as a child of all of the nodeables.
                # This function takes care of only adding it as a child for
                # nodeables that are marked.
                _add_op_as_child(op, *nodeables)

                # Attach the Operation node to its children (if it is not None)
                def _foo(nodeable: NodeMixin):
                    # FIXME: make sure they are not returning a nodeable that
                    # is already in the dag. May be related to checking that the graph
                    # is acyclic.
                    if not nodeable.has_inode():
                        inode_id = (
                            None if not isinstance(nodeable, Store) else nodeable.id
                        )
                        nodeable.attach_to_inode(
                            nodeable.create_inode(inode_id=inode_id)
                        )

                    if op is not None:
                        op.inode.add_child(nodeable.inode)

                _nested_apply(result, _foo)

            return result

        setattr(wrapper, "__wrapper__", _REACTIVE_FN)
        return wrapper

    return __reactive(fn)


def is_reactive_fn(fn: Callable) -> bool:
    """Check if a function is wrapped by the `reactive` decorator."""
    return (
        hasattr(fn, "__wrapped__")
        and hasattr(fn, "__wrapper__")
        and fn.__wrapper__ == _REACTIVE_FN
    )


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


def _add_op_as_child(op: Operation, *nodeables: NodeMixin):
    """Add the operation as a child of the nodeables.

    Args:
        op: The operation to add as a child.
        nodeables: The nodeables to add the operation as a child.
    """
    for nodeable in nodeables:
        # Add the operation as a child of the nodeable
        triggers = nodeable.marked if isinstance(nodeable, MarkableMixin) else True
        nodeable.inode.add_child(op.inode, triggers=triggers)


def _wrap_outputs(obj):
    from meerkat.interactive.graph.store import Store, _IteratorStore

    if isinstance(obj, NodeMixin):
        return obj
    elif isinstance(obj, Iterator):
        return _IteratorStore(obj)
    return Store(obj)


def _create_nodes_for_nodeables(*nodeables: NodeMixin):
    from meerkat.interactive.graph.store import Store

    for nodeable in nodeables:
        assert isinstance(nodeable, NodeMixin)
        # Make a node for this nodeable if it doesn't have one
        if not nodeable.has_inode():
            inode_id = None if not isinstance(nodeable, Store) else nodeable.id
            nodeable.attach_to_inode(nodeable.create_inode(inode_id=inode_id))


def _any_inputs_marked(*args, **kwargs) -> bool:
    """Returns True if any of the inputs are reactive.

    Note: This function does not recursively check the arguments for
    reactive inputs.
    """

    def _is_marked(obj):
        return isinstance(obj, MarkableMixin) and obj.marked

    return any(_is_marked(arg) for arg in args) or any(
        _is_marked(arg) for arg in kwargs.values()
    )
