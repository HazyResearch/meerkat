import warnings
from functools import partial, wraps
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union, cast

from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import ModelField
from tqdm import tqdm
from wrapt import ObjectProxy

from meerkat.dataframe import DataFrame
from meerkat.interactive.modification import (
    DataFrameModification,
    Modification,
    StoreModification,
)
from meerkat.interactive.node import Node, NodeMixin, _topological_sort
from meerkat.interactive.types import Primitive, Storeable, T
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.state import state


def _update_result(
    result: Union[list, tuple, dict, "Store", Primitive],
    update: Union[list, tuple, dict, "Store", Primitive],
    modifications: List[Modification],
) -> Union[list, tuple, dict, "Store", Primitive]:
    """Update the result object with the update object. This recursive function
    will perform a nested update to the result with the update. This function
    will also update the modifications list with the changes made to the result
    object.

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
        modifications.append(DataFrameModification(id=inode.id, scope=update.columns))

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


def trigger() -> List[Modification]:
    """Trigger the computation graph of an interface based on a list of
    modifications.

    Return:
        List[Modification]: The list of modifications that resulted from running the
            computation graph.
    """
    modifications = state.modification_queue.queue

    # build a graph rooted at the stores and refs in the modifications list
    root_nodes = [mod.node for mod in modifications if mod.node is not None]

    # Sort the nodes in topological order, and keep the Operation nodes
    order = [
        node.obj
        for node in _topological_sort(root_nodes)
        if isinstance(node.obj, Operation)
    ]
    new_modifications = []
    if len(order) > 0:
        print(f"triggered pipeline: {'->'.join([node.fn.__name__ for node in order])}")
        with tqdm(total=len(order)) as pbar:
            # Go through all the operations in order: run them and add
            # their modifications
            # to the new_modifications list
            for op in order:
                pbar.set_postfix_str(f"Running {op.fn.__name__}")
                mods = op()
                # TODO: check this
                # mods = [mod for mod in mods if not isinstance(mod, StoreModification)]
                new_modifications.extend(mods)
                pbar.update(1)
        print("done")

    # Clear out the modification queue
    state.modification_queue.clear()
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


def _wrap_outputs(obj):
    if isinstance(obj, NodeMixin):
        return obj
    return Store(obj)


def _create_nodes_for_nodeables(*nodeables: NodeMixin):
    for nodeable in nodeables:
        assert isinstance(nodeable, NodeMixin)
        # Make a node for this nodeable if it doesn't have one
        if not nodeable.has_inode():
            inode_id = None if not isinstance(nodeable, Store) else nodeable.id
            nodeable.attach_to_inode(nodeable.create_inode(inode_id=inode_id))


def _add_op_as_child(
    op: "Operation",
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


def _nested_apply(obj: object, fn: callable):
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


def reactive(
    fn: Callable = None,
    nested_return: bool = None,
) -> Callable:
    """Decorator that is used to mark a function as an interface operation.

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


def _replace_nodeables_with_nodes(obj):
    if isinstance(obj, NodeMixin):
        obj = obj.inode
    elif isinstance(obj, list) or isinstance(obj, tuple):
        obj = type(obj)(_replace_nodeables_with_nodes(x) for x in obj)
    elif isinstance(obj, dict):
        obj = {
            _replace_nodeables_with_nodes(k): _replace_nodeables_with_nodes(v)
            for k, v in obj.items()
        }
    return obj


def _replace_nodes_with_nodeables(obj):
    if isinstance(obj, Node):
        obj = obj.obj
    elif isinstance(obj, list) or isinstance(obj, tuple):
        obj = type(obj)(_replace_nodes_with_nodeables(x) for x in obj)
    elif isinstance(obj, dict):
        obj = {
            _replace_nodes_with_nodeables(k): _replace_nodes_with_nodeables(v)
            for k, v in obj.items()
        }
    return obj


# A stack that manages if reactive mode is enabled
# The stack is reveresed so that the top of the stack is
# the last index in the list.
class _ReactiveState:
    def __init__(self, *, reactive: bool, nested_return: Optional[bool]) -> None:
        self.reactive = reactive
        self.kwargs = dict(nested_return=nested_return)

    def __bool__(self):
        return self.reactive


_IS_REACTIVE: List[_ReactiveState] = []


def is_reactive():
    return len(_IS_REACTIVE) > 0 and _IS_REACTIVE[-1]


def get_reactive_kwargs() -> Dict[str, Any]:
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

    def __init__(self, reactive: bool = True, *, nested_return: bool = None):
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
    def __init__(self, nested_return: bool = None):
        super().__init__(reactive=False, nested_return=nested_return)


class StoreFrontend(BaseModel):
    store_id: str
    value: Any
    has_children: bool
    is_store: bool = True


# ObjectProxy must be the last base class
class Store(IdentifiableMixin, NodeMixin, Generic[T], ObjectProxy):

    _self_identifiable_group: str = "stores"

    def __init__(self, wrapped: T, backend_only: bool = False):
        super().__init__(wrapped=wrapped)
        # Set up these attributes so we can create the
        # schema and detail properties.
        self._self_schema = None
        self._self_detail = None
        self._self_value = None
        self._self_backend_only = backend_only

    @property
    def value(self):
        return self.__wrapped__

    def to_json(self):
        return self.__wrapped__

    @property
    def frontend(self):
        return StoreFrontend(
            store_id=self.id,
            value=self.__wrapped__,
            has_children=self.inode.has_children() if self.inode else False,
        )

    @property
    def detail(self):
        return f"Store({self.__wrapped__}) has id {self.id} and node {self.inode}"

    def set(self, new_value: T):
        """Set the value of the store."""
        if isinstance(new_value, Store):
            # if the value is a store, then we need to unpack so it can be sent to the
            # frontend
            new_value = new_value.__wrapped__

        mod = StoreModification(id=self.id, value=new_value)
        self.__wrapped__ = new_value
        mod.add_to_queue()

    def __repr__(self) -> str:
        return f"Store({self.__wrapped__})"

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.__wrapped__, name)

        # Only executing functions/methods should make the output reactifiable.
        # TODO: See if we want accessing attributes/properties to be reactifiable.
        # The reason they are not reactifiable now is that it is not clear what
        # storing the attributes as a state would give us.
        return reactive(attr) if callable(attr) else attr

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field: ModelField):
        if not isinstance(v, cls):
            if not field.sub_fields:
                # Generic parameters were not provided so we don't try to validate
                # them and just return the value as is
                return cls(v)
            else:
                # Generic parameters were provided so we try to validate them
                # and return a Store object
                v, error = field.sub_fields[0].validate(v, {}, loc="value")
                if error:
                    raise ValidationError(error)
                return cls(v)
        return v

    @reactive
    def __lt__(self, other):
        return super().__lt__(other)

    @reactive
    def __le__(self, other):
        return super().__le__(other)

    @reactive
    def __eq__(self, other):
        return super().__eq__(other)

    @reactive
    def __ne__(self, other):
        return super().__ne__(other)

    @reactive
    def __gt__(self, other):
        return super().__gt__(other)

    @reactive
    def __ge__(self, other):
        return super().__ge__(other)

    # def __hash__(self):
    #     return hash(self.__wrapped__)

    @reactive
    def __nonzero__(self):
        return super().__nonzero__()

    @reactive
    def __bool__(self):
        return super().__bool__()

    @reactive
    def __add__(self, other):
        return super().__add__(other)

    @reactive
    def __sub__(self, other):
        return super().__sub__(other)

    @reactive
    def __mul__(self, other):
        return super().__mul__(other)

    @reactive
    def __div__(self, other):
        return super().__div__(other)

    @reactive
    def __truediv__(self, other):
        return super().__truediv__(other)

    @reactive
    def __floordiv__(self, other):
        return super().__floordiv__(other)

    @reactive
    def __mod__(self, other):
        return super().__mod__(other)

    @reactive
    def __divmod__(self, other):
        return super().__divmod__(other)

    @reactive
    def __pow__(self, other, *args):
        return super().__pow__(other, *args)

    @reactive
    def __lshift__(self, other):
        return super().__lshift__(other)

    @reactive
    def __rshift__(self, other):
        return super().__rshift__(other)

    @reactive
    def __and__(self, other):
        return super().__and__(other)

    @reactive
    def __xor__(self, other):
        return super().__xor__(other)

    @reactive
    def __or__(self, other):
        return super().__or__(other)

    @reactive
    def __radd__(self, other):
        return super().__radd__(other)

    @reactive
    def __rsub__(self, other):
        return super().__rsub__(other)

    @reactive
    def __rmul__(self, other):
        return super().__rmul__(other)

    @reactive
    def __rdiv__(self, other):
        return super().__rdiv__(other)

    @reactive
    def __rtruediv__(self, other):
        return super().__rtruediv__(other)

    @reactive
    def __rfloordiv__(self, other):
        return super().__rfloordiv__(other)

    @reactive
    def __rmod__(self, other):
        return super().__rmod__(other)

    @reactive
    def __rdivmod__(self, other):
        return super().__rdivmod__(other)

    @reactive
    def __rpow__(self, other, *args):
        return super().__rpow__(other, *args)

    @reactive
    def __rlshift__(self, other):
        return super().__rlshift__(other)

    @reactive
    def __rrshift__(self, other):
        return super().__rrshift__(other)

    @reactive
    def __rand__(self, other):
        return super().__rand__(other)

    @reactive
    def __rxor__(self, other):
        return super().__rxor__(other)

    @reactive
    def __ror__(self, other):
        return super().__ror__(other)

    # We do not need to decorate i-methods because they call their
    # out-of-place counterparts, which are reactive
    def __iadd__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__iadd__ is out-of-place. Use __add__ instead."
        )
        return self.__add__(other)

    def __isub__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__isub__ is out-of-place. Use __sub__ instead."
        )
        return self.__sub__(other)

    def __imul__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__imul__ is out-of-place. Use __mul__ instead."
        )
        return self.__mul__(other)

    def __idiv__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__idiv__ is out-of-place. Use __div__ instead."
        )
        return self.__div__(other)

    def __itruediv__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__itruediv__ is out-of-place. "
            "Use __truediv__ instead."
        )
        return self.__truediv__(other)

    def __ifloordiv__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__ifloordiv__ is out-of-place. "
            "Use __floordiv__ instead."
        )
        return self.__floordiv__(other)

    def __imod__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__imod__ is out-of-place. Use __mod__ instead."
        )
        return self.__mod__(other)

    def __ipow__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__ipow__ is out-of-place. Use __pow__ instead."
        )
        return self.__pow__(other)

    def __ilshift__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__ilshift__ is out-of-place. "
            "Use __lshift__ instead."
        )
        return self.__lshift__(other)

    def __irshift__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__irshift__ is out-of-place. "
            "Use __rshift__ instead."
        )
        return self.__rshift__(other)

    def __iand__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__iand__ is out-of-place. Use __and__ instead."
        )
        return self.__and__(other)

    def __ixor__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__ixor__ is out-of-place. Use __xor__ instead."
        )
        return self.__xor__(other)

    def __ior__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__ior__ is out-of-place. Use __or__ instead."
        )
        return self.__or__(other)

    @reactive
    def __neg__(self):
        return super().__neg__()

    @reactive
    def __pos__(self):
        return super().__pos__()

    @reactive
    def __abs__(self):
        return super().__abs__()

    @reactive
    def __invert__(self):
        return super().__invert__()

    @reactive
    def __int__(self):
        return super().__int__()

    @reactive
    def __long__(self):
        return super().__long__()

    @reactive
    def __float__(self):
        return super().__float__()

    @reactive
    def __complex__(self):
        return super().__complex__()

    @reactive
    def __oct__(self):
        return super().__oct__()

    @reactive
    def __hex__(self):
        return super().__hex__()

    @reactive
    def __index__(self):
        return super().__index__()

    @reactive
    def __len__(self):
        return super().__len__()

    @reactive
    def __contains__(self, value):
        return super().__contains__(value)

    @reactive
    def __getitem__(self, key):
        return super().__getitem__(key)

    @reactive
    def __setitem__(self, key, value):
        # Make a shallow copy of the value because this operation is not in-place.
        obj = self.__wrapped__.copy()
        obj[key] = value
        warnings.warn(f"{type(self).__name__}.__setitem__ is out-of-place.")
        return type(self)(obj, backend_only=self._self_backend_only)

    @reactive
    def __delitem__(self, key):
        obj = self.__wrapped__.copy()
        del obj[key]
        warnings.warn(f"{type(self).__name__}.__delitem__ is out-of-place.")
        return type(self)(obj, backend_only=self._self_backend_only)

    @reactive
    def __getslice__(self, i, j):
        return super().__getslice__(i, j)

    @reactive
    def __setslice__(self, i, j, value):
        obj = self.__wrapped__.copy()
        obj[i:j] = value
        warnings.warn(f"{type(self).__name__}.__setslice__ is out-of-place.")
        return type(self)(obj, backend_only=self._self_backend_only)

    @reactive
    def __delslice__(self, i, j):
        obj = self.__wrapped__.copy()
        del obj[i:j]
        warnings.warn(f"{type(self).__name__}.__delslice__ is out-of-place.")
        return type(self)(obj, backend_only=self._self_backend_only)

    # def __enter__(self):
    #     return self.__wrapped__.__enter__()

    # def __exit__(self, *args, **kwargs):
    #     return self.__wrapped__.__exit__(*args, **kwargs)

    # def __iter__(self):
    #     return iter(self.__wrapped__)


def store_field(value: str) -> Field:
    """Utility for creating a pydantic field with a default factory that
    creates a Store object wrapping the given value.

    TODO (karan): Take a look at this again. I think we might be able to
    get rid of this in favor of just passing value.
    """
    return Field(default_factory=lambda: Store(value))


def make_store(value: Union[str, Storeable]) -> Store:
    """Make a Store.

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
        """Execute the operation. Unpack the arguments and keyword arguments
        and call the function. Then, update the result Reference with the
        result and return a list of modifications.

        These modifications describe the delta changes made to the
        result Reference, and are used to update the state of the GUI.
        """
        # Dereference the nodes.
        args = _replace_nodes_with_nodeables(self.args)
        kwargs = _replace_nodes_with_nodeables(self.kwargs)

        update = self.fn(*args, **kwargs)

        modifications = []
        self.result = _update_result(self.result, update, modifications=modifications)

        return modifications
