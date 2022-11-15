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
from meerkat.ops.sliceby.sliceby import SliceBy
from meerkat.tools.utils import nested_apply


class ReferenceConfig(BaseModel):
    ref_id: str
    type: str = "DataFrame"
    is_store: bool = True


class Reference(IdentifiableMixin, NodeMixin, Generic[T]):
    _self_identifiable_group: str = "refs"

    def __init__(self, obj: T):
        super().__init__()
        self._obj = obj

    @property
    def config(self):
        return ReferenceConfig(ref_id=self.id, type="DataFrame")

    @property
    def _(self):
        return self.obj

    @_.setter
    def _(self, obj: T):
        self.obj = obj

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, obj: T):
        # TODO: make it this
        # mod = ReferenceModification(id=self.id, scope=self.obj.columns)
        # mod.add_to_queue()
        self._obj = obj

    # def __getattr__(self, name):
    #     if name == "_obj":
    #         return self._obj
    #     return getattr(self.obj, name)

    # def __getitem__(self, key):
    #     return self.obj[key]

    def __repr__(self):
        return f"Reference({self.obj})"


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


# class Store(IdentifiableMixin, NodeMixin, Generic[T]):
#     identifiable_group: str = "stores"

#     def __init__(self, value: Any):
#         super().__init__()
#         self._value = value

#     @property
#     def config(self):
#         return StoreConfig(
#             store_id=self.id,
#             value=self.value,
#             has_children=self.has_children(),
#         )

#     @property
#     def _(self):
#         return self.value

#     @_.setter
#     def _(self, value):
#         self.value = value

#     @property
#     def value(self):
#         return self._value

#     @value.setter
#     def value(self, value):
#         mod = StoreModification(id=self.id, value=value)
#         self._value = value
#         mod.add_to_queue()

#     def __repr__(self) -> str:
#         return f"Store({self._})"

# @classmethod
# def __get_validators__(cls):
#     # one or more validators may be yielded which will be called in the
#     # order to validate the input, each validator will receive as an input
#     # the value returned from the previous validator
#     yield cls.validate


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


def make_ref(value: Union[any, Reference]) -> Reference:
    """
    Make a Reference.

    If value is a Reference, return it. Otherwise, return a
    new Reference that wraps value.

    Args:
        value (Union[any, Reference]): The value to wrap.

    Returns:
        Reference: The Reference wrapping value.
    """
    return value if isinstance(value, Reference) else Reference(value)


class Operation(NodeMixin):
    def __init__(
        self,
        fn: Callable,
        args: List[Reference],
        kwargs: Dict[str, Reference],
        result: Reference,
        on=None,  # TODO: add support for on
    ):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.result = result
        self.on = on

        # Add the result as a child of this Operation node
        # nested_apply(self.result, self.add_child)

    def __call__(self) -> List[Modification]:
        """
        Execute the operation. Unpack the arguments and keyword arguments
        and call the function. Then, update the result Reference with the result
        and return a list of modifications.

        These modifications describe the delta changes made to the result Reference,
        and are used to update the state of the GUI.
        """
        # unpacked_args, unpacked_kwargs, _, _ = _unpack_refs_and_stores(
        #     *self.args, **self.kwargs
        # )
        # update = self.fn(*unpacked_args, **unpacked_kwargs)

        update = self.fn(*self.args, **self.kwargs)

        modifications = []
        self.result = _update_result(self.result, update, modifications=modifications)

        return modifications


def _update_result(
    result: Union[list, tuple, dict, Reference, Store, Primitive],
    update: Union[list, tuple, dict, Reference, Store, Primitive],
    modifications: List[Modification],
) -> Union[list, tuple, dict, Reference, Store, Primitive]:
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
    # elif isinstance(result, Reference):
    #     # If the result is a Reference, then we need to update the Reference's object
    #     # and return a ReferenceModification
    #     result.obj = update
    #     if isinstance(result.obj, DataFrame):
    #         modifications.append(
    #             ReferenceModification(id=result.id, scope=result.obj.columns)
    #         )
    #     return result
    elif isinstance(result, DataFrame):
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

    # order = [
    #     node for node in _topological_sort(root_nodes) if isinstance(node, Operation)
    # ]
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


def _unpack_refs_and_stores(*args, **kwargs):
    # TODO(Sabri): this should be nested
    refs = []
    stores = []
    unpacked_args = []
    for arg in args:
        if isinstance(arg, Reference):
            refs.append(arg)
            unpacked_args.append(arg.obj)
        elif isinstance(arg, Store):
            stores.append(arg)
            # unpacked_args.append(arg.value)
            unpacked_args.append(arg)
        elif isinstance(arg, list) or isinstance(arg, tuple):
            unpacked_args_i, _, refs_i, stores_i = _unpack_refs_and_stores(*arg)
            unpacked_args.append(unpacked_args_i)
            refs.extend(refs_i)
            stores.extend(stores_i)
        elif isinstance(arg, dict):
            _, unpacked_kwargs_i, refs_i, stores_i = _unpack_refs_and_stores(**arg)
            unpacked_args.append(unpacked_kwargs_i)
            refs.extend(refs_i)
            stores.extend(stores_i)
        else:
            unpacked_args.append(arg)

    unpacked_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, Reference):
            refs.append(v)
            unpacked_kwargs[k] = v.obj
        elif isinstance(v, Store):
            stores.append(v)
            # unpacked_kwargs[k] = v.value
            unpacked_kwargs[k] = v
        elif isinstance(v, list) or isinstance(v, tuple):
            unpacked_args_i, _, refs_i, stores_i = _unpack_refs_and_stores(*v)
            unpacked_kwargs[k] = unpacked_args_i
            refs.extend(refs_i)
            stores.extend(stores_i)
        elif isinstance(v, dict):
            _, unpacked_kwargs_i, refs_i, stores_i = _unpack_refs_and_stores(**v)
            unpacked_kwargs[k] = unpacked_kwargs_i
            refs.extend(refs_i)
            stores.extend(stores_i)
        else:
            unpacked_kwargs[k] = v

    return unpacked_args, unpacked_kwargs, refs, stores


def _has_ref_or_store(arg):
    if isinstance(arg, Reference) or isinstance(arg, Store):
        return True
    elif isinstance(arg, list) or isinstance(arg, tuple):
        return any([_has_ref_or_store(a) for a in arg])
    elif isinstance(arg, dict):
        return any([_has_ref_or_store(a) for a in arg.values()])
    else:
        return False


def _nested_apply(obj: object, fn: callable, return_type: type = None):
    if return_type is Store or return_type is Reference:
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


def _pack_refs_and_stores(obj, return_type: type = None):
    if return_type is Store:
        return Store(obj)
    elif return_type is Reference:
        return Reference(obj)

    if isinstance(obj, (DataFrame, SliceBy)):
        return Reference(obj)

    # TODO(Sabri): we should think more deeply about how to handle nested outputs
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return Store(obj)
    return obj


def _wrap_outputs(obj, return_type: type = None):
    if isinstance(obj, NodeMixin):
        return obj
    return Store(obj)


# def _add_op_as_child(
#     op: Operation,
#     *refs_and_stores: Union[Reference, Store],
#     triggers: bool = True,
# ):
#     """
#     Add the operation as a child of the refs and stores.

#     Args:
#         op: The operation to add as a child.
#         refs_and_stores: The refs and stores to add the operation as a child
#             of.
#         triggers: Whether the operation is triggered by changes in the refs
#             and stores.
#     """
#     for ref_or_store in refs_and_stores:
#         if isinstance(ref_or_store, (Reference, Store)):
#             ref_or_store.add_child(op, triggers=triggers)


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

        # Make a node for the operation if it doesn't have one
        if not op.has_inode():
            op.attach_to_inode(op.create_inode())

        # Add the operation as a child of the nodeable
        nodeable.inode.add_child(op.inode, triggers=triggers)


def interface_op(
    fn: Callable = None,
    nested_return: bool = True,
    return_type: type = None,
    first_call: Any = None,
    on: Union[Reference, Store, str, List[Union[Reference, Store, str]]] = None,
    also_on: Union[Reference, Store, List[Union[Reference, Store]]] = None,
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
        on: A Reference or Store, or a list of References or Stores. When these are
            modified, the function will be called. *This will prevent the function from
            being triggered when its inputs are modified.*

            Also accepts strings in addition to References and Stores. If a string is
            passed, then the function argument with the same name will be used as the
            Reference or Store. For example, if the function has an argument `df`,
            then you can pass in `on="df"` to trigger the function when `df` is
            modified.
        also_on: A Reference or Store, or a list of References or Stores. When these are
            modified, the function will be called. *The function will continue to be
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
            operation graph and returns a `Reference` object that wraps the
            output of the function.

            Subsequent calls to the function will be handled by the graph.
            """
            # TODO(karan): have to make `on` and `also_on` nonlocal otherwise
            # it throws an UnboundLocalError. But why doesn't this happen for
            # `first_call`?!
            nonlocal on, also_on

            # TODO(Sabri): this should be nested
            nodeables = _get_nodeables(*args, **kwargs)
            print("Nodeables", nodeables)
            # unpacked_args, unpacked_kwargs, refs, stores = _unpack_refs_and_stores(
            #     *args, **kwargs
            # )

            # if first_call is not None:
            #     # For expensive functions, the user can specify a first call value
            #     # that allows us to setup the Operation without running the function
            #     if isinstance(first_call, Callable):
            #         result = first_call(*unpacked_args, **unpacked_kwargs)
            #     else:
            #         result = first_call
            # else:
            #     # By default, run the function to produce a result
            #     # Call the function on the unpacked args and kwargs
            #     result = fn(*unpacked_args, **unpacked_kwargs)

            # Call the function on the args and kwargs
            result = fn(*args, **kwargs)

            # Setup an Operation node if any of the
            # args or kwargs were refs or stores
            # if (
            #     (len(refs) > 0 or len(stores) > 0)
            #     or on is not None
            #     or also_on is not None
            # ):
            if (len(nodeables) > 0) or on is not None or also_on is not None:

                # FIXME: the result should be possible to put as nodes in the graph
                # and if they're not, wrap them in Store and make them nodes
                # So classes that are Nodeable

                # The result should be placed inside a Store or Reference
                # (or a nested object) containing Stores and References.
                # Then we can update the contents of this result when the
                # function is called again.
                if nested_return:
                    derived = _nested_apply(result, fn=_wrap_outputs)
                # elif isinstance(result, (DataFrame, SliceBy)):
                # TODO: this needs to be removed
                # Instead, we should directly return the result
                # derived = Reference(result)
                # pass
                elif isinstance(result, NodeMixin):
                    derived = result
                else:
                    derived = Store(result)

                # Create the Operation node
                op = Operation(fn=fn, args=args, kwargs=kwargs, result=derived)

                if on is None:
                    # Add this Operation node as a child of all of the refs and stores
                    # regardless of the value of `also_on`
                    # TODO: refs should be all the DataFrame / SliceBy / Column objects
                    # that are now "References"
                    # _add_op_as_child(op, *refs, *stores, triggers=True)
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

                    nested_apply(derived, _foo)
                else:
                    # Add this Operation node as a child of all of the refs and stores
                    # that are passed into the function. However, these children
                    # are non-triggering, meaning that the Operation node will
                    # only be called when the refs and stores in `on` are modified,
                    # and not otherwise.
                    # _add_op_as_child(op, *refs, *stores, triggers=False)

                    # Add this Operation node as a child of the refs and stores
                    # passed into `on`. These will trigger the Operation!
                    # TODO(Sabri): this should be nested
                    if isinstance(on, (Reference, Store)):
                        on = [on]
                    _, _, refs, stores = _unpack_refs_and_stores(*on)
                    _add_op_as_child(op, *refs, *stores, triggers=True)

                    # Find all the str elements in `on`
                    # These are the names of fn arguments that were passed into `on`
                    # We can first analyze the fn signature to figure out which
                    # argument names are bound to what values
                    # (e.g. Reference, Store, etc.)

                    # Analyze the fn signature to figure out which argument names
                    # are bound
                    fn_signature = inspect.signature(fn)
                    if on:
                        assert all(
                            [
                                e in fn_signature.parameters
                                for e in on
                                if isinstance(e, str)
                            ]
                        ), "All strings passed into `on` must be arguments of the \
                            decorated function."
                    fn_bound_arguments = fn_signature.bind(*args, **kwargs).arguments
                    # Now we pull out the values of the arguments that were passed
                    # into `on` and unpack them to get the refs and stores
                    _, _, refs, stores = _unpack_refs_and_stores(
                        *[fn_bound_arguments[e] for e in on if isinstance(e, str)]
                    )
                    # ...and add this Operation node as a child of these
                    # refs and stores
                    _add_op_as_child(op, *refs, *stores, triggers=True)

                if also_on is not None:
                    # Add this Operation node as a child of the refs and stores
                    # passed into `also_on`
                    # TODO(Sabri): this should be nested
                    if isinstance(also_on, (Reference, Store)):
                        also_on = [also_on]
                    _, _, refs, stores = _unpack_refs_and_stores(*also_on)
                    _add_op_as_child(op, *refs, *stores, triggers=True)

                return derived

            return result

        return wrapper

    return _interface_op(fn)
