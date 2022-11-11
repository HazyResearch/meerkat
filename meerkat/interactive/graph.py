import inspect
from abc import ABC
from collections import defaultdict
from functools import partial, wraps
from typing import Any, Callable, Dict, Generic, List, TypeVar, Union

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

    This mixin is used in Reference, Store and Operation to make
    them part of a computation graph.
    """

    def __init__(self):
        # The children of this node: this is a dictionary
        # mapping children to a boolean indicating whether
        # the child is triggered when this node is triggered.
        self.children: Dict[Operation, bool] = dict()

    def add_child(self, child, triggers=True):
        """Adds a child to this node.

        Args:
            child: The child to add.
            triggers: If True, this child is triggered
                when this node is triggered.
        """
        if child not in self.children:
            self.children[child] = triggers

        # Don't overwrite triggers=True with triggers=False
        self.children[child] = triggers | self.children[child]

    @property
    def trigger_children(self):
        """Returns the children that are triggered."""
        return [child for child, triggers in self.children.items() if triggers]

    def __hash__(self):
        """Hash is based on the id of the node."""
        return hash(id(self))

    def __eq__(self, other):
        """Two nodes are equal if they have the same id."""
        return id(self) == id(other)

    def has_children(self):
        """Returns True if this node has children."""
        return len(self.children) > 0

    def has_trigger_children(self):
        """Returns True if this node has children that are triggered."""
        return any(self.children.values())


def _topological_sort(root_nodes: List[NodeMixin]) -> List[NodeMixin]:
    """
    Perform a topological sort on a graph.

    Args:
        root_nodes (List[NodeMixin]): The root nodes of the graph.

    Returns:
        List[NodeMixin]: The topologically sorted nodes.
    """
    # get a mapping from node to the children of each node
    # only get the children that are triggered by the node
    # i.e. ignore children that use the node as a dependency
    # but are not triggered by the node
    parents = defaultdict(set)
    nodes = set()
    while root_nodes:
        node = root_nodes.pop(0)
        for child in node.trigger_children:
            parents[child].add(node)
            nodes.add(node)
            root_nodes.append(child)

    current = [
        node for node in nodes if not parents[node]
    ]  # get a set of all the nodes without an incoming edge

    while current:
        node: NodeMixin = current.pop(0)
        yield node

        for child in node.trigger_children:
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


class ReferenceConfig(BaseModel):
    ref_id: str
    type: str = "DataFrame"
    is_store: bool = True


T = TypeVar("T", "DataFrame", "SliceBy")


class Reference(IdentifiableMixin, NodeMixin, Generic[T]):
    identifiable_group: str = "refs"

    def __init__(self, obj):
        super().__init__()
        self._obj = obj

    @property
    def config(self):
        return ReferenceConfig(ref_id=self.id, type="DataFrame")

    @property
    def _(self):
        return self.obj

    @_.setter
    def _(self, obj: object):
        self.obj = obj

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, obj: object):
        # TODO: make it this
        # mod = ReferenceModification(id=self.id, scope=self.obj.columns)
        # mod.add_to_queue()
        self._obj = obj

    def __getattr__(self, name):
        return getattr(self.obj, name)

    def __getitem__(self, key):
        return self.obj[key]

    def __repr__(self):
        return f"Reference({self.obj})"


class StoreConfig(BaseModel):
    store_id: str
    value: Any
    has_children: bool
    is_store: bool = True


class EndpointConfig(BaseModel):
    endpoint_id: Union[str, None]


class Store(IdentifiableMixin, NodeMixin, Generic[T]):
    identifiable_group: str = "stores"

    def __init__(self, value: Any):
        super().__init__()
        self._value = value

    @property
    def config(self):
        return StoreConfig(
            store_id=self.id,
            value=self.value,
            has_children=self.has_children(),
        )

    @property
    def _(self):
        return self.value

    @_.setter
    def _(self, value):
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        mod = StoreModification(id=self.id, value=value)
        self._value = value
        mod.add_to_queue()

    def __repr__(self) -> str:
        return f"Store({self._})"


class Endpoint(IdentifiableMixin, NodeMixin, Generic[T]):
    identifiable_group: str = "endpoints"

    def __init__(self, fn: Callable = None):
        super().__init__()
        if fn is None:
            self.id = None
        self.fn = fn

    @property
    def config(self):
        return EndpointConfig(
            endpoint_id=self.id,
        )

    def __call__(self, *args, **kwargs):
        # Any Stores or References that are passed in as arguments
        # should have this Endpoint as a non-triggering child
        for arg in args:
            if isinstance(arg, (Store, Reference)):
                arg.add_child(self, triggers=False)

        for kwarg in kwargs.values():
            if isinstance(kwarg, (Store, Reference)):
                kwarg.add_child(self, triggers=False)

        return partial(self.fn, *args, **kwargs)

    @staticmethod
    def get(id: str) -> Store:
        """Get an Endpoint using its id."""
        from meerkat.state import state

        return state.identifiables.get(group="endpoints", id=id)


def make_endpoint(endpoint_or_fn: Union[Callable, Endpoint, None]) -> Endpoint:
    """Make an Endpoint."""
    return (
        endpoint_or_fn
        if isinstance(endpoint_or_fn, Endpoint)
        else Endpoint(endpoint_or_fn)
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


class Modification(BaseModel, ABC):
    """
    Base class for modifications.

    Modifications are used to track changes to Reference and Store nodes
    in the graph.

    Attributes:
        id (str): The id of the Reference or Store.
    """

    id: str

    @property
    def node(self):
        """The Reference or Store node that this modification is for."""
        raise NotImplementedError()

    def add_to_queue(self):
        """Add this modification to the queue."""
        # Get the queue
        from meerkat.state import state

        state.modification_queue.add(self)


class ReferenceModification(Modification):
    scope: List[str]
    type: str = "ref"

    @property
    def node(self) -> Reference:
        from meerkat.state import state

        return state.identifiables.get(group="refs", id=self.id)


class StoreModification(Modification):
    value: Any  # : Storeable # TODO(karan): Storeable prevents
    # us from storing objects in the store
    type: str = "store"

    @property
    def node(self) -> Store:
        from meerkat.state import state

        return state.identifiables.get(group="stores", id=self.id)


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
        nested_apply(self.result, self.add_child)

    def __call__(self) -> List[Modification]:
        """
        Execute the operation. Unpack the arguments and keyword arguments
        and call the function. Then, update the result Reference with the result
        and return a list of modifications.

        These modifications describe the delta changes made to the result Reference,
        and are used to update the state of the GUI.
        """
        unpacked_args, unpacked_kwargs, _, _ = _unpack_refs_and_stores(
            *self.args, **self.kwargs
        )
        update = self.fn(*unpacked_args, **unpacked_kwargs)

        modifications = []
        _update_result(self.result, update, modifications=modifications)

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
    elif isinstance(result, Reference):
        # If the result is a Reference, then we need to update the Reference's object
        # and return a ReferenceModification
        result.obj = update
        if isinstance(result.obj, DataFrame):
            modifications.append(
                ReferenceModification(id=result.id, scope=result.obj.columns)
            )
        return result
    elif isinstance(result, Store):
        # If the result is a Store, then we need to update the Store's value
        # and return a StoreModification
        # TODO(karan): now checking if the value is the same
        # This is assuming that all values put into Stores have an __eq__ method
        # defined that can be used to check if the value has changed.
        # print(result.value, update)
        if isinstance(result.value, (str, int, float, bool, type(None), tuple)):
            # We can just check if the value is the same
            if result.value != update:
                result.value = update
                modifications.append(StoreModification(id=result.id, value=update))
        else:
            # We can't just check if the value is the same if the Store contains
            # a list, dict or object, since they are mutable (and it would just
            # return True).
            result.value = update
            modifications.append(StoreModification(id=result.id, value=update))
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
            mods = [mod for mod in mods if not isinstance(mod, StoreModification)]
            new_modifications.extend(mods)
            pbar.update(1)
    return modifications + new_modifications


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
            unpacked_args.append(arg.value)
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
            unpacked_kwargs[k] = v.value
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


def _add_op_as_child(
    op: Operation,
    *refs_and_stores: Union[Reference, Store],
    triggers: bool = True,
):
    """
    Add the operation as a child of the refs and stores.

    Args:
        op: The operation to add as a child.
        refs_and_stores: The refs and stores to add the operation as a child
            of.
        triggers: Whether the operation is triggered by changes in the refs
            and stores.
    """
    for ref_or_store in refs_and_stores:
        if isinstance(ref_or_store, (Reference, Store)):
            ref_or_store.add_child(op, triggers=triggers)


def endpoint(fn: Callable = None):
    """
    Decorator to mark a function as an endpoint.

    An endpoint is a function that can be called to
        - update the value of a Store (e.g. incrementing a counter)
        - update an object referenced by a Reference (e.g. editing the
            contents of a DataFrame)
        - run a computation and return its result to the frontend
        - run a function in response to a frontend event (e.g. button
            click)

    Endpoints differ from operations in that they are not automatically
    triggered by changes in their inputs. Instead, they are triggered by
    explicit calls to the endpoint function.

    The Store and Reference objects that are modified inside the endpoint
    function will automatically trigger operations in the graph that
    depend on them.

    Warning: Due to this, we do not recommend running endpoints manually
    in your Python code. This can lead to unexpected behavior e.g.
    running an endpoint inside an operation may change a Store or
    Reference  that causes the operation to be triggered repeatedly,
    leading to an infinite loop.

    Almost all use cases can be handled by using the frontend to trigger
    endpoints.

    .. code-block:: python

        @endpoint
        def increment(count: Store, step: int = 1):
            count._ += step
            # ^ update the count Store, which will trigger operations
            #   that depend on it

            # return the updated value to the frontend
            return count._

        # Now you can create a button that calls the increment endpoint
        counter = Store(0)
        button = Button(on_click=increment(counter))
        # ^ read this as: call the increment endpoint with the counter
        # Store when the button is clicked

    Args:
        fn: The function to decorate.

    Returns:
        The decorated function, as an Endpoint object.
    """

    def _endpoint(fn: Callable):

        stores = set()
        references = set()
        for name, annot in inspect.getfullargspec(fn).annotations.items():
            if isinstance(annot, type) and issubclass(annot, Store):
                stores.add(name)
            if isinstance(annot, type) and issubclass(annot, Reference):
                references.add(name)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            """
            This `wrapper` function is only run once. It creates a node in the
            operation graph and returns a `Reference` object that wraps the
            output of the function.

            Subsequent calls to the function will be handled by the graph.
            """
            # Keep the arguments that were not annotated to be stores or
            # references
            fn_signature = inspect.signature(fn)
            fn_bound_arguments = fn_signature.bind(*args, **kwargs).arguments

            # Unpack the args and kwargs to get the refs and stores
            fn_args_to_unpack = {
                k: v
                for k, v in fn_bound_arguments.items()
                if k not in stores and k not in references
            }
            args, kwargs, _, _ = _unpack_refs_and_stores(**fn_args_to_unpack)

            # Don't unpack the refs and stores that were type hinted
            fn_args_as_is = {
                k: v
                for k, v in fn_bound_arguments.items()
                if k in stores or k in references
            }
            assert all(
                [isinstance(v, (Store, Reference)) for v in fn_args_as_is.values()]
            ), "All arguments that are type hinted as stores or references \
                must be Store or Reference objects."

            # Run the function
            result = fn(*args, **{**kwargs, **fn_args_as_is})

            # Update the refs and stores and return modifications
            # Get the modifications from the queue
            from meerkat.state import state

            modifications = state.modification_queue.queue

            # Trigger the modifications
            modifications = trigger(modifications)
            return result

        # Register the endpoint and return it
        return Endpoint(wrapper)

    return _endpoint(fn)


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
            unpacked_args, unpacked_kwargs, refs, stores = _unpack_refs_and_stores(
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
            # args or kwargs were refs or stores
            if (
                (len(refs) > 0 or len(stores) > 0)
                or on is not None
                or also_on is not None
            ):

                # The result should be placed inside a Store or Reference
                # (or a nested object) containing Stores and References.
                # Then we can update the contents of this result when the
                # function is called again.
                if nested_return:
                    derived = _nested_apply(
                        result, fn=_pack_refs_and_stores, return_type=return_type
                    )
                elif isinstance(result, (DataFrame, SliceBy)):
                    derived = Reference(result)
                else:
                    derived = Store(result)

                # Create the Operation node
                op = Operation(fn=fn, args=args, kwargs=kwargs, result=derived)

                if on is None:
                    # Add this Operation node as a child of all of the refs and stores
                    # regardless of the value of `also_on`
                    _add_op_as_child(op, *refs, *stores, triggers=True)
                else:
                    # Add this Operation node as a child of all of the refs and stores
                    # that are passed into the function. However, these children
                    # are non-triggering, meaning that the Operation node will
                    # only be called when the refs and stores in `on` are modified,
                    # and not otherwise.
                    _add_op_as_child(op, *refs, *stores, triggers=False)

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


@interface_op
def head(df: "DataFrame", n: int = 5):
    new_df = df.head(n)
    import numpy as np

    new_df["head_column"] = np.zeros(len(new_df))
    return new_df
