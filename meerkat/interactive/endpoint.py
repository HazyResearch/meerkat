import inspect
from functools import partial, wraps
from typing import Any, Callable, Generic, Union

from fastapi import APIRouter, Body
from pydantic import BaseModel, create_model

from meerkat.interactive.graph import Reference, Store, _unpack_refs_and_stores, trigger
from meerkat.interactive.node import NodeMixin
from meerkat.interactive.types import T
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.state import state


class SingletonRouter(type):
    """
    A metaclass that ensures that only one instance of a router is created
    *for a given prefix*.

    A prefix is a string that is used to identify a router. For example,
    the prefix for the router that handles endpoints is "/endpoint". We
    want to ensure that only one router is created for each prefix.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        prefix = kwargs["prefix"]
        # Look up if this (cls, prefix) pair has been created before
        if (cls, prefix) not in cls._instances:
            # If not, we let a new instance be created
            cls._instances[(cls, prefix)] = super(SingletonRouter, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[(cls, prefix)]


class SimpleRouter(IdentifiableMixin, APIRouter, metaclass=SingletonRouter):
    """
    A very simple FastAPI router.

    Only one instance of this router will be created *for a given prefix*, so
    you can call this router multiple times in your code and it will always
    return the same instance.

    This router allows you to pass in arbitrary keyword arguments that are
    passed to the FastAPI router, and sets sensible defaults for the
    prefix, tags, and responses.

    Attributes:
        prefix (str): The prefix for this router.
        **kwargs: Arbitrary keyword arguments that are passed to the FastAPI
            router.

    Example:
        >>> from meerkat.interactive.api.routers import SimpleRouter
        >>> router = SimpleRouter(prefix="/endpoint")
        >>> router = SimpleRouter(prefix="/endpoint")
        >>> router is SimpleRouter(prefix="/endpoint")
        True
    """

    identifiable_group: str = "routers"

    def __init__(self, prefix: str, **kwargs):
        super().__init__(
            prefix=prefix,
            tags=[prefix.strip("/").replace("/", "-")],
            responses={404: {"description": "Not found"}},
            id=prefix,
            **kwargs,
        )


class EndpointConfig(BaseModel):
    endpoint_id: Union[str, None]


class Endpoint(IdentifiableMixin, NodeMixin, Generic[T]):
    """
    Create an endpoint from a function in Meerkat.

    Typically, you will not need to call this class directly, but
    instead use the `endpoint` decorator.

    Attributes:
        fn (Callable): The function to create an endpoint from.
        prefix (str): The prefix for this endpoint.
        route (str): The route for this endpoint.

    Note:
    All endpoints can be hit with a POST request at
    /{endpoint_id}/dispatch/
    The request needs a JSON body with the following keys:
        - kwargs: a dictionary of keyword arguments to be
            passed to the endpoint function `fn`
        - payload: additional payload, if any

    Optionally, the user can customize how endpoints are
    organized by specifying a prefix and a route. The prefix
    is a string that is used to identify a router. For example,
    the prefix for the router that handles endpoints is "/endpoint".
    The route is a string that is used to identify an endpoint
    within a router. For example, the route for the endpoint
    that handles the `get` function could be "/get".

    If only a prefix is specified, then the route will be the
    name of the function e.g. "my_endpoint". If both a prefix
    and a route are specified, then the route will be the
    specified route e.g. "/specific/route/".

    Refer to the FastAPI documentation for more information
    on how to create routers and endpoints.
    """

    EmbeddedBody = partial(Body, embed=True)

    identifiable_group: str = "endpoints"

    def __init__(
        self,
        fn: Callable = None,
        prefix: Union[str, APIRouter] = None,
        route: str = None,
    ):
        super().__init__()
        if fn is None:
            self.id = None
        self.fn = fn

        if prefix is None:
            # No prefix, no router
            self.router = None
        else:
            # Make the router
            if isinstance(prefix, APIRouter):
                self.router = prefix
            else:
                self.router = SimpleRouter(prefix=prefix)

        self.prefix = prefix
        self.route = route

    @property
    def config(self):
        return EndpointConfig(
            endpoint_id=self.id,
        )

    def run(self, *args, **kwargs) -> Any:
        """
        Actually run the endpoint function `fn`.

        Args:
            *args: Positional arguments to pass to `fn`.
            **kwargs: Keyword arguments to pass to `fn`.

        Returns:
            The return value of `fn`.
        """
        # Check if self.fn has any arguments left to be filled
        signature = inspect.signature(self.fn)
        bound_args = signature.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        # If there are no arguments left to be filled, then we can
        # call the function
        if len(bound_args.arguments) == len(signature.parameters) == 0:
            return self.fn()

        # Raise an error if there are still arguments left to be filled
        raise ValueError(
            f"Endpoint {self.id} still has arguments left to be \
            filled: {bound_args.arguments}. Ensure that all arguments \
                are passed in when calling `.run()` on this endpoint."
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

        return Endpoint(
            fn=partial(self.fn, *args, **kwargs),
            prefix=self.prefix,
            route=self.route,
        )

    def add_route(self, method: str = "POST") -> None:
        """
        Add a FastAPI route for this endpoint to the router. This
        function will not do anything if the router is None (i.e.
        no prefix was specified).

        This function is called automatically when the endpoint
        is created using the `endpoint` decorator.
        """
        if self.router is None:
            return

        if self.route is None:
            # The route will be postfixed with the fn name
            self.route = f"/{self.fn.__name__}/"

            # Analyze the function signature of `fn` to
            # construct a dictionary, mapping argument names
            # to their types and default values for creating a
            # Pydantic model.

            # During this we also
            # - make sure that args are either type-hinted or
            #   annotated with a default value (can't create
            #  a Pydantic model without a type hint or default)
            # - replace arguments that have type-hints which
            #   are subclasses of `IdentifiableMixin` with
            #   strings (i.e. the id of the Identifiable)
            #   (e.g. `Store` -> `str`, `Reference` -> `str`)
            signature = inspect.signature(self.fn)

            pydantic_model_params = {}
            for p in signature.parameters:
                annot = signature.parameters[p].annotation
                default = signature.parameters[p].default
                has_default = default is not inspect._empty

                if annot is inspect.Parameter.empty:
                    assert (
                        has_default
                    ), f"Parameter {p} must have a type annotation or a default value."
                elif isinstance(annot, type) and issubclass(annot, IdentifiableMixin):
                    # e.g. Stores, References must be referred to by str ids when
                    # passed into the API
                    pydantic_model_params[p] = (str, ...)
                else:
                    pydantic_model_params[p] = (
                        (annot, default) if has_default else (annot, ...)
                    )

            # Allow arbitrary types in the Pydantic model
            class Config:
                arbitrary_types_allowed = True

            # Create the Pydantic model, named `{fn_name}Model`
            FnPydanticModel = create_model(
                f"{self.fn.__name__.capitalize()}Model",
                __config__=Config,
                **pydantic_model_params,
            )

            # Create a wrapper function, with kwargs that conform to the
            # Pydantic model, and a return annotation that matches `fn`
            def _fn(kwargs: FnPydanticModel) -> signature.return_annotation:
                return self.fn(**kwargs.dict())

            # def _fn(kwargs: Endpoint.EmbeddedBody()) -> signature.return_annotation:
            #     return self.fn(**kwargs)

            # Name the wrapper function the same as `fn`, so it looks nice
            # in the docs
            _fn.__name__ = self.fn.__name__
        else:
            signature = inspect.signature(self.fn)
            for p in signature.parameters:
                annot = signature.parameters[p].annotation

                # If annot is a subclass of `IdentifiableMixin`, replace
                # it with the `str` type (i.e. the id of the Identifiable)
                # (e.g. `Store` -> `str`, `Reference` -> `str`)
                if isinstance(annot, type) and issubclass(annot, IdentifiableMixin):
                    self.fn.__annotations__[p] = str

            _fn = self.fn

        # Make FastAPI endpoint for POST requests
        self.router.add_api_route(
            self.route + "/" if not self.route.endswith("/") else self.route,
            _fn,
            methods=[method],
        )

        # Must add the router to the app again, everytime a new route is added
        # otherwise, the new route does not show up in the docs
        from meerkat.interactive.api.main import app

        app.include_router(self.router)

        print(self.router, self.prefix, self.route, method)


def make_endpoint(endpoint_or_fn: Union[Callable, Endpoint, None]) -> Endpoint:
    """Make an Endpoint."""
    return (
        endpoint_or_fn
        if isinstance(endpoint_or_fn, Endpoint)
        else Endpoint(endpoint_or_fn)
    )


def endpoint(
    fn: Callable = None,
    prefix: Union[str, APIRouter] = None,
    route: str = None,
    method: str = "POST",
):
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
    if fn is None:
        return partial(endpoint, prefix=prefix, route=route, method=method)

    def _endpoint(fn: Callable):
        # Gather up all the arguments that are hinted as Stores and References
        stores = set()
        references = set()
        # Also gather up the hinted arguments that subclass IdentifiableMixin
        # e.g. Store, Reference, Endpoint, Interface, etc.
        identifiables = {}
        for name, annot in inspect.getfullargspec(fn).annotations.items():
            if isinstance(annot, type) and issubclass(annot, Store):
                stores.add(name)
            elif isinstance(annot, type) and issubclass(annot, Reference):
                references.add(name)

            # Add all the identifiables
            if isinstance(annot, type) and issubclass(annot, IdentifiableMixin):
                identifiables[name] = annot

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

            # Unpack the arguments that were *not* annotated to be Stores or
            # References. Unpacking takes Stores or References, and grabs
            # the `._` attribute, which is the value of the Store or
            # the object the Reference points to.
            fn_args_to_unpack = {
                k: v
                for k, v in fn_bound_arguments.items()
                if k not in stores and k not in references
            }
            args, kwargs, _, _ = _unpack_refs_and_stores(**fn_args_to_unpack)

            # Identifiables that are passed into the function
            # may be passed in as a string id, or as the object itself
            # If they are passed in as a string id, we need to get the object
            # from the registry
            fn_args_as_is = {
                k: v if not isinstance(v, str) else identifiables[k].from_id(v)
                for k, v in fn_bound_arguments.items()
                if k in identifiables
            }

            # # Cautionary early check to make sure that the arguments that
            # # were hinted as Store and Reference arguments are indeed
            # # Store and Reference objects
            # assert all(
            #     [isinstance(v, (Store, Reference)) for v in fn_args_as_is.values()]
            # ), "All arguments that are type hinted as stores or references \
            #     must be Store or Reference objects."

            # Run the function
            result = fn(*args, **{**kwargs, **fn_args_as_is})

            # Get the modifications from the queue
            modifications = state.modification_queue.queue

            # Trigger the modifications
            modifications = trigger(modifications)

            # Return the result of the function
            return result

        # Register the endpoint and return it
        endpoint = Endpoint(
            fn=wrapper,
            prefix=prefix,
            route=route,
        )
        endpoint.add_route(method)
        return endpoint

    return _endpoint(fn)
