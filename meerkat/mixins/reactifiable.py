import inspect
from typing import Any


class ReactifiableMixin:
    """A class that reactifies all attribute accesses.

    When an attribute accessor is "reactified" and is called within
    a :cls:`mk.gui.react` context, it will add accessing the attribute
    to the graph. This means if the object changes, the attribute
    will be fetched again.

    Outside of this context, the method will not add operations and the outputs
    to the graph.
    """

    # TODO: Clean and investigate failure points of this function.

    _reactive: bool = False

    def react(self):
        """Converts the object to a reactive object in-place."""
        # TODO: Determine if this should be out-of-place, similar to torch.
        self._reactive = True
        return self

    def no_react(self):
        """Converts the object to a non-reactive object in-place."""
        self._reactive = False
        return self

    def __getattribute__(self, name: str) -> Any:
        from meerkat.interactive.graph import (
            _reactive,
            is_reactive,
            is_reactive_fn,
            no_react,
        )

        # We assume accessing the attribute twice will not result in different values.
        # We dont explicitly check for this because it is expensive.
        with no_react():
            is_obj_reactive = super().__getattribute__("_reactive")
            attr = super().__getattribute__(name)
            is_method_or_fn = inspect.ismethod(attr) or inspect.isfunction(attr)
            _is_reactive_fn = is_method_or_fn and is_reactive_fn(attr)

        # If the attribute is a method or function that is decorated with @reactive,
        # then we need to determine if we should return a reactive version.
        #   1. If the object is reactive, then we do not need to do anything.
        #      The function is already decorated with @reactive and will be
        #      handled accordingly.
        #   2. If the object is not reactive, then we need to return
        #      a non-reactive function. We can achieve this by wrapping the function
        #      with @no_react.
        # NOTE: We only have to do (2) when the object is not reactive. We should not
        #       check the is_reactive() state when the function is fetched (i.e. here),
        #       but rather when the function is called. This will be handled based
        #       on the is_reactive(). Only when the object is not reactive, do we
        #       need to handle this case.
        if _is_reactive_fn:
            if not is_obj_reactive:
                return no_react()(attr)
            return attr

        # TODO: Handle functions that are stored as attributes.
        # These functions should be wrapped in reactive when the object is reactive.
        # For ordinary attributes, we need to check if reactive=True.
        # FIXME: We may be able to get rid of this distinction by decorating
        # Store.__call__ with @reactive.
        if (
            is_obj_reactive
            and is_reactive()
            and not is_method_or_fn
            # Ignore dunder attributes.
            and not name.startswith("__")
            # Ignore all node-related attributes. These should never be accessed
            # in a reactive way.
            and name not in ("_self_inode", "inode", "inode_id")
        ):
            # Only build the function if we are in a reactive context.
            # TODO: Cache this function so that it is faster.
            def _fn(_obj):
                return super().__getattribute__(name)

            _fn.__name__ = name
            _fn = _reactive(_fn, nested_return=False)

            return _fn(self)
        else:
            return attr
