import inspect
import warnings
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
        from meerkat.interactive.graph.reactivity import (
            _reactive,
            is_noreact_fn,
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
            _is_noreact_fn = is_method_or_fn and is_noreact_fn(attr)

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
        if is_method_or_fn:
            if not is_obj_reactive or name in ["react", "no_react", "attach_to_inode"]:
                return no_react()(attr)
            # TODO: Verify this check needs to be valid for both _reactive
            # and @no_react decorators.
            elif not _is_reactive_fn and not _is_noreact_fn:
                # If the object is reactive, but the function is not decorated with
                # @reactive, then we need to wrap the function with @reactive.
                # This is because the object is reactive and we want the function
                # to be reactive.
                return _reactive(attr)
            else:
                return attr

        # TODO: Handle functions that are stored as attributes.
        # These functions should be wrapped in reactive when the object is reactive.
        # For ordinary attributes, we need to check if reactive=True.
        # FIXME: We may be able to get rid of this distinction by decorating
        # Store.__call__ with @reactive.
        if (
            is_obj_reactive
            and is_reactive()
            # Ignore dunder attributes.
            and not name.startswith("__")
            # Ignore all node-related attributes. These should never be accessed
            # in a reactive way.
            and name not in ("_self_inode", "inode", "inode_id", "_reactive")
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

    def _reactive_warning(self, name, placeholder="obj"):
        from meerkat.interactive.graph import is_reactive

        if is_reactive():
            warnings.warn(
                f"Calling {name}({placeholder}) is not reactive. "
                f"Use `mk.{name}({placeholder})` to get"
                "a reactive variable (i.e. a Store). "
                f"`mk.{name}({placeholder})` behaves exactly"
                f"like {name}({placeholder}) outside of this difference."
            )

    def __len__(self):
        self._reactive_warning("len")
        return super().__len__()

    def __int__(self):
        self._reactive_warning("int")
        return super().__int__()

    def __long__(self):
        self._reactive_warning("long")
        return super().__long__()

    def __float__(self):
        self._reactive_warning("float")
        return super().__float__()

    def __complex__(self):
        self._reactive_warning("complex")
        return super().__complex__()

    def __oct__(self):
        self._reactive_warning("oct")
        return super().__oct__()

    def __hex__(self):
        self._reactive_warning("hex")
        return super().__hex__()
