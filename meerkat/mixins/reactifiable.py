import inspect
import warnings
from typing import Any

from meerkat.interactive.graph.marking import (
    is_unmarked_context,
    is_unmarked_fn,
    unmarked,
)


class MarkableMixin:
    """A class that can be marked."""

    _self_marked: bool = False

    @property
    def marked(self):
        return self._self_marked

    def mark(self):
        """Mark this object.

        When marked, this object will trigger a reactive function when
        passed in as an argument.
        """
        self._self_marked = True
        return self

    def unmark(self):
        """Unmark this object.

        When unmarked, this object will not trigger a reactive function
        when passed in as an argument.
        """
        self._self_marked = False
        return self


class ReactifiableMixin(MarkableMixin):
    """A class that reactifies all attribute accesses.

    When an attribute accessor is "reactified" and is called within
    a :cls:`mk.gui.react` context, it will add accessing the attribute
    to the graph. This means if the object changes, the attribute
    will be fetched again.

    Outside of this context, the method will not add operations and the outputs
    to the graph.
    """

    # TODO: Clean and investigate failure points of this function.
    def __getattribute__(self, name: str) -> Any:
        from meerkat.interactive.graph.reactivity import is_reactive_fn, reactive

        # We assume accessing the attribute twice will not result in different values.
        # We dont explicitly check for this because it is expensive.
        with unmarked():
            is_self_marked = super().__getattribute__("_self_marked")
            attr = super().__getattribute__(name)
            is_method_or_fn = inspect.ismethod(attr) or inspect.isfunction(attr)
            attr_is_reactive_fn = is_method_or_fn and is_reactive_fn(attr)
            attr_is_unmarked_fn = is_method_or_fn and is_unmarked_fn(attr)

        if is_method_or_fn:
            if name in ["_reactive_warning"]:
                # There are some functions we never want to wrap because the function
                # needs access to the current reactive state (i.e. is_reactive()).
                # These functions should all be declared here.

                # For example, `_reactive_warning` needs to know if we are in a
                # reactive context, in order to determine if we should raise a warning.
                return attr

            if name in ["mark", "unmark", "attach_to_inode"]:
                # Regardless of whether `self` is marked or not, these methods should
                # never be wrapped in @reactive or run any statements inside them
                # reactively. All such methods should be declared here.
                # So far, we include methods in MarkableMixin and NodeMixin.

                # Therefore, we explicitly wrap them in @unmarked, so that they can
                # never be reactive.
                return unmarked()(attr)

            if not attr_is_reactive_fn and not attr_is_unmarked_fn:
                # By default, all methods of a ReactifiableMixin are unmarked.
                # This makes it cleaner because the developer only has to decorate
                # methods with @reactive if they want them to be reactive.
                # The alternative here would be to do nothing - i.e. to leave methods
                # that are undecorated, undecorated. However, this has some
                # consequences: particularly, that code inside the method could still
                # be reactive if the method was passed in a marked argument.
                # Here's an example:
                #
                # def foo(self, x: int):
                #    # this will be reactive when a is marked, because
                #    # properties are reactive
                #    a = self.property
                #    # this will add an operation to the graph if x is marked
                #    b = reactive(fn)(x)
                #
                # By using the @unmarked decorator, we can avoid random lines of code
                # becoming operations on the graph and being rerun - i.e. the method
                # behaves like a single unit, rather than separate lines of code.
                # For methods of a class, this makes sense because the user should
                # not be privvy to the internals of the method. They should only
                # see what they put in and what comes out - so the method effectively
                # should behave like a single black box.
                #
                # The user can always define their own reactive function that wraps a
                # method like `foo` and make it reactive!
                return unmarked()(attr)

            return attr

        # TODO: Handle functions that are stored as attributes.
        # These functions should be wrapped in reactive when the object is reactive.
        # For ordinary attributes, we need to check if reactive=True.
        # FIXME: We may be able to get rid of this distinction by decorating
        # Store.__call__ with @reactive.
        if (
            # Don't do anything when the object is not marked, it should behave
            # as normal in that case.
            is_self_marked
            and not is_unmarked_context()
            # Ignore private attributes.
            and not name.startswith("_")
            # Ignore dunder attributes.
            and not name.startswith("__")
            # Ignore all node-related attributes. These should never be accessed
            # in a reactive way.
            # NOTE: There seems to be some dependence between NodeMixin and this class.
            # Specially, anything that is a ReactifiableMixin should also be a
            # NodeMixin. Consider having this class inherit from NodeMixin.
            and name
            not in ("_self_inode", "inode", "inode_id", "_self_marked", "marked", "id")
        ):
            # Only build the function if we are in a reactive context.
            # TODO: Cache this function so that it is faster.
            def _fn(_obj):
                return super().__getattribute__(name)

            _fn.__name__ = name
            _fn = reactive(_fn, nested_return=False)

            return _fn(self)
        else:
            return attr

    def _reactive_warning(self, name, placeholder="obj"):
        if not is_unmarked_context() and self.marked:
            warnings.warn(
                f"Calling {name}({placeholder}) is not reactive. "
                f"Use `mk.{name}({placeholder})` to get"
                "a reactive variable (i.e. a Store). "
                f"`mk.{name}({placeholder})` behaves exactly"
                f"like {name}({placeholder}) outside of this difference."
            )

    """
    We include the following dunder methods as examples of how to correctly
    implement them when using this Mixin.
    """

    def __len__(self):
        with unmarked():
            out = super().__len__()
        self._reactive_warning("len")
        return out

    def __int__(self):
        with unmarked():
            out = super().__int__()
        self._reactive_warning("int")
        return out

    def __long__(self):
        with unmarked():
            out = super().__long__()
        self._reactive_warning("long")
        return out

    def __float__(self):
        with unmarked():
            out = super().__float__()
        self._reactive_warning("float")
        return out

    def __complex__(self):
        with unmarked():
            out = super().__complex__()
        self._reactive_warning("complex")
        return out

    def __oct__(self):
        with unmarked():
            out = super().__oct__()
        self._reactive_warning("oct")
        return out

    def __hex__(self):
        with unmarked():
            out = super().__hex__()
        self._reactive_warning("hex")
        return out
