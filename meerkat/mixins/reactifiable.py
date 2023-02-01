from typing import Any, Callable


class ReactifiableMixin:
    """A class that reactifies all attribute accesses.

    When an attribute accessor is "reactified" and is called within
    a :cls:`mk.gui.react` context, it will add accessing the attribute
    to the graph. This means if the object changes, the attribute
    will be fetched again.


    Outside of this context, the method will not add operations and the outputs
    to the graph.

    TODO: Clean and investigate failure points of this function.
    """

    def __getattribute__(self, name: str) -> Any:
        from meerkat.interactive.graph import is_reactive, no_react, reactive

        # We assume accessing the attribute twice will not result in different values.
        # We dont explicitly check for this because it is expensive.
        with no_react():
            attr = super().__getattribute__(name)

        if (
            is_reactive()
            and not isinstance(attr, Callable)
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
            _fn = reactive(_fn, nested_return=False)

            return _fn(self)
        else:
            return attr
