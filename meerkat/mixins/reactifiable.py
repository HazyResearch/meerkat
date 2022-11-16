from typing import Any, Callable


class ReactifiableMixin:
    def __getattribute__(self, name: str) -> Any:
        """Reactify instance methods.

        When a method is "reactified" and is called within a :cls:`mk.gui.react`
        context, it will return reactive containers
        (e.g. :cls:`mk.gui.Store` or :cls:`mk.gui.Reference`) and add the method
        (i.e. operation) and the outputs to the graph.

        Outside of this context, the method will not add operations and the outputs
        to the graph.

        Currently properties and attributes are not reactified.
        """
        from meerkat.interactive.graph import reactify

        if name == 'keys':
            out = super().__getattribute__(name)
            if isinstance(out, Callable):
                out = reactify(out)
            return out
        else:
            return super().__getattribute__(name)

        out = super().__getattribute__(name)
        if isinstance(out, Callable):
            out = reactify(out)
        return out
