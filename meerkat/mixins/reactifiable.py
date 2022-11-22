from typing import Any, Callable

_REACTIVE_KEYS = [
    "keys",
    "to_jsonl",
]


class ReactifiableMixin:
    def __getattribute__(self, name: str) -> Any:
        """Reactify instance methods.

        When a method is "reactified" and is called within a :cls:`mk.gui.react`
        context, it will return reactive containers
        (e.g. :cls:`mk.gui.Store`) and add the method
        (i.e. operation) and the outputs to the graph.

        Outside of this context, the method will not add operations and the outputs
        to the graph.

        Currently properties and attributes are not reactified.
        """
        from meerkat.interactive.graph import is_reactive, reactive

        if name in _REACTIVE_KEYS:  # FIXME
            out = super().__getattribute__(name)
            if isinstance(out, Callable) and is_reactive():
                out = reactive(out)
            return out
        else:
            return super().__getattribute__(name)
