from meerkat.interactive.endpoint import Endpoint, endpoint
from meerkat.interactive.graph import Store

from ..abstract import Component


@endpoint
def _toggle(value: bool, store: Store[bool]) -> bool:
    """
    A wrapper to convert the value to positional arguments.
    This allows the user to write an endpoint without having
    the same parameter name as is passed from the frontend.

    AFAIK, pydantic does not allow positional arguments.

    TODO: Figure out if we can pass positional args to pydantic.
    """
    store.set(value)
    return value


class Toggle(Component):
    title: str = ""
    value: bool = False

    on_toggle: Endpoint = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        on_toggle = _toggle.partial(store=self.value)
        if self.on_toggle is not None:
            on_toggle = on_toggle.compose(self.on_toggle)
        self.on_toggle = on_toggle
