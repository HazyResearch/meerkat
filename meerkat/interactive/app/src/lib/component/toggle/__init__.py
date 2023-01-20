from meerkat.interactive.endpoint import Endpoint, endpoint

from ..abstract import AutoComponent


@endpoint
def _toggle(value: bool) -> bool:
    """
    A wrapper to convert the value to positional arguments.
    This allows the user to write an endpoint without having
    the same parameter name as is passed from the frontend.

    AFAIK, pydantic does not allow positional arguments.

    TODO: Figure out if we can pass positional args to pydantic.
    """
    return value


class Toggle(AutoComponent):
    title: str = ""

    on_toggle: Endpoint

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.on_toggle = _toggle.compose(self.on_toggle)
