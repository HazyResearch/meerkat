import logging
from typing import Any, List, Literal

from meerkat.interactive.endpoint import Endpoint, endpoint, get_signature
from meerkat.interactive.graph import Store

from ..abstract import AutoComponent

logger = logging.getLogger(__name__)


@endpoint
def _select_value(
    index: int, value_store: Store[Any], choices: Store[List], return_value: bool
):
    """Endpoint to set the value of a choice component.

    We pass the index of the value rather than the value itself to perserve
    the type of the input. This is necessary because the frontend will convert
    all inputs to strings.
    """
    value = choices[index]
    logger.debug(f"Choice: Setting value of {value_store} to {value}")
    value_store.set(value)
    if return_value:
        return value


class Choice(AutoComponent):
    """Component to select a value from a list of choices.

    The ``value`` store will be set to the selected value.
    It will be automatically updated when the user selects a new value.
    Functions that should trigger when the value changes should take
    ``Choice.value`` as an input.

    Attributes:
        choices: A list of choices to select from.
        value: The value of the selected choice.
        gui_type: The type of GUI to use. Can be either "dropdown" or "radio".
        title: The title of the component.
        on_select: An custom endpoint to call when a choice is selected.
            This will be called in addition to setting the value.

    Examples:

        .. code-block:: python

            import meerkat as mk

            @mk.gui.endpoint
            def on_select(value):
                print("Selected", value)
            choice = mk.gui.Choice(choices=["a", "b"], on_select=on_select)
    """

    choices: list
    value: str
    gui_type: Literal["dropdown", "radio"] = "dropdown"
    title: str = None

    on_select: Endpoint = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Determine if the endpoint should return a value.
        return_value = False
        if self.on_select is not None:
            sig = get_signature(self.on_select)
            if len(sig.parameters) > 1:
                raise ValueError("on_select must have at most one argument.")
            elif len(sig.parameters) == 1:
                return_value = True

        on_select = _select_value.partial(
            value_store=self.value, choices=self.choices, return_value=return_value
        )
        if self.on_select is not None:
            on_select = on_select.compose(self.on_select)
        self.on_select = on_select
