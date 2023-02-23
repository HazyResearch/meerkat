from typing import Any, List, Optional, Union

from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.event import EventInterface


class OnChangeSelect(EventInterface):
    value: Union[str, int, float, bool, None]


class Select(Component):
    """A selection dropdown that can be used to select a single value from a
    list of options.

    Args:
        values (List[Any]): A list of values to select from.
        labels (List[str]): A list of labels to display for each value.
        value (Any): The selected value.
        disabled (bool): Whether the select is disabled.
        classes (str): The Tailwind classes to apply to the select.

        on_change: The `Endpoint` to call when the selected value changes. \
            It must have the following signature:

            `(value: Union[str, int, float, bool, None])`

            with
                value (Union[str, int, float, bool, None]): The value of the \
                selected radio button.
    """

    values: List[Any]
    labels: List[str] = None
    value: Any = None
    disabled: bool = False
    classes: str = ""

    on_change: Optional[Endpoint[OnChangeSelect]] = None

    def __init__(
        self,
        values: List[Any],
        *,
        labels: List[str] = None,
        value: Any = None,
        disabled: bool = False,
        classes: str = "",
        on_change: Optional[Endpoint[OnChangeSelect]] = None,
    ):
        if labels is None:
            labels = values

        super().__init__(
            values=values,
            labels=labels,
            value=value,
            disabled=disabled,
            classes=classes,
            on_change=on_change,
        )
