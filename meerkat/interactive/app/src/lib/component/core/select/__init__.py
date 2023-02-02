from typing import Any, List, Optional, Union

from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint


class Select(Component):
    """
    A selection dropdown, which can be used to select a single value from a list of options.

    Args:
        values (List[Any]): A list of values to select from.
        labels (List[str]): A list of labels to display for each value.
        value (Any): The selected value.
        disabled (bool): Whether the select is disabled.
        classes (str): The Tailwind classes to apply to the select.

        on_change: An endpoint to call when the value changes.
    """

    values: List[Any]
    labels: List[str]
    value: Any = None
    disabled: bool = False
    classes: str = ""

    on_change: Optional[Endpoint] = None
