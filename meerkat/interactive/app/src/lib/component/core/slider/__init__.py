from typing import Any, Optional, Union

from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.event import EventInterface


class OnChangeSlider(EventInterface):
    value: Any


class Slider(Component):
    """A slider that allows the user to select a value from a range.

    Args:
        value: The current value of the slider.
        min: The minimum value of the slider.
        max: The maximum value of the slider.
        step: The step size of the slider.
        disabled: Whether the slider is disabled.
        classes: The Tailwind classes to apply to the component.
    """

    value: Union[float, int] = 0.0
    min: Union[float, int] = 0.0
    max: Union[float, int] = 100.0
    step: Union[float, int] = 1.0
    disabled: bool = False
    classes: str = "bg-violet-50 px-4 py-1 rounded-lg"

    on_change: Optional[Endpoint[OnChangeSlider]] = None
