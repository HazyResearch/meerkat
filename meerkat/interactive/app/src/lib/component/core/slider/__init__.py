from typing import Any, Optional

from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.event import EventInterface


class OnChangeSlider(EventInterface):
    value: Any


class Slider(Component):
    """
    A slider that allows the user to select a value from a range.

    Args:
        value (float): The current value of the slider.
        min (float): The minimum value of the slider.
        max (float): The maximum value of the slider.
        step (float): The step size of the slider.
        disabled (bool): Whether the slider is disabled.
        classes (str): The Tailwind classes to apply to the component.
    """

    value: float = 0
    min: float = 0.0
    max: float = 100.0
    step: float = 1.0
    disabled: bool = False
    classes: str = "bg-violet-50 px-4 py-1 rounded-lg"

    on_change: Optional[Endpoint[OnChangeSlider]] = None
