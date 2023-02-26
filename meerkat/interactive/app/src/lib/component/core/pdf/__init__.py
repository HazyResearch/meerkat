from typing import Any, Dict

from meerkat.interactive.app.src.lib.component.abstract import Component


class PDF(Component):
    data: str
    classes: str = ""