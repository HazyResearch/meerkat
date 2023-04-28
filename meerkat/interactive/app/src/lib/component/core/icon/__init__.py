from typing import Optional

from meerkat.interactive.app.src.lib.component.abstract import Component


class Icon(Component):
    data: str = ""
    name: str = "Globe2"
    fill: Optional[str] = None
    size: int = 16
