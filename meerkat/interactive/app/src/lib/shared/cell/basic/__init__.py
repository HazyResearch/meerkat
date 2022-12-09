from typing import Any
from meerkat.interactive.graph import Store


from meerkat.interactive.app.src.lib.component.abstract import Component


class Text(Component):

    data: Store[str]
    dtype: str = None
    precision: int = 3
    percentage: bool = False
