
from meerkat.interactive.graph import Store

from ..abstract import Component


class Text(Component):

    data: Store[str]
    dtype: str = None
    precision: int = 3
    percentage: bool = False
