from typing import Dict, Mapping, Union

from ..abstract import Component

Stat = Union[int, float, str]


class Stats(Component):
    data: Mapping[str, Stat]
    specs: Mapping[str, Mapping] = None
