from typing import Dict, Mapping, Union

from ..abstract import AutoComponent

Stat = Union[int, float, str]


class Stats(AutoComponent):
    data: Mapping[str, Stat]
    specs: Mapping[str, Mapping] = None
