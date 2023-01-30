from typing import Mapping, Union

from ..abstract import Component


class Stats(Component):
    data: Mapping[str, Union[int, float]]
