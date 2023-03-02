from typing import Mapping, Union

from meerkat.interactive.app.src.lib.component.abstract import Component


class Stats(Component):
    data: Mapping[str, Union[int, float]]
