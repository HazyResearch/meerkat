from typing import Dict, Mapping, Union

from meerkat.interactive.graph import Store, make_store

from ..abstract import Component

Stat = Union[int, float, str]


class Stats(Component):
    name: str = "Stats"

    def __init__(
        self,
        data: Store[Dict[str, Union[Store[Stat], Stat]]],
        specs: Mapping[str, Mapping[str, any]] = None,
    ) -> None:
        super().__init__()
        self.data = {k: make_store(v) for k, v in data.items()}
        self.specs = specs

    @property
    def props(self):
        props = {
            "data": {k: v.config for k, v in self.data.items()},
            "specs": self.specs,
        }
        return props
