from typing import Union

from meerkat.interactive.graph import Pivot, Store, make_store

from ..abstract import Component


class Match(Component):

    name = "Match"

    def __init__(
        self, pivot: Pivot, against: Union[Store, str], col: Union[Store, str] = ""
    ):
        super().__init__()
        self.pivot = pivot
        self.against: Store = make_store(against)
        self.col = make_store(col)
        self.text = make_store("")

    @property
    def props(self):
        return {
            "against": self.against.config,
            "dp": self.pivot.config,
            "col": self.col.config,
            "text": self.text.config,
        }
