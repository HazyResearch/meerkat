import warnings
from typing import Union

from meerkat.interactive.graph import Pivot, Store, make_box, make_store

from ..abstract import Component


class Match(Component):

    name = "Match"

    def __init__(
        self,
        pivot: Pivot,
        against: Union[Store, str],
        col: Union[Store, str] = "",
        title: str = "",
    ):
        super().__init__()
        if not isinstance(pivot, Pivot):
            warnings.warn("input is not a Pivot - this may cause errors")
        self.pivot = make_box(pivot)
        self.against: Store = make_store(against)
        self.col = make_store(col)
        self.text = make_store("")
        self.title = title

    @property
    def props(self):
        return {
            "against": self.against.config,
            "df": self.pivot.config,
            "col": self.col.config,
            "text": self.text.config,
            "title": self.title,
        }
