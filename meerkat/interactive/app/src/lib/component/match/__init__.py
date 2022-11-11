import warnings
from typing import Union

from meerkat.interactive.graph import Reference, Store, make_ref, make_store

from ..abstract import Component


class Match(Component):

    name = "Match"

    def __init__(
        self,
        pivot: Reference,
        against: Union[Store, str],
        col: Union[Store, str] = "",
        title: str = "",
    ):
        super().__init__()
        if not isinstance(pivot, Reference):
            warnings.warn("input is not a Reference - this may cause errors")
        self.pivot = make_ref(pivot)
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
