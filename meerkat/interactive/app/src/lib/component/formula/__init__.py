"""

match("smiling", column("image"))
"""
import warnings
from typing import Union

from meerkat.interactive.graph import Pivot, Store, make_store

from ..abstract import Component




class FormulaBar(Component):

    pivot: Store 
    against: Store = make_store(against)
    col: Store[str]
    text = make_store("")
    title = title

def __init__(
        self, pivot: Pivot, against: Union[Store, str], col: Union[Store, str] = "",
        title: str = ""
    ):
        super().__init__()
        if not isinstance(pivot, Pivot):
            warnings.warn("input is not a Pivot - this may cause errors")
        self.pivot = pivot
        self.against: Store = make_store(against)
        self.col = make_store(col)
        self.text = make_store("")
        self.title = title

    @property
    def props(self):
        return {
            "against": self.against.config,
            "dp": self.pivot.config,
            "col": self.col.config,
            "text": self.text.config,
            "title": self.title
        }
