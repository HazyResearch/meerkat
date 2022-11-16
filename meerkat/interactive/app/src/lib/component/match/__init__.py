from typing import Union

from meerkat.dataframe import DataFrame
from meerkat.interactive.graph import Store, make_store

from ..abstract import Component


class Match(Component):

    name = "Match"

    def __init__(
        self,
        df: DataFrame,
        against: Union[Store, str],
        col: Union[Store, str] = "",
        title: str = "",
    ):
        super().__init__()
        self.df = df
        self.against: Store = make_store(against)
        self.col = make_store(col)
        self.text = make_store("")
        self.title = title

    @property
    def props(self):
        return {
            "against": self.against.config,
            "df": self.df.config,  # FIXME
            "col": self.col.config,
            "text": self.text.config,
            "title": self.title,
        }
