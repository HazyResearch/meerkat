from typing import List, Union

import numpy as np

from meerkat.interactive.edit import EditTarget
from meerkat.interactive.graph import Box, Store, make_box, make_store

from ..abstract import Component


class Editor(Component):

    name = "Editor"

    def __init__(
        self,
        df: Box,
        col: Union[Store, str],
        target: EditTarget = None,
        selected: Store[List[int]] = None,
        primary_key: str = None,
    ) -> None:
        super().__init__()
        self.col = make_store(col)
        self.text = make_store("")
        self.primary_key = primary_key

        self.df = make_box(df)
        if target is None:
            df["_edit_id"] = np.arange(len(df))
            target = EditTarget(self.df, "_edit_id", "_edit_id")
        self.target = target
        self.selected = selected

    @property
    def props(self):
        return {
            "df": self.df.config,
            "target": self.target.config,
            "col": self.col.config,
            "text": self.text.config,
            "selected": self.selected.config,
            "primary_key": self.primary_key,
        }
