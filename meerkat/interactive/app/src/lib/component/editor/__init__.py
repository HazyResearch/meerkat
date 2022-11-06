from typing import List, Union
from meerkat.interactive.graph import Box, Store, make_box, make_store
from meerkat.interactive.edit import EditTarget
import numpy as np

from ..abstract import Component


class Editor(Component):

    name = "Editor"

    def __init__(
        self,
        dp: Box,
        col: Union[Store, str],
        target: EditTarget = None,
        selected: Store[List[int]] = None,
        primary_key: str = None,
        title: str = None,
    ) -> None:
        super().__init__()
        self.col = make_store(col)
        self.text = make_store("")
        self.primary_key = primary_key

        self.dp = make_box(dp)
        if target is None:
            dp["_edit_id"] = np.arange(len(dp))
            target = EditTarget(self.dp, "_edit_id", "_edit_id")
        self.target = target
        self.selected = selected
        self.title = title if title is not None else ""

    @property
    def props(self):
        return {
            "dp": self.dp.config,
            "target": self.target.config,
            "col": self.col.config,
            "text": self.text.config,
            "selected": self.selected.config,
            "primary_key": self.primary_key,
            "title": self.title,
        }
