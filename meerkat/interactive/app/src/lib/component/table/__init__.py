from dataclasses import dataclass

from meerkat.interactive.graph import Box, make_box
from meerkat.interactive.edit import EditTarget
import numpy as np

from ..abstract import Component


class Table(Component):

    name = "Table"

    def __init__(self, dp: Box, edit_target: EditTarget = None) -> None:
        super().__init__()

        self.dp = make_box(dp)
        if edit_target is None:
            dp["_edit_id"] = np.arange(len(dp))
            edit_target = EditTarget(self.dp, "_edit_id", "_edit_id")
        self.edit_target = edit_target

    @property
    def props(self):
        return {
            "dp": self.dp.config,
            "edit_target": self.edit_target.config,
        }
