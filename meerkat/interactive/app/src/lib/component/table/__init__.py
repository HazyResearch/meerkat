import numpy as np

from meerkat.interactive.edit import EditTarget
from meerkat.interactive.graph import Box, make_box

from ..abstract import Component


class Table(Component):

    name = "Table"

    def __init__(
        self,
        dp: Box,
        edit_target: EditTarget = None,
        per_page: int = 100,
        column_widths: list = None,
    ) -> None:
        super().__init__()

        self.dp = make_box(dp)
        if edit_target is None:
            self.dp.obj["_edit_id"] = np.arange(len(self.dp.obj))
            edit_target = EditTarget(self.dp, "_edit_id", "_edit_id")
        self.edit_target = edit_target

        self.per_page = per_page
        self.column_widths = column_widths

    @property
    def props(self):
        return {
            "dp": self.dp.config,
            "edit_target": self.edit_target.config,
            "per_page": self.per_page,
            "column_widths": self.column_widths,
        }
