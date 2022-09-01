from dataclasses import dataclass

from meerkat.interactive import Box
from meerkat.interactive.graph import Pivot

from ..abstract import Component


@dataclass
class EditTarget:
    pivot: Pivot
    pivot_id_column: str
    id_column: str

    @property
    def config(self):
        return {
            "pivot": self.pivot.config,
            "pivot_id_column": self.pivot_id_column,
            "id_column": self.id_column,
        }


class Table(Component):

    name = "Table"

    def __init__(self, dp: Box, edit_target: EditTarget = None) -> None:
        super().__init__()
        self.dp = dp
        self.edit_target = edit_target

    @property
    def props(self):
        return {
            "dp": self.dp.config,
            "edit_target": self.edit_target.config,
        }
