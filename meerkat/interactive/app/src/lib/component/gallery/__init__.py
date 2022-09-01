from dataclasses import dataclass
from typing import List

from meerkat.interactive import Box, make_store
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


class Gallery(Component):

    name = "Gallery"

    def __init__(
        self,
        dp: Box,
        main_column: str,
        tag_columns: List[str],
        edit_target: EditTarget = None,
    ) -> None:
        super().__init__()
        self.dp = dp
        self.main_column = make_store(main_column)
        self.tag_columns = make_store(tag_columns)
        self.edit_target = edit_target

    @property
    def props(self):
        return {
            "dp": self.dp.config,
            "main_column": self.main_column.config,
            "tag_columns": self.tag_columns.config,
            "edit_target": self.edit_target.config,
        }
