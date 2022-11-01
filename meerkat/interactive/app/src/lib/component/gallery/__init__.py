from dataclasses import dataclass
from typing import List

from meerkat.interactive.graph import Box, Pivot, Store, make_box, make_store

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
        selected: Store[List[int]] = None,
        primary_key: str = None,
    ) -> None:
        super().__init__()
        self.dp = make_box(dp)
        self.main_column = make_store(main_column)
        self.tag_columns = make_store(tag_columns)
        self.primary_key = primary_key

        if edit_target is None:
            # TODO: primary key - make this based on primary keys once that is
            # implemented
            edit_target = EditTarget(self.dp, self.primary_key, self.primary_key)
        self.edit_target = edit_target

        self.primary_key = primary_key
        if selected is None:
            selected = []
        self.selected = make_store(selected)

    @property
    def props(self):
        props = {
            "dp": self.dp.config,
            "main_column": self.main_column.config,
            "tag_columns": self.tag_columns.config,
            "edit_target": self.edit_target.config,
            "selected": self.selected.config,
            "primary_key": self.primary_key,
        }
        return props
