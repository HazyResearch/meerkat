import numpy as np

from meerkat.dataframe import DataFrame
from meerkat.interactive.edit import EditTarget

from ..abstract import Component


class Table(Component):

    name = "Table"

    def __init__(
        self,
        df: DataFrame,
        edit_target: EditTarget = None,
        per_page: int = 100,
        column_widths: list = None,
    ) -> None:
        super().__init__()

        self.df = df
        if edit_target is None:
            self.df.obj["_edit_id"] = np.arange(len(self.df.obj))
            edit_target = EditTarget(self.df, "_edit_id", "_edit_id")
        self.edit_target = edit_target

        self.per_page = per_page
        self.column_widths = column_widths

    @property
    def props(self):
        return {
            "df": self.df.config,  # FIXME
            "edit_target": self.edit_target.config,
            "per_page": self.per_page,
            "column_widths": self.column_widths,
        }
