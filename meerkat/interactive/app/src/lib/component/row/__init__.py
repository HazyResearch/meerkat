import numpy as np
from meerkat.dataframe import DataFrame

from meerkat.interactive.edit import EditTarget
from meerkat.interactive.graph import Store

from ..abstract import Component


class Row(Component):

    name = "Row"

    def __init__(
        self,
        df: DataFrame,
        idx: Store[int],
        target: EditTarget = None,
        cell_specs: dict = None,
        title: str = "",
    ):
        super().__init__()
        self.df = df
        self.idx = idx
        if target is None:
            df["_edit_id"] = np.arange(len(df))
            target = EditTarget(self.df, "_edit_id", "_edit_id")
        self.target = target

        if cell_specs is None:
            cell_specs = {}
        self.cell_specs = cell_specs
        self.title = title

    @property
    def props(self):
        return {
            "df": self.df.config,  # FIXME
            "idx": self.idx.config,
            "target": self.target.config,
            "cell_specs": self.cell_specs,
            "title": self.title,
        }
