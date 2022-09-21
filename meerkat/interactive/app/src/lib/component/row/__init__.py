from typing import Union

import numpy as np 

from meerkat.interactive.edit import EditTarget
from meerkat.interactive.graph import Pivot, Store, make_store

from ..abstract import Component


class Row(Component):

    name = "Row"

    def __init__(
        self,
        dp: Pivot,
        idx: Store[int],
        target: EditTarget = None,
    ):
        super().__init__()
        self.dp = dp
        self.idx = idx
        if target is None:
            dp["_edit_id"] = np.arange(len(dp))
            target = EditTarget(self.dp, "_edit_id", "_edit_id")
        self.target = target

    @property
    def props(self):
        return {
            "dp": self.dp.config,
            "idx": self.idx.config,
            "target": self.target.config,
        }
