from typing import List, Union
from meerkat.interactive.graph import Box, Store, make_box, make_store
from meerkat.interactive.edit import EditTarget
import numpy as np

from ..editor import Editor


class StatsLabeler(Editor):

    name = "StatsLabeler"

    def __init__(
        self,
        dp: Box,
        col: Union[Store, str],
        target: EditTarget = None,
        selected: Store[List[int]] = None,
        primary_key: str = None,
        mode: Union[Store, str] = "train"
    ) -> None:
        super().__init__(dp=dp, col=col, target=target, selected=selected, primary_key=primary_key)
        self.mode = make_store(mode)

    @property
    def props(self):
        _props = super().props
        _props["mode"] = self.mode.config
        return _props