from typing import Callable, Dict, List, Union

import meerkat as mk
from meerkat.datapanel import DataPanel
from meerkat.interactive.graph import Box, make_store
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.ops.sliceby.sliceby import SliceBy

from ..abstract import Component


class Aggregation(IdentifiableMixin):

    identifiable_group: str = "aggregations"

    def __init__(self, func: Callable[[mk.DataPanel], Union[int, float, str]]):
        self.func = func
        super().__init__()

    def __call__(self, dp: mk.DataPanel) -> Union[int, float, str]:
        return self.func(dp)

    @property
    def config(self):
        return {
            "id": self.id,
        }


class SliceByCards(Component):

    name = "SliceByCards"

    def __init__(
        self,
        sliceby: Box[SliceBy],
        dp: Box[DataPanel],
        main_column: str,
        tag_columns: List[str] = None,
        aggregations: Dict[
            str, Callable[[mk.DataPanel], Union[int, float, str]]
        ] = None,
    ) -> None:
        assert sliceby.obj.data is dp.obj

        super().__init__()
        self.sliceby = sliceby
        self.dp = dp

        if aggregations is None:
            aggregations = {}

        self.aggregations = {k: Aggregation(v) for k, v in aggregations.items()}

        self.main_column = make_store(main_column)
        self.tag_columns = make_store(tag_columns)

    @property
    def props(self):
        return {
            "sliceby": self.sliceby.config,
            "dp": self.dp.config,
            "main_column": self.main_column.config,
            "tag_columns": self.tag_columns.config,
            "aggregations": {k: v.config for k, v in self.aggregations.items()},
        }
