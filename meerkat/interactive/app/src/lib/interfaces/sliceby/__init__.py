from typing import Callable, Dict, List, Union

import meerkat as mk
from meerkat.interactive.app.src.lib.component.slicebycards import SliceByCards
from meerkat.interactive.app.src.lib.component.table import EditTarget, Table
from meerkat.interactive.graph import Pivot, Store, head, interface_op
from meerkat.ops.sliceby.sliceby import SliceBy
from meerkat.state import state

from ..abstract import Interface, InterfaceConfig


@interface_op
def simple_op(col: str):
    return col + "!"


@interface_op
def make_selection_dp(dp, id, selection):
    """An out of place operation to take a store and make it a derived
    datapanel."""
    return mk.DataPanel({id: dp[id][selection]})


class SliceByInterface(Interface):
    def __init__(
        self,
        sliceby: SliceBy,
        main_column: str,
        tag_columns: List[str] = None,
        aggregations: Dict[
            str, Callable[[mk.DataPanel], Union[int, float, str]]
        ] = None,
    ):

        if main_column not in sliceby.data:
            raise ValueError(f"The column {main_column} is not in the sliceby.")

        for tag_column in tag_columns:
            if tag_column not in sliceby.data:
                raise ValueError(f"The column {tag_column} is not in the sliceby.")

        super().__init__()
        self.sliceby = sliceby
        self.main_column = main_column
        self.tag_columns = tag_columns
        self.aggregations = aggregations

        self._layout()

    def layout(self):
        # Setup pivots
        sliceby = self.pivot(self.sliceby)
        dp = self.pivot(self.sliceby.data)
        component = SliceByCards(
            sliceby=sliceby,
            dp=dp,
            main_column=self.main_column,
            tag_columns=self.tag_columns,
            aggregations=self.aggregations,
        )
