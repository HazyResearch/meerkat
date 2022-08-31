from typing import Callable, Dict, List, Union

import meerkat as mk
from meerkat.interactive.app.src.lib.component.sliceby import SliceBy
from meerkat.interactive.app.src.lib.component.table import EditTarget, Table
from meerkat.interactive.graph import Pivot, Store, head, interface_op
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
        super().__init__()
        self.sliceby = sliceby
        self.main_column = main_column
        self.tag_columns = tag_columns
        self.aggregations = aggregations

        self._layout()

    def layout(self):
        # Setup pivots
        sliceby = self.pivot(self.dp)
        component = SliceBy(
            sliceby,
            main_column=self.main_column,
            tag_columns=self.tag_columns,
            aggregations=self.aggregations,
        )
