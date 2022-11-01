import meerkat as mk
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.app.src.lib.component.match import Match
from meerkat.interactive.app.src.lib.component.plot import Plot
from meerkat.interactive.app.src.lib.component.table import EditTarget, Table
from meerkat.interactive.graph import Store, interface_op

from ..abstract import Interface


@interface_op
def simple_op(col: str):
    return col + "!"


@interface_op
def make_selection_dp(dp, id, selection):
    """An out of place operation to take a store and make it a derived
    datapanel."""
    return mk.DataPanel({id: dp[id][selection]})


class PlotInterface(Interface):
    def __init__(
        self,
        dp: mk.DataPanel,
        id_column: str,
    ):
        super().__init__()
        self.id_column = id_column
        self.dp = dp

    def layout(self):
        # Setup pivots
        dp_pivot = self.pivot(self.dp)

        # Setup stores
        against = Store("")
        selection = Store([])

        selection_dp = make_selection_dp(dp_pivot, self.id_column, selection)

        # Setup components
        match_x: Component = Match(dp_pivot, against=against, col="label")
        match_y: Component = Match(dp_pivot, against=against)

        # Setup computation graph
        merge_derived = mk.merge(left=dp_pivot, right=selection_dp, on=self.id_column)
        sort_derived = mk.sort(dp_pivot, by=match_x.col, ascending=False)

        simple_op(against)

        gallery: Component = Table(  # noqa: F841
            sort_derived,
            edit_target=EditTarget(dp_pivot, self.id_column, self.id_column),
        )

        selected_table: Component = Table(  # noqa: F841
            merge_derived,
            edit_target=EditTarget(dp_pivot, self.id_column, self.id_column),
        )

        plot: Component = Plot(  # noqa: F841
            dp_pivot,
            selection=selection,
            x=match_x.col,
            y=match_y.col,
            x_label=match_x.text,
            y_label=match_y.text,
        )
