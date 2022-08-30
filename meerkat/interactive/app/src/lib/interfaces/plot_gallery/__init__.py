import meerkat as mk
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.app.src.lib.component.gallery import Gallery
from meerkat.interactive.app.src.lib.component.match import Match
from meerkat.interactive.app.src.lib.component.plot import Plot
from meerkat.interactive.app.src.lib.component.table import EditTarget, Table
from meerkat.interactive.graph import Pivot, Store, head, interface_op
from meerkat.state import state

from ..abstract import Interface, InterfaceConfig


@interface_op
def simple_op(col: str):
    return col + "!"


class PlotInterface(Interface):
    def __init__(
        self,
        dp: mk.DataPanel,
        id_column: str,
    ):
        super().__init__()
        self.id_column = id_column

        self.pivots = []
        self.dp = dp

        # with context
        self._layout()

    def pivot(self, obj):
        # checks whether the object is valid pivot

        pivot = Pivot(obj)
        self.pivots.append(pivot)

        return pivot

    def _layout(self):
        # Setup pivots
        dp_pivot = self.pivot(self.dp)
        selection_dp = mk.DataPanel({self.id_column: []})
        selection_pivot = self.pivot(selection_dp)

        # Setup stores
        against = Store("")

        # Setup components
        match_x: Component = Match(dp_pivot, against=against, col="label")
        match_y: Component = Match(dp_pivot, against=against)

        # Setup computation graph
        # merge_derived: Derived = mk.merge(
        #     left=dp_pivot, right=selection_pivot, on=self.id_column
        # )
        sort_derived = mk.sort(dp_pivot, by=match_x.col, ascending=False)

        result = simple_op(against)

        gallery: Component = Table(
            sort_derived,
            edit_target=EditTarget(dp_pivot, self.id_column, self.id_column),
        )

        plot: Component = Plot(
            dp_pivot,
            selection=selection_pivot,
            x=match_x.col,
            y=match_y.col,
            x_label=match_x.text,
            y_label=match_y.text,
        )

        # TODO: make this more magic
        self.components = [match_x, match_y, plot, gallery]

    @property
    def config(self):
        return InterfaceConfig(
            pivots=[pivot.config for pivot in self.pivots],
            components=[component.config for component in self.components],
        )
