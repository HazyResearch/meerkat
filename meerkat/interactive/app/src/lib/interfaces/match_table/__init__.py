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


class MatchTableInterface(Interface):
    def __init__(self, dp: mk.DataPanel, id_column: str, against: str = None):
        super().__init__()
        self.id_column = id_column
        self.against = against

        self.pivots = []
        self.dp = dp

        # with context
        self._layout()

    def _layout(self):
        # Setup pivots
        dp_pivot = self.pivot(self.dp)

        # Setup components
        match: Component = Match(dp_pivot, against=self.against, col=self.id_column)

        sort_derived = mk.sort(dp_pivot, by=match.col, ascending=False)

        gallery: Component = Table(
            sort_derived,
            edit_target=EditTarget(dp_pivot, self.id_column, self.id_column),
        )

        # TODO: make this more magic
        self.components = [match, gallery]

    @property
    def config(self):
        return InterfaceConfig(
            pivots=[pivot.config for pivot in self.pivots],
            components=[component.config for component in self.components],
        )
