import meerkat as mk
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.app.src.lib.component.match import Match
from meerkat.interactive.app.src.lib.component.table import EditTarget, Table
from meerkat.interactive.graph import reactive

from ..abstract import Interface, InterfaceConfig


@reactive
def simple_op(col: str):
    return col + "!"


class MatchTableInterface(Interface):
    def __init__(self, df: mk.DataFrame, id_column: str, against: str = None):
        super().__init__()
        self.id_column = id_column
        self.against = against

        self.pivots = []
        self.df = df

        # with context
        self._layout()

    def _layout(self):
        # Setup pivots
        df_pivot = self.pivot(self.df)

        # Setup components
        match: Component = Match(df_pivot, against=self.against, col=self.id_column)

        sort_derived = mk.sort(df_pivot, by=match.col, ascending=False)

        gallery: Component = Table(
            sort_derived,
            edit_target=EditTarget(df_pivot, self.id_column, self.id_column),
        )

        # TODO: make this more magic
        self.components = [match, gallery]

    @property
    def config(self):
        return InterfaceConfig(
            pivots=[pivot.config for pivot in self.pivots],
            components=[component.config for component in self.components],
            name="MatchTable",
        )
