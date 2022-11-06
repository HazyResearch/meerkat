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
def make_selection_df(df, id, selection):
    """An out of place operation to take a store and make it a derived
    dataframe."""
    return mk.DataFrame({id: df[id][selection]})


class PlotInterface(Interface):
    def __init__(
        self,
        df: mk.DataFrame,
        id_column: str,
    ):
        super().__init__()
        self.id_column = id_column
        self.df = df

    def layout(self):
        # Setup pivots
        df_pivot = self.pivot(self.df)

        # Setup stores
        against = Store("")
        selection = Store([])

        selection_df = make_selection_df(df_pivot, self.id_column, selection)

        # Setup components
        match_x: Component = Match(df_pivot, against=against, col="label")
        match_y: Component = Match(df_pivot, against=against)

        # Setup computation graph
        merge_derived = mk.merge(left=df_pivot, right=selection_df, on=self.id_column)
        sort_derived = mk.sort(df_pivot, by=match_x.col, ascending=False)

        simple_op(against)

        gallery: Component = Table(  # noqa: F841
            sort_derived,
            edit_target=EditTarget(df_pivot, self.id_column, self.id_column),
        )

        selected_table: Component = Table(  # noqa: F841
            merge_derived,
            edit_target=EditTarget(df_pivot, self.id_column, self.id_column),
        )

        plot: Component = Plot(  # noqa: F841
            df_pivot,
            selection=selection,
            x=match_x.col,
            y=match_y.col,
            x_label=match_x.text,
            y_label=match_y.text,
        )
