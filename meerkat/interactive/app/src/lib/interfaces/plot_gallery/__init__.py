import meerkat as mk
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.app.src.lib.component.gallery import Gallery
from meerkat.interactive.app.src.lib.component.match import Match
from meerkat.interactive.app.src.lib.component.plot import Plot
from meerkat.interactive.graph import Pivot, Store, head
from meerkat.state import state

from ..abstract import Interface, InterfaceConfig


class PlotInterface(Interface):
    def __init__(
        self,
        dp: mk.DataPanel,
        id_column: str,
    ):
        super().__init__()
        self.id_column = id_column

        self.pivots = []
        self.stores = []
        self.dp = dp

        self._layout()

    def pivot(self, obj):
        # checks whether the object is valid pivot

        pivot = Pivot(obj)
        self.pivots.append(pivot)

        return pivot

    def store(self, obj):
        # checks whether the object is valid store

        store = Store(obj)
        self.stores.append(store)

        return store

    def _layout(self):

        # Setup pivots
        dp_pivot = self.pivot(self.dp)
        selection_dp = mk.DataPanel({self.id_column: []})
        selection_pivot = self.pivot(selection_dp)

        # Setup stores
        against = self.store("image")

        # Setup computation graph
        # merge_derived: Derived = mk.merge(
        #     left=dp_pivot, right=selection_pivot, on=self.id_column
        # )
        merge_derived = head(dp_pivot, n=5)

        # Setup components
        match_x: Component = Match(dp_pivot, against=against)
        match_y: Component = Match(dp_pivot, against=against)
        plot: Component = Plot(
            dp_pivot,
            selection=selection_pivot,
            x=match_x.col,
            y=match_y.col,
            x_label=match_x.text,
            y_label=match_y.text,
        )
        gallery: Component = Gallery(merge_derived)

        # TODO: make this more magic
        self.components = [match_x, match_y, plot, gallery]

    @property
    def config(self):
        return InterfaceConfig(
            pivots=[pivot.config for pivot in self.pivots],
            stores=[store.config for store in self.stores],
            components=[component.config for component in self.components],
        )
