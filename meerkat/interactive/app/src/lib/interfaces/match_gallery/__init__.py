from typing import List

import meerkat as mk
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.app.src.lib.component.filter import Filter
from meerkat.interactive.app.src.lib.component.gallery import Gallery
from meerkat.interactive.app.src.lib.component.match import Match
from meerkat.interactive.app.src.lib.component.table import EditTarget
from meerkat.interactive.graph import Pivot, interface_op

from ..abstract import Interface, InterfaceConfig


@interface_op
def simple_op(col: str):
    return col + "!"


class MatchGalleryInterface(Interface):
    def __init__(
        self,
        dp: mk.DataPanel,
        id_column: str,
        main_column: str,
        tag_columns: List[str],
        against: str = None,
    ):
        super().__init__()
        self.id_column = id_column
        self.against = against
        self.main_column = main_column
        self.tag_columns = tag_columns

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

        # Setup components
        match: Component = Match(dp_pivot, against=self.against, col="label")

        filter: Component = Filter(dp_pivot)

        sort_derived = mk.sort(filter.derived(), by=match.col, ascending=False)

        gallery: Component = Gallery(
            sort_derived,
            main_column=self.main_column,
            tag_columns=self.tag_columns,
            edit_target=EditTarget(dp_pivot, self.id_column, self.id_column),
        )

        # TODO: make this more magic
        # FIXME: Do these have to be ordered?
        self.components = [match, filter, gallery]

    @property
    def config(self):
        return InterfaceConfig(
            pivots=[pivot.config for pivot in self.pivots],
            components=[component.config for component in self.components],
        )
