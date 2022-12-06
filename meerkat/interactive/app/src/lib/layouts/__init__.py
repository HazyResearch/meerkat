from typing import Sequence
from meerkat.interactive.app.src.lib.component.abstract import Component


class Flex(Component):

    components: Sequence[Component]
    classes: str = "flex-col"


class Grid(Component):

    components: Sequence[Component]
    classes: str = "grid-cols-2"


class Div(Component):

    component: Component
    classes: str = ""


class AutoLayout(Component):

    components: Sequence[Component]
    classes: str = ""


class RowLayout(Grid):

    classes: str = "grid-cols-1"


class ColumnLayout(Flex):

    classes: str = "flex-row"
