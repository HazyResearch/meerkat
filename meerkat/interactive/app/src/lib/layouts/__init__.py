from meerkat.interactive.app.src.lib.component.abstract import (
    AutoComponent,
    Slottable,
)


class Flex(Slottable, AutoComponent):

    classes: str = "flex-col"


class Grid(Slottable, AutoComponent):

    classes: str = "grid-cols-2"


class Div(Slottable, AutoComponent):

    classes: str = ""


class AutoLayout(Slottable, AutoComponent):

    classes: str = ""


class RowLayout(Grid):

    classes: str = "grid-cols-1"


class ColumnLayout(Flex):

    classes: str = "flex-row"
