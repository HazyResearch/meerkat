from typing import Any

from meerkat.interactive.app.src.lib.component.abstract import Component, Slottable


class Flex(Slottable, Component):

    classes: str = "flex-col"


class Grid(Slottable, Component):

    classes: str = "grid-cols-2"


class Div(Slottable, Component):

    classes: str = ""


class AutoLayout(Slottable, Component):

    classes: str = ""


class RowLayout(Grid):

    classes: str = "grid-cols-1"


class ColumnLayout(Flex):

    classes: str = "flex-row"


class Brace(Component):

    data: Any
