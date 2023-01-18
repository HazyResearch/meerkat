from meerkat.interactive.app.src.lib.component.abstract import (
    Component,
)


class Image(Component):

    data: str
    height: str = "100%"
    width: str = "100%"
    layout: str = "object-cover"
