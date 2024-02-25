from meerkat.interactive.app.src.lib.component.abstract import Component


class Image(Component):
    data: str
    classes: str = ""
    enable_zoom: bool = False
    enable_pan: bool = False
