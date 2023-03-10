from meerkat.interactive.app.src.lib.component.abstract import Component


class RawHTML(Component):
    html: str
    view: str = "full"
    sanitize: bool = True
    classes: bool = "rounded-md shadow-md"
