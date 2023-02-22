from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.formatter.base import BaseFormatter


class Website(Component):
    data: str
    height: int


class WebsiteFormatter(BaseFormatter):
    component_class: type = Website
    data_prop: str = "data"

    def __init__(self, height: int = 50):
        super().__init__(height=height)

    def _encode(self, url: str) -> str:
        return url
