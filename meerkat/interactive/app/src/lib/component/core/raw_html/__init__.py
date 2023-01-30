import base64
from io import BytesIO

from meerkat.columns.deferred.base import DeferredCell
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.formatter.base import Formatter, Variant


class RawHTML(Component):
    html: str
    view: str = "full"


class HTMLFormatter(Formatter):

    component_class: type = RawHTML
    data_prop: str = "html"

    variants: dict = {
        "small": Variant(
            props={"view": "thumbnail"},
            encode_kwargs={},
        ),
        "tiny": Variant(
            props={"view": "logo"},
            encode_kwargs={},
        )
    }
    
    def _encode(self, data: str, thumbnail: bool = False) -> str:
        return data

    def html(self, cell: str) -> str:
        return "HTML"
