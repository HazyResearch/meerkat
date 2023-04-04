from typing import Optional

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.tools.utils import classproperty

from ...abstract import Component


class Histogram(Component):
    df: DataFrame
    x: str
    color: Optional[str]
    title: Optional[str]
    nbins: Optional[int]
    on_click: EndpointProperty = None

    def __init__(
        self,
        df: DataFrame,
        *,
        x: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        nbins: Optional[int] = None,
        on_click: EndpointProperty = None,
    ):
        super().__init__(
            df=df, x=x, on_click=on_click, color=color, title=title, nbins=nbins
        )

    @classproperty
    def namespace(cls):
        return "plotly"
