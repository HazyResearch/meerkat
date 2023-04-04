from typing import Optional

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.tools.utils import classproperty

from ...abstract import Component


class ScatterMapbox(Component):
    df: DataFrame
    lat: str
    lon: str
    title: Optional[str] = None
    on_click: EndpointProperty = None

    def __init__(
        self,
        df: DataFrame,
        *,
        lat: str,
        lon: str,
        title: Optional[str] = None,
        on_click: EndpointProperty = None,
    ):
        super().__init__(df=df, lat=lat, lon=lon, on_click=on_click, title=title)

    @classproperty
    def namespace(cls):
        return "plotly"
