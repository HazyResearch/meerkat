from typing import List

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint, EndpointProperty
from meerkat.tools.utils import classproperty


class ScatterPlot(Component):
    df: DataFrame
    x: str
    y: str
    hue: str = None
    title: str = ""
    selected: List[str] = []

    on_click: EndpointProperty = None
    on_select: Endpoint = None

    def __init__(
        self,
        df: DataFrame,
        *,
        x: str,
        y: str,
        hue: str = None,
        title: str = "",
        selected: List[str] = [],
        on_click: EndpointProperty = None,
        on_select: Endpoint = None,
    ):
        super().__init__(
            df=df,
            x=x,
            y=y,
            hue=hue,
            title=title,
            selected=selected,
            on_click=on_click,
            on_select=on_select,
        )

    @classproperty
    def namespace(cls):
        return "plotly"

    def _get_ipython_height(self):
        return "400px"
