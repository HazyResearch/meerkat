from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.tools.utils import classproperty

from ...abstract import Component


class BarPlot(Component):
    df: DataFrame
    x: str
    y: str
    on_click: EndpointProperty = None

    def __init__(
        self,
        df: DataFrame,
        *,
        x: str,
        y: str,
        on_click: EndpointProperty = None,
    ):
        super().__init__(df=df, x=x, y=y, on_click=on_click)

    @classproperty
    def namespace(cls):
        return "plotly"
