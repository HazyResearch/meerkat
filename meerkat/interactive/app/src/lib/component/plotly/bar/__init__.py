from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.tools.utils import classproperty

from ...abstract import Component


class BarPlot(Component):
    df: DataFrame
    x: str
    y: str
    on_click: EndpointProperty = None

    @classproperty
    def namespace(cls):
        return "plotly"
