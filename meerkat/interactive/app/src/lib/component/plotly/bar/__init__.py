from meerkat.dataframe import DataFrame
from meerkat.mixins.identifiable import classproperty
from ...abstract import Component
from meerkat.interactive.endpoint import EndpointProperty


class BarPlot(Component):

    df: DataFrame
    x: str
    y: str
    on_click: EndpointProperty = None

    @classproperty
    def namespace(cls):
        return "plotly"
