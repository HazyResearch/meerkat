from meerkat.dataframe import DataFrame
from ...abstract import AutoComponent
from meerkat.interactive.endpoint import EndpointProperty


class BarPlot(AutoComponent):

    df: DataFrame
    x: str
    y: str
    on_click: EndpointProperty = None