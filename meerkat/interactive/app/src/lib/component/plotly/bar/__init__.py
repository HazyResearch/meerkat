from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.tools.utils import classproperty


class BarPlot(Component):
    df: DataFrame
    x: str
    y: str
    title: str
    on_click: EndpointProperty = None

    def __init__(
        self,
        df: DataFrame,
        *,
        x: str,
        y: str,
        title: str = "",
        on_click: EndpointProperty = None,
    ):
        if len(df[x].unique()) != len(df):
            df = df.groupby(x)[[x, y]].mean()
        super().__init__(df=df, x=x, y=y, on_click=on_click, title=title)

    @classproperty
    def namespace(cls):
        return "plotly"
