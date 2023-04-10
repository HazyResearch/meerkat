from typing import Optional

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.tools.utils import classproperty


class Bar(Component):
    df: DataFrame
    x: str
    y: str
    title: str
    orientation: str = "v"
    on_click: EndpointProperty = None

    # TODO (dean): convert to json_desc
    def __init__(
        self,
        df: DataFrame,
        *,
        x: str,
        y: str,
        title: str = "",
        orientation: str = "v",
        on_click: EndpointProperty = None,
    ):
        if df.primary_key_name is None:
            raise ValueError("Dataframe must have a primary key")
        if len(df[x].unique()) != len(df):
            df = df.groupby(x)[[x, y]].mean()
            df.create_primary_key("id")
        super().__init__(
            df=df,
            x=x,
            y=y,
            title=title,
            orientation=orientation,
            on_click=on_click,
        )

    @classproperty
    def namespace(cls):
        return "plotly"

    def _get_ipython_height(self):
        return "600px"
