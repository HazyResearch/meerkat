from typing import List, Union

from meerkat import env
from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import Endpoint, EndpointProperty
from meerkat.tools.lazy_loader import LazyLoader
from meerkat.tools.utils import classproperty, requires

from ...abstract import Component

px = LazyLoader("plotly.express")


class Area(Component):
    df: DataFrame
    keyidxs: List[Union[str, int]]
    on_click: EndpointProperty = None
    selected: List[str] = []
    on_select: Endpoint = None

    json_desc: str = ""

    @requires("plotly.express")
    def __init__(
        self,
        df: DataFrame,
        *,
        x=None,
        y=None,
        line_group=None,
        color=None,
        on_click: EndpointProperty = None,
        selected: List[str] = [],
        on_select: Endpoint = None,
        **kwargs,
    ):
        """See https://plotly.com/python-api-
        reference/generated/plotly.express.area.html for more details."""

        if not env.is_package_installed("plotly"):
            raise ValueError(
                "Plotly components require plotly. Install with `pip install plotly`."
            )

        if df.primary_key_name is None:
            raise ValueError("Dataframe must have a primary key")

        fig = px.area(
            df.to_pandas(),
            x=x,
            y=y,
            line_group=line_group,
            color=color,
            **kwargs,
        )

        super().__init__(
            df=df,
            keyidxs=df.primary_key.values.tolist(),
            on_click=on_click,
            selected=selected,
            on_select=on_select,
            json_desc=fig.to_json(),
        )

    @classproperty
    def namespace(cls):
        return "plotly"

    def _get_ipython_height(self):
        return "800px"
