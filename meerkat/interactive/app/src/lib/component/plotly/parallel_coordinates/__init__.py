from typing import List, Union

import plotly.express as px

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import Endpoint, EndpointProperty
from meerkat.tools.utils import classproperty

from ...abstract import Component


class ParallelCoordinates(Component):
    df: DataFrame
    keyidxs: List[Union[str, int]]
    on_click: EndpointProperty = None
    selected: List[str] = []
    on_select: Endpoint = None

    json_desc: str = ""

    def __init__(
        self,
        df: DataFrame,
        *,
        dimensions=None,
        color=None,
        on_click: EndpointProperty = None,
        selected: List[str] = [],
        on_select: Endpoint = None,
        **kwargs,
    ):
        """See https://plotly.com/python-api-reference/generated/plotly.express.parallel_categories.html
        for more details."""

        fig = px.parallel_categories(
            df.to_pandas(),
            dimensions=dimensions,
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
