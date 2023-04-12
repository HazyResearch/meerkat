from typing import List, Union

import plotly.express as px

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import Endpoint, EndpointProperty
from meerkat.tools.utils import classproperty

from ...abstract import Component


class DensityMapbox(Component):
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
        lat=None,
        lon=None,
        z=None,
        on_click: EndpointProperty = None,
        selected: List[str] = [],
        on_select: Endpoint = None,
        **kwargs,
    ):
        """See https://plotly.com/python-api-reference/generated/plotly.express.density_mapbox.html
        for more details."""

        if df.primary_key_name is None:
            raise ValueError("Dataframe must have a primary key")

        fig = px.density_mapbox(df.to_pandas(), lat=lat, lon=lon, z=z, **kwargs)

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
