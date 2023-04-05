from typing import Optional

import plotly.express as px

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.tools.utils import classproperty

from ...abstract import Component


class ScatterMapbox(Component):
    df: DataFrame
    lat: str
    lon: str
    title: Optional[str] = None
    on_click: EndpointProperty = None
    json_desc: str = ""

    def __init__(
        self,
        df: DataFrame,
        *,
        lat: str,
        lon: str,
        title: Optional[str] = None,
        on_click: EndpointProperty = None,
    ):
        fig = px.scatter_mapbox(
            df.to_pandas(),
            lat=lat,
            lon=lon,
            # hover_name="Make",
            # hover_data=["Model Year", "Make", "Model", "Electric Vehicle Type"],
            # color="Electric Range",
            # color_continuous_scale=[(0, "orange"), (1, "red")],
            # size="Electric Range",
            zoom=8,
            height=800,
            width=800,
            title=title,
            mapbox_style="open-street-map",
        )

        super().__init__(
            df=df, lat=lat, lon=lon, json_desc=fig.to_json(), on_click=on_click
            # df=df, lat=lat, lon=lon, json_desc=fig.to_json(), on_click=on_click
        )

    @classproperty
    def namespace(cls):
        return "plotly"

    def _get_ipython_height(self):
        return "800px"
