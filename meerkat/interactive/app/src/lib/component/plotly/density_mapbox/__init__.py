import plotly.express as px

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.tools.utils import classproperty

from ...abstract import Component


class DensityMapbox(Component):
    df: DataFrame
    on_click: EndpointProperty = None

    json_desc: str = ""

    def __init__(
        self,
        df: DataFrame,
        *,
        lat=None,
        lon=None,
        z=None,
        on_click: EndpointProperty = None,
        **kwargs,
    ):
        """See https://plotly.com/python-api-reference/generated/plotly.express.density_mapbox.html
        for more details."""
        
        fig = px.density_mapbox(df.to_pandas(), lat=lat, lon=lon, z=z, **kwargs)

        super().__init__(df=df, on_click=on_click, json_desc=fig.to_json())

    @classproperty
    def namespace(cls):
        return "plotly"

    def _get_ipython_height(self):
        return "800px"
