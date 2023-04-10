import plotly.express as px

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.tools.utils import classproperty

from ...abstract import Component


class Icicle(Component):
    df: DataFrame
    on_click: EndpointProperty = None

    json_desc: str = ""

    def __init__(
        self,
        df: DataFrame,
        *,
        names=None,
        values=None,
        parents=None,
        path=None,
        ids=None,
        color=None,
        on_click: EndpointProperty = None,
        **kwargs,
    ):
        """See https://plotly.com/python-api-reference/generated/plotly.express.icicle.html
        for more details."""

        fig = px.icicle(
            df.to_pandas(),
            names=names,
            values=values,
            parents=parents,
            path=path,
            ids=ids,
            color=color,
            **kwargs,
        )

        super().__init__(df=df, on_click=on_click, json_desc=fig.to_json())

    @classproperty
    def namespace(cls):
        return "plotly"

    def _get_ipython_height(self):
        return "800px"
