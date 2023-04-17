from typing import List, Union
import json

from meerkat import env
from meerkat.dataframe import DataFrame
from meerkat.interactive.graph import Store
from meerkat.dataframe import DataFrame
from meerkat.interactive.graph import reactive
from meerkat.interactive.endpoint import Endpoint, EndpointProperty, endpoint
from meerkat.interactive.event import EventInterface
from meerkat.tools.lazy_loader import LazyLoader
from meerkat.tools.utils import classproperty, requires

from ...abstract import Component

px = LazyLoader("plotly.express")


class OnRelayoutInterface(EventInterface):
    """Defines the interface for an event.

    Subclass this to define the interface for a new event type.
    The class will specify the keyword arguments returned by an event from the
    frontend to any endpoint that has subscribed to it.

    All endpoints that are expected to receive an event of this type should
    ensure they have a signature that matches the keyword arguments defined
    in this class.
    """

    x_range: List[float]
    y_range: List[float]




class DynamicScatter(Component):
    df: DataFrame
    keyidxs: List[Union[str, int]]
    on_click: EndpointProperty = None
    on_relayout: EndpointProperty = None
    selected: List[str] = []
    on_select: Endpoint = None

    data: str = ""
    layout: str = ""

    filtered_df: DataFrame = None


    @requires("plotly.express")
    def __init__(
        self,
        df: DataFrame,
        *,
        x=None,
        y=None,
        color=None,
        max_points: int = 1_000,
        on_click: EndpointProperty = None,
        on_relayout: EndpointProperty = None,
        selected: List[str] = [],
        on_select: Endpoint = None,
        **kwargs,
    ):
        """See
        https://plotly.com/python-api-reference/generated/plotly.express.scatter.html
        for more details."""

        if not env.is_package_installed("plotly"):
            raise ValueError(
                "Plotly components require plotly. Install with `pip install plotly`."
            )

        if df.primary_key_name is None:
            raise ValueError("Dataframe must have a primary key")

        fig_df = df if len(df) <= max_points else df.sample(max_points)
        fig = px.scatter(fig_df.to_pandas(), x=x, y=y, color=color, **kwargs)
        
        json_desc = json.loads(fig.to_json())

        axis_range = Store({
            "x0": None, 
            "x1": None, 
            "y0": None,
            "y1": None
        })

        @endpoint()
        def on_relayout(axis_range: Store[dict], x_range, y_range):
            axis_range.set({
                "x0": x_range[0],
                "x1": x_range[1],
                "y0": y_range[0],
                "y1": y_range[1]
            })

        @reactive()
        def filter_df(df: DataFrame, axis_range: dict):
            df = df.view()
            if axis_range["x0"] is not None:
                df = df[(axis_range["x0"]  < df[x])]
            if axis_range["x1"] is not None:
                df = df[df[x] < axis_range["x1"]]
            if axis_range["y0"] is not None:
                df = df[(axis_range["y0"] < df[y])]
            if axis_range["y1"] is not None:
                df = df[df[y] < axis_range["y1"]]            
            return df
        
        @reactive()
        def sample_df(df: DataFrame):
            df = df.view()
            if len(df) > max_points:
                df = df.sample(max_points)
            return df
        
        @reactive()
        def compute_plotly(df: DataFrame):
            fig = px.scatter(df.to_pandas(), x=x, y=y, color=color, **kwargs)
            return json.dumps(json.loads(fig.to_json())["data"])
        df.mark()
        filtered_df = filter_df(df=df, axis_range=axis_range)
        sampled_df = sample_df(df=filtered_df)
        plotly_data = compute_plotly(df=sampled_df)
        super().__init__(
            df=df,
            keyidxs=df.primary_key.values.tolist(),
            on_click=on_click,
            selected=selected,
            on_select=on_select,
            on_relayout=on_relayout.partial(axis_range=axis_range),
            data=plotly_data,
            layout=json.dumps(json_desc["layout"]),
            filtered_df=filtered_df,
        )

    @classproperty
    def namespace(cls):
        return "plotly"

    def _get_ipython_height(self):
        return "800px"
