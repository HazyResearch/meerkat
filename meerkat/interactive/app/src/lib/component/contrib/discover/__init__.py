from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import EndpointProperty, endpoint

from ...abstract import Component


@endpoint
def get_discover_schema(df: DataFrame):
    import meerkat as mk
    from meerkat.interactive.api.routers.dataframe import (
        SchemaResponse,
        _get_column_infos,
    )

    columns = [
        k
        for k, v in df.items()
        if isinstance(v, mk.TorchTensorColumn) and len(v.shape) == 2
    ]
    return SchemaResponse(
        id=df.id,
        columns=_get_column_infos(df, columns),
        nrows=len(df),
    )


@endpoint
def discover(df: DataFrame, by: str, target: str, pred: str):
    eb = df.explainby(
        by,
        target={"targets": target, "pred_probs": pred},
        n_slices=10,
        n_mixture_components=10,
        n_pca_components=256,
        use_cache=False,
    )
    return eb


class Discover(Component):
    df: DataFrame
    by: str
    target: str = None
    pred: str = None
    on_discover: EndpointProperty = None
    get_discover_schema: EndpointProperty = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.get_discover_schema = get_discover_schema.partial(
            df=self.df,
        )

        on_discover = discover.partial(df=self.df, target=self.target, pred=self.pred)
        if self.on_discover is not None:
            on_discover = on_discover.compose(self.on_discover)
        self.on_discover = on_discover
