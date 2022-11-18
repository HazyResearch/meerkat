from dataclasses import dataclass
from ..abstract import Component
import re
from typing import Callable, List, Optional, Tuple
import ast

from fastapi import HTTPException
import numpy as np

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import Endpoint, endpoint

# from meerkat.interactive.modification import Modification

@endpoint
def get_match_schema(df: DataFrame, encoder: str):
    import meerkat as mk
    from meerkat.interactive.api.routers.dataframe import (
        _get_column_infos,
        SchemaResponse,
    )

    columns = [
        k
        for k, v in df.items()
        if isinstance(v, mk.NumpyArrayColumn) and len(v.shape) == 2
        # TODO: We should know the provenance of embeddings and where they came from,
        # to explicitly check whether the encoder will match it in size.
    ]
    return SchemaResponse(
        id=df.id,
        columns=_get_column_infos(df, columns),
        nrows=len(df),
    )


_SUPPORTED_BIN_OPS = {
    "Add": lambda x, y: x + y,
    "Sub": lambda x, y: x - y,
    "Mult": lambda x, y: x * y,
    "Div": lambda x, y: x / y,
    "Pow": lambda x, y: x**y,
}

_SUPPORTED_CALLS = {
    "concat": lambda *args: np.concatenate(args, axis=1),
}

def parse_query(query: str):
    return _parse_query(ast.parse(query, mode="eval").body)


def _parse_query(
    node: ast.AST,
):
    import meerkat as mk

    if isinstance(node, ast.BinOp):
        return _SUPPORTED_BIN_OPS[node.op.__class__.__name__](
            _parse_query(node.left), _parse_query(node.right)
        )
    elif isinstance(node, ast.Call):
        return _SUPPORTED_CALLS[node.func.id](*[_parse_query(arg) for arg in node.args])
    elif isinstance(node, ast.Constant):
        return mk.embed(
            data=mk.PandasSeriesColumn([node.value]),
            encoder="clip",
            num_workers=0,
            pbar=False,
        )
    else:
        raise ValueError(f"Unsupported query node {node}")


@endpoint
def match(
    df: DataFrame,
    against: str = Endpoint.EmbeddedBody(),
    query: str = Endpoint.EmbeddedBody(),
    encoder: str = Endpoint.EmbeddedBody(None),
):
    """Match a query string against a DataFrame column.

    The `dataframe_id` remains the same as the original request.
    """
    import meerkat as mk

    if not isinstance(df, DataFrame):
        raise HTTPException(
            status_code=400, detail="`match` expects a ref containing a dataframe"
        )

    try:
        data_embedding = df[against]
        query_embedding = parse_query(query)
        print(query_embedding.shape)

        scores = (data_embedding @ query_embedding.T).squeeze()
        print(scores.shape)
        col_name = f"match({against}, {query})"
        df[col_name] = scores

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return col_name


@dataclass
class Match(Component):

    df: "DataFrame"
    against: str
    text: str = ""
    encoder: str = "clip"
    on_match: Endpoint = None
    title: str = "Match"

    def __post_init__(self):
        super().__post_init__()

        # we do not add the against or the query to the partial, because we don't
        # want them to be maintained on the backend
        # if they are maintained on the backend, then a store update dispatch will
        # run on every key stroke

        self.get_match_schema = get_match_schema.partial(
            df=self.df,
            encoder=self.encoder,
        )

        on_match = match.partial(
            df=self.df,
            encoder=self.encoder,
        )
        if self.on_match is not None:
            on_match = on_match.compose(self.on_match)
        self.on_match = on_match
