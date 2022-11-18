from dataclasses import dataclass

from meerkat.interactive.graph import Store
from ..abstract import Component
import re
from typing import Callable, ClassVar, List, Optional, Tuple
import ast

from fastapi import HTTPException
import numpy as np

from meerkat.dataframe import DataFrame
from meerkat.interactive.graph import reactive
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
def set_criterion(
    df: DataFrame,
    query: str = Endpoint.EmbeddedBody(),
    against: str = Endpoint.EmbeddedBody(),
    criterion: str = Endpoint.EmbeddedBody(),
    encoder: str = Endpoint.EmbeddedBody(None),
):
    """Match a query string against a DataFrame column.

    The `dataframe_id` remains the same as the original request.
    """

    if not isinstance(df, DataFrame):
        raise HTTPException(
            status_code=400, detail="`match` expects a ref containing a dataframe"
        )

    try:
        query_embedding = parse_query(query)

        criterion.set(
            MatchCriterion(
                query=query,
                against=against,
                query_embedding=query_embedding,
                name=f"match({against}, {query})",
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return criterion.__wrapped__



@dataclass
class MatchCriterion:
    against: str
    query: str
    name: str
    query_embedding: np.ndarray = None



@reactive
def compute_match_scores(df: DataFrame, criterion: MatchCriterion):
    df = df.view() 
    if criterion == None: 
        return df

    data_embedding = df[criterion.against]
    scores = (data_embedding @ criterion.query_embedding.T).squeeze()
    df[criterion.name] = scores
    return df 


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

        self.criterion: MatchCriterion = Store(None)

        on_match = set_criterion.partial(
            df=self.df,
            encoder=self.encoder,
            criterion=self.criterion,
        )
        if self.on_match is not None:
            on_match = on_match.compose(self.on_match)
        self.on_match = on_match

    @property
    def _backend_only(self):
        return ["criterion"] + super()._backend_only
    
    def __call__(self, df: DataFrame = None) -> DataFrame:
        if df is None:
            df = self.df

        return compute_match_scores(df, self.criterion)
    

