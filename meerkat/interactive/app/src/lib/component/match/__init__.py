from dataclasses import dataclass
from ..abstract import Component
import re
from typing import Callable, List, Optional, Tuple

from fastapi import HTTPException

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import Endpoint, endpoint

# from meerkat.interactive.modification import Modification


_SUPPORTED_MATCH_OPS = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "**": lambda x, y: x**y,
}

def _parse_concat(query: str) -> List[str]:
    
    pass 

def _regex_parse_query(query: str) -> Tuple[List[str], Optional[Callable]]:
    """Parse a query string into a list of columns and operations to perform.

    This parser is not exhaustive and should only be used to parse simple
    operations (e.g. addition, subtraction, division, multiplication) between
    two columns.

    To run an operation between two query results, each query must be wrapped
    in double quotation marks (e.g. "query1" + "query2"). Single quotation marks
    will be ignored.
    """
    def _process_queries(queries):
        # Remove quotation marks from queries.
        return [q.replace('"', "") for q in queries]

    queries = re.findall('"[^"]*"', query)
    if not queries:
        return [query], None

    # Remove quotation marks from queries.
    if len(queries) == 1:
        return _process_queries(queries), None

    if len(queries) != 2:
        raise ValueError(
            f"Invalid query string - expected two columns, got {len(queries)}"
        )
    op = query.replace(queries[0], "").replace(queries[1], "").replace('"', "").strip()
    if op not in _SUPPORTED_MATCH_OPS:
        raise ValueError(f"Invalid query string - unsupported operation {op}")
    queries = _process_queries(queries)
    return queries, op


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


@endpoint
def match(
    df: DataFrame,
    against: str = Endpoint.EmbeddedBody(),
    query: str = Endpoint.EmbeddedBody(),
    encoder: str = Endpoint.EmbeddedBody(),
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

        # TODO: This is remarkably hacky. Needs a much better fix soon.
        sub_queries = _parse_concat(query)
        queries_to_concat = []
        for sub_query in sub_queries:
            subsub_queries, op = mk.PandasSeriesColumn(_regex_parse_query(sub_query))
            subsubquery_embs = mk.embed(
                data=subsub_queries, 
                encoder=encoder, 
                num_workers=0, 
                pbar=False
            )
            if len(subsubquery_embs) > 1:
                subquery_emb = _SUPPORTED_MATCH_OPS[op](
                    subsubquery_embs[0], subsubquery_embs[1]
                )
            else: 
                subquery_emb = subsubquery_embs[0]
            queries_to_concat.apend(subquery_emb)
        query_emb = mk.concat(queries_to_concat)
        
        scores = data_embedding @ query_emb.T
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
