import re
from typing import Callable, List, Optional, Tuple

from fastapi import HTTPException

import meerkat as mk
from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import Endpoint, endpoint
from meerkat.interactive.graph import Store, trigger
from meerkat.interactive.modification import (
    DataFrameModification,
    Modification,
    StoreModification,
)

_SUPPORTED_MATCH_OPS = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "**": lambda x, y: x**y,
}


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


@endpoint(prefix="/ops", route="/{df}/match/")
def match(
    df: DataFrame,
    input: str = Endpoint.EmbeddedBody(),
    query: str = Endpoint.EmbeddedBody(),
    col_out: Store = Endpoint.EmbeddedBody(None),
) -> List[Modification]:
    """Match a query string against a DataFrame column.

    The `dataframe_id` remains the same as the original request.
    """
    if not isinstance(df, DataFrame):
        raise HTTPException(
            status_code=400, detail="`match` expects a ref containing a dataframe"
        )

    try:
        # Parse the string to see if we should be running some operation on it.
        # TODO (arjundd): Support more than one op.
        # Potentially parse the string in order?
        queries, op = _regex_parse_query(query)
        df, match_columns = mk.match(
            data=df, query=queries, input=input, return_column_names=True
        )
        if len(match_columns) > 1:
            assert op is not None
            col = f"_match_{input}_{query}"
            df[col] = _SUPPORTED_MATCH_OPS[op](
                df[match_columns[0]], df[match_columns[1]]
            )
            match_columns = [col] + match_columns

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    modifications = [
        DataFrameModification(id=df.inode.id, scope=match_columns),  # FIXME: check
    ]

    if col_out is not None:
        # TODO: match probably will only need to return one column in the future
        col_out.set(match_columns[0])
        modifications.append(
            StoreModification(id=col_out.id, value=col_out),  # FIXME: check
        )

    modifications = trigger(modifications)
    return modifications


@endpoint(prefix="/ops", route="/{df}/add/")
def add_column(df: DataFrame, column: str = Endpoint.EmbeddedBody()):
    import numpy as np

    df[column] = np.zeros(len(df))
    modifications = [
        DataFrameModification(id=df.inode.id, scope=[column])
    ]  # FIXME: check
    modifications = trigger(modifications)
    return modifications
