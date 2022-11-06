import functools
import re
from typing import Callable, List, Optional, Tuple

from fastapi import APIRouter, Body, HTTPException

import meerkat as mk
from meerkat.dataframe import DataFrame
from meerkat.interactive import Modification, trigger
from meerkat.interactive.graph import BoxModification, StoreModification
from meerkat.state import state

EmbeddedBody = functools.partial(Body, embed=True)

router = APIRouter(
    prefix="/ops",
    tags=["ops"],
    responses={404: {"description": "Not found"}},
)
_SUPPORTED_MATCH_OPS = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "**": lambda x, y: x ** y,
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


@router.post("/{box_id}/match/")
def match(
    box_id: str, input: str = Body(), query: str = Body(), col_out: str = Body(None)
) -> List[Modification]:
    """Match a query string against a DataFrame column.

    The `dataframe_id` remains the same as the original request.
    """
    box = state.identifiables.get(group="boxes", id=box_id)

    df = box.obj
    if not isinstance(df, DataFrame):
        raise HTTPException(
            status_code=400, detail="`match` expects a box containing a dataframe"
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
        print(e)
        raise HTTPException(status_code=404, detail=str(e))

    modifications = [
        BoxModification(id=box_id, scope=match_columns),
    ]

    if col_out is not None:
        col_out_store = state.identifiables.get(group="stores", id=col_out)
        col_out_store.value = match_columns[
            0
        ]  # TODO: match probably will only need to return one column in the future
        modifications.append(
            StoreModification(id=col_out, value=col_out_store.value),
        )

    modifications = trigger(modifications)
    return modifications
    # return SchemaResponse(
    #     id=pivot.dataframe_id,
    #     columns=_get_column_infos(df, match_columns)
    # )


@router.post("/{pivot_id}/add/")
def add_column(pivot_id: str, column: str = EmbeddedBody()):
    pivot = state.identifiables.get(group="boxes", id=pivot_id)
    df = pivot.obj

    import numpy as np

    df[column] = np.zeros(len(df))

    modifications = [Modification(box_id=pivot_id, scope=[column])]
    modifications = trigger(modifications)
    return modifications
