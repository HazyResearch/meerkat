import functools
from typing import Any, Dict, List, Union

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

import meerkat as mk
from meerkat.datapanel import DataPanel
from meerkat.interactive import Modification, trigger
from meerkat.interactive.graph import BoxModification, StoreModification
from meerkat.state import state

from ....tools.utils import convert_to_python
from .datapanel import SchemaResponse

EmbeddedBody = functools.partial(Body, embed=True)

router = APIRouter(
    prefix="/ops",
    tags=["ops"],
    responses={404: {"description": "Not found"}},
)


@router.post("/{box_id}/match/")
def match(
    box_id: str, input: str = Body(), query: str = Body(), col_out: str = Body(None)
) -> List[Modification]:
    """Match a query string against a DataPanel column.

    The `datapanel_id` remains the same as the original request.
    """
    box = state.identifiables.get(group="boxes", id=box_id)

    dp = box.obj
    if not isinstance(dp, DataPanel):
        raise HTTPException(
            status_code=400, detail="`match` expects a box containing a datapanel"
        )

    try:
        dp, match_columns = mk.match(
            data=dp, query=query, input=input, return_column_names=True
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

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
    # return SchemaResponse(id=pivot.datapanel_id, columns=_get_column_infos(dp, match_columns))


@router.post("/{pivot_id}/add/")
def add_column(pivot_id: str, column: str = EmbeddedBody()):
    pivot = state.identifiables.get(group="boxes", id=pivot_id)
    dp = pivot.obj

    import numpy as np

    dp[column] = np.zeros(len(dp))

    modifications = [Modification(box_id=pivot_id, scope=[column])]
    modifications = trigger(modifications)
    return modifications
