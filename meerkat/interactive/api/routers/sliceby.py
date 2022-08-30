from multiprocessing.sharedctypes import Value
from time import sleep
from typing import Any, Dict, List, Union

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

import meerkat as mk
from meerkat.datapanel import DataPanel
from meerkat.interactive.formatter import BasicFormatter
from meerkat.ops.sliceby.sliceby import SliceBy, SliceKey
from meerkat.state import state

from .datapanel import RowsResponse, _get_column_infos


def get_sliceby(sliceby_id: str) -> SliceBy:
    try:

        datapanel = state.identifiables.slicebys[sliceby_id]

    except KeyError:
        raise HTTPException(
            status_code=404, detail="No datapanel with id {}".format(sliceby_id)
        )
    return datapanel


router = APIRouter(
    prefix="/sliceby",
    tags=["sliceby"],
    responses={404: {"description": "Not found"}},
)


class InfoResponse(BaseModel):
    id: str
    type: str
    n_slices: int
    slice_keys: List[SliceKey]

    class Config:
        # need a smart union here to avoid casting ints to strings in SliceKey
        # https://pydantic-docs.helpmanual.io/usage/types/#unions
        smart_union = True


@router.get("/{sliceby_id}/info/")
def get_info(sliceby_id: str) -> InfoResponse:
    if sliceby_id == "test":
        sliceby_id = list(state.identifiables.slicebys.keys())[0]
    sb = get_sliceby(sliceby_id=sliceby_id)

    return InfoResponse(
        id=sliceby_id,
        type=type(sb).__name__,
        n_slices=len(sb),
        slice_keys=sb.slice_keys,
    )


class SliceByRowsRequest(BaseModel):
    # TODO (sabri): add support for data validation
    slice_key: SliceKey
    start: int = None
    end: int = None
    indices: List[int] = None
    columns: List[str] = None

    class Config:
        smart_union = True


@router.post("/{sliceby_id}/rows/")
def get_rows(
    sliceby_id: str,
    request: SliceByRowsRequest,
) -> RowsResponse:
    """Get rows from a DataPanel as a JSON object."""
    sb = get_sliceby(sliceby_id=sliceby_id)
    slice_key = request.slice_key
    full_length = sb.get_slice_length(slice_key)
    column_infos = _get_column_infos(sb.data, request.columns)

    sb = sb[[info.name for info in column_infos]]

    if request.indices is not None:
        dp = sb.slice[slice_key, request.indices]
        indices = request.indices
    elif request.start is not None:
        if request.end is None:
            request.end = len(dp)
        dp = sb.slice[slice_key, request.start : request.end]
        indices = list(range(request.start, request.end))
    else:
        raise ValueError()

    rows = []
    for row in dp.lz:
        rows.append(
            [dp[info.name].formatter.encode(row[info.name]) for info in column_infos]
        )
    return RowsResponse(
        column_infos=column_infos,
        rows=rows,
        full_length=full_length,
        indices=indices,
    )


@router.post("/{sliceby_id}/aggregate/")
def aggregate(
    sliceby_id: str,
    aggregation_id: str = Body(None),
    aggregation: str = Body(None),
    accepts_dp: bool = Body(False),
    columns: List[str] = Body(None),
) -> Dict:
    # print("agg_id" + aggregation_id)
    sliceby = state.identifiables.get(group="slicebys", id=sliceby_id)

    if columns is not None:
        sliceby = sliceby[columns]

    if (aggregation_id is None) == (aggregation is None):
        raise HTTPException(
            status_code=400,
            detail="Must specify either aggregation_id or aggregation",
        )

    if aggregation_id is not None:
        aggregation = state.identifiables.get(id=aggregation_id, group="aggregations")
        value = sliceby.aggregate(aggregation, accepts_dp=accepts_dp)

    else:
        if aggregation not in ["mean", "sum", "min", "max"]:
            raise HTTPException(
                status_code=400, detail=f"Invalid aggregation {aggregation}"
            )
        value = sliceby.aggregate(aggregation)

    # convert to dict format for output
    df = value.to_pandas().set_index(sliceby.by)
    dct = df.applymap(BasicFormatter().encode).to_dict()
    return dct
