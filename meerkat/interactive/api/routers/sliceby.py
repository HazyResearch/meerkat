from typing import Dict, List

from fastapi import HTTPException
from pydantic import BaseModel

from meerkat.interactive.endpoint import Endpoint, endpoint
from meerkat.interactive.formatter import BasicFormatter
from meerkat.ops.sliceby.sliceby import SliceBy, SliceKey
from meerkat.state import state

from .dataframe import RowsResponse, _get_column_infos


class InfoResponse(BaseModel):
    id: str
    type: str
    n_slices: int
    slice_keys: List[SliceKey]

    class Config:
        # need a smart union here to avoid casting ints to strings in SliceKey
        # https://pydantic-docs.helpmanual.io/usage/types/#unions
        smart_union = True


@endpoint(prefix="/sliceby", route="/{sb}/info/", method="GET")
def get_info(sb: SliceBy) -> InfoResponse:
    # FIXME Make sure the SliceBy object works for this endpoint
    return InfoResponse(
        id=sb.id,
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


@endpoint(prefix="/sliceby", route="/{sb}/rows/")
def get_rows(
    sb: SliceBy,
    request: SliceByRowsRequest,
) -> RowsResponse:
    """Get rows from a DataFrame as a JSON object."""
    # FIXME Make sure the SliceBy object works for this endpoint
    slice_key = request.slice_key
    full_length = sb.get_slice_length(slice_key)
    column_infos = _get_column_infos(sb.data, request.columns)

    sb = sb[[info.name for info in column_infos]]

    if request.indices is not None:
        df = sb.slice[slice_key, request.indices]
        indices = request.indices
    elif request.start is not None:
        df = sb.slice[slice_key, request.start : request.end]
        indices = list(range(request.start, request.end))
    else:
        raise ValueError()

    rows = []
    for row in df:
        rows.append(
            [df[info.name].formatter.encode(row[info.name]) for info in column_infos]
        )
    return RowsResponse(
        columnInfos=column_infos,
        rows=rows,
        fullLength=full_length,
        indices=indices,
    )


@endpoint(prefix="/sliceby", route="/{sb}/aggregate/")
def aggregate(
    sb: SliceBy,
    aggregation_id: str = Endpoint.EmbeddedBody(None),
    aggregation: str = Endpoint.EmbeddedBody(None),
    accepts_df: bool = Endpoint.EmbeddedBody(False),
    columns: List[str] = Endpoint.EmbeddedBody(None),
) -> Dict:
    # FIXME Make sure the SliceBy object works for this endpoint
    if columns is not None:
        sb = sb[columns]

    if (aggregation_id is None) == (aggregation is None):
        raise HTTPException(
            status_code=400,
            detail="Must specify either aggregation_id or aggregation",
        )

    if aggregation_id is not None:
        aggregation = state.identifiables.get(id=aggregation_id, group="aggregations")
        value = sb.aggregate(aggregation, accepts_df=accepts_df)

    else:
        if aggregation not in ["mean", "sum", "min", "max"]:
            raise HTTPException(
                status_code=400, detail=f"Invalid aggregation {aggregation}"
            )
        value = sb.aggregate(aggregation)

    # convert to dict format for output
    df = value.to_pandas().set_index(sb.by)
    dct = df.applymap(BasicFormatter().encode).to_dict()
    return dct
