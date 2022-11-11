from typing import Dict, List

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from meerkat.interactive.formatter import BasicFormatter
from meerkat.ops.sliceby.sliceby import SliceBy, SliceKey
from meerkat.state import state

from .dataframe import RowsResponse, _get_column_infos


def get_sliceby(sliceby_id: str) -> SliceBy:
    try:

        dataframe = state.identifiables.slicebys[sliceby_id]

    except KeyError:
        raise HTTPException(
            status_code=404, detail="No dataframe with id {}".format(sliceby_id)
        )
    return dataframe


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


@router.get("/{ref_id}/info/")
def get_info(ref_id: str) -> InfoResponse:
    ref = state.identifiables.get(group="refs", id=ref_id)
    sb = ref.obj

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


@router.post("/{ref_id}/rows/")
def get_rows(
    ref_id: str,
    request: SliceByRowsRequest,
) -> RowsResponse:
    """Get rows from a DataFrame as a JSON object."""
    ref = state.identifiables.get(group="refs", id=ref_id)
    sb = ref.obj

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
    for row in df.lz:
        rows.append(
            [df[info.name].formatter.encode(row[info.name]) for info in column_infos]
        )
    return RowsResponse(
        column_infos=column_infos,
        rows=rows,
        full_length=full_length,
        indices=indices,
    )


@router.post("/{ref_id}/aggregate/")
def aggregate(
    ref_id: str,
    aggregation_id: str = Body(None),
    aggregation: str = Body(None),
    accepts_df: bool = Body(False),
    columns: List[str] = Body(None),
) -> Dict:
    ref = state.identifiables.get(group="refs", id=ref_id)
    sb = ref.obj

    sliceby = sb
    if columns is not None:
        sliceby = sliceby[columns]

    if (aggregation_id is None) == (aggregation is None):
        raise HTTPException(
            status_code=400,
            detail="Must specify either aggregation_id or aggregation",
        )

    if aggregation_id is not None:
        aggregation = state.identifiables.get(id=aggregation_id, group="aggregations")
        value = sliceby.aggregate(aggregation, accepts_df=accepts_df)

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
