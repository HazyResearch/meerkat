from multiprocessing.sharedctypes import Value
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import meerkat as mk
from meerkat.datapanel import DataPanel
from meerkat.state import state


def get_datapanel(datapanel_id: str):
    try:
        if datapanel_id == "test-imagenette":
            return mk.get("imagenette", version="320px")

        datapanel = state.identifiables.datapanels[datapanel_id]

    except KeyError:
        raise HTTPException(
            status_code=404, detail="No datapanel with id {}".format(datapanel_id)
        )
    return datapanel


router = APIRouter(
    prefix="/dp",
    tags=["dp"],
    responses={404: {"description": "Not found"}},
)


class ColumnInfoResponse(BaseModel):

    name: str
    type: str
    cell_component: str
    cell_props: Dict[str, Any]


@router.get("/{datapanel_id}/column_info/")
def get_column_info(
    datapanel_id: int,
):
    dp = get_datapanel(datapanel_id)
    return _get_column_infos(dp, dp.columns)


def _get_column_infos(dp: DataPanel, columns: List[str] = None):
    if columns is None:
        columns = dp.columns

    return [
        ColumnInfoResponse(
            name=col,
            type=str(type(dp[col])),
            cell_component=dp[col].formatter.cell_component,
            cell_props=dp[col].formatter.cell_props,
        )
        for col in columns
    ]


class DataPanelResponse(BaseModel):
    column_infos: List[ColumnInfoResponse]
    indices: List[int] = None
    rows: List[List[Any]]
    full_length: int


class DataPanelRequest(BaseModel):
    # TODO (sabri): add support for data validation
    start: int = None
    end: int = None
    indices: List[int] = None
    columns: List[str] = None


@router.post("/{datapanel_id}/rows/")
def get_rows(
    datapanel_id: str,
    request: DataPanelRequest,
) -> DataPanelResponse:
    """
    Get rows from a DataPanel as a JSON object.
    """
    dp = get_datapanel(datapanel_id)
    full_length = len(dp)
    column_infos = _get_column_infos(dp, request.columns)

    dp = dp[[info.name for info in column_infos]]

    if request.indices is not None:
        dp = dp.lz[request.indices]
        indices = request.indices
    elif request.start is not None:
        if request.end is None:
            request.end = len(dp)
        dp = dp.lz[request.start : request.end]
        indices = list(range(request.start, request.end))
    else:
        raise ValueError()

    rows = []
    for row in dp.lz:
        rows.append(
            [dp[info.name].formatter.encode(row[info.name]) for info in column_infos]
        )
    return DataPanelResponse(
        column_infos=column_infos,
        rows=rows,
        full_length=full_length,
        indices=indices,
    )
