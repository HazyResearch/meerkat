import base64
from io import BytesIO
from typing import Any, List, Type

import PIL
from fastapi import APIRouter
from pydantic import BaseModel

from meerkat.columns.image_column import ImageColumn
from meerkat.columns.list_column import ListColumn
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.datapanel import DataPanel

from .interface import get_interface


router = APIRouter(
    prefix="/dp",
    tags=["dp"],
    responses={404: {"description": "Not found"}},
)


def image_to_base64(image: PIL.Image) -> str:
    with BytesIO() as buffer:
        image.save(buffer, "jpeg")
        return "data:image/jpeg;base64,{im_base_64}".format(
            im_base_64=base64.b64encode(buffer.getvalue()).decode()
        )


FORMATTERS = {
    "image": image_to_base64,
    "number": float,
    "string": str,
}


# TODO (Sabri): add support for multiple dps per interface, will probably need to add dp id


class ColumnInfoResponse(BaseModel):

    name: str
    type: str
    formatter: str


@router.get("/{interface_id}/column_info/")
def get_column_info(
    interface_id: int,
):
    interface = get_interface(interface_id)
    dp = interface.data
    return _get_column_info(dp, dp.columns)


def _get_column_info(dp: DataPanel, columns: List[str] = None):
    if columns is None:
        columns = dp.columns

    return [
        ColumnInfoResponse(
            name=col,
            type=str(type(dp[col])),
            formatter="image"
            if isinstance(dp[col], ImageColumn)
            else "string",  # TODO(sabri): add formatter
        )
        for col in columns
    ]


class DataPanelResponse(BaseModel):
    column_info: List[ColumnInfoResponse]
    indices: List[int] = None
    rows: List[List[Any]]
    full_length: int


class DataPanelRequest(BaseModel):
    # TODO (sabri): add support for data validation
    start: int = None
    end: int = None
    indices: List[int] = None
    columns: List[str] = None


@router.post("/{interface_id}/rows/")
def get_rows(
    interface_id: int,
    request: DataPanelRequest,
) -> DataPanelResponse:
    """
    Get rows from a DataPanel as a JSON object.
    """
    dp = get_interface(interface_id).data
    column_info = _get_column_info(dp, request.columns)

    dp = dp[[info.name for info in column_info]]

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
    for row in dp:
        rows.append(
            [FORMATTERS[info.formatter](row[info.name]) for info in column_info]
        )

    return DataPanelResponse(
        column_info=column_info,
        rows=rows,
        full_length=len(dp),
        indices=indices,
    )
