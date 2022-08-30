import functools
from multiprocessing.sharedctypes import Value
from typing import Any, Dict, List, Union

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

import meerkat as mk
from meerkat.datapanel import DataPanel
from meerkat.interactive import Modification, trigger
from meerkat.interactive.graph import BoxModification
from meerkat.state import state

from ....tools.utils import convert_to_python

router = APIRouter(
    prefix="/dp",
    tags=["dp"],
    responses={404: {"description": "Not found"}},
)

EmbeddedBody = functools.partial(Body, embed=True)


class ColumnInfo(BaseModel):

    name: str
    type: str
    cell_component: str
    cell_props: Dict[str, Any]


class SchemaRequest(BaseModel):
    columns: List[str] = None


class SchemaResponse(BaseModel):
    id: str
    columns: List[ColumnInfo]
    nrows: int = None


@router.post("/{pivot_id}/schema/")
def schema(pivot_id: str, request: SchemaRequest) -> SchemaResponse:
    pivot = state.identifiables.get(group="boxes", id=pivot_id)

    dp = state.identifiables.get(group="datapanels", id=pivot.obj.id)
    columns = dp.columns if request is None else request.columns
    return SchemaResponse(
        id=pivot.obj.id, columns=_get_column_infos(dp, columns), nrows=len(dp)
    )


def _get_column_infos(dp: DataPanel, columns: List[str] = None):
    if columns is None:
        columns = dp.columns
    else:
        missing_columns = set(columns) - set(dp.columns)
        if len(missing_columns) > 0:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Requested columns {columns} do not exist in datapanel"
                    f" with id {dp.id}"
                ),
            )

    # TODO: remove this and fix
    columns = [
        column for column in columns if column not in ["clip(img)", "clip(image)"]
    ]

    return [
        ColumnInfo(
            name=col,
            type=type(dp[col]).__name__,
            cell_component=dp[col].formatter.cell_component,
            cell_props=dp[col].formatter.cell_props,
        )
        for col in columns
    ]


class RowsResponse(BaseModel):
    column_infos: List[ColumnInfo]
    indices: List[int] = None
    rows: List[List[Any]]
    full_length: int


@router.post("/{box_id}/rows/")
def rows(
    box_id: str,
    start: int = Body(None),
    end: int = Body(None),
    indices: List[int] = Body(None),
    columns: List[str] = Body(None),
) -> RowsResponse:
    """Get rows from a DataPanel as a JSON object."""
    box = state.identifiables.get(group="boxes", id=box_id)
    dp = box.obj

    full_length = len(dp)
    column_infos = _get_column_infos(dp, columns)

    dp = dp[[info.name for info in column_infos]]

    if indices is not None:
        dp = dp.lz[indices]
        indices = indices
    elif start is not None:
        if end is None:
            end = len(dp)
        dp = dp.lz[start:end]
        indices = list(range(start, end))
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


@router.post("/{box_id}/edit/")
def edit(
    box_id: str,
    value=Body(),  # don't set type
    column: str = Body(),
    row_id=Body(),
    id_column: str = Body(),
) -> List[Modification]:

    box = state.identifiables.get(group="boxes", id=box_id)
    dp = box.obj

    mask = dp[id_column] == row_id
    if mask.sum() == 0:
        raise HTTPException(f"Row with id {row_id} not found in column {id_column}")
    dp[column][mask] = value

    modifications = trigger(modifications=[BoxModification(id=box_id, scope=[column])])
    return modifications


# TODO: (Sabri/Arjun) Make this more robust and less hacky
curr_dp: mk.DataPanel = None


@router.post("/{datapanel_id}/sort/")
def sort(datapanel_id: str, by: str = EmbeddedBody()):
    dp = state.identifiables.get(group="datapanels", id=datapanel_id)
    dp = mk.sort(data=dp, by=by, ascending=False)
    global curr_dp
    curr_dp = dp
    return SchemaResponse(id=dp.id, columns=_get_column_infos(dp))


@router.post("/{datapanel_id}/aggregate/")
def aggregate(
    datapanel_id: str,
    aggregation_id: str = Body(None),
    aggregation: str = Body(None),
    accepts_dp: bool = Body(False),
    columns: List[str] = Body(None),
) -> Union[float, int, str]:
    dp = state.identifiables.get(group="datapanels", id=datapanel_id)

    if columns is not None:
        dp = dp[columns]

    if (aggregation_id is None) == (aggregation is None):
        raise HTTPException(
            status_code=400,
            detail="Must specify either aggregation_id or aggregation",
        )

    if aggregation_id is not None:
        aggregation = state.identifiables.get(id=aggregation_id, group="aggregations")
        value = dp.aggregate(aggregation, accepts_dp=accepts_dp)

    else:
        if aggregation not in ["mean", "sum", "min", "max"]:
            raise HTTPException(
                status_code=400, detail=f"Invalid aggregation {aggregation}"
            )
        value = dp.aggregate(aggregation)

    # convert value to native python type
    value = convert_to_python(value)

    return value


# TODO: Filter to take filtering criteria objects
# TODO: update for filtering row
class FilterRequest(BaseModel):
    columns: List[str]
    values: List[Any]
    ops: List[str]


# Map a string operator to a function that takes a value and returns a boolean.
_operator_str_to_func = {
    "==": lambda x, y: x == y,  # equality
    "!=": lambda x, y: x != y,  # inequality
    ">": lambda x, y: x > y,  # greater than
    "<": lambda x, y: x < y,  # less than
    ">=": lambda x, y: x >= y,  # greater than or equal to
    "<=": lambda x, y: x <= y,  # less than or equal to
    "in": lambda x, y: x in y,  # in
    "not in": lambda x, y: x not in y,  # not in
}


@router.post("/{datapanel_id}/filter")
def filter(datapanel_id: str, request: FilterRequest) -> SchemaResponse:
    # TODO(karan): untested change as earlier version called a function
    # that didn't exist
    dp = state.identifiables.get(group="datapanels", id=datapanel_id)

    supported_column_types = (mk.PandasSeriesColumn, mk.NumpyArrayColumn)
    if not all(
        isinstance(dp[column], supported_column_types) for column in request.columns
    ):
        raise HTTPException(
            f"Only {supported_column_types} are supported for filtering."
        )

    # Filter pandas series columns.
    all_series = [
        _operator_str_to_func[op](dp[col], value)
        for col, value, op in zip(request.columns, request.values, request.ops)
    ]
    mask = functools.reduce(lambda x, y: x & y, all_series)
    dp = dp.lz[mask]

    global curr_dp
    curr_dp = dp
    return SchemaResponse(
        id=dp.id, columns=_get_column_infos(dp, ["img", "path", "label"])
    )
