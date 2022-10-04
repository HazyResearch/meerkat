import functools
from typing import Any, Dict, List, Union

import numpy as np
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, StrictInt, StrictStr

import meerkat as mk
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.columns.numpy_column import NumpyArrayColumn
from meerkat.datapanel import DataPanel
from meerkat.interactive import Modification, trigger
from meerkat.interactive.edit import EditTargetConfig
from meerkat.interactive.graph import BoxModification
from meerkat.state import state
import pandas as pd
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
    key_column: str = Body(None),
    keys: List[Union[StrictInt, StrictStr]] = Body(None),
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
    elif keys is not None:
        if key_column is None:
            # TODO(sabri): when we add support for primary keys this should defualt to 
            # the primary key
            raise ValueError("Must provide key_column if keys are provided")
        
        # FIXME(sabri): this will only work if key_column is a pandas column
        dp = dp.lz[dp[key_column].isin(keys)]
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


@router.post("/{box_id}/remove_row_by_index/")
def remove_row_by_index(
    box_id: str, row_index: int = EmbeddedBody()
) -> List[Modification]:
    box = state.identifiables.get(group="boxes", id=box_id)

    dp = box.obj
    dp = dp.lz[np.arange(len(dp)) != row_index]
    # this is an out-of-place operation, so the box should be updated
    # TODO(karan): double check this
    box.obj = dp

    modifications = trigger(
        modifications=[BoxModification(id=box_id, scope=dp.columns)]
    )
    return modifications


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


@router.post("/{box_id}/edit_target/")
def edit_target(
    box_id: str,
    target: EditTargetConfig = Body(),
    value=Body(),  # don't set type
    column: str = Body(),
    row_indices: List[int] = Body(None),
    row_keys: List[Union[StrictInt, StrictStr]] = Body(None),
    primary_key: str = Body(None),
    metadata: Dict[str, Any] = Body(None),
):
    """Edit a target datapanel.

    Args:
        metadata (optional): Additional metadata to write.
            This should be a mapping from column_name -> value.
            Currently only unitary values are supported.
    """
    if (row_indices is None) == (row_keys is None):
        raise HTTPException(
            status_code=400,
            detail="Exactly one of row_indices or row_keys must be specified",
        )

    dp = state.identifiables.get(group="boxes", id=box_id).obj

    target_dp = state.identifiables.get(group="boxes", id=target.target.box_id).obj

    if row_indices is not None:
        source_ids = dp[target.source_id_column][row_indices]
    else:
        if primary_key is None:
            # TODO(): make this work once we've implemented primary_key
            raise NotImplementedError()
            # primary_key = target_dp.primary_key
        source_ids = dp[target.source_id_column].lz[np.isin(dp[primary_key], row_keys)]

    mask = np.isin(target_dp[target.target_id_column], source_ids)

    if mask.sum() != (len(row_keys) if row_keys is not None else len(row_indices)):
        breakpoint()
        raise HTTPException(
            status_code=500, detail="Target datapanel does not contain all source ids."
        )
    target_dp[column][mask] = value

    # TODO: support making a column if the column does not exist.
    # This requires deducing the column type and the default value
    # to fill in.
    if metadata is not None:
        for column_name, col_value in metadata.items():
            value = col_value
            default = None
            if isinstance(value, dict):
                value = col_value["value"]
                default = col_value["default"]
            if column_name not in target_dp.columns:
                if default is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Column {column_name} does not exist in target datapanel",
                    )
                default_col = np.full(len(target_dp), default)
                if isinstance(default, str):
                    default_col = PandasSeriesColumn([default] * len(target_dp))
                else:
                    default_col = NumpyArrayColumn(np.full(len(target_dp), default))
                target_dp[column_name] = default_col
            target_dp[column_name][mask] = value

    modifications = trigger(
        modifications=[BoxModification(id=target.target.box_id, scope=[column])]
    )
    return modifications


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
