import functools
from typing import Any, Dict, List, Union

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

import meerkat as mk
from meerkat.dataframe import DataFrame
from meerkat.state import state

from ....tools.utils import convert_to_python

router = APIRouter(
    prefix="/box",
    tags=["box"],
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


@router.post("/{dataframe_id}/schema/")
def get_schema(dataframe_id: str, request: SchemaRequest) -> SchemaResponse:
    df = state.identifiables.get(group="dataframes", id=dataframe_id)
    columns = df.columns if request is None else request.columns
    return SchemaResponse(id=dataframe_id, columns=_get_column_infos(df, columns))


def _get_column_infos(df: DataFrame, columns: List[str] = None):
    if columns is None:
        columns = df.columns
    else:
        missing_columns = set(columns) - set(df.columns)
        if len(missing_columns) > 0:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Requested columns {columns} do not exist in dataframe"
                    f" with id {df.id}"
                ),
            )

    # TODO: remove this and fix
    columns = [
        column for column in columns if column not in ["clip(img)", "clip(image)"]
    ]

    return [
        ColumnInfo(
            name=col,
            type=type(df[col]).__name__,
            cell_component=df[col].formatter.cell_component,
            cell_props=df[col].formatter.cell_props,
        )
        for col in columns
    ]


class RowsResponse(BaseModel):
    column_infos: List[ColumnInfo]
    indices: List[int] = None
    rows: List[List[Any]]
    full_length: int


class RowsRequest(BaseModel):
    # TODO (sabri): add support for data validation
    start: int = None
    end: int = None
    indices: List[int] = None
    columns: List[str] = None


@router.post("/{box_id}/rows/")
def get_rows(
    box_id: str,
    request: RowsRequest,
) -> RowsResponse:
    """Get rows from a DataFrame as a JSON object."""
    box = state.identifiables.get(group="boxes", id=box_id)

    if not isinstance(box.obj, DataFrame):
        raise HTTPException("`get_rows` expects a box holding a Datapanel.")
    df = box.obj

    full_length = len(df)
    column_infos = _get_column_infos(df, request.columns)

    df = df[[info.name for info in column_infos]]

    if request.indices is not None:
        df = df.lz[request.indices]
        indices = request.indices
    elif request.start is not None:
        if request.end is None:
            request.end = len(df)
        df = df.lz[request.start : request.end]
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


class MatchRequest(BaseModel):
    input: str  # The name of the input column.
    query: str  # The query text to match against.


@router.post("/{dataframe_id}/match/")
def match(
    dataframe_id: str, input: str = EmbeddedBody(), query: str = EmbeddedBody()
) -> SchemaResponse:
    """Match a query string against a DataFrame column.

    The `dataframe_id` remains the same as the original request.
    """
    df = state.identifiables.get(group="dataframes", id=dataframe_id)
    # write the query to a file
    with open("/tmp/query.txt", "w") as f:
        f.write(query)
    try:
        df, match_columns = mk.match(
            data=df, query=query, input=input, return_column_names=True
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return SchemaResponse(id=dataframe_id, columns=_get_column_infos(df, match_columns))


@router.post("/{dataframe_id}/sort/")
def sort(dataframe_id: str, by: str = EmbeddedBody()):
    df = state.identifiables.get(group="dataframes", id=dataframe_id)
    df = mk.sort(data=df, by=by, ascending=False)
    global curr_df
    curr_df = df
    return SchemaResponse(id=df.id, columns=_get_column_infos(df))


@router.post("/{dataframe_id}/aggregate/")
def aggregate(
    dataframe_id: str,
    aggregation_id: str = Body(None),
    aggregation: str = Body(None),
    accepts_df: bool = Body(False),
    columns: List[str] = Body(None),
) -> Union[float, int, str]:
    df = state.identifiables.get(group="dataframes", id=dataframe_id)

    if columns is not None:
        df = df[columns]

    if (aggregation_id is None) == (aggregation is None):
        raise HTTPException(
            status_code=400,
            detail="Must specify either aggregation_id or aggregation",
        )

    if aggregation_id is not None:
        aggregation = state.identifiables.get(id=aggregation_id, group="aggregations")
        value = df.aggregate(aggregation, accepts_df=accepts_df)

    else:
        if aggregation not in ["mean", "sum", "min", "max"]:
            raise HTTPException(
                status_code=400, detail=f"Invalid aggregation {aggregation}"
            )
        value = df.aggregate(aggregation)

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


def _filter(
    df: DataFrame, columns: List[str], values: List[Any], ops: List[str]
) -> DataFrame:
    supported_column_types = (mk.PandasSeriesColumn, mk.NumpyArrayColumn)
    if not all(isinstance(df[column], supported_column_types) for column in columns):
        raise HTTPException(
            f"Only {supported_column_types} are supported for filtering."
        )
    all_series = [
        _operator_str_to_func[op](df[col], value)
        for col, value, op in zip(columns, values, ops)
    ]
    mask = functools.reduce(lambda x, y: x & y, all_series)
    return df.lz[mask]


class Op(BaseModel):
    box_id: str
    op_id: str


@router.post("/{box_id}/filter")
def filter(box_id: str, request: FilterRequest) -> Op:
    # TODO(karan): untested change as earlier version called a function
    # that didn't exist
    box = state.identifiables.get(group="boxes", id=box_id)

    op = box.apply(
        _filter, columns=request.columns, values=request.values, ops=request.ops
    )

    return Op(box_id=box.id, op_id=op.id)


@router.post("/{box_id}/undo")
def undo(box_id: str, operation_id: str = EmbeddedBody()) -> Op:
    box = state.identifiables.get(group="boxes", id=box_id)
    box.undo(operation_id=operation_id)
    return Op(box_id=box_id, op_id="meerkat")  # TODO fix this
