from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, StrictInt, StrictStr

from meerkat.columns.scalar import ScalarColumn
from meerkat.columns.tensor.numpy import NumPyTensorColumn
from meerkat.dataframe import DataFrame
from meerkat.interactive.edit import EditTargetConfig
from meerkat.interactive.endpoint import Endpoint, endpoint
from meerkat.interactive.graph import trigger
from meerkat.interactive.modification import DataFrameModification
from meerkat.state import state


class ColumnInfo(BaseModel):
    name: str
    type: str
    cell_component: str
    cell_props: Dict[str, Any]
    cell_data_prop: str


class SchemaResponse(BaseModel):
    id: str
    columns: List[ColumnInfo]
    nrows: int = None
    primary_key: str = None


@endpoint(prefix="/df", route="/{df}/schema/")
def schema(
    df: DataFrame, columns: List[str] = None, variants: List[str] = None
) -> SchemaResponse:
    columns = df.columns if columns is None else columns
    return SchemaResponse(
        id=df.id,
        columns=_get_column_infos(df, columns, variants=variants),
        nrows=len(df),
        primary_key=df.primary_key_name,
    )


def _get_column_infos(
    df: DataFrame, columns: List[str] = None, variants: List[str] = None
):

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

    columns = [column for column in columns if not column.startswith("_")]
    if df.primary_key_name is not None and df.primary_key_name not in columns:
        columns += [df.primary_key_name]
    return [
        ColumnInfo(
            name=col,
            type=type(df[col]).__name__,
            cell_component=df[col].formatter.component_class.alias,
            cell_props=df[col].formatter.get_props(variants=variants),
            cell_data_prop=df[col].formatter.data_prop,
        )
        for col in columns
    ]


class RowsResponse(BaseModel):
    column_infos: List[ColumnInfo]
    posidxs: List[int] = None
    rows: List[List[Any]]
    full_length: int
    # primary key
    primary_key: Optional[str] = None


@endpoint(prefix="/df", route="/{df}/rows/")
def rows(
    df: DataFrame,
    start: int = Endpoint.EmbeddedBody(None),
    end: int = Endpoint.EmbeddedBody(None),
    posidxs: List[int] = Endpoint.EmbeddedBody(None),
    key_column: str = Endpoint.EmbeddedBody(None),
    keyidxs: List[Union[StrictInt, StrictStr]] = Endpoint.EmbeddedBody(None),
    columns: List[str] = Endpoint.EmbeddedBody(None),
    variants: List[str] = Endpoint.EmbeddedBody(None),
) -> RowsResponse:
    """Get rows from a DataFrame as a JSON object."""

    full_length = len(df)
    column_infos = _get_column_infos(df, columns, variants=variants)

    df = df[[info.name for info in column_infos]]

    if posidxs is not None:
        df = df[posidxs]
        posidxs = posidxs
    elif start is not None:
        if end is None:
            end = len(df)
        else:
            end = min(end, len(df))
        df = df[start:end]
        posidxs = list(range(start, end))
    elif keyidxs is not None:
        if key_column is None:
            if df.primary_key is None:
                raise ValueError(
                    "Must provide key_column if keyidxs are provided and no "
                    "primary_key on dataframe."
                )
            df = df.loc[keyidxs]
        else:
            # FIXME(sabri): this will only work if key_column is a pandas column
            df = df[df[key_column].isin(keyidxs)]
    else:
        raise ValueError()

    rows = []
    for row in df:
        rows.append(
            [
                df[info.name].formatter.encode(row[info.name], variants=variants)
                for info in column_infos
            ]
        )
    return RowsResponse(
        column_infos=column_infos,
        rows=rows,
        full_length=full_length,
        posidxs=posidxs,
        primary_key=df.primary_key_name,
    )


@endpoint(prefix="/df", route="/{df}/remove_row_by_index/")
def remove_row_by_index(df: DataFrame, row_index: int = Endpoint.EmbeddedBody()):
    df = df[np.arange(len(df)) != row_index]

    # Set the df so Meerkat knows it changed
    df.set(df)


@endpoint(prefix="/df", route="/{df}/edit/")
def edit(
    df: DataFrame,
    value=Endpoint.EmbeddedBody(),  # don't set type
    column: str = Endpoint.EmbeddedBody(),
    row_id=Endpoint.EmbeddedBody(),
    id_column: str = Endpoint.EmbeddedBody(),
):

    mask = df[id_column] == row_id
    if mask.sum() == 0:
        raise HTTPException(f"Row with id {row_id} not found in column {id_column}")
    df[column][mask] = value

    # Set the df so Meerkat knows it changed
    df.set(df)
