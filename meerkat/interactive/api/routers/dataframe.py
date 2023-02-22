from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException
from pydantic import BaseModel, StrictInt, StrictStr

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import Endpoint, endpoint


class ColumnInfo(BaseModel):
    name: str
    type: str
    cellComponent: str
    cellProps: Dict[str, Any]
    cellDataProp: str


class SchemaResponse(BaseModel):
    id: str
    columns: List[ColumnInfo]
    nrows: int = None
    primaryKey: str = None


@endpoint(prefix="/df", route="/{df}/schema/")
def schema(
    df: DataFrame,
    columns: List[str] = Endpoint.EmbeddedBody(None),
    formatter: Union[str, Dict[str, str]] = Endpoint.EmbeddedBody("base"),
) -> SchemaResponse:
    columns = df.columns if columns is None else columns
    if isinstance(formatter, str):
        formatter = {column: formatter for column in columns}
    return SchemaResponse(
        id=df.id,
        columns=_get_column_infos(df, columns, formatter_placeholders=formatter),
        nrows=len(df),
        primaryKey=df.primary_key_name,
    )


def _get_column_infos(
    df: DataFrame,
    columns: List[str] = None,
    formatter_placeholders: Dict[str, str] = None,
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

    # TODO: Check if this is the right fix.
    if formatter_placeholders is None:
        formatter_placeholders = {}
    elif isinstance(formatter_placeholders, str):
        formatter_placeholders = {
            column: formatter_placeholders for column in df.columns
        }

    out = [
        ColumnInfo(
            name=col,
            type=type(df[col]).__name__,
            cellComponent=df[col]
            .formatters[formatter_placeholders.get(col, "base")]
            .component_class.alias,
            cellProps=df[col].formatters[formatter_placeholders.get(col, "base")].props,
            cellDataProp=df[col]
            .formatters[formatter_placeholders.get(col, "base")]
            .data_prop,
        )
        for col in columns
    ]
    return out


class RowsResponse(BaseModel):
    columnInfos: List[ColumnInfo]
    posidxs: List[int] = None
    rows: List[List[Any]]
    fullLength: int
    primaryKey: Optional[str] = None


@endpoint(prefix="/df", route="/{df}/rows/")
def rows(
    df: DataFrame,
    start: int = Endpoint.EmbeddedBody(None),
    end: int = Endpoint.EmbeddedBody(None),
    posidxs: List[int] = Endpoint.EmbeddedBody(None),
    key_column: str = Endpoint.EmbeddedBody(None),
    keyidxs: List[Union[StrictInt, StrictStr]] = Endpoint.EmbeddedBody(None),
    columns: List[str] = Endpoint.EmbeddedBody(None),
    formatter: Union[str, Dict[str, str]] = Endpoint.EmbeddedBody("base"),
    shuffle: bool = Endpoint.EmbeddedBody(False),
) -> RowsResponse:
    """Get rows from a DataFrame as a JSON object."""
    if columns is None:
        columns = df.columns
    if isinstance(formatter, str):
        formatter = {column: formatter for column in columns}
    full_length = len(df)
    column_infos = _get_column_infos(df, columns, formatter_placeholders=formatter)

    if shuffle:
        df = df.shuffle()

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
                df[info.name]
                .formatters[formatter.get(info.name, "base")]
                .encode(row[info.name])
                for info in column_infos
            ]
        )
    return RowsResponse(
        columnInfos=column_infos,
        rows=rows,
        fullLength=full_length,
        posidxs=posidxs,
        primaryKey=df.primary_key_name,
    )
