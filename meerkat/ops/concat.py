from itertools import combinations
from typing import Sequence, Tuple, Union

import cytoolz as tz

from meerkat import DataFrame
from meerkat.columns.abstract import Column
from meerkat.errors import ConcatError
from meerkat.interactive.graph.reactivity import reactive

from .decorators import check_primary_key


@reactive()
@check_primary_key
# @capture_provenance(capture_args=["axis"])
def concat(
    objs: Union[Sequence[DataFrame], Sequence[Column]],
    axis: Union[str, int] = "rows",
    suffixes: Tuple[str] = None,
    overwrite: bool = False,
) -> Union[DataFrame, Column]:
    """Concatenate a sequence of columns or a sequence of `DataFrame`s. If
    sequence is empty, returns an empty `DataFrame`.

    - If concatenating columns, all columns must be of the same type (e.g. all
    `ListColumn`).
    - If concatenating `DataFrame`s along axis 0 (rows), all `DataFrame`s must have the
    same set of columns.
    - If concatenating `DataFrame`s along axis 1 (columns), all `DataFrame`s must have
    the same length and cannot have any of the same column names.


    Args:
        objs (Union[Sequence[DataFrame], Sequence[AbstractColumn]]): sequence of columns
            or DataFrames.
        axis (Union[str, int]): The axis along which to concatenate. Ignored if
            concatenating columns.

    Returns:
        Union[DataFrame, AbstractColumn]: concatenated DataFrame or column
    """
    if len(objs) == 0:
        return DataFrame()

    if not all([type(objs[0]) == type(obj) for obj in objs[1:]]):
        _any_object_empty = any([len(obj) == 0 for obj in objs])
        if _any_object_empty:
            raise ConcatError(
                """All objects passed to concat must be of same type.
This error may be because you have empty `objs`.
Try running `<objs>.filter(lambda x: len(x) > 0)` before calling mk.concat."""
            )
        raise ConcatError("All objects passed to concat must be of same type.")

    if isinstance(objs[0], DataFrame):
        if axis == 0 or axis == "rows":
            # append new rows
            columns = objs[0].columns
            if not all([set(df.columns) == set(columns) for df in objs]):
                raise ConcatError(
                    "Can only concatenate DataFrames along axis 0 (rows) if they have "
                    " the same set of columns names."
                )
            return objs[0]._clone(
                {column: concat([df[column] for df in objs]) for column in columns}
            )
        elif axis == 1 or axis == "columns":
            # append new columns
            length = len(objs[0])
            if not all([len(df) == length for df in objs]):
                raise ConcatError(
                    "Can only concatenate DataFrames along axis 1 (columns) if they "
                    "have the same length."
                )

            # get all column names that appear in more than one DataFrame
            shared = set()
            for df1, df2 in combinations(objs, 2):
                shared |= set(df1.columns) & set(df2.columns)

            if shared and not overwrite:
                if suffixes is None:
                    raise ConcatError("Must ")
                data = tz.merge(
                    {k + suffixes[idx] if k in shared else k: v for k, v in df.items()}
                    for idx, df in enumerate(objs)
                )
            else:
                data = tz.merge(dict(df.items()) for df in objs)

            return objs[0]._clone(data=data)
        else:
            raise ConcatError(f"Invalid axis `{axis}` passed to concat.")
    elif isinstance(objs[0], Column):
        # use the concat method of the column
        return objs[0].concat(objs)
    else:
        raise ConcatError(
            "Must pass a sequence of dataframes or a sequence of columns to concat."
        )
