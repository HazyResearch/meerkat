from itertools import combinations
from typing import Sequence, Tuple, Union

import cytoolz as tz

from meerkat import DataPanel
from meerkat.columns.abstract import AbstractColumn
from meerkat.errors import ConcatError
from meerkat.provenance import capture_provenance


@capture_provenance(capture_args=["axis"])
def concat(
    objs: Union[Sequence[DataPanel], Sequence[AbstractColumn]],
    axis: Union[str, int] = "rows",
    suffixes: Tuple[str] = None,
    overwrite: bool = False,
) -> Union[DataPanel, AbstractColumn]:
    """Concatenate a sequence of columns or a sequence of `DataPanel`s. If
    sequence is empty, returns an empty `DataPanel`.

    - If concatenating columns, all columns must be of the same type (e.g. all
    `ListColumn`).
    - If concatenating `DataPanel`s along axis 0 (rows), all `DataPanel`s must have the
    same set of columns.
    - If concatenating `DataPanel`s along axis 1 (columns), all `DataPanel`s must have
    the same length and cannot have any of the same column names.

    Args:
        objs (Union[Sequence[DataPanel], Sequence[AbstractColumn]]): sequence of columns
            or DataPanels.
        axis (Union[str, int]): The axis along which to concatenate. Ignored if
            concatenating columns.

    Returns:
        Union[DataPanel, AbstractColumn]: concatenated DataPanel or column
    """
    if len(objs) == 0:
        return DataPanel()

    if not all([type(objs[0]) == type(obj) for obj in objs[1:]]):
        raise ConcatError("All objects passed to concat must be of same type.")

    if isinstance(objs[0], DataPanel):
        if axis == 0 or axis == "rows":
            # append new rows
            columns = set(objs[0].columns)
            if not all([set(dp.columns) == columns for dp in objs]):
                raise ConcatError(
                    "Can only concatenate DataPanels along axis 0 (rows) if they have "
                    " the same set of columns names."
                )
            return objs[0].from_batch(
                {column: concat([dp[column] for dp in objs]) for column in columns}
            )
        elif axis == 1 or axis == "columns":
            # append new columns
            length = len(objs[0])
            if not all([len(dp) == length for dp in objs]):
                raise ConcatError(
                    "Can only concatenate DataPanels along axis 1 (columns) if they "
                    "have the same length."
                )

            # get all column names that appear in more than one DataPanel
            shared = set()
            for dp1, dp2 in combinations(objs, 2):
                shared |= set(dp1.columns) & set(dp2.columns)

            # TODO (sabri): I removed the index column for now to address
            # https://github.com/robustness-gym/meerkat/issues/65, but when we refactor
            # index with https://github.com/robustness-gym/meerkat/issues/117 we should
            # take this out
            shared -= {"index"}
            if shared and not overwrite:
                if suffixes is None:
                    raise ConcatError("Must ")
                data = tz.merge(
                    {k + suffixes[idx] if k in shared else k: v for k, v in dp.items()}
                    for idx, dp in enumerate(objs)
                )
            else:
                data = tz.merge(dict(dp.items()) for dp in objs)

            return objs[0]._clone(data=data)
        else:
            raise ConcatError(f"Invalid axis `{axis}` passed to concat.")
    elif isinstance(objs[0], AbstractColumn):
        # use the concat method of the column
        return objs[0].concat(objs)
    else:
        raise ConcatError(
            "Must pass a sequence of datapanels or a sequence of columns to concat."
        )
