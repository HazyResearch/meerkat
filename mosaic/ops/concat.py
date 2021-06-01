from typing import Sequence, Union

import cytoolz as tz

from mosaic import DataPanel
from mosaic.columns.abstract import AbstractColumn
from mosaic.errors import ConcatError


def concat(
    objs: Union[Sequence[DataPanel], Sequence[AbstractColumn]],
    axis: Union[str, int] = "rows",
) -> Union[DataPanel, AbstractColumn]:
    """Concatenate a sequence of columns or a sequence of datapanels.

    If sequence is empty, returns an empty DataPanels.

    Args:
        objs (Union[Sequence[DataPanel], Sequence[AbstractColumn]]): [description]

    Returns:
        Union[DataPanel, AbstractColumn]: concatenated datapanel or column, depending on
            type of objs.
    """
    if len(objs) == 0:
        return DataPanel()

    if not all([type(objs[0]) == type(obj) for obj in objs[1:]]):
        raise ConcatError("All objects passed to concat must be of same type.")

    if isinstance(objs[0], DataPanel):
        if axis == 0 or axis == "rows":
            # append new rows
            columns = set(objs[0].visible_columns)
            if not all([set(dp.visible_columns) == columns for dp in objs]):
                raise ConcatError(
                    "Can only concatenate DataPanels along axis 0 (rows) if they have "
                    " the same set of columns names."
                )
            return DataPanel.from_batch(
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

            columns = list(tz.concat((dp.visible_columns for dp in objs)))
            columns.remove("index")  # need to remove index before checking distinct
            if not tz.isdistinct(columns):
                raise ConcatError(
                    "Can only concatenate DataPanels along axis 1 (columns) if they "
                    "have distinct column names."
                )

            data = tz.merge(*(dict(dp.items()) for dp in objs))
            return DataPanel.from_batch(data)
    elif isinstance(objs[0], AbstractColumn):
        # use the concat method of the column
        return objs[0].concat(objs)
    else:
        raise ConcatError(
            "Must pass a sequence of datapanels or a sequence of columns to concat."
        )
