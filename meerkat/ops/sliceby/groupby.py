from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

from meerkat.datapanel import DataPanel

from .sliceby import SliceBy


class GroupBy(SliceBy):
    def __init__(
        self,
        data: DataPanel,
        by: Union[List[str], str],
        sets: Dict[Union[str, Tuple[str]], np.ndarray] = None,
    ):
        super().__init__(data=data, by=by, sets=sets)


def groupby(
    data: DataPanel,
    by: Union[str, Sequence[str]] = None,
) -> GroupBy:
    """Perform a groupby operation on a DataPanel or Column (similar to a
    `DataFrame.groupby` and `Series.groupby` operations in Pandas).j.

    Args:
        data (Union[DataPanel, AbstractColumn]): The data to group.
        by (Union[str, Sequence[str]]): The column(s) to group by. Ignored if ``data``
            is a Column.

    Returns:
        Union[DataPanelGroupBy, AbstractColumnGroupBy]: A GroupBy object.
    """
    if isinstance(by, str):
        by = [by]
    return GroupBy(data=data, sets=data[by].to_pandas().groupby(by).indices, by=by)
