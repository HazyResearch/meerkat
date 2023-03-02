from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from meerkat.dataframe import DataFrame
from meerkat.interactive.graph.reactivity import reactive

from .sliceby import SliceBy


class GroupBy(SliceBy):
    def __init__(
        self,
        data: DataFrame,
        by: Union[List[str], str],
        sets: Dict[Union[str, Tuple[str]], np.ndarray] = None,
    ):
        super().__init__(data=data, by=by, sets=sets)


@reactive()
def groupby(
    data: DataFrame,
    by: Union[str, Sequence[str]] = None,
) -> GroupBy:
    """Perform a groupby operation on a DataFrame or Column (similar to a
    `DataFrame.groupby` and `Series.groupby` operations in Pandas).j.

    Args:
        data (Union[DataFrame, AbstractColumn]): The data to group.
        by (Union[str, Sequence[str]]): The column(s) to group by. Ignored if ``data``
            is a Column.

    Returns:
        Union[DataFrameGroupBy, AbstractColumnGroupBy]: A GroupBy object.
    """
    if isinstance(by, str):
        by = [by]
    return GroupBy(data=data, sets=data[by].to_pandas().groupby(by).indices, by=by)
