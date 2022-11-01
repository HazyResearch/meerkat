from typing import List, Union

import numpy as np

from meerkat import DataPanel
from meerkat.interactive.graph import interface_op


@interface_op
def sort(
    data: DataPanel,
    by: Union[str, List[str]],
    ascending: Union[bool, List[bool]] = True,
    kind: str = "quicksort",
) -> DataPanel:
    """Sort a DataPanel or Column. If a DataPanel, sort by the values in the
    specified columns. Similar to ``sort_values`` in pandas.

    Args:
        data (Union[DataPanel, AbstractColumn]): DataPanel or Column to sort.
        by (Union[str, List[str]]): The columns to sort by. Ignored if data is a Column.
        ascending (Union[bool, List[bool]]): Whether to sort in ascending or
            descending order. If a list, must be the same length as `by`.Defaults
            to True.
        kind (str): The kind of sort to use. Defaults to 'quicksort'. Options
            include 'quicksort', 'mergesort', 'heapsort', 'stable'.

    Return:
        DataPanel: A sorted view of DataPanel.
    """
    by = [by] if isinstance(by, str) else by

    if isinstance(ascending, bool):
        ascending = [ascending] * len(by)

    if len(ascending) != len(by):
        raise ValueError(
            f"Length of `ascending` ({len(ascending)}) must be the same as "
            f"length of `by` ({len(by)})."
        )

    df = data[by].to_pandas()
    df["_sort_idx_"] = np.arange(len(df))
    df = df.sort_values(by=by, ascending=ascending, kind=kind, inplace=False)
    sorted_indices = df["_sort_idx_"]

    return data.lz[sorted_indices]
