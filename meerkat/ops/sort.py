from typing import List, Union

import numpy as np

from meerkat import DataFrame
from meerkat.interactive.graph import reactive


@reactive()
def sort(
    data: DataFrame,
    by: Union[str, List[str]],
    ascending: Union[bool, List[bool]] = True,
    kind: str = "quicksort",
) -> DataFrame:
    """Sort a DataFrame or Column. If a DataFrame, sort by the values in the
    specified columns. Similar to ``sort_values`` in pandas.

    Args:
        data (Union[DataFrame, AbstractColumn]): DataFrame or Column to sort.
        by (Union[str, List[str]]): The columns to sort by. Ignored if data is a Column.
        ascending (Union[bool, List[bool]]): Whether to sort in ascending or
            descending order. If a list, must be the same length as `by`.Defaults
            to True.
        kind (str): The kind of sort to use. Defaults to 'quicksort'. Options
            include 'quicksort', 'mergesort', 'heapsort', 'stable'.

    Return:
        DataFrame: A sorted view of DataFrame.
    """
    # Use "==" because `by` can be a Store.
    # Store(None) == None is True, but `Store(None) is None` evalutes to False.
    if by is None or by == None:  # noqa: E711
        return data.view()

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

    return data[sorted_indices]
