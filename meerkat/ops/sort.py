from typing import List, Union

import numpy as np

from meerkat import DataPanel


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

    if len(by) > 1:  # Sort with multiple column
        keys = []
        for col in by[::-1]:
            keys.append(data[col].to_numpy())

        sorted_indices = np.lexsort(keys=keys)

        if ascending is False:
            sorted_indices = sorted_indices[::-1].copy()

    else:  # Sort with single column
        sorted_indices = data[by[0]].argsort(ascending=ascending, kind=kind)

    return data.lz[sorted_indices]
