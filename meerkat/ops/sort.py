from typing import List, Union

from meerkat import AbstractColumn, DataPanel


def sort(
    self,
    data: Union[DataPanel, AbstractColumn],
    by: Union[str, List[str]],
    ascending: Union[bool, List[bool]] = True,
    kind: str = "quicksort",
) -> DataPanel:
    """Sort a DataPanel or Column. If a DataPanel, sort by the values in the
    specified columns. Similar to ``sort_values`` in pandas.

    TODO(Hannah): Implement this method and add tests for it in
    tests/meerkat/ops/test_sort.py
    This method will rely on implementations of sort and argsort for each of the
    column types:
    e.g. NumpyArrayColumn.sort(), PandasSeriesColumn.sort(), TensorColumn.sort().
    Recommend implementing this in a top down manner – so start with this implementation
    of sort, assuming you have access to implementations of argsort and sort for each
    column type. Then implement sort for those column types. Ask questions if things are
    unclear or you're stuck!

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
    raise NotImplementedError
