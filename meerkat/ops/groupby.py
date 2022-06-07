from __future__ import annotations
from abc import ABC, abstractmethod

from meerkat.datapanel import DataPanel
import pandas as pd
from typing import Union, List, Sequence, Callable

class BaseGroupBy(ABC):
    def __init__(self, indices, data, by, keys) -> None:
        self.indices = indices
        self.data = data
        self.by = by
        self.keys = keys

    def mean(self, *args, **kwargs):
        return self._reduce(lambda x: x.mean(*args, **kwargs))


    def _reduce(self, f: Callable):
        # inputs: self.indices are a dictionary of {
        #   labels : [indices]
        # }
        labels = list(self.indices.keys())

        # sorting them so that they appear in a nice order.
        labels.sort()

        # Means will be a list of dictionaries where each element in the dict

        means = []
        for label in labels:
            indices_l = self.indices[label]
            relevant_rows_where_by_is_label = self.data[indices_l]
            m = f(relevant_rows_where_by_is_label) # TODO : Use reduce function.
            means.append(m)

        from meerkat.datapanel import DataPanel

        # Create data panel as a list of rows.
        out = DataPanel(means)


        assert isinstance(self.by, list)

        # Add the by columns.
        if len(labels) > 0:
            if isinstance(labels[0], tuple):
                columns = list(zip(*labels))

                for i, col in enumerate(self.by):
                    out[col] = columns[i]
            else:
                # This is the only way that this can occur.
                assert(len(self.by) == 1)
                col = self.by[0]
                out[col] = labels
        return out




def groupby(
    data: Union[DataPanel, AbstractColumn],
    by: Union[str, Sequence[str]] = None,
) -> Union[DataPanelGroupBy, AbstractColumnGroupBy]:
    """Perform a groupby operation on a DataPanel or Column (similar to a
    `DataFrame.groupby` and `Series.groupby` operations in Pandas).

    TODO (Sam): I put down a very rough scaffolding of how you could setup the class
    hierarchy for this. It is inspired by the way pandas has things setup: check out
    https://github.com/pandas-dev/pandas/tree/a8968bfa696d51f73769c54f2630a9530488236a/pandas/core/groupby
    for some inspiration.

    I'd recommend starting with small simple datapanels. e.g. a datapanel of all numpy
    array columns. For example,
    ```
    dp = DataPanel({
        'a': NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
        'b': NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
        'c': NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6])
    })

    groupby(dp, by="a")["c"].mean()
    ```

    Eventually we'll want to support a bunch of different aggregations, but for the time
    being let's just focus on mean, sum, and count.

    Note: we'll also want to implement methods `DataPanel.groupby` or
    `AbstractColumn.groupby` eventually, but we also want a functional version that could
    be called like `mk.groupby(dp, by="class")`. I'd suggest putting most of the implementation
    here, and then making the methods just wrappers. See merge as an example.

    Args:
        data (Union[DataPanel, AbstractColumn]): The data to group.
        by (Union[str, Sequence[str]]): The column(s) to group by. Ignored if ``data``
            is a Column.

    Returns:
        Union[DataPanelGroupBy, AbstractColumnGroupBy]: A GroupBy object.
    """



    # what work do we want pandas to do what do we want to do ourselves? 
    # this does throw exceptions

    # must pass two arguments (columns - by, by), 
    # by -> is a dictionary, a map, all distinct group_ids to indicies. 
    # pass DataPanelGroupBy()

    try:
        if isinstance(by, str):
            by = [by]
        return DataPanelGroupBy(data[by].to_pandas().groupby(by).indices, data, by, data.columns)
    except Exception as e:
        # future work needed here.
        print("dataPanel group by error", e)
        raise NotImplementedError()


class DataPanelGroupBy(BaseGroupBy):

    def __getitem__(
        self, key: Union[str, Sequence[str]]
    ) -> Union[DataPanelGroupBy, AbstractColumnGroupBy]:
        indices = self.indices 
        # TODO: weak reference? 
        
        if isinstance(key, str):
            key = [key]

        return DataPanelGroupBy(indices, self.data[key], self.by, key)



