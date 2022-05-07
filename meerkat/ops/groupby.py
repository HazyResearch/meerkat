from __future__ import annotations
from abc import ABC, abstractmethod

from meerkat.mixins.groupbyable import BaseGroupBy
from meerkat.datapanel import DataPanel
# from meerkat.columns.abstract import AbstractColumn

import pandas as pd
from typing import Union, List, Sequence

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


    # def __init__(self, pd_gb, by, dp, keys) -> None:

    #     # the underlying group by object
    #     self._pd_group_by = pd_gb

    #     # the original datapanel
    #     self._main_dp = dp

    #     # by is what the group by is sorted on
    #     self._by = by

    #     # the queried keys.
    #     self._keys = keys

    # TODO must write accumulators like sum and mean.

    # def mean(self) -> DataPanel:
    #     # this is a datapanelgroup by and it has has a list of columns
    #     means = []
    #     indices = self.indices

    #     s_indices = list(indices.keys())
    #     s_indices.sort()
    #     labels = s_indices
    #     for key in s_indices:
    #         indices_l = indices[key]
    #         appropriate_slice = self.data[indices_l]

    #         # Hmm what do we want to do here? 

    #         # Mean of a data panel? Do we want to combine columns?
    #         mean_slice = appropriate_slice.mean()

    #         # This is throwing an error.
    #         means.append(mean_slice)

    #     queries = self.keys
    #     d = {}

    #     for query in queries:
    #         d[query] = means

    #     if isinstance(self.by, str):
    #         d[self.by] = labels
    #         return DataPanel(d)
    #     else:
    #         # self.by is a list
    #         if len(self.by) == 1:
    #             d[self.by[0]] = labels
    #         else:
    #             unzipped = list(zip(*labels))
    #             for i, l in enumerate(unzipped):
    #                 d[self.by[i]] = l
                
    #         return DataPanel(d)



    def __getitem__(
        self, key: Union[str, Sequence[str]]
    ) -> Union[DataPanelGroupBy, AbstractColumnGroupBy]:
        indices = self.indices 

        # pass in groups instead: keys are stable. 

        # what if you sort it? 
        # TODO: weak reference? 

        if isinstance(key, str):
            # assuming key is just one string
            column = self.data[key]

            # NumpyArrayColumn has no attr to_group_by
            # return column.to_group_by()

            # TODO: File structure is preventing me from doing what I want to do here. 
            # return NumpyArrayColumn.to_group_by(column, indices, self.by, key)
            return column.to_group_by(indices, self.by) # needs to be implemented else where. 
        else:
            return DataPanelGroupBy(indices,  self.data, self.by, self.keys)



