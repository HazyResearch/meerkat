from __future__ import annotations

from abc import ABC
from typing import Callable, Sequence, Union
from collections import Counter
import numpy as np
import torch

from meerkat.datapanel import DataPanel
from meerkat import embed

class AbstractColumnGroupBy:
    pass


def mode(l):
    item, count = Counter(l).most_common(1)[0]
    return item, count / len(l)

class BaseGroupBy(ABC):
    def __init__(self, indices, data, by, keys, is_soft = False) -> None:
        self.indices = indices
        self.data = data
        self.by = by
        self.keys = keys
        self.is_soft = is_soft


    def describe(self) -> str:
        if self.is_soft:


            classes = list(set(self.data["label"].data.values))


            correct_col = DataPanel({"class" : [c for c in classes], "label" : [f"a photo of a {c}" for c in classes]})
            embed(correct_col, "label", modality = "text", out_col = ".emb", num_workers = 0)
            embedded_images = self.data[".emb"]
            embedding_label = correct_col[".emb"]




            for i in range(len(embedded_images)):
                im = torch.Tensor(embedded_images[i])
                im /= im.norm(dim = -1, keepdim = True)
                embedded_images[i] = im
# d = {}

            t_embedded_ims = torch.Tensor(embedded_images)

            scores = torch.zeros((len(embedded_images), len(classes)))
            for i in range(embedding_label.shape[0]):

                emb = correct_col[i][".emb"] 

                t = torch.Tensor(emb)
                t /= t.norm(dim = -1, keepdim = True)

                out = (t_embedded_ims * t).sum(axis = 1)

                scores[:, i] = out



            preds = scores.softmax(dim = -1).argmax(dim = -1)



            class_preds = np.array([classes[i] for i in preds])
            print("Description of generated clusters.")
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            


            print(f"Created {len(self.indices)} clusters")
            for i, group in enumerate(self.indices):
                preds_by_group = class_preds[self.indices[group]]
                item, freq = mode(preds_by_group)
                print(f"Cluster {i} is made up of {round(freq * 100, 1)} {item}s")
                
                # print(f"Generating description for id: {group}")
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            return ""
        else:
            return "You already know this because it's a normal categorical value."


    def mean(self, *args, **kwargs):
        return self._reduce(lambda x: x.mean(*args, **kwargs))

    def _reduce(self, f: Callable):
        """self.indices are a dictionary of {labels : [indices]}"""
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
            relevant_rows_where_by_is_label = self.data.lz[indices_l]
            m = f(relevant_rows_where_by_is_label)
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
                assert len(self.by) == 1
                col = self.by[0]
                out[col] = labels
        return out


def groupby(
    data: DataPanel,
    by: Union[str, Sequence[str]] = None,
    is_soft: bool = False
) -> DataPanelGroupBy:
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
    `AbstractColumn.groupby` eventually, but we also want a functional version
     that could be called like `mk.groupby(dp, by="class")`. I'd suggest
     putting most of the implementation here,
      and then making the methods just wrappers. See merge as an example.

    Args:
        data (Union[DataPanel, AbstractColumn]): The data to group.
        by (Union[str, Sequence[str]]): The column(s) to group by. Ignored if ``data``
            is a Column.

    Returns:
        Union[DataPanelGroupBy, AbstractColumnGroupBy]: A GroupBy object.
    """

    # must pass two arguments (columns - by, by),
    # by -> is a dictionary, a map, all distinct group_ids to indicies.
    # pass DataPanelGroupBy()

    try:
        if isinstance(by, str):
            by = [by]
        return DataPanelGroupBy(
            data[by].to_pandas().groupby(by).indices, data, by, data.columns, is_soft
        )
    except Exception as e:
        # future work needed here.
        print("dataPanel group by error", e)
        raise NotImplementedError()


class DataPanelGroupBy(BaseGroupBy):

    def describe(self) -> str:
        return super().describe()

    def get_assignments(self):
        return self.indices

    def __getitem__(
        self, key: Union[str, Sequence[str]]
    ) -> Union[DataPanelGroupBy, AbstractColumnGroupBy]:
        indices = self.indices
        # TODO: weak reference?

        if isinstance(key, str):
            key = [key]

        return DataPanelGroupBy(indices, self.data[key], self.by, key, self.is_soft)
