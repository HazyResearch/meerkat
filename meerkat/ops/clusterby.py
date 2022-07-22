
import meerkat as mk
import numpy as np
from meerkat import DataPanel, NumpyArrayColumn
from typing import Union, Sequence
from meerkat.ops.groupby import DataPanelGroupBy, groupby
from sklearn.cluster import KMeans 
from sklearn.base import ClusterMixin

def embed( img):
    return np.array(img).mean(axis = 0).mean(axis = 0)


def get_edge_indices(dp : DataPanel):
    cluster = clusterby(dp, "img")
    cluster.describe()
    return cluster.data[".class_preds"] != cluster.data["img"]


def _clusterby(embedded_data: DataPanel,
    by: Sequence[str] = None,
    alg : Union[str, ClusterMixin]  = "kmeans"):

    if len(by) > 1:
        raise NotImplementedError()

    arr = embedded_data[by[0]].data
    groups = None
    if isinstance(alg, str):
        if alg == "kmeans":
            alg = KMeans(n_clusters=10, random_state=0)
        else:
            raise NotImplementedError("Please consider passing in a custom object via the sklearn interface.")
    if isinstance(alg, ClusterMixin):
        groups = alg.fit(arr).labels_

    if groups is not None:
        embedded_data['.group'] = groups
        return groupby(embedded_data, by = ".group", is_soft = True)
    else:
        raise NotImplementedError()


def clusterby(
    data: DataPanel,
    by: Union[str, Sequence[str]] = None,
    alg : Union[str, ClusterMixin]  = "kmeans", 
    does_embed: bool = True
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

    if isinstance(by, str):
        by = [by]
    elif isinstance(by, list):
        pass
    else:
        raise NotImplementedError("Please pass in a list or a string for by.")

    if does_embed:
        by = by[0]
        embedded_data = mk.embed(data, input=by, encoder="clip", out_col=".emb")
        by = [".emb"]
        return _clusterby(embedded_data, by, alg)
    else:
        return _clusterby(data, by, alg)


def main():



    clusterby_intermediate = mk.DataPanel.read("imagenette_clip_embedded_full.dp")
    clusters = _clusterby(clusterby_intermediate, [".emb"])

    clusters.describe()

    ind = get_edge_indices()

    
    # for class_id in gb.indices:
        # print(class_id, ":", gb.indices[class_id])

if __name__ == "__main__":
    main()