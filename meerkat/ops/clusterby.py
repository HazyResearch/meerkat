
import meerkat as mk
import numpy as np
from meerkat import DataPanel, NumpyArrayColumn
from typing import Union, Sequence
from meerkat.ops.groupby import DataPanelGroupBy, groupby
from sklearn.cluster import KMeans 

def embed( img):
    return np.array(img).mean(axis = 0).mean(axis = 0)

def clusterby(
    data: DataPanel,
    by: Union[str, Sequence[str]] = None,
    alg : str = "kmeans", 
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

    if does_embed:
        data[".emb"] = data[by].map(embed, num_workers=4, pbar=True)
        by = [".emb"]

    if alg == "kmeans":
        if isinstance(by, str):
            by = [by]
        
        if len(by) > 1:
            raise NotImplementedError()
        

        arr = data[by[0]].data
        n_clusters = 10

        groups = KMeans(n_clusters=n_clusters, random_state=0).fit(arr).labels_
        data['.group'] = groups

        return groupby(data, by = ".group")



def main():

    dp = mk.get("imagenette")[:100]

    
    gb = clusterby(dp, by = 'img')["img"] # Numpy Array Column
    
    for class_id in gb.indices:
        print(class_id, ":", gb.indices[class_id])




    
    # print(dp.columns)


if __name__ == "__main__":
    main()