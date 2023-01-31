from typing import TYPE_CHECKING, Optional, Tuple, Union

from meerkat import Column, DataFrame, ScalarColumn, TensorColumn
from meerkat.tools.lazy_loader import LazyLoader

if TYPE_CHECKING:
    from sklearn.base import ClusterMixin


skcluster = LazyLoader("sklearn.cluster")


def cluster(
    data: Union[Column, DataFrame],
    input: Optional[str] = None,
    method: Union[str, "ClusterMixin"] = "KMeans",
    encoder: str = "clip",  # add support for auto selection of encoder
    modality: str = None,
    **kwargs,
) -> Tuple[ScalarColumn, "ClusterMixin"]:
    """Cluster the data in a column. If the column is an unstructured type,
    (e.g. image), the column is first embedded then clustered.

    Args:
        data (Union[DataFrame, AbstractColumn]): The column to cluster or a dataframe
            containing the column to cluster.
        input (Union[str, Sequence[str]]): The column(s) to cluster by. These columns
            will be embedded using the ``encoder`` and the resulting embedding
            will be used. Ignored if ``data`` is a Column.
        method (Union[str, ClusterMixin]): The clustering method to use.
        encoder (str): The encoder to use for the embedding. Defaults to ``clip``.
        modality (Union[str, Sequence[str])): The modality to of the
        **kwargs: Additional keyword arguments to pass to the clustering method.

    Returns:
        (Union[NumpyArrayColumn, DataFrame], ClusterMixin): A tuple containing the
            clustered column and the fit clusterer. If ``data`` is a DataFrame, the
            clustered column is added to the DataFrame and it is returned.
    """
    if isinstance(data, DataFrame):
        col = data[input]
        output_col = f"{method}({input})"
    else:
        col = data

    if not isinstance(col, TensorColumn) and len(col.shape) != 2:
        raise ValueError("Must pass 2D TensorColumn.")

    if isinstance(method, str):
        method = getattr(skcluster, method)(**kwargs)

    clusters = method.fit_predict(col.data)

    if isinstance(data, DataFrame):
        data[output_col] = clusters
        return data, method

    return clusters, method
