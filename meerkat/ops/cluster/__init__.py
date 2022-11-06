from typing import Optional, Tuple, Union

import sklearn.cluster as skcluster
from sklearn.base import ClusterMixin

from meerkat import AbstractColumn, DataFrame, NumpyArrayColumn, embed


def cluster(
    data: Union[AbstractColumn, DataFrame],
    input: Optional[str] = None,
    method: Union[str, ClusterMixin] = "KMeans",
    encoder: str = "clip",  # add support for auto selection of encoder
    modality: str = None,
    **kwargs,
) -> Tuple[NumpyArrayColumn, ClusterMixin]:
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
        # TODO (sabri): Give the user the option to specify the output column.
        cluster_column = f"{method}({input})"
        embed_col = f"{encoder}({input})"

        # TODO (sabri): Remove this guard once caching is supported.
        if embed_col not in data:
            data = embed(
                data=data,
                input=input,
                encoder=encoder,
                out_col=embed_col,
                modality=modality,
            )
        data_embedding = data[embed_col]
    else:
        raise NotImplementedError

    data_embedding = data[embed_col]
    if isinstance(method, str):
        method = getattr(skcluster, method)(**kwargs)

    clusters = method.fit_predict(data_embedding.data)

    if isinstance(data, DataFrame):
        data[cluster_column] = clusters
        return data, method
    return clusters, method
