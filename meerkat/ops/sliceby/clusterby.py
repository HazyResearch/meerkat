from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Union

import numpy as np

from meerkat.dataframe import DataFrame
from meerkat.interactive.graph.reactivity import reactive
from meerkat.ops.cluster import cluster

from .sliceby import SliceBy

if TYPE_CHECKING:
    from sklearn.base import ClusterMixin


class ClusterBy(SliceBy):
    def __init__(
        self,
        data: DataFrame,
        by: Union[List[str], str],
        sets: Dict[Union[str, Tuple[str]], np.ndarray] = None,
    ):
        super().__init__(data=data, by=by, sets=sets)


@reactive()
def clusterby(
    data: DataFrame,
    by: Union[str, Sequence[str]],
    method: Union[str, "ClusterMixin"] = "KMeans",
    encoder: str = "clip",  # add support for auto selection of encoder
    modality: str = None,
    **kwargs,
) -> ClusterBy:
    """Perform a clusterby operation on a DataFrame.

    Args:
        data (DataFrame): The dataframe to cluster.
        by (Union[str, Sequence[str]]): The column(s) to cluster by. These columns will
            be embedded using the ``encoder`` and the resulting embedding will be used.
        method (Union[str, "ClusterMixin"]): The clustering method to use.
        encoder (str): The encoder to use for the embedding. Defaults to ``clip``.
        modality (Union[str, Sequence[str])): The modality to of the
        **kwargs: Additional keyword arguments to pass to the clustering method.

    Returns:
        ClusterBy: A ClusterBy object.
    """
    out_col = f"{method}({by})"
    # TODO (sabri): Give the user the option to specify the output and remove this guard
    # once caching is supported

    if not isinstance(by, str):
        raise NotImplementedError

    if out_col not in data:
        data, _ = cluster(
            data=data,
            input=by,
            method=method,
            encoder=encoder,
            modality=modality,
            **kwargs,
        )
    clusters = data[out_col]
    cluster_indices = {key: np.where(clusters == key)[0] for key in np.unique(clusters)}

    if isinstance(by, str):
        by = [by]
    return ClusterBy(data=data, sets=cluster_indices, by=by)
