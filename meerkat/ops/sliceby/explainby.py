from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import domino
import numpy as np

from meerkat.datapanel import DataPanel
from meerkat.ops.explain import explain

from .sliceby import SliceBy


class ExplainBy(SliceBy):
    def __init__(
        self,
        data: DataPanel,
        by: Union[List[str], str],
        scores: Dict[Union[str, Tuple[str]], np.ndarray] = None,
        sets: Dict[Union[str, Tuple[str]], np.ndarray] = None,
    ):
        super().__init__(data=data, by=by, sets=sets, scores=scores)


def explainby(
    data: DataPanel,
    by: Union[str, Sequence[str]],
    target: Union[str, Sequence[str]],
    method: Union[str, domino.Slicer] = "MixtureSlicer",
    encoder: str = "clip",  # add support for auto selection of encoder
    modality: str = None,
    scores: bool = False,
    **kwargs,
) -> ExplainBy:
    """Perform a clusterby operation on a DataPanel.

    Args:
        data (DataPanel): The datapanel to cluster.
        by (Union[str, Sequence[str]]): The column(s) to cluster by. These columns will
            be embedded using the ``encoder`` and the resulting embedding will be used.
        method (Union[str, domino.Slicer]): The clustering method to use.
        encoder (str): The encoder to use for the embedding. Defaults to ``clip``.
        modality (Union[str, Sequence[str])): The modality to of the
        **kwargs: Additional keyword arguments to pass to the clustering method.

    Returns:
        ExplainBy: A ExplainBy object.
    """
    out_col = f"{method}({by},{target})"
    # TODO (sabri): Give the user the option to specify the output and remove this guard
    # once caching is supported

    if not isinstance(by, str):
        raise NotImplementedError

    if out_col not in data:
        data, _ = explain(
            data=data,
            input=by,
            target=target,
            method=method,
            encoder=encoder,
            modality=modality,
            **kwargs,
        )

    if scores:

        slice_scores = data[out_col]
        slice_scores = {
            key: slice_scores[:, key] for key in range(slice_scores.shape[1])
        }
        return ExplainBy(data=data, scores=slice_scores, by=by)
    else:
        slice_sets = data[out_col]
        slice_sets = {
            key: np.where(slice_sets[:, key] == 1)[0]
            for key in range(slice_sets.shape[1])
        }
        return ExplainBy(data=data, sets=slice_sets, by=by)
