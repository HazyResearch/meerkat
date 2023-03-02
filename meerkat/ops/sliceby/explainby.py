from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np

from meerkat.dataframe import DataFrame
from meerkat.interactive.graph.reactivity import reactive
from meerkat.ops.explain import explain

from .sliceby import SliceBy

if TYPE_CHECKING:
    import domino.Slicer


class ExplainBy(SliceBy):
    def __init__(
        self,
        data: DataFrame,
        by: Union[List[str], str],
        scores: Dict[Union[str, Tuple[str]], np.ndarray] = None,
        sets: Dict[Union[str, Tuple[str]], np.ndarray] = None,
    ):
        super().__init__(data=data, by=by, sets=sets, scores=scores)


@reactive()
def explainby(
    data: DataFrame,
    by: Union[str, Sequence[str]],
    target: Union[str, Mapping[str]],
    method: Union[str, "domino.Slicer"] = "MixtureSlicer",
    encoder: str = "clip",  # add support for auto selection of encoder
    modality: str = None,
    scores: bool = False,
    use_cache: bool = True,
    output_col: str = None,
    **kwargs,
) -> ExplainBy:
    """Perform a clusterby operation on a DataFrame.

    Args:
        data (DataFrame): The dataframe to cluster.
        by (Union[str, Sequence[str]]): The column(s) to cluster by. These columns will
            be embedded using the ``encoder`` and the resulting embedding will be used.
        method (Union[str, domino.Slicer]): The clustering method to use.
        encoder (str): The encoder to use for the embedding. Defaults to ``clip``.
        modality (Union[str, Sequence[str])): The modality to of the
        **kwargs: Additional keyword arguments to pass to the clustering method.

    Returns:
        ExplainBy: A ExplainBy object.
    """
    if output_col is None:
        output_col = f"{method}({by},{target})"
    # TODO (sabri): Give the user the option to specify the output and remove this guard
    # once caching is supported

    if not isinstance(by, str):
        raise NotImplementedError

    if output_col not in data or not use_cache:
        data, _ = explain(
            data=data,
            input=by,
            target=target,
            method=method,
            encoder=encoder,
            modality=modality,
            output_col=output_col,
            **kwargs,
        )

    if scores:
        slice_scores = data[output_col]
        slice_scores = {
            key: slice_scores[:, key] for key in range(slice_scores.shape[1])
        }
        return ExplainBy(data=data, scores=slice_scores, by=by)
    else:
        slice_sets = data[output_col]
        slice_sets = {
            key: np.where(slice_sets[:, key] == 1)[0]
            for key in range(slice_sets.shape[1])
        }
        return ExplainBy(data=data, sets=slice_sets, by=by)
