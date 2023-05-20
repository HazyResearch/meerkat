from typing import TYPE_CHECKING

import numpy as np

from meerkat import DataFrame, NumPyTensorColumn, TensorColumn, TorchTensorColumn
from meerkat.env import is_torch_available
from meerkat.interactive.graph.reactivity import reactive
from meerkat.tools.lazy_loader import LazyLoader

torch = LazyLoader("torch")

if TYPE_CHECKING:
    import torch


@reactive
def search(
    data: DataFrame,
    query: np.ndarray,
    by: str = None,
    k: int = None,
    metric: str = "dot",
    score_column: str = None,
) -> DataFrame:
    """Search by a query in a DataFrame.

    Args:
        data: The DataFrame to search.
        query: The query to search with.
        by: The column to compare the query against.
        k: The number of results to return.
        metric: The metric to use for comparison.
        score_column: The name of the column to store the scores in.
            If ``None``, the scores will not be stored.

    Return:
        DataFrame: A sorted view of DataFrame.
    """
    if len(data) <= 1:
        raise ValueError("Dataframe must have at least 2 rows.")

    by = data[by]

    if not isinstance(by, TensorColumn):
        raise ValueError("")

    # convert query to same backend as by
    if isinstance(by, TorchTensorColumn):
        import torch

        if not torch.is_tensor(query):
            query = torch.tensor(query)
        query = query.to(by.device)

        fn = _torch_search

    elif isinstance(by, NumPyTensorColumn):
        if is_torch_available():
            import torch

            if torch.is_tensor(query):
                query = query.detach().cpu().numpy()
        elif not isinstance(query, np.ndarray):
            query = np.array(query)

        fn = _numpy_search
    else:
        raise ValueError("")

    scores, indices = fn(query=query, by=by.data, metric=metric, k=k)
    data = data[indices]
    if score_column is not None:
        data[score_column] = scores
    return data


def _torch_search(
    query: "torch.Tensor", by: "torch.Tensor", metric: str, k: int
) -> "torch.Tensor":
    with torch.no_grad():
        if len(query.shape) == 1:
            query = query.unsqueeze(0)

        if metric == "dot":
            scores = (by @ query.T).squeeze()
        else:
            raise ValueError("")

        scores, indices = torch.topk(scores, k=k)
    return scores.to("cpu").numpy(), indices.to("cpu")


def _numpy_search(query: np.ndarray, by: np.ndarray, metric: str, k: int) -> np.ndarray:
    if query.ndim == 1:
        query = query[np.newaxis, ...]

    if metric == "dot":
        scores = np.squeeze(by @ query.T)
    else:
        raise ValueError("")

    if k is not None:
        indices = np.argpartition(scores, -k)[-k:]
        indices = indices[np.argsort(-scores[indices])]
        scores = scores[indices]
    else:
        indices = np.argsort(-scores)
        scores = scores[indices]

    return scores, indices
