from typing import TYPE_CHECKING

import numpy as np

from meerkat import DataFrame, NumPyTensorColumn, TensorColumn, TorchTensorColumn
from meerkat.tools.lazy_loader import LazyLoader

torch = LazyLoader("torch")

if TYPE_CHECKING:
    import torch


def search(
    data: DataFrame,
    query: np.ndarray,
    by: str = None,
    k: int = None,
    metric: str = "dot",
) -> DataFrame:
    """Compute a sort a DataFrame. If a DataFrame, sort by the values in the
    specified columns. Similar to ``sort_values`` in pandas.

    Args:
        data (Union[DataFrame, AbstractColumn]): DataFrame or Column to sort.
        by (Union[str, List[str]]): The columns to sort by. Ignored if data is a Column.
        ascending (Union[bool, List[bool]]): Whether to sort in ascending or
            descending order. If a list, must be the same length as `by`.Defaults
            to True.
        kind (str): The kind of sort to use. Defaults to 'quicksort'. Options
            include 'quicksort', 'mergesort', 'heapsort', 'stable'.

    Return:
        DataFrame: A sorted view of DataFrame.
    """
    by = data[by]

    if not isinstance(by, TensorColumn):
        raise ValueError("")

    # convert query to same backend as by
    if isinstance(by, TorchTensorColumn):
        import torch

        if not torch.is_tensor(query):
            query = torch.tensor(query)

        fn = _torch_search

    elif isinstance(by, NumPyTensorColumn):
        if torch.is_tensor(query):
            query = query.detach().cpu().numpy()
        elif not isinstance(query, np.ndarray):
            query = np.array(query)

        fn = _numpy_search
    else:
        raise ValueError("")

    _, indices = fn(query=query, by=by.data, metric=metric, k=k)
    return data[indices]


def _torch_search(
    query: "torch.Tensor", by: "torch.Tensor", metric: str, k: int
) -> "torch.Tensor":
    if len(query.shape) == 1:
        query = query.unsqueeze(0)

    if metric == "dot":
        scores = torch.matmul(by, query.T).squeeze()
    else:
        raise ValueError("")

    scores, indices = torch.topk(scores, k=k)
    return scores, indices


def _numpy_search(query: "torch.Tensor", by: "torch.Tensor", metric: str, k: int):
    raise NotImplementedError()
