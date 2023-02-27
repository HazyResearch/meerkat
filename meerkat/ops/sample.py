from typing import Union

import numpy as np

from meerkat import Column, DataFrame
from meerkat.interactive.graph.reactivity import reactive


@reactive
def sample(
    data: Union[DataFrame, Column],
    n: int = None,
    frac: float = None,
    replace: bool = False,
    weights: Union[str, np.ndarray] = None,
    random_state: Union[int, np.random.RandomState] = None,
) -> Union[DataFrame, Column]:
    """Select a random sample of rows from DataFrame or Column. Roughly
    equivalent to ``sample`` in Pandas
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html.

    Args:
        data (Union[DataFrame, AbstractColumn]): DataFrame or Column to sample from.
        n (int): Number of samples to draw. If `frac` is specified, this parameter
            should not be passed. Defaults to 1 if `frac` is not passed.
        frac (float): Fraction of rows to sample. If `n` is specified, this parameter
            should not be passed.
        replace (bool): Sample with or without replacement. Defaults to False.
        weights (Union[str, np.ndarray]): Weights to use for sampling. If `None`
            (default), the rows will be sampled uniformly. If a numpy array, the
            sample will be weighted accordingly. If a string and `data` is a DataFrame,
            the sampled_df will be applied to the rows based on the column with the name
            specified. If weights do not sum to 1 they will be normalized to sum to 1.
        random_state (Union[int, np.random.RandomState]): Random state or seed to use
            for sampling.

    Return:
        Union[DataFrame, AbstractColumn]: A random sample of rows from DataFrame or
            Column.
    """
    import pandas.core.common as com
    from pandas.core.sample import process_sampling_size
    from pandas.core.sample import sample as _sample

    if isinstance(weights, str):
        if isinstance(data, Column):
            raise ValueError(
                "Weights passed to `sample` must be a numpy array if data is a Column."
            )
        weights = data[weights].to_numpy()

    rs = com.random_state(random_state)
    n = process_sampling_size(n=n, frac=frac, replace=replace)
    if frac is not None:
        n = round(frac * len(data))

    sampled_indices = _sample(
        obj_len=len(data),
        size=n,
        replace=replace,
        weights=weights,
        random_state=rs,
    )
    return data[sampled_indices]
