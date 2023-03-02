from typing import Union

import numpy as np

from meerkat import Column, DataFrame
from meerkat.interactive.graph.reactivity import reactive


@reactive()
def shuffle(data: Union[DataFrame, Column], seed=None) -> Union[DataFrame, Column]:
    """Shuffle the rows of a DataFrame or Column.

    Shuffling is done out-of-place and with numpy.

    Args:
        data (Union[DataFrame, Column]): DataFrame or Column to shuffle.
        seed (int): Seed to use for shuffling.

    Returns:
        Union[DataFrame, Column]: Shuffled DataFrame or Column.
    """
    idx = np.arange(len(data))
    state = np.random.RandomState(seed) if seed is not None else np.random
    state.shuffle(idx)
    return data[idx]
