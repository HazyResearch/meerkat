from typing import Callable, Dict, Type

import numpy as np
import pandas as pd
import rich

from meerkat.env import is_torch_available
from meerkat.interactive.graph.reactivity import reactive
from meerkat.tools.lazy_loader import LazyLoader

torch = LazyLoader("torch")

print = reactive(rich.print)


def get_custom_json_encoder() -> Dict[Type, Callable]:
    from meerkat.columns.abstract import Column
    from meerkat.interactive.graph.store import Store

    custom_encoder = {
        np.ndarray: lambda v: v.tolist(),
        pd.Series: lambda v: v.tolist(),
        Column: lambda v: v.to_json(),
        np.int64: lambda v: int(v),
        np.float64: lambda v: float(v),
        np.int32: lambda v: int(v),
        np.bool_: lambda v: bool(v),
        np.bool8: lambda v: bool(v),
        Store: lambda v: v.to_json(),
    }

    if is_torch_available():
        custom_encoder[torch.Tensor] = lambda v: v.tolist()
    return custom_encoder
