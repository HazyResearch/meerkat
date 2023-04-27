from typing import Callable, Dict, Mapping, Type

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
    from meerkat.interactive.endpoint import Endpoint
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
        Endpoint: lambda v: v.to_json(),
    }

    if is_torch_available():
        custom_encoder[torch.Tensor] = lambda v: v.tolist()
    return custom_encoder


def is_equal(a, b):
    """Recursively check equality of two objects.

    This also verifies that the types of the objects are the same.

    Args:
        a: The first object.
        b: The second object.

    Returns:
        True if the objects are equal, False otherwise.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, type(b)) and isinstance(b, type(a)) and np.all(a, b)
    elif isinstance(a, pd.Series) or isinstance(b, pd.Series):
        return isinstance(a, type(b)) and isinstance(b, type(a)) and np.all(a == b)
    elif isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (
            isinstance(a, type(b))
            and isinstance(b, type(a))
            and len(a) == len(b)
            and all(is_equal(_a, _b) for _a, _b in zip(a, b))
        )
    elif isinstance(a, Mapping) or isinstance(b, Mapping):
        a_keys = a.keys()
        b_keys = b.keys()
        return (
            isinstance(a, type(b))
            and isinstance(b, type(a))
            and len(a) == len(b)
            and a_keys == b_keys
            and all(is_equal(a[k], b[k]) for k in a_keys)
        )
    else:
        return a == b
