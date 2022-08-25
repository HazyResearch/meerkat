from functools import wraps
from itertools import product
from typing import Any, Dict, Sequence

import pytest


@wraps(pytest.mark.parametrize)
def product_parametrize(params: Dict[str, Sequence[Any]], **kwargs):
    """Wrapper around pytest.mark.parametrize with a simpler interface."""
    argvalues, ids = zip(
        *[
            (v, ",".join(map(str, v))) if len(v) > 1 else (v[0], str(v[0]))
            for v in product(*params.values())
        ]
    )
    params = {
        "argnames": ",".join(params.keys()),
        "argvalues": argvalues,
        "ids": ids,
    }

    return pytest.mark.parametrize(
        **params,
        **kwargs,
    )
