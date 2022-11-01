import warnings
from typing import Any, Callable, Dict, Union

import meerkat as mk

from ...mixins.aggregate import AggregationError


def aggregate(
    data: mk.DataPanel,
    function: Union[Callable, str],
    nuisance: str = "drop",
    accepts_dp: bool = False,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """"""
    if nuisance not in ["drop", "raise", "warn"]:
        raise ValueError(f"{nuisance} is not a valid nuisance option")

    if accepts_dp and not isinstance(function, Callable):
        raise ValueError("Must pass a callable to aggregate if accepts_dp is True")

    if accepts_dp:
        return {"dp": function(data, *args, **kwargs)}

    result = {}

    for name, column in data.items():
        try:
            result[name] = column.aggregate(function, *args, **kwargs)
        except AggregationError as e:
            if nuisance == "drop":
                continue
            elif nuisance == "raise":
                raise e
            elif nuisance == "warn":
                warnings.warn(str(e))
                continue
    return result
