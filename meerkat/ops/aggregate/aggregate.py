import warnings
from typing import Any, Callable, Dict, Union

import meerkat as mk
from meerkat.interactive.graph.reactivity import reactive

from ...mixins.aggregate import AggregationError


@reactive()
def aggregate(
    data: mk.DataFrame,
    function: Union[Callable, str],
    nuisance: str = "drop",
    accepts_df: bool = False,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """"""
    if nuisance not in ["drop", "raise", "warn"]:
        raise ValueError(f"{nuisance} is not a valid nuisance option")

    if accepts_df and not isinstance(function, Callable):
        raise ValueError("Must pass a callable to aggregate if accepts_df is True")

    if accepts_df:
        return {"df": function(data, *args, **kwargs)}

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
