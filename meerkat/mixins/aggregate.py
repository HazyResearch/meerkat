from typing import Callable, Union


class AggregationError(ValueError):
    pass


class AggregateMixin:
    AGGREGATIONS = [
        "mean",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__()

    def aggregate(self, function: Union[Callable, str], *args, **kwargs):
        if isinstance(function, str):
            if function not in self.AGGREGATIONS:
                raise ValueError(f"{function} is not a valid aggregation")
            return getattr(self, function)(*args, **kwargs)
        else:
            return function(self, *args, **kwargs)

    def mean(self, *args, **kwargs):
        raise AggregationError(
            f"Aggregation 'mean' not implemented for column of type {type(self)}."
        )
