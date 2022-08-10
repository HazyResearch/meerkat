from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

from meerkat.datapanel import DataPanel
from meerkat.mixins.identifiable import IdentifiableMixin


def sets_only(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        if self.slice_type == "sets":
            return fn(self, *args, **kwargs)
        else:
            raise ValueError("This method is only valid for sets")

    return wrapped


class SliceBy(IdentifiableMixin):

    identifiable_group: str = "sliceby"

    def __init__(
        self,
        data: DataPanel,
        by: Union[List[str], str],
        sets: Dict[Union[str, Tuple[str]], np.ndarray] = None,
        scores: Dict[Union[str, Tuple[str], np.ndarray]] = None,
    ):
        # exactly one of sets and scores must be provided
        if (sets is None) == (scores is None):
            raise ValueError("Exactly one of sets and scores must be provided")

        self.slice_type = "scores" if scores is not None else "sets"

        self.sets = sets
        self.scores = scores
        self.data = data
        self.by = by

    def mean(self, *args, **kwargs):
        return self._reduce(lambda x: x.mean(*args, **kwargs))

    @sets_only
    def count(self, *args, **kwargs):
        return self._reduce(lambda x: x.count(*args, **kwargs))

    @sets_only
    def median(self, *args, **kwargs):
        return self._reduce(lambda x: x.median(*args, **kwargs))

    def _reduce(self, f: Callable):
        """self.sets are a dictionary of {labels : [sets]}"""

        # sorting them so that they appear in a nice order.
        slice_keys = sorted(list(self.sets.keys()))

        # means will be a list of dictionaries where each element in the dict
        out = []
        for slice_key in slice_keys:
            if self.slice_type == "scores":
                pass
            else:
                slice_dp = self.data.lz[self.sets[slice_key]]
                slice_values: Dict[str, Any] = f(slice_dp)
            out.append(slice_values)

        from meerkat.datapanel import DataPanel

        # create DataPanel as a list of rows.
        out = DataPanel(out)

        # add the by columns.
        if len(slice_keys) > 0:
            if len(self.by) > 1:
                columns = list(zip(*slice_keys))
                for i, col in enumerate(self.by):
                    out[col] = columns[i]
            else:
                col = self.by[0]
                out[col] = slice_keys
        return out

    def __getitem__(self, key: Union[str, Sequence[str]]) -> SliceBy:
        if isinstance(key, str):
            key = [key]

        return self.__class__(data=self.data[key], sets=self.sets, by=self.by)
