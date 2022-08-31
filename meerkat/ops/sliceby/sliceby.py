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


SliceKey = Union[str, int]


class SliceBy(IdentifiableMixin):

    identifiable_group: str = "slicebys"

    def __init__(
        self,
        data: DataPanel,
        by: Union[List[str], str],
        sets: Dict[Union[SliceKey, Tuple[SliceKey]], np.ndarray] = None,
        scores: Dict[Union[SliceKey, Tuple[SliceKey], np.ndarray]] = None,
    ):
        super().__init__()
        # exactly one of sets and scores must be provided
        if (sets is None) == (scores is None):
            raise ValueError("Exactly one of sets and scores must be provided")

        self.slice_type = "scores" if scores is not None else "sets"
        self.slices = scores if scores is not None else sets
        self.data = data
        self.by = by

        # prepare the gui object
        from meerkat.interactive.gui import SliceByGUI

        self.gui = SliceByGUI(self)
        self.slice = SliceIndexer(self)

    def __len__(self) -> int:
        return len(self.slices)

    def mean(self, *args, **kwargs) -> DataPanel:
        return self._aggregate(lambda x: x.mean(*args, **kwargs))

    @sets_only
    def count(self, *args, **kwargs) -> DataPanel:
        return self._aggregate(lambda x: x.count(*args, **kwargs))

    @sets_only
    def median(self, *args, **kwargs) -> DataPanel:
        return self._aggregate(lambda x: x.median(*args, **kwargs))

    @sets_only
    def aggregate(self, function: Callable, accepts_dp: bool = False) -> DataPanel:
        """_summary_

        Args:
            function (Callable): _description_
            accepts_dp (bool, optional): _description_. Defaults to False.

        Returns:
            DataPanel: _description_
        """
        return self._aggregate(f=function, accepts_dp=accepts_dp)

    @property
    def slice_keys(self):
        return sorted(list(self.slices.keys()))

    def _aggregate(self, f: Callable, accepts_dp: bool = False) -> DataPanel:
        """self.sets are a dictionary of {labels : [sets]}"""

        # means will be a list of dictionaries where each element in the dict
        out = []
        for slice_key in self.slice_keys:
            if self.slice_type == "scores":
                pass
            else:
                slice_dp = self.data.lz[self.slices[slice_key]]
                slice_values: Dict[str, Any] = slice_dp.aggregate(
                    f, accepts_dp=accepts_dp
                )

            out.append(slice_values)

        from meerkat.datapanel import DataPanel

        # create DataPanel as a list of rows.
        out = DataPanel(out)

        # add the by columns.
        if len(self.slice_keys) > 0:
            if len(self.by) > 1:
                columns = list(zip(*self.slice_keys))
                for i, col in enumerate(self.by):
                    out[col] = columns[i]
            else:
                col = self.by[0]
                out[col] = self.slice_keys
        return out

    def _get(self, slice_key: str, index, materialize: bool = False) -> List[Any]:
        """Get rows from a slice by."""
        if self.slice_type == "sets":
            return self.data._get(
                self.slices[slice_key][index], materialize=materialize
            )
        else:
            raise NotImplemented

    def get_slice_length(self, slice_key: SliceKey) -> int:
        if self.slice_type == "sets":
            return len(self.slices[slice_key])
        else:
            return len(self.data)

    def __getitem__(self, column: Union[str, Sequence[str]]) -> SliceBy:
        if isinstance(column, str):
            column = [column]

        if self.slice_type == "sets":
            return self.__class__(data=self.data[column], sets=self.slices, by=self.by)
        else:
            return self.__class__(
                data=self.data[column], scores=self.slices, by=self.by
            )


class SliceIndexer:
    def __init__(self, obj: object):
        self.obj = obj

    def __getitem__(self, index):
        key, index = index
        return self.obj._get(key, index, materialize=False)
