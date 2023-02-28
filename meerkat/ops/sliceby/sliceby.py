from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

from meerkat.dataframe import DataFrame
from meerkat.interactive.graph.reactivity import reactive
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
    _self_identifiable_group: str = "slicebys"

    def __init__(
        self,
        data: DataFrame,
        by: Union[List[str], str],
        sets: Dict[Union[SliceKey, Tuple[SliceKey]], np.ndarray] = None,
        scores: Dict[Union[SliceKey, Tuple[SliceKey], np.ndarray]] = None,
        masks: Dict[Union[SliceKey, Tuple[SliceKey]], np.ndarray] = None,
    ):
        super().__init__()
        # exactly one of sets and scores must be provided
        if (sets is None) == (scores is None):
            raise ValueError("Exactly one of sets and scores must be provided")

        self.slice_type = "scores" if scores is not None else "sets"
        self.slices = scores if scores is not None else sets
        self.data = data
        if isinstance(by, str):
            by = [by]
        self.by = by

        # # prepare the gui object
        # from meerkat.interactive.gui import SliceByGUI

        # self.gui = SliceByGUI(self)
        self.slice = SliceIndexer(self)

    def __len__(self) -> int:
        return len(self.slices)

    def mean(self, *args, **kwargs) -> DataFrame:
        return self._aggregate(lambda x: x.mean(*args, **kwargs))

    @sets_only
    def count(self, *args, **kwargs) -> DataFrame:
        return self._aggregate(lambda x: len(x))

    @sets_only
    def median(self, *args, **kwargs) -> DataFrame:
        return self._aggregate(lambda x: x.median(*args, **kwargs))

    @sets_only
    def aggregate(self, function: Callable, accepts_df: bool = False) -> DataFrame:
        """_summary_

        Args:
            function (Callable): _description_
            accepts_df (bool, optional): _description_. Defaults to False.

        Returns:
            DataFrame: _description_
        """
        return self._aggregate(f=function, accepts_df=accepts_df)

    @property
    def slice_keys(self):
        return sorted(list(self.slices.keys()))

    def _aggregate(self, f: Callable, accepts_df: bool = False) -> DataFrame:
        """self.sets are a dictionary of {labels : [sets]}"""

        # means will be a list of dictionaries where each element in the dict
        out = []

        # TODO (Sabri): This is an extremely slow way of doing this – we need to
        # vectorize it
        for slice_key in self.slice_keys:
            if self.slice_type == "scores":
                raise NotImplementedError
            else:
                slice_df = self.data[self.slices[slice_key]]
                slice_values: Dict[str, Any] = slice_df.aggregate(
                    f, accepts_df=accepts_df
                )

            out.append(slice_values)

        from meerkat.dataframe import DataFrame

        # create DataFrame as a list of rows.
        out = DataFrame(out)

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
            sorted = self.data[np.argsort(-np.array(self.slices[slice_key]))]
            return sorted._get(index, materialize=materialize)

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


@reactive
def sliceby(
    data: DataFrame,
    by: Union[str, Sequence[str]] = None,
    key_mapping: Dict[int, str] = None,
) -> SliceBy:
    """Perform a groupby operation on a DataFrame or Column (similar to a
    `DataFrame.groupby` and `Series.groupby` operations in Pandas).j.

    Args:
        data (Union[DataFrame, AbstractColumn]): The data to group.
        by (Union[str, Sequence[str]]): The column(s) to group by. Ignored if ``data``
            is a Column.

    Returns:
        Union[DataFrameGroupBy, AbstractColumnGroupBy]: A GroupBy object.
    """
    if isinstance(by, str):
        by = [by]
    return SliceBy(
        data=data,
        by="slice",
        sets={curr_by: np.where(data[curr_by] == 1)[0] for curr_by in by},
    )
