import collections.abc
from typing import List, Sequence, Union

import numpy as np

from meerkat import DataPanel, ListColumn
from meerkat.columns.cell_column import CellColumn
from meerkat.columns.numpy_column import NumpyArrayColumn
from meerkat.columns.tensor_column import TensorColumn
from meerkat.errors import MergeError
from meerkat.provenance import capture_provenance


@capture_provenance(capture_args=["left_on", "on", "right_on", "how"])
def merge(
    left: DataPanel,
    right: DataPanel,
    how: str = "inner",
    on: Union[str, List[str]] = None,
    left_on: Union[str, List[str]] = None,
    right_on: Union[str, List[str]] = None,
    sort: bool = False,
    suffixes: Sequence[str] = ("_x", "_y"),
    validate=None,
    keep_indexes: bool = False,
):
    if how == "cross":
        raise ValueError("DataPanel does not support cross merges.")  # pragma: no cover

    if (on is None) and (left_on is None) and (right_on is None):
        raise MergeError("Merge expects either `on` or `left_on` and `right_on`")

    left_on = on if left_on is None else left_on
    right_on = on if right_on is None else right_on
    # cast `left_on` and `right_on` to lists
    left_on = [left_on] if isinstance(left_on, str) else left_on
    right_on = [right_on] if isinstance(right_on, str) else right_on

    # ensure we can merge on specified columns
    _check_merge_columns(left, left_on)
    _check_merge_columns(right, right_on)

    # convert datapanels to dataframes so we can apply Pandas merge
    # (1) only include columns we are joining on
    left_df = left[left_on].to_pandas()
    right_df = right[right_on].to_pandas()
    # (2) add index columns, which we'll use to reconstruct the columns we excluded from
    # the Pandas merge
    if ("__right_indices__" in right_df) or ("__left_indices__" in left_df):
        raise MergeError(
            "The column names '__right_indices__' and '__left_indices__' cannot appear "
            "in the right and left panels respectively. They are used by merge."
        )
    left_df["__left_indices__"] = np.arange(len(left_df))
    right_df["__right_indices__"] = np.arange(len(right_df))

    # apply pandas merge
    merged_df = left_df.merge(
        right_df,
        how=how,
        left_on=left_on,
        right_on=right_on,
        sort=sort,
        validate=validate,
        suffixes=suffixes,
    )
    left_indices = merged_df.pop("__left_indices__").values
    right_indices = merged_df.pop("__right_indices__").values
    merged_df = merged_df[set(left_on) & set(right_on)]

    # reconstruct other columns not in the `left_on & right_on` using `left_indices`
    # and `right_indices`, the row order returned by merge
    def _cols_to_construct(dp: DataPanel):
        # don't construct columns in both `left_on` and `right_on` because we use
        # `merged_df` for these
        return [k for k in dp.keys() if k not in (set(left_on) & set(right_on))]

    new_left = _construct_from_indices(left[_cols_to_construct(left)], left_indices)
    new_right = _construct_from_indices(right[_cols_to_construct(right)], right_indices)

    # concatenate the two new datapanels
    merged_dp = new_left.append(new_right, axis="columns", suffixes=suffixes)

    # add columns in both `left_on` and `right_on`, casting to the column type in left
    for name, column in merged_df.iteritems():
        merged_dp.add_column(name, left[name]._clone(data=column.values))
        merged_dp.visible_columns = (
            merged_dp.visible_columns[-1:] + merged_dp.visible_columns[:-1]
        )
    if (
        not keep_indexes
        and ("index" + suffixes[0]) in merged_dp
        and ("index" + suffixes[1]) in merged_dp
    ):
        merged_dp.remove_column("index" + suffixes[0])
        merged_dp.remove_column("index" + suffixes[1])

    return merged_dp


def _construct_from_indices(dp: DataPanel, indices: np.ndarray):
    if np.isnan(indices).any():
        # when performing "outer", "left", and "right" merges, column indices output
        # by pandas merge can include `nan` in rows corresponding to merge keys that
        # only appear in one of the two panels. For these columns, we convert the
        # column to  ListColumn, and fill with "None" wherever indices is "nan".
        data = {
            name: ListColumn(
                [None if np.isnan(index) else col.lz[int(index)] for index in indices]
            )
            for name, col in dp.items()
        }
        return dp._clone(data=data)
    else:
        # if there are no `nan`s in the indices, then we can just lazy index the
        # original column
        return dp.lz[indices]


def _check_merge_columns(dp: DataPanel, on: List[str]):
    for name in on:
        column = dp[name]
        if isinstance(column, NumpyArrayColumn) or isinstance(column, TensorColumn):
            if len(column.shape) > 1:
                raise MergeError(
                    f"Cannot merge on column `{name}`, has more than one dimension."
                )
        elif isinstance(column, ListColumn):
            if not all(
                [isinstance(cell, collections.abc.Hashable) for cell in column.lz]
            ):
                raise MergeError(
                    f"Cannot merge on column `{name}`, contains unhashable objects."
                )

        elif isinstance(column, CellColumn):
            if not all(
                [isinstance(cell, collections.abc.Hashable) for cell in column.lz]
            ):
                raise MergeError(
                    f"Cannot merge on column `{name}`, contains unhashable cells."
                )
