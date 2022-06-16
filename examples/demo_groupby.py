from typing import Sequence, Union
from meerkat import DataPanel, PandasSeriesColumn, TensorColumn, NumpyArrayColumn
from meerkat.clever.utils import DeferredOp
import pandas as pd
import numpy as np
from pandas.core.groupby import SeriesGroupBy
import torch

from domino import DominoSlicer

from meerkat.columns.abstract import AbstractColumn


def groupby(dp: DataPanel, by: Union[str, Sequence[str]]):
    if isinstance(by, str):
        by = [by]

    # apply deferred ops
    updated_by = []
    for column in by:
        if isinstance(column, DeferredOp):
            column(data=dp)
            updated_by.append(column.out_col)
        else:
            updated_by.append(column)
    by = updated_by

    if len(by) == 1:
        # TODO: this is exceedingly hacky and will only work in cases where `by`` is a
        # numpy array column with contiguous integers starting at 0
        groups = dp[by[0]].data
        group_name = by[0]
    else:
        method = DominoSlicer(
            y_log_likelihood_weight=5,
            y_hat_log_likelihood_weight=5,
            n_slices=n_groups,
            n_mixture_components=n_groups,
        )

        method.fit(data=dp, embeddings=by[0], targets=by[1], pred_probs=by[2])
        groups = method.predict(
            data=dp, embeddings=by[0], targets=by[1], pred_probs=by[2]
        )
        groups = groups.argmax(axis=-1)

        group_name = "-".join(by)
        dp[group_name] = groups

    return DataPanelGroupBy(dp=dp, groups=groups, group_name=group_name)


class DataPanelGroupBy:
    def __init__(
        self,
        dp: DataPanel,
        groups: Union[str, NumpyArrayColumn],
        group_name: str = "group",
    ):

        self.dp = dp
        if isinstance(groups, str):
            self.groups = dp[groups]
        else:
            self.groups = groups

        self.group_name = group_name

        def _include_column(column) -> bool:
            return (
                isinstance(column, PandasSeriesColumn)
                or (isinstance(column, NumpyArrayColumn) and len(column.shape) == 1)
                or (isinstance(column, TensorColumn) and len(column.shape) == 1)
            )

        def _prepare_column(column) -> pd.Series:
            series = pd.Series(
                column.data
                if not isinstance(column, TensorColumn)
                else column.data.numpy()
            )
            return series.reset_index(drop=True)

        self.df_groupby = pd.DataFrame(
            {
                self.group_name: self.groups,
                **{
                    name: _prepare_column(column)
                    for name, column in dp.items()
                    if _include_column(column)
                },
            }
        ).groupby(by=self.group_name)

    def mean(self):
        dp = {}
        for name in self.dp.columns:
            res = self[name].mean()
            if self.group_name not in dp:
                print(self.group_name)
                dp[self.group_name] = res[self.group_name]

            dp[name] = res[name]
        return DataPanel(dp)

    def __getitem__(self, column: Union[str, Sequence[str]]):
        if isinstance(column, str):
            try:
                # TODO return object that returns datapanel
                return PandasGroupBy(self.df_groupby[column])
            except KeyError:
                column_name = column
                column = self.dp[column]
                if isinstance(column, NumpyArrayColumn):
                    return NumPyArrayGroupBy(
                        column=column,
                        groups=self.groups,
                        column_name=column_name,
                        group_name=self.group_name,
                    )
                elif isinstance(column, TensorColumn):
                    return TensorGroupBy(
                        column=column,
                        groups=self.groups,
                        column_name=column_name,
                        group_name=self.group_name,
                    )

            column = self.dp[column]
            raise ValueError(
                f"GroupBy not implemented for column of type {type(column)}."
            )
        else:
            return DataPanelGroupBy(
                dp=self.dp[column], groups=self.groups, group_name=self.group_name
            )


class PandasGroupBy:
    def __init__(self, groupby: SeriesGroupBy):
        self.groupby = groupby

    def mean(self):
        return DataPanel.from_pandas(self.groupby.mean().reset_index())


class ColumnGroupBy:
    def __init__(
        self,
        column: TensorColumn,
        groups: NumpyArrayColumn,
        column_name: str = "value",
        group_name: str = "group",
    ) -> None:

        self.column = column
        self.groups = torch.tensor(groups.data)
        self.column_name = column_name
        self.group_name = group_name

    def _wrap_in_dp(self, values: AbstractColumn):
        return DataPanel(
            {
                self.group_name: np.unique(self.groups.numpy()),
                self.column_name: values,
            }
        )


class NumPyArrayGroupBy(ColumnGroupBy):
    """
    Considered an implementationn like this, but hit a roadblock because there's no
    scatter_add in numpy:
    ```
            import numpy as np

        samples = np.array([
            [0.1, 0.1],    #-> group / class 1
            [0.2, 0.2],    #-> group / class 2
            [0.4, 0.4],    #-> group / class 2
            [0.0, 0.0]     #-> group / class 0
        ])

        labels = np.array([1, 2, 2, 0])
        labels = np.repeat(labels[:, np.newaxis], repeats=2,axis=1)

        unique_labels, labels_count = np.unique(labels, axis=0, return_counts=True)

        res = np.zeros_like(unique_labels, dtype=float)
        res = np.take_along_axis(
            arr=samples,
            axis=0,
            indices=labels,
            #values=samples
        )
        res = res / labels_count[:, np.newaxis]
        res
    ```
    TODO: figure this out
    """

    def mean(self):
        return self._wrap_in_dp(
            _tensor_mean_groupby(
                data=torch.tensor(self.column.data), groups=self.groups
            ).numpy()
        )

    def sum(self):
        return self._wrap_in_dp(
            _tensor_sum_groupby(
                data=torch.tensor(self.column.data), groups=self.groups
            ).numpy()
        )


class TensorGroupBy(ColumnGroupBy):
    def mean(self):
        return self._wrap_in_dp(
            _tensor_mean_groupby(data=self.column.data, groups=self.groups)
        )

    def sum(self):
        return self._wrap_in_dp(
            _tensor_sum_groupby(data=self.column.data, groups=self.groups)
        )


def _tensor_mean_groupby(data, groups):
    assert len(data) == len(groups)
    assert len(data.shape) == 2
    res, groups_count = _tensor_sum_groupby(
        data=data, groups=groups, return_counts=True
    )
    res = res / groups_count.float().unsqueeze(1)

    return res


def _tensor_sum_groupby(data, groups, return_counts: bool = False):
    assert len(data) == len(groups)
    assert len(data.shape) == 2

    groups = groups.view(groups.size(0), 1).expand(-1, data.size(1)).to(torch.long)

    if return_counts:
        unique_groups, groups_count = groups.unique(dim=0, return_counts=return_counts)
    else:
        unique_groups = groups.unique(dim=0, return_counts=return_counts)

    res = torch.zeros_like(unique_groups, dtype=data.dtype).scatter_add_(
        dim=0, index=groups, src=data
    )

    return res, groups_count if return_counts else res
