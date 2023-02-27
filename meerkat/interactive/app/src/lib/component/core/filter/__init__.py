from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from meerkat.columns.abstract import Column
from meerkat.columns.scalar import ScalarColumn
from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.graph import Store, reactive


def filter_by_operator(*args, **kwargs):
    raise NotImplementedError()


def _in(column: Column, value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    if isinstance(column, Column):
        column = column.data

    if not isinstance(column, pd.Series):
        data = pd.Series(column.data)
    else:
        data = column
    return data.isin(value)


# Map a string operator to a function that takes a value and returns a boolean.
_operator_str_to_func = {
    "==": lambda x, y: x == y,  # equality
    "!=": lambda x, y: x != y,  # inequality
    ">": lambda x, y: x > y,  # greater than
    "<": lambda x, y: x < y,  # less than
    ">=": lambda x, y: x >= y,  # greater than or equal to
    "<=": lambda x, y: x <= y,  # less than or equal to
    # TODO (arjundd): Add support for "in" and "not in" operators.
    "in": _in,
    "not in": lambda x, y: ~_in(x, y),
}


class FilterCriterion(BaseModel):
    is_enabled: bool
    column: str
    op: str
    value: Any
    source: Optional[str] = ""
    is_fixed: bool = False


def parse_filter_criterion(criterion: str) -> Dict[str, Any]:
    """Parse the filter criterion from the string.

    Args:
        criterion: The string representation of the criterion.
            Examples: "label == data"

    Returns:
        Dict[str, Any]: The column, op, and value dicts required
            to construct the FilterCriterion.
    """
    # Parse all the longer op keys first.
    # This is to avoid split on a shorter key that could be a substring of a larger key.
    operators = sorted(_operator_str_to_func.keys(), key=lambda x: len(x), reverse=True)
    column = None
    value = None
    for op in operators:
        if op not in criterion:
            continue
        candidates = criterion.split(op)
        if len(candidates) != 2:
            raise ValueError(
                "Expected format: <column> <op> <value> (e.g. 'label == car')."
            )

        column, value = tuple(candidates)
        value = value.strip()
        if "," in value:
            value = [x.strip() for x in value.split(",")]
        return dict(column=column.strip(), value=value, op=op)
    return None
    # raise ValueError(f"Could not find any operation in the string {criterion}")


def _format_criteria(
    criteria: List[Union[FilterCriterion, Dict[str, Any]]]
) -> List[FilterCriterion]:
    # since the criteria can either be a list of dictionary or of FilterCriterion
    # we need to convert them to FilterCriterion
    return [
        criterion
        if isinstance(criterion, FilterCriterion)
        else FilterCriterion(**criterion)
        for criterion in criteria
    ]


@reactive()
def filter(
    data: Union["DataFrame", "Column"],
    criteria: Sequence[Union[FilterCriterion, Dict[str, Any]]],
):
    """Filter data based on operations.

    This operation adds q columns to the dataframe where q is the number of queries.
    Note, if data is a dataframe, this operation is performed in-place.

    TODO (arjundd): Filter numpy and pandas columns first because of speed.

    Args:
        data: A dataframe or column containing the data to embed.
        query: A single or multiple query strings to match against.
        input: If ``data`` is a dataframe, the name of the column
            to embed. If ``data`` is a column, then the parameter is ignored.
            Defaults to None.
        input_modality: The input modality. If None, infer from the input column.
        query_modality: The query modality. If None, infer from the query column.
        return_column_names: Whether to return the names of columns added based
            on match.

    Returns:
        mk.DataFrame: A view of ``data`` with a new column containing the embeddings.
        This column will be named according to the ``out_col`` parameter.
    """
    # since the criteria can either be a list of dictionary or of FilterCriterion
    # we need to convert them to FilterCriterion
    criteria = _format_criteria(criteria)

    # Filter out criteria that are disabled.
    criteria = [criterion for criterion in criteria if criterion.is_enabled]

    if len(criteria) == 0:
        # we view so that the result is a different dataframe than the input
        return data.view()

    # Filter pandas series columns.c
    # TODO (arjundd): Make this more efficient to perform filtering sequentially.
    all_masks = []
    for criterion in criteria:
        col = data[criterion.column]

        # values should be split by "," when using in/not-in operators.
        if "in" in criterion.op:
            if isinstance(criterion.value, str):
                value = [x.strip() for x in criterion.value.split(",")]
            elif isinstance(criterion.value, list):
                value = criterion.value
            else:
                raise ValueError(
                    "Expected a list or comma-separated string "
                    f"for value {criterion.value}."
                )
        else:
            if col.dtype in (np.bool_, bool):
                value = criterion.value not in ("False", "false", "0")
            else:
                value = col.dtype.type(criterion.value)

        # Remove trailing and leading "" if the value is a string.
        if isinstance(value, str):
            value = value.strip('"').strip("'")

        # TODO: this logic will fail when the column is a boolean column
        # beacuse all values will be rendered as strings. If the string
        # is not empty, col.dtype will cast the string to True.
        # e.g. np.asarray("False", dtype=np.bool) --> True
        if isinstance(col, ScalarColumn):
            value = np.asarray(value, dtype=col.dtype)
        if "in" in criterion.op:
            value = value.tolist()

        # FIXME: Figure out why we cannot pass col for PandasSeriesColumn.
        # the .data accessor is an interim solution.
        mask = _operator_str_to_func[criterion.op](col.data, value)
        all_masks.append(np.asarray(mask))
    mask = np.stack(all_masks, axis=1).all(axis=1)
    return data[mask]


class Filter(Component):
    df: DataFrame = None
    criteria: Store[List[FilterCriterion]] = Field(
        default_factory=lambda: Store(list())
    )
    operations: Store[List[str]] = Field(
        default_factory=lambda: Store(list(_operator_str_to_func.keys()))
    )

    title: str = "Filter"

    def __init__(
        self,
        df: DataFrame = None,
        *,
        criteria: List[FilterCriterion] = [],
        operations: List[str] = list(_operator_str_to_func.keys()),
    ):
        """Filter a dataframe based on a list of filter criteria.

        Filtering criteria are maintained in a Store. On change of
        values in the store, the dataframe is filtered.
        """
        super().__init__(
            df=df,
            criteria=criteria,
            operations=operations,
        )

    def __call__(self, df: DataFrame = None) -> DataFrame:
        if df is None:
            df = self.df
        return filter(df, self.criteria)
