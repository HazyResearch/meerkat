from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.interactive.graph import Store, interface_op, make_store

from ..abstract import Component

if TYPE_CHECKING:
    from meerkat import DataFrame


def _in(column: AbstractColumn, value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    if not isinstance(column, PandasSeriesColumn):
        data = pd.Series(column.data)
    else:
        data = column.data
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


@interface_op
def filter_by_operator(
    data: Union["DataFrame", "AbstractColumn"],
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
    import meerkat as mk

    # since the criteria can either be a list of dictionary or of FilterCriterion
    # we need to convert them to FilterCriterion
    criteria = [
        criterion
        if isinstance(criterion, FilterCriterion)
        else FilterCriterion(**criterion)
        for criterion in criteria
    ]

    # Filter out criteria that are disabled.
    criteria = [criterion for criterion in criteria if criterion.is_enabled]

    if len(criteria) == 0:
        # FIXME: Do we need to return a new DataFrame so that it does not point
        # to the pivot?
        return data

    (mk.PandasSeriesColumn, mk.NumpyArrayColumn)
    # if not all(
    #     isinstance(data[column], supported_column_types) for column in input_columns
    # ):
    #     raise ValueError(f"All columns must be one of {supported_column_types}")

    # Filter pandas series columns.
    # TODO (arjundd): Make this more efficient to perform filtering sequentially.
    all_masks = []
    for criterion in criteria:
        col = data[criterion.column]

        # values should be split by "," when using in/not-in operators.
        if "in" in criterion.op:
            value = [x.strip() for x in criterion.value.split(",")]
            if isinstance(col, mk.NumpyArrayColumn):
                value = np.asarray(value, dtype=col.dtype).tolist()
        else:
            value = col.dtype.type(criterion.value)
            if isinstance(col, mk.NumpyArrayColumn):
                value = np.asarray(value, dtype=col.dtype)

        mask = _operator_str_to_func[criterion.op](col, value)
        all_masks.append(np.asarray(mask))
    mask = np.stack(all_masks, axis=1).all(axis=1)

    return data.lz[mask]


class Filter(Component):
    """This component handles filtering of the pivot dataframe.

    Filtering criteria are maintained in a Store. On change of values
    in the store, the dataframe is filtered.

    This component will return a Reference object, which can be used downstream.

    We recommend performing filtering before other out-of-place operations,
    like sorting, to avoid unnecessary computation.
    """

    name = "Filter"

    def __init__(
        self,
        df: DataFrame,
        criteria: Union[Store[List[FilterCriterion]], List[FilterCriterion]] = None,
        title: str = "",
    ):
        super().__init__()
        self.df = df

        if criteria is None:
            criteria = []

        self.criteria = make_store(criteria)  # Dict[str, List[Any]]
        self.operations = list(_operator_str_to_func.keys())
        self.title = title

    def derived(self):
        """Return a derived object that filters the pivot dataframe."""
        return filter_by_operator(self.df, self.criteria)

    @property
    def props(self):
        return {
            "df": self.df.config,  # FIXME
            "criteria": self.criteria.config,
            "operations": self.operations,
            "title": self.title,
        }
