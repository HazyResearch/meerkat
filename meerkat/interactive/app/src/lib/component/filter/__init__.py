from dataclasses import dataclass
from ..abstract import Component
from typing import TYPE_CHECKING, Dict, Any, Optional, Sequence
from meerkat.interactive.graph import Box, Store, make_store

from typing import List, Union, Any

from meerkat.interactive.graph import interface_op
import functools
import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from meerkat import AbstractColumn, DataPanel


# Map a string operator to a function that takes a value and returns a boolean.
_operator_str_to_func = {
    "==": lambda x, y: x == y,  # equality
    "!=": lambda x, y: x != y,  # inequality
    ">": lambda x, y: x > y,  # greater than
    "<": lambda x, y: x < y,  # less than
    ">=": lambda x, y: x >= y,  # greater than or equal to
    "<=": lambda x, y: x <= y,  # less than or equal to
    # TODO (arjundd): Add support for "in" and "not in" operators.
    # "in": lambda x, y: x in y,  # in
    # "not in": lambda x, y: x not in y,  # not in
}


class FilterCriterion(BaseModel):
    is_enabled: bool
    column: str
    op: str
    value: Any
    source: Optional[str] = ""
    is_fixed: bool = False


@interface_op
def filter_by_operator(
    data: Union["DataPanel", "AbstractColumn"],
    criteria: Sequence[Union[FilterCriterion, Dict[str, Any]]],
):
    """Filter data based on operations.

    This operation adds q columns to the datapanel where q is the number of queries.
    Note, if data is a datapanel, this operation is performed in-place.

    TODO (arjundd): Filter numpy and pandas columns first because of speed.

    Args:
        data: A datapanel or column containing the data to embed.
        query: A single or multiple query strings to match against.
        input: If ``data`` is a datapanel, the name of the column
            to embed. If ``data`` is a column, then the parameter is ignored.
            Defaults to None.
        input_modality: The input modality. If None, infer from the input column.
        query_modality: The query modality. If None, infer from the query column.
        return_column_names: Whether to return the names of columns added based
            on match.

    Returns:
        mk.DataPanel: A view of ``data`` with a new column containing the embeddings.
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
        # FIXME: Do we need to return a new DataPanel so that it does not point
        # to the pivot?
        return data

    supported_column_types = (mk.PandasSeriesColumn, mk.NumpyArrayColumn)
    # if not all(
    #     isinstance(data[column], supported_column_types) for column in input_columns
    # ):
    #     raise ValueError(f"All columns must be one of {supported_column_types}")

    # Filter pandas series columns.
    # TODO (arjundd): Make this more efficient to perform filtering sequentially.
    all_masks = []
    for criterion in criteria:
        col = data[criterion.column]
        value = col.dtype.type(criterion.value)
        if isinstance(col, mk.NumpyArrayColumn):
            value = np.asarray(value, dtype=col.dtype)

        mask = _operator_str_to_func[criterion.op](col, value)
        all_masks.append(np.asarray(mask))
    mask = np.stack(all_masks, axis=1).all(axis=1)

    return data.lz[mask]


class Filter(Component):
    """This component handles filtering of the pivot datapanel.

    Filtering criteria are maintained in a Store. On change of values
    in the store, the datapanel is filtered.

    This component will return a Derived object, which can be used downstream.

    We recommend performing filtering before other out-of-place operations,
    like sorting, to avoid unnecessary computation.
    """

    name = "Filter"

    def __init__(
        self,
        dp: Box["DataPanel"],
        criteria: Union[Store[List[FilterCriterion]], List[FilterCriterion]] = None,
    ):
        super().__init__()
        self.dp = dp

        if criteria is None:
            criteria = []

        self.criteria = make_store(criteria)  # Dict[str, List[Any]]
        self.operations = list(_operator_str_to_func.keys())

    def derived(self):
        """Return a derived object that filters the pivot datapanel."""
        return filter_by_operator(self.dp, self.criteria)

    @property
    def props(self):
        return {
            "dp": self.dp.config,
            "criteria": self.criteria.config,
            "operations": self.operations,
        }
