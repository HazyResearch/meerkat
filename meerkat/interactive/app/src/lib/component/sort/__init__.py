from typing import Any, Dict, List, Sequence, Union

from pydantic import BaseModel

from meerkat.dataframe import DataFrame
from meerkat.interactive.graph import Reference, Store, interface_op, make_store

from ..abstract import Component


class SortCriterion(BaseModel):
    id: str
    is_enabled: bool
    column: str
    ascending: bool
    source: str = ""


@interface_op
def sort_by_criteria(
    data: DataFrame,
    criteria: Sequence[Union[SortCriterion, Dict[str, Any]]],
):
    """Wrapper around mk.sort that adds unpacking of store to the DAG."""
    import meerkat as mk

    # since the criteria can either be a list of dictionary or of FilterCriterion
    # we need to convert them to FilterCriterion
    criteria = [
        criterion
        if isinstance(criterion, SortCriterion)
        else SortCriterion(**criterion)
        for criterion in criteria
    ]

    # Filter out criteria that are disabled.
    criteria = [criterion for criterion in criteria if criterion.is_enabled]
    if len(criteria) == 0:
        return data

    sort_by = [criterion.column for criterion in criteria]
    ascending = [criterion.ascending for criterion in criteria]
    return mk.sort(data, by=sort_by, ascending=ascending)


class Sort(Component):
    """This component handles a sort_by list and a sort_order list.

    Sorting criteria are maintained in a Store. On change of these
    values, the dataframe is sorted.

    This component will return a Reference object, which is a sorted
    view of the dataframe. The sort operation is out-of-place, so a
    new dataframe will be returned as a result of the op.
    """

    name = "Sort"

    def __init__(
        self,
        df: Reference["DataFrame"],
        criteria: Union[Store[List[str]], List[str]] = None,
        title: str = "",
    ):
        super().__init__()
        self.df = df

        if criteria is None:
            criteria = []

        self.criteria = make_store(criteria)  # Dict[str, List[Any]]
        self.title = title

    def derived(self):
        # TODO (arjundd): Add option to configure ascending / descending.
        return sort_by_criteria(self.df, self.criteria)

    @property
    def props(self):
        return {
            "df": self.df.config,
            "criteria": self.criteria.config,
            "title": self.title,
        }
