import uuid
from typing import Any, Dict, List, Sequence, Union

from pydantic import BaseModel

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.graph import reactive


class SortCriterion(BaseModel):
    id: str
    is_enabled: bool
    column: str
    ascending: bool
    source: str = ""


def _format_criteria(
    criteria: Sequence[Union[SortCriterion, Dict[str, Any]]]
) -> List[SortCriterion]:
    # since the criteria can either be a list of dictionary or of SortCriterion
    # we need to convert them to SortCriterion
    return [
        criterion
        if isinstance(criterion, SortCriterion)
        else SortCriterion(**criterion)
        for criterion in criteria
    ]


@reactive()
def sort_by_criteria(
    data: DataFrame,
    criteria: Sequence[Union[SortCriterion, Dict[str, Any]]],
) -> DataFrame:
    """Wrapper around mk.sort that adds unpacking of store to the DAG."""
    import meerkat as mk

    # Since the criteria can either be a list of dictionary or of SortCriterion
    # we need to convert them to SortCriterion
    criteria = _format_criteria(criteria)

    # Filter out criteria that are disabled.
    criteria = [criterion for criterion in criteria if criterion.is_enabled]
    if len(criteria) == 0:
        return data.view()

    sort_by = [criterion.column for criterion in criteria]
    ascending = [criterion.ascending for criterion in criteria]

    return mk.sort(data, by=sort_by, ascending=ascending)


class Sort(Component):
    df: DataFrame
    criteria: Union[List[SortCriterion], SortCriterion] = []
    title: str = "Sort"

    def __init__(
        self,
        df: DataFrame,
        *,
        criteria: List[SortCriterion] = [],
        title: str = "Sort",
    ):
        """This component handles a sort_by list and a sort_order list.

        Sorting criteria are maintained in a Store. On change of these
        values, the dataframe is sorted.

        This component will return a Reference object, which is a sorted
        view of the dataframe. The sort operation is out-of-place, so a
        new dataframe will be returned as a result of the op.

        Args:
            df: The dataframe to sort.
            criteria: The sorting criteria to use.
        """
        super().__init__(df=df, criteria=criteria, title=title)

    def __call__(self, df: DataFrame = None) -> DataFrame:
        if df is None:
            df = self.df
        return sort_by_criteria(df, self.criteria)

    @staticmethod
    @reactive()
    def create_criterion(column: str, ascending: bool, source: str = ""):
        return SortCriterion(
            id=str(uuid.uuid4()),
            is_enabled=True,
            column=column,
            ascending=ascending,
            source=source,
        )
