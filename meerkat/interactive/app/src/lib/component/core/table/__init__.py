from typing import Any, List

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.interactive.event import EventInterface
from meerkat.interactive.formatter.base import register_placeholder


class OnSelectTable(EventInterface):
    selected: List[Any]


class Table(Component):
    df: DataFrame
    selected: List[str] = []
    single_select: bool = False

    on_edit: EndpointProperty = None
    on_select: EndpointProperty[OnSelectTable] = None

    def __init__(
        self,
        df: DataFrame,
        *,
        selected: List[int] = [],
        single_select: bool = False,
        on_edit: EndpointProperty = None,
        on_select: EndpointProperty = None
    ):
        """Gallery view of a DataFrame.

        Args:
            df (DataFrame): The DataFrame to display.
            main_column (str): The column to display in the main gallery view.
            tag_columns (List[str], optional): The columns to display as tags. \
                Defaults to [].
            selected (List[int], optional): The indices of the rows selected in the \
                gallery. Useful for labeling and other tasks. Defaults to [].
            allow_selection (bool, optional): Whether to allow the user to select \
                rows. Defaults to False.
            single_select: Whether to allow the user to select only one row.
        """
        super().__init__(
            df=df,
            selected=selected,
            single_select=single_select,
            on_edit=on_edit,
            on_select=on_select,
        )


register_placeholder(
    name="table",
    fallbacks=["thumbnail"],
    description="Formatter to be used in a gallery view.",
)
