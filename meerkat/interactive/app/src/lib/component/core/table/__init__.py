from typing import List

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.interactive.formatter.base import register_placeholder


class Table(Component):
    df: DataFrame
    selected: List[int] = []
    allow_selection: bool = False
    on_edit: EndpointProperty = None

    def __init__(
        self,
        df: DataFrame,
        *,
        selected: List[int] = [],
        allow_selection: bool = False,
        cell_size: int = 24,
        on_edit: EndpointProperty = None
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
        """
        super().__init__(
            df=df, selected=selected, allow_selection=allow_selection, on_edit=on_edit
        )


register_placeholder(
    name="table",
    fallbacks=["thumbnail"],
    description="Formatter to be used in a gallery view.",
)
