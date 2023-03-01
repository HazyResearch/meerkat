from typing import List

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.formatter.base import register_placeholder


class Gallery(Component):
    df: DataFrame
    main_column: str
    tag_columns: List[str] = []
    selected: List[str] = []
    allow_selection: bool = False
    cell_size: int = 24

    def __init__(
        self,
        df: DataFrame,
        *,
        main_column: str = None,
        tag_columns: List[str] = None,
        selected: List[int] = [],
        allow_selection: bool = False,
        cell_size: int = 24,
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
        if main_column is None:
            main_column = df.columns[0]

        if tag_columns is None:
            tag_columns = []
        super().__init__(
            df=df,
            main_column=main_column,
            tag_columns=tag_columns,
            selected=selected,
            allow_selection=allow_selection,
            cell_size=cell_size,
        )

    def _get_ipython_height(self):
        return "600px"


register_placeholder(
    name="gallery",
    fallbacks=["thumbnail"],
    description="Formatter to be used in a gallery view.",
)
