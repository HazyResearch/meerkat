from typing import List

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component


class Gallery(Component):
    df: DataFrame
    main_column: str
    tag_columns: List[str] = []
    selected: List[int] = []

    def __init__(
        self,
        df: DataFrame,
        *,
        main_column: str,
        tag_columns: List[str] = [],
        selected: List[int] = [],
    ):
        super().__init__(
            df=df,
            main_column=main_column,
            tag_columns=tag_columns,
            selected=selected,
        )
