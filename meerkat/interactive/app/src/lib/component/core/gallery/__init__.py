from typing import List

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component


class Gallery(Component):

    df: DataFrame
    main_column: str
    tag_columns: List[str] = []
    selected: List[int] = []
