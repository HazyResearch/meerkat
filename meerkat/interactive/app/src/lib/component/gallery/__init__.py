from dataclasses import dataclass
from typing import List

from pydantic import Field

from meerkat.dataframe import DataFrame

from ..abstract import AutoComponent


@dataclass
class EditTarget:
    df: DataFrame
    pivot_id_column: str
    id_column: str

    @property
    def config(self):
        return {
            "pivot": self.pivot.config,
            "pivot_id_column": self.pivot_id_column,
            "id_column": self.id_column,
        }


class Gallery(AutoComponent):

    df: DataFrame
    main_column: str
    tag_columns: List[str] = []
    selected: List[int] = []