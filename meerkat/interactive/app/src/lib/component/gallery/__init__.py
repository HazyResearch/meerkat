from dataclasses import dataclass
from typing import List
from pydantic import Field

from meerkat.dataframe import DataFrame
from meerkat.interactive.graph import Store, make_store

from ..abstract import Component


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


class Gallery(Component):

    df: DataFrame
    main_column: Store[str]
    tag_columns: Store[List[str]] = Field(default_factory=lambda: Store(list()))
    selected: Store[List[int]] = Field(default_factory=lambda: Store(list()))
