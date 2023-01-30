from typing import Optional

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.graph import Store


class Document(Component):

    df: DataFrame
    text_column: str
    paragraph_column: Optional[str] = None
    label_column: Optional[str] = None
    id_column: Optional[str] = None

    @classmethod
    def events(cls):
        return ["label"]
