from meerkat.dataframe import DataFrame
from meerkat.interactive.graph import Store

from typing import Optional

from ..abstract import AutoComponent


class Document(AutoComponent):

    df: DataFrame
    text_column: str
    paragraph_column: str = None
    label_column: str = None
    id_column: str = None
    
    @classmethod
    def events(cls):
        return ['label']
