from dataclasses import dataclass

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import Endpoint

from ..abstract import Component


@dataclass
class Document(Component):

    df: DataFrame
    text_column: str
    paragraph_column: str = None
    label_column: str = None
    id_column: str = None
    on_sentence_label: Endpoint = None
