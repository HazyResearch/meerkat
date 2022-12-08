from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.graph import Store

from ..abstract import Component


class Document(Component):

    df: DataFrame
    text_column: Store[str]
    paragraph_column: Store[str] = None
    label_column: Store[str] = None
    id_column: Store[str] = None
    on_sentence_label: Endpoint = None
