from typing import Sequence, Union
from dataclasses import dataclass
from meerkat.dataframe import DataFrame
from meerkat.interactive.graph import Store, make_store

from ..abstract import Component
from meerkat.interactive.endpoint import Endpoint, endpoint

@dataclass
class Plot(Component):
    # name: str = "Plot"

    df: "DataFrame"
    x: str
    y: str
    id_col: str
    x_label: str = None
    y_label: str = None
    type: str = "scatter"
    slot: str = None
    keys_to_remove: list = None
    metadata_columns: list = None

    on_select: Endpoint = None

    def __post_init__(self):
        super().__post_init__()

        if self.x_label is None:
            self.x_label = self.x
        if self.y_label is None:
            self.y_label = self.y

        if self.metadata_columns is None:
            self.metadata_columns = []

        if self.keys_to_remove is None:
            self.keys_to_remove = []
        
        self.selection = Store([0])


    @property
    def props(self):
        print(self.metadata_columns)
        return super().props
