from typing import Sequence, Union
from dataclasses import dataclass, field
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
    primary_key: str = None
    x_label: str = None
    y_label: str = None
    type: str = "scatter"
    slot: str = None
    keys_to_remove: list = field(default_factory=list)
    metadata_columns: list = field(default_factory=list)

    on_select: Endpoint = None

    def __post_init__(self):
        super().__post_init__()

        if self.x_label.__wrapped__ is None:
            self.x_label = self.x
        if self.y_label.__wrapped__ is None:
            self.y_label = self.y

        if self.primary_key.__wrapped__ is not None:
            self.df = self.df.set_primary_key(self.primary_key)
        self.primary_key = Store(self.df.primary_key_name)
        print(self.primary_key)

        self.selection = Store([0])

    @property
    def props(self):
        print(self.metadata_columns)
        return super().props
