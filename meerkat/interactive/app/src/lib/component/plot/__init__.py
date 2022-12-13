from pydantic import Field

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.graph import Store

from ..abstract import Component


def is_none(x):
    return (isinstance(x, Store) and x.__wrapped__ is None) or x is None


class Plot(Component):
    # name: str = "Plot"

    df: "DataFrame"
    x: Store[str]
    y: Store[str]
    primary_key: Store[str] = None
    x_label: Store[str] = None
    y_label: Store[str] = None
    type: Store[str] = Store("scatter")
    slot: str = None
    keys_to_remove: Store[list] = Field(default_factory=lambda: Store(list()))
    metadata_columns: list = Field(default_factory=list)

    on_select: Endpoint = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # FIXME: this is buggy code, will create two stores for x and
        # x_label if x is a string, and x_label is None
        # and will create a single store for x and x_label if x is a store
        # and x_label is None
        if is_none(self.x_label):
            self.x_label = self.x
        if is_none(self.y_label):
            self.y_label = self.y

        if not is_none(self.primary_key):
            self.df = self.df.set_primary_key(self.primary_key)
        self.primary_key = Store(self.df.primary_key_name)
        self.selection = Store([0])
