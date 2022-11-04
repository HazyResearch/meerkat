from typing import Sequence, Union

from meerkat.dataframe import DataFrame
from meerkat.interactive.graph import Pivot, Store, make_store

from ..abstract import Component


class Plot(Component):
    name: str = "Plot"

    def __init__(
        self,
        data: Pivot[DataFrame],
        selection: Union[list, Store],
        x: Union[str, Store],
        y: Union[str, Store],
        x_label: Union[str, Store],
        y_label: Union[str, Store],
        id: Union[str, Store] = "key",
        type: str = "scatter",
        slot: str = None,
        keys_to_remove: Union[str, Store] = None,
        metadata_columns: Sequence[str] = None,
        can_remove: bool = True,
    ) -> None:
        super().__init__()
        self.data = data
        self.selection = make_store(selection)
        self.x = make_store(x)
        self.y = make_store(y)
        self.id = id
        self.x_label = make_store(x_label)
        self.y_label = make_store(y_label)
        self.type = type
        self.slot = slot
        self.can_remove = can_remove

        if metadata_columns is None:
            metadata_columns = []
        self.metadata_columns = metadata_columns

        if keys_to_remove is None:
            keys_to_remove = []
        self.keys_to_remove = make_store(keys_to_remove)

    @property
    def props(self):
        props = {
            "df": self.data.config,
            "selection": self.selection.config,
            "x": self.x.config,
            "y": self.y.config,
            "x_label": self.x_label.config,
            "y_label": self.y_label.config,
            "type": self.type,
            "id": self.id if isinstance(self.id, str) else self.id.config,
            "keys_to_remove": self.keys_to_remove.config,
            "can_remove": self.can_remove,
        }
        if self.slot is not None:
            props["slot"] = self.slot
        if self.metadata_columns is not None:
            props["metadata_columns"] = self.metadata_columns
        return props
