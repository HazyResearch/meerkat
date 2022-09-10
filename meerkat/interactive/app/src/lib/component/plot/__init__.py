from typing import Union

from meerkat.datapanel import DataPanel
from meerkat.interactive.graph import Pivot, Store, make_store
from meerkat.ops.sliceby.sliceby import SliceBy

from ..abstract import Component


class Plot(Component):
    name: str = "Plot"

    def __init__(
        self,
        data: Pivot[DataPanel],
        selection: Union[list, Store],
        x: Union[str, Store],
        y: Union[str, Store],
        x_label: Union[str, Store],
        y_label: Union[str, Store],
        type: str = "scatter",
    ) -> None:
        super().__init__()
        self.data = data
        self.selection = make_store(selection)
        self.x = make_store(x)
        self.y = make_store(y)
        self.x_label = make_store(x_label)
        self.y_label = make_store(y_label)
        self.type = type

    @property
    def props(self):
        return {
            "dp": self.data.config,
            "selection": self.selection.config,
            "x": self.x.config,
            "y": self.y.config,
            "x_label": self.x_label.config,
            "y_label": self.y_label.config,
            "type": self.type,
        }
