from __future__ import annotations

from pandas import Timestamp

from meerkat.cells.abstract import AbstractCell


class TimestampCell(AbstractCell):
    def __init__(
        self,
        timestamp: Timestamp,
        *args,
        **kwargs,
    ):
        super(TimestampCell, self).__init__(*args, **kwargs)

        # Put some data into the cell
        self.timestamp = timestamp

    def get(self, *args, **kwargs):
        return self.timestamp

    def __getitem__(self, index):
        return self.get()[index]

    def __getattr__(self, item):
        try:
            return getattr(self.get(), item)
        except AttributeError:
            raise AttributeError(f"Attribute {item} not found.")

    def __repr__(self):
        return self.get().__repr__()
