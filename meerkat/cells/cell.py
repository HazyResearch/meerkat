from __future__ import annotations

from meerkat.cells.abstract import AbstractCell


class Cell(AbstractCell):
    def __init__(self, data: object):
        # Put some data into the cell
        self.data = data

    @property
    def loader(self, *args, **kwargs):
        return self.data

    def get(self, *args, **kwargs):
        return self.loader(*args, **kwargs)
