from __future__ import annotations

import abc

from yaml.representer import Representer

from meerkat.mixins.io import CellIOMixin

Representer.add_representer(abc.ABCMeta, Representer.represent_name)


class AbstractCell(CellIOMixin, abc.ABC):
    def __init__(self, *args, **kwargs):
        super(AbstractCell, self).__init__(*args, **kwargs)

    def get(self, *args, **kwargs) -> object:
        """Get me the thing that this cell exists for."""
        raise NotImplementedError("Must implement `get`.")

    @property
    def data(self) -> object:
        """Get the data associated with this cell."""
        return NotImplemented

    @property
    def metadata(self) -> dict:
        """Get the metadata associated with this cell."""
        return {}

    def loader(self, *args, **kwargs) -> object:
        return self

    def __getitem__(self, index):
        return self.get()[index]

    def __getattr__(self, item):
        try:
            return getattr(self.get(), item)
        except AttributeError:
            raise AttributeError(f"Attribute {item} not found.")

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __eq__(self, other):
        raise NotImplementedError
