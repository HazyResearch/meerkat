from __future__ import annotations

import abc


class AbstractCell(abc.ABC):
    def __init__(self, *args, **kwargs):
        super(AbstractCell, self).__init__(*args, **kwargs)

    def get(self, *args, **kwargs) -> object:
        """Get me the thing that this cell exists for."""
        raise NotImplementedError("Must implement `get`.")

    @property
    def metadata(self) -> dict:
        """Get the metadata associated with this cell."""
        return {}

    def __getitem__(self, index):
        return self.get()[index]

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __eq__(self, other):
        raise NotImplementedError
