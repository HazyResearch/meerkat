from __future__ import annotations

import abc
import os
from abc import abstractmethod

import dill
import yaml
from yaml.representer import Representer

Representer.add_representer(abc.ABCMeta, Representer.represent_name)


class AbstractCell(abc.ABC):
    def __init__(self, *args, **kwargs):
        super(AbstractCell, self).__init__(*args, **kwargs)

    @abstractmethod
    def get(self, *args, **kwargs):
        """Get me the thing that this cell exists for."""
        raise NotImplementedError("Must implement `get`.")

    def loader(self, *args, **kwargs) -> object:
        return self

    @property
    def data(self) -> object:
        """Get the data associated with this cell."""
        return NotImplemented

    @property
    def metadata(self) -> dict:
        """Get the metadata associated with this cell."""
        return {}

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

    def get_state(self) -> object:
        """
        Encode `self` in order to specify what information is important to
        store.

        By default, we just return `self` so the entire object is
        stored. For complex objects (e.g. Spacy Doc), we may want to
        return a compressed representation of the object here.
        """
        return self

    @classmethod
    def from_state(cls, state) -> AbstractCell:
        """Recover the object from its compressed representation.

        By default, we don't change the state.
        """
        return state

    def write(self, path: str) -> None:
        """Actually write the encoded object to disk."""
        # Create the paths
        os.makedirs(path, exist_ok=True)
        data_path = os.path.join(path, "data.dill")
        metadata_path = os.path.join(path, "meta.yaml")

        state = self.get_state()
        yaml.dump(
            {
                "dtype": type(self),
                **self.metadata(),
            },
            open(metadata_path, "w"),
        )
        return dill.dump(state, open(data_path, "wb"))

    @classmethod
    def read(cls, path: str, *args, **kwargs) -> AbstractCell:
        """Read the cell from disk."""
        assert os.path.exists(path), f"`path` {path} does not exist."

        # Create the paths
        data_path = os.path.join(path, "data.dill")
        metadata_path = os.path.join(path, "meta.yaml")

        # Load the metadata in
        metadata = dict(yaml.load(open(metadata_path, "r"), Loader=yaml.FullLoader))
        return metadata["dtype"].from_state(
            dill.load(open(data_path, "rb")), *args, **kwargs
        )

    @classmethod
    def read_metadata(cls, path: str, *args, **kwargs) -> dict:
        """Lightweight alternative to full read."""
        meta_path = os.path.join(path, "meta.yaml")
        return dict(yaml.load(open(meta_path, "r"), Loader=yaml.FullLoader))
