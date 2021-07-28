import os

import dill
import yaml


class CellStorageMixin:
    def write(self, path: str) -> None:
        """Write a cell to disk."""

        # Create the paths
        os.makedirs(path, exist_ok=True)
        data_path = os.path.join(path, "data.dill")
        metadata_path = os.path.join(path, "meta.yaml")

        # Get the object state
        state = self.get_state()

        # Store the metadata
        yaml.dump(
            {
                "dtype": type(self),
                **self.metadata,
            },
            open(metadata_path, "w"),
        )

        # Store the data
        dill.dump(state, open(data_path, "wb"))

    @classmethod
    def read(cls, path: str, *args, **kwargs) -> object:
        """Read the cell from disk."""
        # Assert that the path exists
        assert os.path.exists(path), f"`path` {path} does not exist."

        # Create the paths
        data_path = os.path.join(path, "data.dill")
        metadata_path = os.path.join(path, "meta.yaml")

        # Load the metadata in
        metadata = dict(yaml.load(open(metadata_path), Loader=yaml.FullLoader))
        return metadata["dtype"].from_state(
            dill.load(open(data_path, "rb")), *args, **kwargs
        )

    @classmethod
    def read_metadata(cls, path: str) -> dict:
        """Lightweight alternative to full read."""
        meta_path = os.path.join(path, "meta.yaml")
        return dict(yaml.load(open(meta_path, "r"), Loader=yaml.FullLoader))


class ColumnStorageMixin:
    _write_together: bool = True

    @property
    def write_together(self):
        return self._write_together

    def write(
        self,
        path: str,
    ) -> None:
        assert hasattr(self, "_data"), f"{self.__class__.__name__} requires `self.data`"

        # Make all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Get the column state
        state = self._get_state()

        metadata = {
            "dtype": type(self),
            "len": len(self),
            **self.metadata,
        }

        # Write the state
        state_path = os.path.join(path, "state.dill")
        dill.dump(state, open(state_path, "wb"))

        # Write the data
        data_path = os.path.join(path, "data.dill")
        dill.dump(self.data, open(data_path, "wb"))

        # Save the metadata as a yaml file
        metadata_path = os.path.join(path, "meta.yaml")
        yaml.dump(metadata, open(metadata_path, "w"))

    @classmethod
    def read(cls, path: str) -> object:
        # Assert that the path exists
        assert os.path.exists(path), f"`path` {path} does not exist."

        # Load in the metadata
        metadata = dict(
            yaml.load(
                open(os.path.join(path, "meta.yaml")),
                Loader=yaml.FullLoader,
            )
        )

        # Load states
        state = dill.load(open(os.path.join(path, "state.dill"), "rb"))
        # Load states
        data = dill.load(open(os.path.join(path, "data.dill"), "rb"))

        col = metadata["dtype"].__new__(metadata["dtype"])
        col._set_state(state)
        col._set_data(data)

        return col

    @classmethod
    def read_metadata(cls, path: str) -> dict:
        """Lightweight alternative to full read."""
        meta_path = os.path.join(path, "meta.yaml")
        return dict(yaml.load(open(meta_path), Loader=yaml.FullLoader))
