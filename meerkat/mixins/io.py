import os

import dill
import yaml


class CellIOMixin:
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


class ColumnIOMixin:
    def write(self, path: str, *args, **kwargs) -> None:
        assert hasattr(self, "_data"), f"{self.__class__.__name__} requires `self.data`"

        # Make all the directories to the path
        os.makedirs(path, exist_ok=True)

        metadata = self._get_meta()

        # Write the state
        self._write_state(path)
        self._write_data(path, *args, **kwargs)

        # Save the metadata as a yaml file
        metadata_path = os.path.join(path, "meta.yaml")
        yaml.dump(metadata, open(metadata_path, "w"))

        return metadata

    def _get_meta(self):
        return {
            "dtype": type(self),
            "len": len(self),
            **self.metadata,
        }

    def _write_data(self, path):
        data_path = os.path.join(path, "data.dill")
        dill.dump(self.data, open(data_path, "wb"))

    def _write_state(self, path):
        state = self._get_state()
        state_path = os.path.join(path, "state.dill")
        dill.dump(state, open(state_path, "wb"))

    @classmethod
    def read(
        cls, path: str, _data: object = None, _meta: object = None, *args, **kwargs
    ) -> object:
        # Assert that the path exists
        assert os.path.exists(path), f"`path` {path} does not exist."

        # Load in the metadata
        meta = (
            dict(
                yaml.load(
                    open(os.path.join(path, "meta.yaml")),
                    Loader=yaml.FullLoader,
                )
            )
            if _meta is None
            else _meta
        )

        col_type = meta["dtype"]
        # Load states
        state = col_type._read_state(path)
        data = col_type._read_data(path, *args, **kwargs) if _data is None else _data

        col = col_type.__new__(col_type)
        col._set_state(state)
        col._set_data(data)

        return col

    @staticmethod
    def _read_state(path: str):
        return dill.load(open(os.path.join(path, "state.dill"), "rb"))

    @staticmethod
    def _read_data(path: str):
        return dill.load(open(os.path.join(path, "data.dill"), "rb"))
