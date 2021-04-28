import os

import dill
import yaml

from robustnessgym.mosaic.mixins.state import StateClass


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
        write_together: bool = None,
    ) -> None:
        assert hasattr(self, "_data"), f"{self.__class__.__name__} requires `self.data`"

        # If unspecified, use the column's property to decide whether to write together
        if write_together is None:
            write_together = self.write_together

        # Make all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Get the column state
        state = self.get_state()
        _data = state["_data"]

        # Remove the data key and put the rest of `state` into a metadata dict
        del state["_data"]
        metadata = {
            "dtype": type(self),
            "data_dtypes": list(map(type, self.data)),
            "len": len(self),
            "write_together": write_together,
            "state": state,
            **self.metadata,
        }

        if write_together:
            # Get the path where data should be stored
            data_path = os.path.join(path, "data.dill")

            # Saving all of the column data in a single dill file
            dill.dump(_data, open(data_path, "wb"))
        else:
            # Save all the elements of the column separately
            data_paths = []
            for index, element in enumerate(self.data):
                data_path = os.path.join(path, f"element_{index}")
                if hasattr(element, "write"):
                    # Element has its own `write` method
                    element.write(data_path)
                else:
                    # No `write` method: default to writing with dill
                    dill.dump(element, open(data_path, "wb"))
                data_paths.append(data_path)

            # Store all the data paths in the metadata dict
            metadata["data_paths"] = data_paths

        # Save the metadata as a yaml file
        metadata_path = os.path.join(path, "meta.yaml")
        yaml.dump(metadata, open(metadata_path, "w"))

    @classmethod
    def read(cls, path: str, *args, **kwargs) -> object:
        # Assert that the path exists
        assert os.path.exists(path), f"`path` {path} does not exist."

        # Load in the metadata
        metadata = dict(
            yaml.load(
                open(os.path.join(path, "meta.yaml")),
                Loader=yaml.FullLoader,
            )
        )

        # Load in the data
        if metadata["write_together"]:
            # All the data is written to a single dill file
            data = dill.load(open(os.path.join(path, "data.dill"), "rb"))

            data = [
                # `dtype` implements `from_state`, run it to get back an instance of
                # dtype
                dtype.from_state(
                    state.state if isinstance(state, StateClass) else state,
                    *args,
                    **kwargs,
                )
                if hasattr(dtype, "from_state")
                # `dtype` doesn't implement `from_state`, just return `state` directly
                else state
                for dtype, state in zip(metadata["data_dtypes"], data)
            ]
        else:
            # Each element of the data is written to individual paths
            data = [
                # `dtype` implements `read`, run it to read back the element instance
                dtype.read(path, *args, **kwargs) if hasattr(dtype, "read")
                # `dtype` doesn't implement `read`, just load with dill directly
                else dill.load(open(path, "rb"))
                for dtype, path in zip(metadata["data_dtypes"], metadata["data_paths"])
            ]

        # Load in the column from the state
        state = metadata["state"]
        state["_data"] = data
        return cls.from_state(state, *args, **kwargs)

    @classmethod
    def read_metadata(cls, path: str) -> dict:
        """Lightweight alternative to full read."""
        meta_path = os.path.join(path, "meta.yaml")
        return dict(yaml.load(open(meta_path), Loader=yaml.FullLoader))
