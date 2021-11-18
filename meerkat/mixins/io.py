import os

import dill
import yaml

from meerkat.tools.utils import MeerkatLoader


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
                    Loader=MeerkatLoader,
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
        try:
            return dill.load(open(os.path.join(path, "state.dill"), "rb"))
        except ModuleNotFoundError:
            dill_str = open(os.path.join(path, "state.dill"), "rb").read()

            if b"meerkat.nn" in dill_str:
                # backwards compatibility
                # TODO (Sabri): remove this in a future release
                dill_str = dill_str.replace(b"meerkat.nn", b"meerkat.ml")
            return dill.loads(dill_str)

    @staticmethod
    def _read_data(path: str):
        return dill.load(open(os.path.join(path, "data.dill"), "rb"))
