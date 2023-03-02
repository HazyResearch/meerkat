import os

import dill

from meerkat.tools.utils import dump_yaml, load_yaml, meerkat_dill_load


class ColumnIOMixin:
    def write(self, path: str, *args, **kwargs) -> None:
        assert hasattr(self, "_data"), f"{self.__class__.__name__} requires `self.data`"

        # Make all the directories to the path
        os.makedirs(path, exist_ok=True)

        metadata = self._get_meta()

        # Write the state
        state = self._get_state()
        metadata["state"] = state
        self._write_data(path, *args, **kwargs)

        # Save the metadata as a yaml file
        metadata_path = os.path.join(path, "meta.yaml")
        dump_yaml(metadata, metadata_path)
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
        # Load in the metadata
        meta = (
            dict(load_yaml(os.path.join(path, "meta.yaml"))) if _meta is None else _meta
        )

        col_type = meta["dtype"]
        # Load states
        if "state" not in meta:
            assert os.path.exists(path), f"`path` {path} does not exist."
            try:
                state = col_type._read_state(path)
            except Exception:
                state = None
        else:
            state = meta["state"]
        data = col_type._read_data(path, *args, **kwargs) if _data is None else _data

        if state is None:
            # KG, Sabri: need to remove this `if-else` in the future,
            # this is only for backwards compatibility.
            # this if statement will not be required.
            col = col_type(data)
        else:
            col = col_type.__new__(col_type)
            col._set_state(state)
            col._set_data(data)

        from meerkat.interactive.formatter import DeprecatedFormatter

        if "_formatters" not in col.__dict__ or isinstance(
            col.formatters, DeprecatedFormatter
        ):
            # FIXME: make deprecated above work with new formatters
            # PATCH: backwards compatability patch for old dataframes
            # saved before v0.2.4
            col.formatters = col._get_default_formatters()

        return col

    @staticmethod
    def _read_state(path: str):
        return meerkat_dill_load(os.path.join(path, "state.dill"))

    @staticmethod
    def _read_data(path: str, *args, **kwargs):
        return meerkat_dill_load(os.path.join(path, "data.dill"))
