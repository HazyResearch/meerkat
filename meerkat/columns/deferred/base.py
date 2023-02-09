from __future__ import annotations

import logging
import os
import warnings
from typing import Callable, Collection, Sequence, Type, Union

import dill
import numpy as np
import yaml

from meerkat.block.abstract import BlockView
from meerkat.block.deferred_block import DeferredBlock, DeferredCellOp, DeferredOp
from meerkat.cells.abstract import AbstractCell
from meerkat.columns.abstract import Column
from meerkat.errors import ConcatWarning, ImmutableError
from meerkat.tools.lazy_loader import LazyLoader

Image = LazyLoader("PIL.Image")


logger = logging.getLogger(__name__)


class DeferredCell(AbstractCell):
    def __init__(self, data: DeferredCellOp):
        self._data = data

    @property
    def data(self) -> object:
        """Get the data associated with this cell."""
        return self._data

    def get(self, *args, **kwargs):
        return self.data._get()

    def __eq__(self, other):
        return (other.__class__ == self.__class__) and (self.data == other.data)

    def __repr__(self):
        name = getattr(self.data.fn, "__qualname__", repr(self.data.fn))
        return f"{self.__class__.__qualname__}(fn={name})"

    def __call__(self):
        return self.data._get()


class DeferredColumn(Column):

    block_class: type = DeferredBlock

    def __init__(
        self,
        data: Union[DeferredOp, BlockView],
        output_type: Type["Column"] = None,
        *args,
        **kwargs,
    ):
        self._output_type = output_type
        super(DeferredColumn, self).__init__(data, *args, **kwargs)

    def __call__(self):
        # TODO(Sabri): Make this a more efficient call
        return self._get(index=np.arange(len(self)), materialize=True)

    def _set(self, index, value):
        raise ImmutableError("LambdaColumn is immutable.")

    @property
    def fn(self) -> Callable:
        """Subclasses like `ImageColumn` should be able to implement their own
        version."""
        return self.data.fn

    def _create_cell(self, data: object) -> DeferredCell:
        return DeferredCell(data=data)

    def _get(self, index, materialize: bool = False, _data: np.ndarray = None):
        index = self._translate_index(index)
        data = self.data._get(index=index, materialize=materialize)

        if isinstance(index, int):
            if materialize:
                return data
            else:
                return self._create_cell(data=data)

        elif isinstance(index, np.ndarray):
            # support for blocks
            if materialize:
                # materialize could change the data in unknown ways, cannot clone
                return self.__class__.from_data(data=self.collate(data))
            else:
                return self._clone(data=data)

    @classmethod
    def _state_keys(cls) -> Collection:
        return super()._state_keys() | {"_output_type"}

    @staticmethod
    def concat(columns: Sequence[DeferredColumn]):
        for c in columns:
            if c.fn != columns[0].fn:
                warnings.warn(
                    ConcatWarning("Concatenating LambdaColumns with different `fn`.")
                )
                break

        return columns[0]._clone(data=DeferredOp.concat([c.data for c in columns]))

    def _write_data(self, path):
        return self.data.write(os.path.join(path, "data"))

    def is_equal(self, other: Column) -> bool:
        if other.__class__ != self.__class__:
            return False
        return self.data.is_equal(other.data)

    @staticmethod
    def _read_data(path: str):
        try:
            return DeferredOp.read(path=os.path.join(path, "data"))
        except KeyError:
            # TODO(Sabri): Remove this in a future version, once we no longer need to
            # support old DataFrames.
            warnings.warn(
                "Reading a LambdaColumn stored in a format that will soon be"
                " deprecated. Please re-write the column to the new format."
            )
            meta = yaml.load(
                open(os.path.join(path, "data", "meta.yaml")),
                Loader=yaml.FullLoader,
            )
            if issubclass(meta["dtype"], Column):
                col = Column.read(os.path.join(path, "data"))
            else:
                raise ValueError(
                    "Support for LambdaColumns based on a DataFrame is deprecated."
                )

            state = dill.load(open(os.path.join(path, "state.dill"), "rb"))

            return DeferredOp(
                args=[col],
                kwargs={},
                fn=state["fn"],
                is_batched_fn=False,
                batch_size=1,
            )

    def _get_default_formatter(self) -> Callable:
        # materialize a sample into a column
        from meerkat.interactive.formatter.base import DeferredFormatter

        col = self._get(index=slice(0, 1, 1), materialize=True)
        return DeferredFormatter(col.formatter)

    def _repr_cell(self, idx):
        return self[idx]
