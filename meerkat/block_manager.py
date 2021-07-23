from collections.abc import MutableMapping
from typing import Any, Dict, Hashable, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from meerkat.columns.abstract import AbstractColumn
from meerkat.mixins.blockable import BlockableMixin


def _infer_column_type_single(data):
    """Convert data to a meerkat column using the appropriate Column
    type."""
    if isinstance(data, AbstractColumn):
        return data.view()

    if isinstance(data, pd.Series):
        from .columns.pandas_column import PandasSeriesColumn

        return PandasSeriesColumn(data)

    if torch.is_tensor(data):
        from .columns.tensor_column import TensorColumn

        return TensorColumn(data)

    if isinstance(data, np.ndarray):
        from .columns.numpy_column import NumpyArrayColumn

        return NumpyArrayColumn(data)

    if isinstance(data, Sequence):
        from .cells.abstract import AbstractCell
        from .cells.imagepath import ImagePath

        if len(data) != 0 and isinstance(data[0], ImagePath):
            from .columns.image_column import ImageColumn

            return ImageColumn.from_cells(data)

        if len(data) != 0 and isinstance(data[0], AbstractCell):
            from .columns.cell_column import CellColumn

            return CellColumn(data)

        if len(data) != 0 and isinstance(
            data[0], (int, float, bool, np.ndarray, np.generic)
        ):
            from .columns.numpy_column import NumpyArrayColumn

            return NumpyArrayColumn(data)
        elif len(data) != 0 and torch.is_tensor(data[0]):
            from .columns.tensor_column import TensorColumn

            return TensorColumn(data)

        from .columns.list_column import ListColumn

        return ListColumn(data)
    else:
        raise ValueError(f"Cannot create column out of data of type {type(data)}")


def _infer_column_type_multiple(data):
    """Convert data to a meerkat column using the appropriate Column
    type."""
    if isinstance(data, AbstractColumn):
        return data.view()

    if isinstance(data, pd.Series):
        from .columns.pandas_column import PandasSeriesColumn

        return PandasSeriesColumn(data)

    if torch.is_tensor(data):
        from .columns.tensor_column import TensorColumn

        return TensorColumn(data)

    if isinstance(data, np.ndarray):
        from .columns.numpy_column import NumpyArrayColumn

        return NumpyArrayColumn(data)

    if isinstance(data, Sequence):
        from .cells.abstract import AbstractCell
        from .cells.imagepath import ImagePath

        if len(data) != 0 and isinstance(data[0], ImagePath):
            from .columns.image_column import ImageColumn

            return ImageColumn.from_cells(data)

        if len(data) != 0 and isinstance(data[0], AbstractCell):
            from .columns.cell_column import CellColumn

            return CellColumn(data)

        if len(data) != 0 and isinstance(
            data[0], (int, float, bool, np.ndarray, np.generic)
        ):
            from .columns.numpy_column import NumpyArrayColumn

            return NumpyArrayColumn(data)
        elif len(data) != 0 and torch.is_tensor(data[0]):
            from .columns.tensor_column import TensorColumn

            return TensorColumn(data)

        from .columns.list_column import ListColumn

        return ListColumn(data)
    else:
        raise ValueError(f"Cannot create column out of data of type {type(data)}")


class BlockManager(MutableMapping):
    """

    This manager manages all blocks.
    """

    def __init__(self) -> None:
        self._columns: Dict[str, AbstractColumn] = {}
        self.blocks: Dict[Hashable, AbstractBlock] = {}
        # Mapping from datapanel column to (block idx, local idx in block)
        self.name_to_block_location: Dict[str, Tuple[Hashable, Any]]

    def insert(self, data, index: Union[str, Sequence[str]] = None):
        """
        Loop through all block instances, and check for a match.

        If match, then insert into that block.
        If not match, create a new block.

        Args:
            data (): a single blockable object, potentially contains multiple columns
        """
        if isinstance(index, str):
            column = _infer_column_type_single(data)
            column_type = type(column)
            columns = {index: column}
        else:
            columns = _infer_column_type_multiple(data, index)
        self._columns.update(columns)

        if not issubclass(column_type, BlockableMixin):
            # These columns are not stored in a block
            return

        column_type.block_type.from_data(data, name_to_)

        return
        column_type = columns
        if column_type.block_type is None:
            # These columns are not stored in a block
            for name, column in name_to_idx.items():
                self._columns[name] = column_constructor(data[data_idx])
        return
        # Convert `data` to a block
        block, name_to_block_idx = column_type.block_type.blockify(
            data, name_to_data_idx
        )
        block_sig = block.signature
        if block_sig in self.blocks:
            # indices of the new part of the block
            name_to_block_idx = self.blocks[block_sig].insert(block, name_to_block_idx)
            # sometimes (fast_insert = insert o blockify) will be faster i.e. run local_indices = self.blocks[block_sig].fast_insert(data)
        else:
            self.blocks[block_sig] = block

        self.name_to_block_location.update(
            {
                name: (block_sig, block_idx)
                for name, block_idx in name_to_block_idx.items()
            }
        )

        self._data.update(
            {
                name: column_constructor(block[block_idx])
                for name, block_idx in name_to_block_idx.items()
            }
        )

    def _get(self, index, materialize=True):
        indexed_blocks = {
            block.id: block._get(index, materialize=materialize)
            for block in self.blocks()
        }

        result = {}
        for name, col in self.items():
            if col.block is None:
                _data = None
            else:
                _data = indexed_blocks[col.block.id].get(col)
            result[name] = col._get(index, materialize=materialize, _data=_data)
        return result

    def __getitem__(self, index: Union[str, Sequence[str]]):
        if isinstance(index, str):
            return self._columns[index]

    def __setitem__(self, index, value: Union[str, Sequence[str]]):
        self.insert(data=value, index=index)

    def __delitem__(self, key):
        del self._columns[key]

    def __len__(self):
        return len(self._columns)

    def __contains__(self, value):
        return value in self._columns

    def __iter__(self):
        return iter(self._columns)

    @classmethod
    def from_dict(cls, data: Mapping[str, Union[AbstractColumn, object]]):
        manager = cls()
        for index, data in data.items():
            manager.insert(data, index)
        return manager


def get_block_signature(data):
    column_type = infer_column_type(data)
    block_type = column_type.block_type
    return block_type.get_block_signature(data)


class AbstractBlock:
    @classmethod
    def match(cls, data) -> bool:
        pass

    def insert(self, data):
        """
        Cannot change block_index of
        """
        pass

    def __getitem__(self, index):
        # The index should be something that was returned by self.insert
        pass

    def get_names(self):
        return None

    def indices(self):
        raise NotImplementedError

    @classmethod
    def from_data(cls, data):
        if not cls.match(data):
            raise ValueError

    @classmethod
    def get_block_signature(data) -> Hashable:
        pass

    @classmethod
    def get_default_column_type(cls):
        raise NotImplementedError
