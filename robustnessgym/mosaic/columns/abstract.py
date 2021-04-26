from __future__ import annotations

import abc
import logging
from abc import abstractmethod
from types import SimpleNamespace
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from robustnessgym.core.identifier import Identifier

logger = logging.getLogger(__name__)


class AbstractColumn(abc.ABC):
    """An abstract class for Mosaic columns."""

    visible_rows: Optional[np.ndarray]
    _data: Sequence

    def __init__(self, num_rows: int, identifier: Identifier = None, *args, **kwargs):
        super(AbstractColumn, self).__init__(*args, **kwargs)

        # Identifier for the column
        self._identifier = (
            Identifier(self.__class__.__name__) if not identifier else identifier
        )

        # Index associated with each element of the column
        self.index = [str(i) for i in range(num_rows)]

        # Whether data in the column is materialized
        self._materialized = False

        # Log creation
        logger.info(f"Created `{self.__class__.__name__}` with {len(self)} rows.")

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    def get_state(self):
        """Get the state of the column."""
        return self

    @classmethod
    def from_state(cls, state) -> AbstractColumn:
        """Create a column from a state."""
        return state

    @abstractmethod
    def write(self, path) -> None:
        """Write a column to disk."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def read(cls) -> AbstractColumn:
        """Read a column from disk."""
        raise NotImplementedError()

    @abstractmethod
    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        **kwargs,
    ) -> AbstractColumn:
        """Map a function over the elements of the column."""
        raise NotImplementedError

    @abstractmethod
    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        **kwargs,
    ) -> AbstractColumn:
        """Filter the elements of the column using a function."""
        raise NotImplementedError

    def _remap_index(self, index):
        if isinstance(index, int):
            return self.visible_rows[index].item()
        elif isinstance(index, slice):
            return self.visible_rows[index].tolist()
        elif isinstance(index, str):
            return index
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            return self.visible_rows[index].tolist()
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            return self.visible_rows[index].tolist()
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

    def set_visible_rows(self, indices: Optional[Sequence]):
        """Set the visible rows of the column."""
        if indices is None:
            self.visible_rows = None
        else:
            if len(indices):
                assert min(indices) >= 0 and max(indices) < len(self), (
                    f"Ensure min index {min(indices)} >= 0 and "
                    f"max index {max(indices)} < {len(self)}."
                )
            if self.visible_rows is not None:
                self.visible_rows = self.visible_rows[np.array(indices, dtype=int)]
            else:
                self.visible_rows = np.array(indices, dtype=int)

    def batch(self, batch_size: int = 32, drop_last_batch: bool = False):
        """Batch the column.

        Args:
            batch_size: integer batch size
            drop_last_batch: drop the last batch if its smaller than batch_size

        Returns:
            batches of data
        """
        for i in range(0, len(self), batch_size):
            if drop_last_batch and i + batch_size > len(self):
                continue
            yield self[i : i + batch_size]

    def _inspect_function(
        self,
        function: Callable,
        with_indices: bool = False,
        batched: bool = False,
    ) -> SimpleNamespace:
        # TODO(Sabri): unify this function with dataset

        # Initialize variables to track
        no_output = dict_output = bool_output = list_output = False

        # Run the function to test it
        if batched:
            if with_indices:
                output = function(self[:2], range(2))
            else:
                output = function(self[:2])

        else:
            if with_indices:
                output = function(self[0], 0)
            else:
                output = function(self[0])

        if isinstance(output, Mapping):
            # `function` returns a dict output
            dict_output = True

        elif output is None:
            # `function` returns None
            no_output = True
        elif isinstance(output, bool):
            # `function` returns a bool
            bool_output = True
        elif isinstance(output, (Sequence, torch.Tensor, np.ndarray)):
            # `function` returns a list
            list_output = True
            if batched and (
                isinstance(output[0], bool)
                or (
                    hasattr(output[0], "dtype")
                    and output[0].dtype in (np.bool, torch.bool)
                )
            ):
                # `function` returns a bool per example
                bool_output = True

        return SimpleNamespace(
            dict_output=dict_output,
            no_output=no_output,
            bool_output=bool_output,
            list_output=list_output,
        )
