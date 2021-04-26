from __future__ import annotations

import copy
import logging
import os
from collections import defaultdict
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import dill
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from robustnessgym.core.tools import convert_to_batch_column_fn
from robustnessgym.mosaic.cells.abstract import AbstractCell
from robustnessgym.mosaic.columns.abstract import AbstractColumn
from robustnessgym.mosaic.mixins.state import StateDictMixin

logger = logging.getLogger(__name__)


def identity_collate(batch: List):
    return batch


class CellColumn(StateDictMixin, AbstractColumn):
    def __init__(
        self,
        cells: Sequence[AbstractCell] = None,
        materialize: bool = True,
        collate_fn: Callable = None,
    ):
        self._cells = cells
        self._materialize = materialize

        if collate_fn is not None:
            self.collate = collate_fn
        else:
            self.collate = identity_collate

        self.visible_rows = None

        super(CellColumn, self).__init__(num_rows=len(self))

    @classmethod
    def from_cells(cls, cells: Sequence[AbstractCell], *args, **kwargs):
        return cls(cells=cells, *args, **kwargs)

    def metadata(self):
        return {}

    def __len__(self):
        # If only a subset of rows are visible
        if self.visible_rows is not None:
            return len(self.visible_rows)

        # If there are columns, len of any column
        if self._cells is not None:
            return len(self._cells)
        return 0

    def __getitem__(self, index):
        if self.visible_rows is not None:
            # Remap the index if only some rows are visible
            index = self._remap_index(index)

        # indices that return a single cell
        if isinstance(index, int) or isinstance(index, np.int):
            cell = self._cells[index]
            if self._materialize:
                return cell.get()
            else:
                return cell

        # indices that return batches
        if isinstance(index, slice):
            # int or slice index => standard list slicing
            cells = self._cells[index]
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            cells = [self._cells[i] for i in index]
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            cells = [self._cells[int(i)] for i in index]
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

        if self._materialize:
            # if materializing, return a batch (by default, a list of objects returned
            # by `Cell.get, otherwise other batch format specified by `self.collate`
            return self.collate([cell.get() for cell in cells])
        else:
            # if not materializing, return a new CellColumn
            return self.from_cells(cells, materialize=self._materialize)

    def batch(
        self,
        batch_size: int = 32,
        drop_last_batch: bool = False,
        collate: bool = True,
        *args,
        **kwargs,
    ):
        if self._materialize:
            return torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                collate_fn=self.collate if collate else lambda x: x,
                drop_last=drop_last_batch,
                *args,
                **kwargs,
            )
        else:
            return super(CellColumn, self).batch(
                batch_size=batch_size, drop_last_batch=drop_last_batch
            )

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        num_proc: Optional[int] = None,
        materialize: bool = None,
        **kwargs,
    ) -> Optional[Union[Dict, List]]:
        """Apply a map over the dataset."""
        # Check if need to materialize:
        # TODO(karan): figure out if we need materialize=False

        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Ensure that num_proc is not None
        if num_proc is None:
            num_proc = 0

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return None

        if not batched:
            # Convert to a batch function
            function = convert_to_batch_column_fn(function, with_indices=with_indices)
            batched = True
            logger.info(f"Converting `function` {function} to a batched function.")

        # # Get some information about the function
        # TODO: discuss whether this is actually required vs. doing it on first pass in
        # loop
        function_properties = self._inspect_function(
            function,
            with_indices,
            batched=batched,
        )

        # Run the map
        logger.info("Running `map`, the dataset will be left unchanged.")
        outputs = defaultdict(list) if function_properties.dict_output else []
        for i, batch in tqdm(
            enumerate(
                self.batch(
                    batch_size=batch_size,
                    drop_last_batch=drop_last_batch,
                    collate=batched,
                )
            ),
            total=(len(self) // batch_size)
            + int(not drop_last_batch and len(self) % batch_size != 0),
        ):

            # Run `function` on the batch
            output = (
                function(
                    batch,
                    range(i * batch_size, min(len(self), (i + 1) * batch_size)),
                )
                if with_indices
                else function(batch)
            )

            # Append the output
            if output is not None:
                if isinstance(output, Mapping):
                    for k in output.keys():
                        outputs[k].extend(output[k])
                else:
                    outputs.extend(output)

        if not len(outputs):
            return None
        elif isinstance(outputs, dict):
            # turns the defaultdict into dict
            return dict(outputs)
        return outputs

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        num_proc: Optional[int] = 64,
        **kwargs,
    ) -> Optional[CellColumn]:
        """Apply a filter over the dataset."""
        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return None

        # Get some information about the function
        function_properties = self._inspect_function(
            function,
            with_indices,
            batched=batched,
        )
        assert function_properties.bool_output, "function must return boolean."

        # Map to get the boolean outputs and indices
        logger.info("Running `filter`, a new dataset will be returned.")
        outputs = self.map(
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            num_proc=num_proc,
        )
        indices = np.where(outputs)[0]

        new_column = self.copy()
        new_column.set_visible_rows(indices)
        return new_column

    @classmethod
    def read(cls, path: str, *args, **kwargs) -> CellColumn:
        # TODO: make this based off of `get_state` `from_state`
        metadata = dict(
            yaml.load(
                open(os.path.join(path, "meta.yaml"), "r"), Loader=yaml.FullLoader
            )
        )
        if metadata["write_together"]:
            data = dill.load(open(os.path.join(path, "data.dill"), "rb"))
            cells = [
                dtype.decode(encoding, *args, **kwargs)
                for dtype, encoding in zip(metadata["cell_dtypes"], data)
            ]
        else:
            cells = [
                dtype.read(path, *args, **kwargs)
                for dtype, path in zip(metadata["cell_dtypes"], metadata["cell_paths"])
            ]

        column = cls()
        state = metadata["state"]
        state["_cells"] = cells
        column.__setstate__(metadata["state"])
        return column

    def write(self, path: str, write_together: bool = True) -> None:
        # TODO: make this based off of `get_state` `from_state`
        metadata_path = os.path.join(path, "meta.yaml")
        state = self.__getstate__()
        del state["_cells"]
        metadata = {
            "dtype": type(self),
            "cell_dtypes": list(map(type, self._cells)),
            "len": len(self),
            "write_together": write_together,
            "state": state,
            **self.metadata(),
        }

        if write_together:
            # Make directory
            os.makedirs(path, exist_ok=True)

            # Get the paths where metadata and data should be stored
            metadata_path = os.path.join(path, "meta.yaml")
            data_path = os.path.join(path, "data.dill")

            # Saving all cell data in a single pickle file
            dill.dump([cell.encode() for cell in self._cells], open(data_path, "wb"))
        else:
            os.makedirs(path, exist_ok=True)
            # Save all the cells separately
            cell_paths = []
            for index, cell in enumerate(self._cells):
                cell_path = os.path.join(path, f"cell_{index}")
                cell.write(cell_path)
                cell_paths.append(cell_path)
            metadata["cell_paths"] = cell_paths

        # Saving the metadata as a yaml
        yaml.dump(metadata, open(metadata_path, "w"))

    def copy(self, deepcopy: bool = False):
        if deepcopy:
            return copy.deepcopy(self)
        else:
            dataset = CellColumn()
            dataset.__dict__ = {k: copy.copy(v) for k, v in self.__dict__.items()}
            return dataset

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {"_materialize", "collate", "_cells"}
