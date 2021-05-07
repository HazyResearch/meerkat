from __future__ import annotations

import abc
import copy
import logging
import os
from collections import defaultdict
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import dill
import numpy as np
import yaml

from robustnessgym.mosaic.columns.cell_column import identity_collate

try:
    import faiss

    _is_faiss_available = True
except ImportError:
    _is_faiss_available = False

from tqdm.auto import tqdm
from yaml.representer import Representer

from robustnessgym.core.tools import convert_to_batch_column_fn
from robustnessgym.mosaic.columns.abstract import AbstractColumn

Representer.add_representer(abc.ABCMeta, Representer.represent_name)

logger = logging.getLogger(__name__)


class EmbeddingsColumn(AbstractColumn):
    def __init__(
        self,
        data: Sequence = None,
        collate_fn: Callable = None,
        *args,
        **kwargs,
    ):
        self._data = data
        if len(self._data) > 0:
            self._dim = len(self._data[0])
        else:
            self._dim = 0
        self._materialize = True
        if collate_fn is not None:
            self.collate = collate_fn
        else:
            self.collate = identity_collate
        self.visible_rows = None

        super(EmbeddingsColumn, self).__init__(num_rows=len(self), *args, **kwargs)

    @classmethod
    def from_array(cls, data: Sequence, *args, **kwargs):
        return cls(data=data, *args, **kwargs)

    def build_faiss_index(self):
        self.index = faiss.IndexFlatL2(self._dim)
        self.index.add(self._data)

    def search(self, query, k):
        return self.index.search(query, k)

    def __getattr__(self, item):
        try:
            return [getattr(doc, item) for doc in self]
        except AttributeError:
            raise AttributeError(f"Attribute {item} not found.")

    def metadata(self):
        return {}

    def __len__(self):
        # If only a subset of rows are visible
        if self.visible_rows is not None:
            return len(self.visible_rows)

        # If there are columns, len of any column
        if self._data is not None:
            return len(self._data)
        return 0

    def __getitem__(self, index):
        if self.visible_rows is not None:
            # Remap the index if only some rows are visible
            index = self._remap_index(index)

        # indices that return a single cell
        if isinstance(index, int) or isinstance(index, np.int):
            return self._data[index]

        # indices that return batches
        if isinstance(index, slice):
            # int or slice index => standard list slicing
            data = self._data[index]
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            data = [self._data[i] for i in index]
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            data = [self._data[int(i)] for i in index]
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

        # TODO(karan): do we need collate in EmbeddingsColumn
        # if self._materialize:  # self._collate
        #     # return a batch
        #     return self.collate([element for element in data])
        # else:
        #     # if not materializing, return a new EmbeddingsColumn
        #     return self.from_list(data)
        return self.from_list(data)

    def batch(
        self,
        batch_size: int = 32,
        drop_last_batch: bool = False,
        # collate: bool = True,
        *args,
        **kwargs,
    ):
        # TODO(karan): do we need collate in EmbeddingsColumn
        # if self._materialize:
        #     return torch.utils.data.DataLoader(
        #         self,
        #         batch_size=batch_size,
        #         collate_fn=self.collate if collate else identity_collate,
        #         drop_last=drop_last_batch,
        #         *args,
        #         **kwargs,
        #     )
        # else:
        #     return super(EmbeddingsColumn, self).batch(
        #         batch_size=batch_size, drop_last_batch=drop_last_batch
        #     )
        return super(EmbeddingsColumn, self).batch(
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
            # TODO: Transfer this fix to other classes
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
                    # collate=batched,
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
    ) -> Optional[EmbeddingsColumn]:
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
    def read(cls, path: str) -> EmbeddingsColumn:
        # Load in the data
        metadata = dict(
            yaml.load(
                open(os.path.join(path, "meta.yaml"), "r"), Loader=yaml.FullLoader
            )
        )
        data = dill.load(open(os.path.join(path, "data.dill"), "rb"))

        column = cls()
        state = metadata["state"]
        state["_data"] = data
        column.__setstate__(metadata["state"])
        return column

    def write(self, path: str) -> None:

        state = self.__getstate__()
        del state["_data"]
        metadata = {
            "dtype": type(self),
            "data_dtypes": list(map(type, self._data)),
            "len": len(self),
            "state": state,
            **self.metadata(),
        }

        # Make directory
        os.makedirs(path, exist_ok=True)

        # Get the paths where metadata and data should be stored
        metadata_path = os.path.join(path, "meta.yaml")
        data_path = os.path.join(path, "data.dill")

        # Saving all cell data in a single pickle file
        dill.dump(self._data, open(data_path, "wb"))

        # Saving the metadata as a yaml
        yaml.dump(metadata, open(metadata_path, "w"))

    def copy(self, deepcopy: bool = False):
        if deepcopy:
            return copy.deepcopy(self)
        else:
            dataset = EmbeddingsColumn()
            dataset.__dict__ = {k: copy.copy(v) for k, v in self.__dict__.items()}
            return dataset

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {"_materialize", "collate", "_data"}

    # TODO: add these state methods to a mixin, and add support for handling changing
    # state keys
    def __getstate__(self) -> Dict:
        """Get the internal state of the dataset."""
        state = {key: getattr(self, key) for key in self._state_keys()}
        self._assert_state_keys(state)
        return state

    @classmethod
    def _assert_state_keys(cls, state: Dict) -> None:
        """Assert that a state contains all required keys."""
        assert (
            set(state.keys()) == cls._state_keys()
        ), f"State must contain all state keys: {cls._state_keys()}."

    def __setstate__(self, state: Dict) -> None:
        """Set the internal state of the dataset."""
        if not isinstance(state, dict):
            raise ValueError(
                f"`state` must be a dictionary containing " f"{self._state_keys()}."
            )

        self._assert_state_keys(state)

        for key in self._state_keys():
            setattr(self, key, state[key])
