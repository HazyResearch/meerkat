import os
from pathlib import Path

import numpy as np

from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.numpy_column import NumpyArrayColumn
from meerkat.writers.abstract import AbstractWriter


class NumpyMemmapWriter(AbstractWriter):
    def __init__(
        self,
        path: str = None,
        dtype: str = "float32",
        mode: str = "r",
        shape: tuple = None,
        output_type: type = NumpyArrayColumn,
        template: AbstractColumn = False,
        *args,
        **kwargs,
    ):
        super(NumpyMemmapWriter, self).__init__(*args, **kwargs)

        # File used to store data
        self.file = None

        # Location of the pointer
        self._pointer = 0

        # If `path` is specified
        self.path = path
        self.dtype = dtype
        self.shape = shape
        if path is not None:
            self.open(path=path, dtype=dtype, mode=mode, shape=shape)

        self.output_type = output_type
        self.template = template

    def open(
        self,
        path: str,
        dtype: str = "float32",
        mode: str = "w+",
        shape: tuple = None,
    ) -> None:
        assert shape is not None, "Must specify `shape`."

        # Make all dirs to path
        os.makedirs(str(Path(path).absolute().parent), exist_ok=True)

        # Open the file as a memmap
        self.file = np.memmap(path, dtype=dtype, mode=mode, shape=shape)
        self._pointer = 0

    def write(self, arr, **kwargs) -> None:
        self.file[self._pointer : self._pointer + len(arr)] = arr
        self._pointer += len(arr)

    def flush(self, *args, **kwargs) -> None:
        self.file.flush()
        return self.output_type.read(
            path=str(self.path), mmap=True, dtype=self.dtype, shape=self.shape
        )

    def close(self, *args, **kwargs) -> None:
        pass

    def finalize(self, *args, **kwargs) -> None:
        pass
