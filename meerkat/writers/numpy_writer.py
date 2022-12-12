import os
from pathlib import Path

from numpy.lib.format import open_memmap

from meerkat.columns.abstract import Column
from meerkat.columns.tensor.numpy import NumPyTensorColumn
from meerkat.writers.abstract import AbstractWriter


class NumpyMemmapWriter(AbstractWriter):
    def __init__(
        self,
        path: str = None,
        dtype: str = "float32",
        mode: str = "r",
        shape: tuple = None,
        output_type: type = NumPyTensorColumn,
        template: Column = None,
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
        self.file = open_memmap(path, dtype=dtype, mode=mode, shape=shape)
        self._pointer = 0
        self.path = path
        self.shape = shape

    def write(self, arr, **kwargs) -> None:
        self.file[self._pointer : self._pointer + len(arr)] = arr
        self._pointer += len(arr)

    def flush(self):
        """Close the mmap file and reopen to release memory."""
        self.file.flush()
        self.file.base.close()
        # ‘r+’ Open existing file for reading and writing.
        self.file = open_memmap(self.file.filename, mode="r+")

    def finalize(self, *args, **kwargs) -> Column:
        self.flush()
        data = self.file
        if self.template is not None:
            if isinstance(data, Column):
                data = data.data
            data = self.template._clone(data=data)
        else:
            data = self.output_type(data)

        return data

    def close(self, *args, **kwargs) -> None:
        pass
