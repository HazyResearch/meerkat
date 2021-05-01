import os
from pathlib import Path

import numpy as np

from robustnessgym.mosaic.writers.abstract import AbstractWriter


class ListWriter(AbstractWriter):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(ListWriter, self).__init__(*args, **kwargs)

    def open(self) -> None:
        self.outputs = []
    
    def write(self, data, **kwargs) -> None:
        self.outputs.extend(data)

    def flush(self, *args, **kwargs):
        return self.outputs

    def close(self, *args, **kwargs):
        pass
