import os
import sys

import numpy as np

import meerkat as mk


def test_meerkat_loader(tmpdir):
    col = mk.NumpyArrayColumn(np.arange(10))
    path = os.path.join(tmpdir, "col.mk")
    col.write(path)
    module = sys.modules.pop("meerkat.columns.numpy_column")
    mk.AbstractColumn.read(path)
    sys.modules["meerkat.columns.numpy_column"] = module
