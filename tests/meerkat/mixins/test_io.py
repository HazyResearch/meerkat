import os
import sys

import numpy as np

import meerkat as mk


def test_meerkat_loader(tmpdir):
    col = mk.NumPyTensorColumn(np.arange(10))
    path = os.path.join(tmpdir, "col.mk")
    col.write(path)
    module = sys.modules.pop("meerkat.columns.tensor.numpy")
    mk.Column.read(path)
    sys.modules["meerkat.columns.tensor.numpy"] = module
