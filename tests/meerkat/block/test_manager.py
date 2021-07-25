import pytest
import meerkat as mk
from meerkat.errors import ConsolidationError
from meerkat.block.manager import BlockManager
import numpy as np


def test_consolidate():
    mgr = BlockManager()

    col1 = mk.NumpyArrayColumn(np.arange(10))
    mgr.add_column(col1, "col1")
    col2 = mk.NumpyArrayColumn(np.arange(10) * 2)
    mgr.add_column(col2, "col2")

    assert len(mgr._block_refs) == 2
    mgr.consolidate()
    assert len(mgr._block_refs) == 1

    # check that the same object backs both the bock
    for name, col in [("col1", col1), ("col2", col2)]:
        assert mgr[name].data.base is list(mgr._block_refs.values())[0].block.data
        assert (mgr[name] == col).all()


def test_consolidate_multiple_types():
    mgr = BlockManager()

    for dtype in [int, float]:
        for idx in range(3):
            col = mk.NumpyArrayColumn(np.arange(10, dtype=dtype))
            mgr.add_column(col, f"col{idx}_{dtype}")

    assert len(mgr._block_refs) == 6
    mgr.consolidate()
    assert len(mgr._block_refs) == 2

@pytest.mark.parametrize(
    "num_blocks", [1, 2, 3]
)
def test_apply_get():
    mgr = BlockManager()

    for dtype in [int, float]:
        for idx in range(3):
            col = mk.NumpyArrayColumn(np.arange(10, dtype=dtype) * idx)
            mgr.add_column(col, f"col{idx}_{dtype}")

    mgr.consolidate()
    slc = slice(2,6,1)
    new_mgr = mgr.apply(method_name="_get", index=slc) 

    for dtype in [int, float]:
        for idx in range(3): 
            pass
        # TODO: revisit design of passing the blockref to the Block, and having the
        # block return a new blockref?
