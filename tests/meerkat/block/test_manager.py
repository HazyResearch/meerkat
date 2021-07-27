from itertools import product

import numpy as np
import pytest

import meerkat as mk
from meerkat.block.manager import BlockManager
from meerkat.errors import ConsolidationError


def test_consolidate():
    mgr = BlockManager()

    col1 = mk.NumpyArrayColumn(data=np.arange(10))
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
    "num_blocks, consolidated",
    product([1, 2, 3], [True, False]),
)
def test_apply_get_multiple(num_blocks, consolidated):
    mgr = BlockManager()

    for dtype in [int, float]:
        for idx in range(num_blocks):
            col = mk.NumpyArrayColumn(np.arange(10, dtype=dtype) * idx)
            mgr.add_column(col, f"col{idx}_{dtype}")
    if consolidated:
        mgr.consolidate()

    for slc in [
        slice(2, 6, 1),
        slice(0, 1, 1),
        slice(2, 8, 3),
        np.array([1, 4, 6]),
        np.array([True, False] * 5),
    ]:
        new_mgr = mgr.apply(method_name="_get", index=slc)
        assert isinstance(new_mgr, BlockManager)
        for dtype in [int, float]:
            for idx in range(num_blocks):
                # check it's equivalent to applying the slice to each column in turn
                assert (
                    new_mgr[f"col{idx}_{dtype}"].data
                    == mgr[f"col{idx}_{dtype}"][slc].data
                ).all()

                # check that the base is the same (since we're just slicing)
                assert (
                    new_mgr[f"col{idx}_{dtype}"].data.base
                    is mgr[f"col{idx}_{dtype}"][slc].data.base
                ) == isinstance(slc, slice)


@pytest.mark.parametrize(
    "num_blocks, consolidated",
    product([1, 2, 3], [True, False]),
)
def test_apply_get_single(num_blocks, consolidated):
    mgr = BlockManager()

    for dtype in [int, float]:
        for idx in range(num_blocks):
            col = mk.NumpyArrayColumn(np.arange(10, dtype=dtype) * idx)
            mgr.add_column(col, f"col{idx}_{dtype}")
    if consolidated:
        mgr.consolidate()

    for slc in [0, 8]:
        result_dict = mgr.apply(method_name="_get", index=slc)
        isinstance(result_dict, dict)
        for dtype in [int, float]:
            for idx in range(num_blocks):
                # check it's equivalent to applying the slice to each column in turn
                assert result_dict[f"col{idx}_{dtype}"] == mgr[f"col{idx}_{dtype}"][slc]
