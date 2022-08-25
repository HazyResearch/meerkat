import numpy as np
import pandas as pd
import pytest

from meerkat import PandasSeriesColumn
from meerkat.block.abstract import BlockView
from meerkat.block.pandas_block import PandasBlock
from meerkat.block.ref import BlockRef
from meerkat.errors import ConsolidationError


def test_signature_hash():
    # check equal
    block1 = PandasBlock(pd.DataFrame({"a": [1, 2, 3], "b": ["4", "5", "6"]}))
    block2 = PandasBlock(pd.DataFrame({"c": [1, 2, 3], "d": ["4", "5", "6"]}))
    assert hash(block1.signature) == hash(block2.signature)

    # check equal
    block1 = PandasBlock(pd.DataFrame({"a": [1, 2, 3], "b": ["4", "5", "6"]}))
    block2 = PandasBlock(pd.DataFrame({"c": [1, 2], "d": ["5", "6"]}))
    assert hash(block1.signature) != hash(block2.signature)


@pytest.mark.parametrize("num_blocks", [1, 2, 3])
def test_consolidate_1(num_blocks):
    # check equal
    blocks = [
        PandasBlock(
            pd.DataFrame(
                {f"a_{idx}": np.arange(10), f"b_{idx}": np.arange(10) * 2},
                index=np.arange(idx, idx + 10),  # need to test with different
            )
        )
        for idx in range(num_blocks)
    ]

    cols = [
        {
            str(slc): PandasSeriesColumn(
                data=BlockView(
                    block=blocks[idx],
                    block_index=slc,
                )
            )
            for slc in [f"a_{idx}", f"b_{idx}"]
        }
        for idx in range(num_blocks)
    ]
    block_refs = [
        BlockRef(block=block, columns=cols) for block, cols in zip(blocks, cols)
    ]
    block_ref = PandasBlock.consolidate(block_refs=block_refs)
    for ref in block_refs:
        block = ref.block
        for name, col in ref.items():
            assert (
                block.data[col._block_index].reset_index(drop=True)
                == block_ref.block.data[block_ref[name]._block_index]
            ).all()


def test_consolidate_empty():
    with pytest.raises(ConsolidationError):
        PandasBlock.consolidate([])


def test_consolidate_mismatched_signature():
    block1 = PandasBlock(pd.DataFrame({"a": [1, 2, 3], "b": ["4", "5", "6"]}))
    block2 = PandasBlock(pd.DataFrame({"c": [1, 2], "d": ["5", "6"]}))
    blocks = [block1, block2]

    slices = [
        ["a", "b"],
        ["c", "d"],
    ]
    cols = [
        {
            str(slc): PandasSeriesColumn(
                data=BlockView(
                    block=blocks[block_idx],
                    block_index=slc,
                )
            )
            for slc in slices[block_idx]
        }
        for block_idx in range(2)
    ]
    block_refs = [
        BlockRef(block=block, columns=cols) for block, cols in zip(blocks, cols)
    ]
    with pytest.raises(ConsolidationError):
        PandasBlock.consolidate(block_refs)


def test_io(tmpdir):
    block = PandasBlock(pd.DataFrame({"a": [1, 2, 3], "b": ["4", "5", "6"]}))
    block.write(tmpdir)
    new_block = block.read(tmpdir)

    assert isinstance(block, PandasBlock)
    assert block.data.equals(new_block.data)

    # test with non-contiguous index, which is not supported by feather
    block = PandasBlock(
        pd.DataFrame({"a": [1, 2, 3], "b": ["4", "5", "6"]}, index=np.arange(1, 4))
    )
    block.write(tmpdir)
    new_block = block.read(tmpdir)

    assert isinstance(block, PandasBlock)
    assert block.data.reset_index(drop=True).equals(new_block.data)
