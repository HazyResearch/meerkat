import numpy as np
import pyarrow as pa
import pytest

from meerkat.block.abstract import BlockView
from meerkat.block.arrow_block import ArrowBlock
from meerkat.block.ref import BlockRef
from meerkat.columns.arrow_column import ArrowArrayColumn
from meerkat.errors import ConsolidationError


def test_signature_hash():
    # check equal
    block1 = ArrowBlock(pa.Table.from_pydict({"a": [1, 2, 3], "b": ["4", "5", "6"]}))
    block2 = ArrowBlock(pa.Table.from_pydict({"c": [1, 2, 3], "d": ["4", "5", "6"]}))
    assert hash(block1.signature) == hash(block2.signature)

    # check not equal
    block1 = ArrowBlock(pa.Table.from_pydict({"a": [1, 2, 3], "b": ["4", "5", "6"]}))
    block2 = ArrowBlock(pa.Table.from_pydict({"c": [1, 2], "d": ["5", "6"]}))
    assert hash(block1.signature) != hash(block2.signature)


@pytest.mark.parametrize("num_blocks", [1, 2, 3])
def test_consolidate_1(num_blocks):
    # check equal
    blocks = [
        ArrowBlock(
            pa.Table.from_pydict(
                {f"a_{idx}": np.arange(10), f"b_{idx}": np.arange(10) * 2},
            )
        )
        for idx in range(num_blocks)
    ]

    cols = [
        {
            str(slc): ArrowArrayColumn(
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
    block_ref = ArrowBlock.consolidate(block_refs=block_refs)
    for ref in block_refs:
        block = ref.block
        for name, col in ref.items():
            assert block.data[col._block_index].equals(
                block_ref.block.data[block_ref[name]._block_index]
            )


def test_consolidate_empty():
    with pytest.raises(ConsolidationError):
        ArrowBlock.consolidate([])


def test_consolidate_mismatched_signature():
    block1 = ArrowBlock(pa.Table.from_pydict({"a": [1, 2, 3], "b": ["4", "5", "6"]}))
    block2 = ArrowBlock(pa.Table.from_pydict({"c": [1, 2], "d": ["5", "6"]}))
    blocks = [block1, block2]

    slices = [
        ["a", "b"],
        ["c", "d"],
    ]
    cols = [
        {
            str(slc): ArrowArrayColumn(
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
        ArrowBlock.consolidate(block_refs)


def test_io(tmpdir):
    block = ArrowBlock(pa.Table.from_pydict({"a": [1, 2, 3], "b": ["4", "5", "6"]}))
    block.write(tmpdir)
    new_block = block.read(tmpdir)

    assert isinstance(block, ArrowBlock)
    assert block.data.equals(new_block.data)
