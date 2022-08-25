import numpy as np
import pytest

from meerkat import NumpyArrayColumn
from meerkat.block.abstract import BlockView
from meerkat.block.numpy_block import NumpyBlock
from meerkat.block.ref import BlockRef
from meerkat.errors import ConsolidationError


def test_signature_hash():
    # check equal
    block1 = NumpyBlock(np.zeros((100, 10)))
    block2 = NumpyBlock(np.ones((100, 10)))
    assert hash(block1.signature) == hash(block2.signature)

    # check differing type
    block1 = NumpyBlock(np.zeros((100, 10), dtype=int))
    block2 = NumpyBlock(np.ones((100, 10), dtype=float))
    assert hash(block1.signature) != hash(block2.signature)

    # check differing column width okay
    block1 = NumpyBlock(np.zeros((100, 13), dtype=int))
    block2 = NumpyBlock(np.ones((100, 10), dtype=int))
    assert hash(block1.signature) == hash(block2.signature)

    # check differing column width okay
    block1 = NumpyBlock(np.zeros((100, 13, 15), dtype=int))
    block2 = NumpyBlock(np.ones((100, 10, 15), dtype=int))
    assert hash(block1.signature) == hash(block2.signature)

    # check differing later dimensions not okay
    block1 = NumpyBlock(np.zeros((100, 10, 15), dtype=int))
    block2 = NumpyBlock(np.ones((100, 10, 20), dtype=int))
    assert hash(block1.signature) != hash(block2.signature)

    # check differing nrows not okay
    block1 = NumpyBlock(np.zeros((90, 10, 15), dtype=int))
    block2 = NumpyBlock(np.ones((100, 10, 20), dtype=int))
    assert hash(block1.signature) != hash(block2.signature)


@pytest.mark.parametrize("num_blocks", [1, 2, 3])
def test_consolidate_1(num_blocks):
    # check equal
    data = np.stack([np.arange(8)] * 12)
    blocks = [NumpyBlock(data.copy()) for _ in range(num_blocks)]

    slices = [
        [0, slice(2, 5, 1)],
        [6, slice(2, 7, 2)],
        [slice(2, 7, 3), slice(1, 8, 1)],
    ]
    cols = [
        {
            str(slc): NumpyArrayColumn(
                data=BlockView(
                    block=blocks[block_idx],
                    block_index=slc,
                )
            )
            for slc in slices[block_idx]
        }
        for block_idx in range(num_blocks)
    ]
    block_refs = [
        BlockRef(block=block, columns=cols) for block, cols in zip(blocks, cols)
    ]
    block_ref = NumpyBlock.consolidate(block_refs=block_refs)
    for ref in block_refs:
        block = ref.block
        for name, col in ref.items():
            assert (
                block.data[:, col._block_index]
                == block_ref.block.data[:, block_ref[name]._block_index]
            ).all()


def test_consolidate_empty():
    with pytest.raises(ConsolidationError):
        NumpyBlock.consolidate([])


def test_consolidate_mismatched_signature():
    data = np.stack([np.arange(8)] * 12)
    blocks = [NumpyBlock(data.astype(int)), NumpyBlock(data.astype(float))]

    slices = [
        [0, slice(2, 5, 1)],
        [6, slice(2, 7, 2)],
    ]
    cols = [
        {
            str(slc): NumpyArrayColumn(
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
        NumpyBlock.consolidate(block_refs)


def test_io(tmpdir):
    np.random.seed(123)
    block = NumpyBlock(np.random.randn(100, 10))
    block.write(tmpdir)
    new_block = NumpyBlock.read(tmpdir)

    assert isinstance(block, NumpyBlock)
    assert (block.data == new_block.data).all()
