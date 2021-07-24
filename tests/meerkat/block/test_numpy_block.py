import pytest
from meerkat.errors import ConsolidationError
from meerkat.block.numpy_block import NumpyBlock
import numpy as np


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

    block_indices = [
        {"0": 0, "2:5:1": slice(2, 5, 1)},
        {"6": 6, "2:7:2": slice(2, 7, 2)},
        {"2:7:2": slice(2, 7, 2), "1:8:1": slice(1, 8, 1)},
    ][:num_blocks]

    new_block, new_indices = NumpyBlock.consolidate(
        blocks=blocks, block_indices=block_indices
    )
    for block, indices in zip(blocks, block_indices):
        for name, index in indices.items():
            assert (block.data[:, index] == new_block.data[:, new_indices[name]]).all()


def test_consolidate_empty():
    with pytest.raises(ConsolidationError):
        NumpyBlock.consolidate([], [])


def test_consolidate_inconsistent():
    with pytest.raises(ConsolidationError):
        NumpyBlock.consolidate([NumpyBlock(np.zeros((10, 10)))], [])


def test_consolidate_mismatched_signature():
    with pytest.raises(ConsolidationError):
        NumpyBlock.consolidate(
            [
                NumpyBlock(np.zeros((10, 10), dtype=float)),
                NumpyBlock(np.zeros((10, 10), dtype=int)),
            ],
            [{}, {}],
        )
