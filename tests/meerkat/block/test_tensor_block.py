import pytest
import torch

from meerkat import TensorColumn
from meerkat.block.abstract import BlockView
from meerkat.block.ref import BlockRef
from meerkat.block.tensor_block import TensorBlock
from meerkat.errors import ConsolidationError


def test_signature_hash():
    # check equal
    block1 = TensorBlock(torch.zeros((100, 10)))
    block2 = TensorBlock(torch.ones((100, 10)))
    assert hash(block1.signature) == hash(block2.signature)

    # check differing type
    block1 = TensorBlock(torch.zeros((100, 10), dtype=int))
    block2 = TensorBlock(torch.ones((100, 10), dtype=float))
    assert hash(block1.signature) != hash(block2.signature)

    # check differing column width okay
    block1 = TensorBlock(torch.zeros((100, 13), dtype=int))
    block2 = TensorBlock(torch.ones((100, 10), dtype=int))
    assert hash(block1.signature) == hash(block2.signature)

    # check differing column width okay
    block1 = TensorBlock(torch.zeros((100, 13, 15), dtype=int))
    block2 = TensorBlock(torch.ones((100, 10, 15), dtype=int))
    assert hash(block1.signature) == hash(block2.signature)

    # check differing later dimensions not okay
    block1 = TensorBlock(torch.zeros((100, 10, 15), dtype=int))
    block2 = TensorBlock(torch.ones((100, 10, 20), dtype=int))
    assert hash(block1.signature) != hash(block2.signature)

    # check differing devices not okay
    block1 = TensorBlock(torch.zeros((100, 10, 15), dtype=int))
    block2 = TensorBlock(torch.ones((100, 10, 20), dtype=int).cpu())
    assert hash(block1.signature) != hash(block2.signature)

    # check differing nrows not okay
    block1 = TensorBlock(torch.zeros((90, 10, 15), dtype=int))
    block2 = TensorBlock(torch.ones((100, 10, 20), dtype=int))
    assert hash(block1.signature) != hash(block2.signature)


@pytest.mark.parametrize("num_blocks", [1, 2, 3])
def test_consolidate_1(num_blocks):
    # check equal
    blocks = [
        TensorBlock(torch.stack([torch.arange(8)] * 12)) for _ in range(num_blocks)
    ]

    slices = [
        [0, slice(2, 5, 1)],
        [6, slice(2, 7, 2)],
        [slice(2, 7, 3), slice(1, 8, 1)],
    ]
    cols = [
        {
            str(slc): TensorColumn(
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
    block_ref = TensorBlock.consolidate(block_refs=block_refs)
    for ref in block_refs:
        block = ref.block
        for name, col in ref.items():
            assert (
                block.data[:, col._block_index]
                == block_ref.block.data[:, block_ref[name]._block_index]
            ).all()


def test_consolidate_empty():
    with pytest.raises(ConsolidationError):
        TensorBlock.consolidate([])


def test_consolidate_mismatched_signature():
    data = torch.stack([torch.arange(8)] * 12)
    blocks = [TensorBlock(data.to(int)), TensorBlock(data.to(float))]

    slices = [
        [0, slice(2, 5, 1)],
        [6, slice(2, 7, 2)],
    ]
    cols = [
        {
            str(slc): TensorColumn(
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
        TensorBlock.consolidate(block_refs)


def test_io(tmpdir):
    torch.manual_seed(123)
    block = TensorBlock(torch.randn(100, 10))
    block.write(tmpdir)
    new_block = TensorBlock.read(tmpdir)

    assert isinstance(block, TensorBlock)
    assert (block.data == new_block.data).all()
