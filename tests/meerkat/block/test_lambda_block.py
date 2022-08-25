import numpy as np

from meerkat import LambdaColumn, NumpyArrayColumn
from meerkat.block.abstract import BlockView
from meerkat.block.lambda_block import LambdaBlock, LambdaCellOp, LambdaOp
from meerkat.block.ref import BlockRef

from ...utils import product_parametrize


def fn(x: int) -> int:
    return x + 1, x + 2, x + 3


@product_parametrize(params={"num_blocks": [1, 2, 3]})
def test_consolidate(num_blocks: int):
    inp = NumpyArrayColumn(np.arange(8))
    op = LambdaOp(args=[inp], fn=fn, kwargs={}, is_batched_fn=False, batch_size=1)

    block_views = [
        LambdaBlock.from_column_data(data=op.with_return_index(i)) for i in range(3)
    ]
    cols = [
        {str(block_view.block_index): LambdaColumn(data=block_view)}
        for block_view in block_views
    ]

    block_ref = LambdaBlock.consolidate(
        block_refs=[
            BlockRef(
                block=block_view.block,
                columns=col,
            )
            for block_view, col in zip(block_views, cols)
        ]
    )

    assert isinstance(block_ref, BlockRef)

    for name, col in block_ref.items():
        assert col._block is block_ref.block
        assert int(name) == col._block_index

    for i in range(num_blocks):
        assert (block_ref[str(i)][:].data == cols[i][str(i)][:].data).all()


def test_consolidate_same_index():
    inp = NumpyArrayColumn(np.arange(8))
    op = LambdaOp(args=[inp], fn=fn, kwargs={}, is_batched_fn=False, batch_size=1)

    block_views = [
        LambdaBlock.from_column_data(data=op.with_return_index(0)),
        LambdaBlock.from_column_data(data=op.with_return_index(0)),
        LambdaBlock.from_column_data(data=op.with_return_index(1)),
    ]
    cols = [
        {str(i): LambdaColumn(data=block_view)}
        for i, block_view in enumerate(block_views)
    ]

    block_ref = LambdaBlock.consolidate(
        block_refs=[
            BlockRef(
                block=block_view.block,
                columns=col,
            )
            for block_view, col in zip(block_views, cols)
        ]
    )

    assert isinstance(block_ref, BlockRef)

    for _, col in block_ref.items():
        assert col._block is block_ref.block

    for i in range(len(block_views)):
        assert (block_ref[str(i)][:].data == cols[i][str(i)][:].data).all()
