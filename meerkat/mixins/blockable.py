from meerkat.block.abstract import AbstractBlock, BlockIndex


class BlockableMixin:
    block_class: type

    def __init__(
        self,
        block: AbstractBlock = None,
        block_index: BlockIndex = None,
        *args,
        **kwargs
    ):
        super(BlockableMixin, self).__init__(*args, **kwargs)
        if (block is None) != (block_index is None):
            raise ValueError("Must pass both `block` and `block_index` or neither.")

        if block is None:
            self._block, self._block_index = self.block_class.from_column(self.data)
        else:
            self._block = block
            self._block_index = block_index
