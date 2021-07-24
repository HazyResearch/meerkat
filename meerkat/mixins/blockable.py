from meerkat.block.abstract import AbstractBlock
from meerkat.block.ref import Index

Index = Union[int, slice, np.ndarray, str]


class BlockableMixin:

    block_class: type

    def __init__(
        self, block: AbstractBlock = None, _block_idx: Index = None, *args, **kwargs
    ):
        super(BlockableMixin, self).__init__(*args, **kwargs)
        if (block is None) != (_block_idx is None):
            raise ValueError("Must pass both `block` and `block_index` or neither.")

        if block is None:
            self._block, self._block_idx = self.block_class.from_column(self.data)
        else:
            self._block = block
            self._block_idx = _block_idx
