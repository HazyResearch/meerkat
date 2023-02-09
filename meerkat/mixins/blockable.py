from meerkat.block.abstract import BlockView


class BlockableMixin:
    def __init__(self, *args, **kwargs):
        super(BlockableMixin, self).__init__(*args, **kwargs)

    block_class: type = None

    @classmethod
    def is_blockable(cls):
        return cls.block_class is not None

    def _unpack_block_view(self, data):
        if isinstance(data, BlockView):
            self._block = data.block
            self._block_index = data.block_index
            data = data.data
        else:
            block_view: BlockView = self.block_class.from_column_data(data)
            self._block, self._block_index = block_view.block, block_view.block_index
            data = block_view.data
        return data

    def _pack_block_view(self):
        return BlockView(block_index=self._block_index, block=self._block)

    def run_block_method(self, method: str, *args, **kwargs):
        result = getattr(self._block.subblock([self._block_index]), method)(
            *args, **kwargs
        )
        return result[self._block_index]
