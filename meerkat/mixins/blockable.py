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
            self._block, self._block_index = self.block_class.from_data(data)
        return data

    def _pack_block_view(self):
        return BlockView(
            data=self.data, block_index=self._block_index, block=self._block
        )
