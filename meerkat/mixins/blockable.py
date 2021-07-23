from typing import Union


class BlockableMixin:

    block_type: Union[None, type] = None

    @property
    def block():
        return None
