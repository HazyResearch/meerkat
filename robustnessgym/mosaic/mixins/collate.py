from collections.abc import Callable
from typing import List


def identity_collate(batch: List):
    return batch


class CollateMixin:
    def __init__(self, collate_fn: Callable = None, *args, **kwargs):
        super(CollateMixin, self).__init__(*args, **kwargs)

        if collate_fn is not None:
            self._collate_fn = collate_fn
        else:
            self._collate_fn = identity_collate

    @property
    def collate_fn(self):
        """Method used to collate."""
        return self._collate_fn

    @collate_fn.setter
    def collate_fn(self, value):
        self._collate_fn = value

    def collate(self, *args, **kwargs):
        """Collate data."""
        return self._collate_fn(*args, **kwargs)
