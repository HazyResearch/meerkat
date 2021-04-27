from __future__ import annotations

import copy as pycopy


class CopyMixin:
    def __init__(self, *args, **kwargs):
        super(CopyMixin, self).__init__(*args, **kwargs)

    def copy(self) -> object:
        """Return a copy of the object."""
        obj = self.__class__()
        obj.__dict__ = {k: pycopy.copy(v) for k, v in self.__dict__.items()}
        return obj

    def deepcopy(self) -> object:
        """Return a deepcopy of the object."""
        return pycopy.deepcopy(self)
