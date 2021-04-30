from __future__ import annotations

import copy as pycopy


class CopyMixin:
    def __init__(self, *args, **kwargs):
        super(CopyMixin, self).__init__(*args, **kwargs)

    def copy(self) -> object:
        """Return a copy of the object."""
        state = {k: pycopy.copy(v) for k, v in self.__dict__.items()}

        try:
            obj = self.__class__(**state)
        except TypeError:
            # use `__new__` to instantiate a bare class, in case __init__ does work
            # we do not want to repeat on copy
            obj = self.__class__.__new__(self.__class__)

        obj.__dict__ = state
        return obj

    def deepcopy(self) -> object:
        """Return a deepcopy of the object."""
        return pycopy.deepcopy(self)
