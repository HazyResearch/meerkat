import hashlib
import weakref
from collections.abc import Mapping
from functools import wraps
from inspect import getcallargs
from typing import Sequence
from uuid import uuid4


class IdentifiableMixin:

    identifiable_group: str

    def __init__(self, id: str = None, *args, **kwargs):
        super(IdentifiableMixin, self).__init__(*args, **kwargs)
        self._set_id(id=id)

    def _set_id(self, id: str = None):
        # get uuid as str
        if id is None:
            self.id = uuid4().hex
        else:
            self.id = id 

        from meerkat.state import state

        state.identifiables.add(self)
