import hashlib
import weakref
from collections.abc import Mapping
from functools import wraps
from inspect import getcallargs
from typing import Sequence
from uuid import uuid4


class IdentifiableMixin:

    identifiable_group: str

    def __init__(self, *args, **kwargs):
        super(IdentifiableMixin, self).__init__(*args, **kwargs)
        self._set_id()

    def _set_id(self):
        # get uuid as str
        self.id = uuid4().hex

        from meerkat.state import state

        state.identifiables.add(self)
