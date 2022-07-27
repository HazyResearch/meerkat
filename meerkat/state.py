import weakref
from typing import Any


class GlobalState:
    def __init__(self):
        pass


state = GlobalState()


def update(key: str, value: Any):
    state.__dict__[key] = value


def weak_update(key: str, value: Any):
    state.__dict__[key] = weakref.ref(value)


def get(key: str):
    if isinstance(state.__dict__[key], weakref.ReferenceType):
        return state.__dict__[key]()
    return state.__dict__[key]
