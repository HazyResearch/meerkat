from dataclasses import dataclass
import weakref
from dataclasses import dataclass
from typing import Any

@dataclass
class GlobalState:
    data_id: int = None 

state = GlobalState()


def update(key: str, value: Any):
    state.__dict__[key] = value


def weak_update(key: str, value: Any):
    state.__dict__[key] = weakref.ref(value)


def get(key: str):
    if isinstance(state.__dict__[key], weakref.ReferenceType):
        return state.__dict__[key]()
    return state.__dict__[key]
