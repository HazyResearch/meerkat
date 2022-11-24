from typing import Type
from pydantic import BaseModel
from abc import ABC, abstractmethod


class FrontendMixin(ABC):
    """A mixin for objects that can be sent to the frontend."""

    @property
    @abstractmethod
    def frontend(self) -> BaseModel:
        raise NotImplementedError()
