from typing import Type
from pydantic import BaseModel
from abc import ABC, abstractmethod


class FrontendMixin(ABC):
    """A mixin for objects that can be sent to the frontend."""

    @property
    @abstractmethod
    def frontend(self) -> BaseModel:
        """Returns a Pydantic model that can be should be sent to the frontend. These
        models are typically named <something>Frontend (e.g. ComponentFrontend, 
        StoreFrontend). 
        """
        raise NotImplementedError()
