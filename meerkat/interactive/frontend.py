from typing import Type
from pydantic import BaseModel
from abc import ABC, abstractmethod


class FrontendMixin(ABC):
    """A mixin for objects that can be sent to the frontend.
    
    The purpose of this mixin is currently just to enable clean `isinstance` checks
    when determining whether an object can be sent to the frontend. Each subclass 
    needs to implement frontend themselves. 
    """

    @property
    @abstractmethod
    def frontend(self) -> BaseModel:
        """Returns a Pydantic model that can be should be sent to the frontend. These
        models are typically named <something>Frontend (e.g. ComponentFrontend, 
        StoreFrontend). 
        """
        raise NotImplementedError()
