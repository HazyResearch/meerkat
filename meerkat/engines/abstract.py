from functools import partial
from typing import Callable, List, Optional, Union

from pydantic import BaseModel, validator

from meerkat.tools.lazy_loader import LazyLoader
from meerkat.ops.watch.abstract import WatchLogger

class BaseEngine:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Engine must implement the __call__ method.")

    @property
    def name(self) -> str:
        """The name of the engine."""
        return self.__class__.__name__

class EngineResponse:
    pass
