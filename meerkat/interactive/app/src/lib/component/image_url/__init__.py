from typing import Callable
from meerkat.interactive.app.src.lib.component.abstract import AutoComponent
from functools import partial


"""Someone wrote this Image component. We want to use this as a formatter."""
class ImageUrl(AutoComponent):
    url: str
    grayscale: bool = False

    @classmethod
    def to_formatter(cls, grayscale: bool = False):
        return NewFormatter(
            encode=partial(cls.encode, grayscale=grayscale),
            component_class=cls,
            component_kwargs=dict(grayscale=grayscale)
        )
    
    @staticmethod
    def encode(data: str, grayscale: bool = False):
        return data

