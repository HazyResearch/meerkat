from typing import Callable
from meerkat.interactive.app.src.lib.component.abstract import AutoComponent
from functools import partial
from meerkat.interactive.formatter import NewFormatter

class FormatterMixin:

    @classmethod
    def to_formatter(cls, data: any) -> NewFormatter:
        raise NotImplementedError()

    
    @staticmethod
    def encode(data: any, **kwargs) -> any:
        return data


"""Someone wrote this Image component. We want to use this as a formatter."""
class ImageUrl(AutoComponent, FormatterMixin):
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
        print(grayscale)
        return data

