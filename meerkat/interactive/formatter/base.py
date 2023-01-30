from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from meerkat.columns.deferred.base import DeferredCell

if TYPE_CHECKING:
    from meerkat.interactive.app.src.lib.component.abstract import BaseComponent


@dataclass
class Variant:
    props: Dict[str, Any]
    encode_kwargs: Dict[str, Any]


class Formatter(ABC):

    component_class: Type["BaseComponent"]
    data_prop: str = "data"
    variants: Dict[str, Variant] = {}

    def __init__(self, encode: Optional[str] = None, **kwargs):
        if encode is not None:
            self._encode = encode

        default_props, required_props = self._get_props()
        if not all(k in kwargs for k in required_props):
            raise ValueError(
                f"Missing required properties {required_props} for {self.__class__.__name__}"
            )

        default_props.update(kwargs)
        self._props = default_props

    def encode(self, cell: Any, variants: List[str] = None, **kwargs):
        """Encode the cell on the backend before sending it to the
        frontend.

        The cell is lazily loaded, so when used on a LambdaColumn,
        ``cell`` will be a ``LambdaCell``. This is important for
        displays that don't actually need to apply the lambda in order
        to display the value.
        """
        if variants is not None:
            for name in variants:
                if name in self.variants:
                    kwargs.update(self.variants[name].encode_kwargs)
                    break
        if self._encode is not None:
            return self._encode(cell, **kwargs)
        return cell

    @property
    def props(self):
        return self._props

    def get_props(self, variants: str = None):
        if variants is None:
            return self.props

        props = self.props.copy()
        for name in variants:
            if name in self.variants:
                props.update(self.variants[name].props)
                break
        return props

    @classmethod
    def _get_props(cls):
        default_props = {}
        required_props = []
        for k, v in cls.component_class.__fields__.items():
            if k == cls.data_prop:
                continue
            if v.required:
                required_props.append(k)
            else:
                default_props[k] = v.default
        return default_props, required_props

    def html(self, cell: Any):
        """When not in interactive mode, objects are visualized using static
        html.

        This method should produce that static html for the cell.
        """
        return str(cell)


class DeferredFormatter(Formatter):
    def __init__(self, formatter: Formatter):
        self.wrapped = formatter

    def encode(self, cell: DeferredCell, variants: List[str] = None, **kwargs):
        return self.wrapped.encode(cell(), variants=variants, **kwargs)

    @property
    def component_class(self):
        return self.wrapped.component_class

    @property
    def data_prop(self):
        return self.wrapped.data_prop

    def get_props(self, variants: str = None):
        return self.wrapped.get_props(variants=variants)

    @property
    def props(self):
        return self.wrapped.props

    def html(self, cell: DeferredCell):
        return self.wrapped.html(cell())
