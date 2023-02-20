from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
import yaml

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union
from meerkat.tools.utils import MeerkatLoader, MeerkatDumper

from meerkat.columns.deferred.base import DeferredCell

if TYPE_CHECKING:
    from meerkat.interactive.app.src.lib.component.abstract import BaseComponent


class Variant:
    def __init__(*args, **kwargs):
        pass


class FormatterGroup(dict):
    """A formatter group is a mapping from formatter placeholders to formatters.

    Data in a Meerkat column sometimes need to be displayed differently in different
    GUI contexts. For example, in a table, we display thumbnails of images, but in a
    carousel view, we display the full image.

    Because most components in Meerkat work on any data type, it is important that
    they are implemented in a formatter-agnostic way. So, instead of specifying
    formatters, components make requests for data specifying a *formatter placeholder*.
    For example, the {class}`mk.gui.Gallery` component requests data using the
    `thumbnail` formatter placeholder.

    For a specific column of data, we specify which formatters to use for each
    placeholder using a *formatter group*. A formatter group is a mapping from
    formatter placeholders to formatters. Each column in Meerkat has a
    `formatter_group` property. A column's formatter group controls how it will be
    displayed in different contexts in Meerkat GUIs.

    Args:
        base (FormatterGroup): The base formatter group to use.
        **kwargs: The formatters to add to the formatter group.
    """

    def __init__(self, base: Formatter, **kwargs):
        # must provide a base formatter
        super().__init__(base=base, **kwargs)

    def __getitem__(self, key: Union[FormatterPlaceholder, str]) -> Formatter:
        """Get the formatter for the given formatter placeholder.

        Args:
            key (FormatterPlaceholder): The formatter placeholder.

        Returns:
            (Formatter) The formatter for the formatter placeholder.
        """
        if isinstance(key, str):
            key = FormatterPlaceholder(key, [])
        if key.name in self:
            return super().__getitem__(key.name)
        for fallback in key.fallbacks:
            if fallback in self:
                return super().__getitem__(fallback)
        return self["base"]


def deferred_formatter_group(group: FormatterGroup) -> FormatterGroup:
    """Wrap all formatters in a FormatterGroup with a DeferredFormatter.

    Args:
        group (FormatterGroup): The FormatterGroup to wrap.

    Returns:
        (FormatterGroup) A new FormatterGroup with all formatters wrapped in a
            DeferredFormatter.
    """
    new_group = FormatterGroup(base=None)
    for name, formatter in group.items():
        new_group[name] = DeferredFormatter(formatter)
    return new_group


class FormatterPlaceholder:
    def __init__(self, name: str, fallbacks: List[FormatterPlaceholder]):
        self.name = name
        self.fallbacks = copy(fallbacks)
        self.fallbacks.append("base")


class Formatter(ABC):
    component_class: Type["BaseComponent"]
    data_prop: str = "data"

    def encode(self, cell: Any, **kwargs):
        """Encode the cell on the backend before sending it to the frontend.

        The cell is lazily loaded, so when used on a LambdaColumn,
        ``cell`` will be a ``LambdaCell``. This is important for
        displays that don't actually need to apply the lambda in order
        to display the value.
        """
        return cell

    @abstractproperty
    def props(self):
        return self._props

    @abstractmethod
    def _get_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _set_state(self, state: Dict[str, Any]):
        pass

    @staticmethod
    def to_yaml(dumper: yaml.Dumper, data: Formatter):
        """This function is called by the YAML dumper to convert a
        :class:`Formatter` object into a YAML node.

        It should not be called directly.
        """
        data = {
            "class": type(data),
            "state": data._get_state(),
        }
        return dumper.represent_mapping("!Formatter", data)

    @staticmethod
    def from_yaml(loader, node):
        """This function is called by the YAML loader to convert a YAML node
        into an :class:`Formatter` object.

        It should not be called directly.
        """
        data = loader.construct_mapping(node)
        formatter = data["class"].__new__(data["class"])
        formatter._set_state(data["state"])
        return formatter

    def html(self, cell: Any):
        """When not in interactive mode, objects are visualized using static
        html.

        This method should produce that static html for the cell.
        """
        return str(cell)


MeerkatDumper.add_multi_representer(Formatter, Formatter.to_yaml)
MeerkatLoader.add_constructor("!Formatter", Formatter.from_yaml)


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

    @property
    def props(self):
        return self.wrapped.props

    def html(self, cell: DeferredCell):
        return self.wrapped.html(cell())

    @staticmethod
    def _get_state(self):
        data = {
            "class": type(self),
            "wrapped": {
                "class": type(self.wrapped),
                "_props": self.wrapped._props,
            },
        }
        return data

    @staticmethod
    def _set_state(state: Dict[str, Any]):
        formatter = state["class"].__new__(state["class"])
        wrapped_state = state["wrapped"]
        wrapped = wrapped_state["class"].__new__(wrapped_state["class"])
        wrapped._props = wrapped_state["_props"]
        formatter.wrapped = wrapped
        return formatter


MeerkatDumper.add_multi_representer(DeferredFormatter, DeferredFormatter.to_yaml)
MeerkatLoader.add_constructor("!DeferredFormatter", DeferredFormatter.from_yaml)
