from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
import collections
from copy import copy
from dataclasses import dataclass
import yaml

from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Type, Union
from meerkat.tools.utils import MeerkatLoader, MeerkatDumper

from meerkat.columns.deferred.base import DeferredCell

if TYPE_CHECKING:
    from meerkat.interactive.app.src.lib.component.abstract import BaseComponent


class Variant:
    def __init__(*args, **kwargs):
        pass


class FormatterGroup(collections.abc.Mapping):
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

    def __init__(self, base: Formatter = None, **kwargs):
        if base is None:
            from meerkat.interactive.app.src.lib.component.core.text import (
                TextFormatter,
            )
            # everything has a str method so this is a safe default   
            base = TextFormatter()

        if not isinstance(base, Formatter):
            raise TypeError("base must be a Formatter")
        for key, value in kwargs.items():
            if key not in formatter_placeholders:
                raise ValueError(
                    f"The key {key} is not a registered formatter "
                    "placeholder. Use `mk.register_formatter_placeholder`"
                )
            if not isinstance(value, Formatter):
                raise TypeError(
                    f"FormatterGroup values must be Formatters, not {type(value)}"
                )

        # must provide a base formatter
        self._dict = dict(base=base, **kwargs)

    def __getitem__(self, key: Union[FormatterPlaceholder, str]) -> Formatter:
        """Get the formatter for the given formatter placeholder.

        Args:
            key (FormatterPlaceholder): The formatter placeholder.

        Returns:
            (Formatter) The formatter for the formatter placeholder.
        """
        if isinstance(key, str):
            if key in formatter_placeholders:
                key = formatter_placeholders[key]
            else:
                key = FormatterPlaceholder(key, [])

        if key.name in self._dict:
            return self._dict.__getitem__(key.name)
        for fallback in key.fallbacks:
            if fallback.name in self._dict:
                return self.__getitem__(fallback)
        return self._dict["base"]

    def __setitem__(
        self, key: Union[FormatterPlaceholder, str], value: Formatter
    ) -> None:
        if key not in formatter_placeholders:
            raise ValueError(
                f"The key {key} is not a registered formatter "
                "placeholder. Use `mk.register_formatter_placeholder`"
            )
        if not isinstance(value, Formatter):
            raise TypeError(
                f"FormatterGroup values must be Formatters, not {type(value)}"
            )
        return self._dict.__setitem__(key, value)

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator:
        return iter(self._dict)

    @staticmethod
    def to_yaml(dumper: yaml.Dumper, data: Formatter):
        """This function is called by the YAML dumper to convert a
        :class:`Formatter` object into a YAML node.

        It should not be called directly.
        """
        data = {
            "class": type(data),
            "dict": data._dict,
        }
        return dumper.represent_mapping("!FormatterGroup", data)

    @staticmethod
    def from_yaml(loader, node):
        """This function is called by the YAML loader to convert a YAML node
        into an :class:`Formatter` object.

        It should not be called directly.
        """
        data = loader.construct_mapping(node)
        formatter = data["class"].__new__(data["class"])
        formatter._dict = data["dict"]
        return formatter


MeerkatDumper.add_multi_representer(FormatterGroup, FormatterGroup.to_yaml)
MeerkatLoader.add_constructor("!FormatterGroup", FormatterGroup.from_yaml)


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
    def __init__(
        self,
        name: str,
        fallbacks: List[Union[str, FormatterPlaceholder]],
        description: str = "",
    ):
        global formatter_placeholders
        self.name = name
        self.fallbacks = [
            fb if isinstance(fb, FormatterPlaceholder) else formatter_placeholders[fb]
            for fb in fallbacks
        ]

        if name != "base":
            self.fallbacks.append(FormatterPlaceholder("base", fallbacks=[]))

        self.description = description


formatter_placeholders = {
    "base": FormatterPlaceholder("base", []),
}


def register_placeholder(
    name: str, fallbacks: List[FormatterPlaceholder] = [], description: str = ""
):
    """Register a new formatter placeholder.

    Args:
        name (str): The name of the formatter placeholder.
        fallbacks (List[FormatterPlaceholder]): The fallbacks for the formatter
            placeholder.
        description (str): A description of the formatter placeholder.
    """
    if name in formatter_placeholders:
        raise ValueError(f"{name} is already a registered formatter placeholder")
    formatter_placeholders[name] = FormatterPlaceholder(
        name=name, fallbacks=fallbacks, description=description
    )


# register core formatter placeholders
register_placeholder("small", fallbacks=[], description="A small version of the data.")
register_placeholder("tiny", fallbacks=["small"], description="A tiny version of the data.")
register_placeholder(
    "thumbnail", fallbacks=["small"], description="A thumbnail of the data."
)
register_placeholder(
    "icon", fallbacks=["tiny"], description="An icon representing the data."
)
register_placeholder(
    "tag",
    fallbacks=["tiny"],
    description="A small version of the data meant to go in a tag field.",
)


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
        data = loader.construct_mapping(node, deep=True)
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

    def encode(self, cell: DeferredCell, **kwargs):
        return self.wrapped.encode(cell(), **kwargs)

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
            "wrapped": {
                "class": self.wrapped.__class__,
                "state": self.wrapped._get_state(),
            },
        }
        return data

    @staticmethod
    def _set_state(state: Dict[str, Any]):
        formatter = state["class"].__new__(state["class"])
        wrapped_state = state["wrapped"]
        wrapped = wrapped_state["class"].__new__(wrapped_state["class"])
        wrapped._set_state(wrapped_state["state"])
        formatter.wrapped = wrapped
        return formatter


MeerkatDumper.add_multi_representer(DeferredFormatter, DeferredFormatter.to_yaml)
MeerkatLoader.add_constructor("!DeferredFormatter", DeferredFormatter.from_yaml)
