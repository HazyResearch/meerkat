import weakref
from dataclasses import dataclass
from typing import Any, Dict, Union

from meerkat.datapanel import DataPanel
from meerkat.columns.abstract import AbstractColumn


@dataclass
class Interface:
    # TODO (all): decide on this schema 
    data: Union[DataPanel, AbstractColumn]
    config: Dict


interfaces = {}


def add_interface(
    data: str,
    config: Dict,
):
    interface = Interface(data=data, config=config)
    interface_id = id(interface)
    interfaces[interface_id] = interface
    return interface_id
