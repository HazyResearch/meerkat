from dataclasses import dataclass
import weakref
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Interface:
    data: str
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
