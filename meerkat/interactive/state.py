import subprocess
import weakref
from dataclasses import dataclass
from typing import Any, Dict, Union

from fastapi import FastAPI

from meerkat.columns.abstract import AbstractColumn
from meerkat.datapanel import DataPanel
from meerkat.interactive.server import Server


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


@dataclass
class NetworkInfo:

    api: FastAPI
    api_server_port: int
    api_server: Server
    npm_server_port: int
    npm_process: subprocess.Popen
    api_server_name: str = "localhost"
    npm_server_name: str = "localhost"

    def __post_init__(self):
        # Hit the npm server _network endpoint with the api url
        # requests.get(url=f"{self.npm_server_url}/network/register",
        # params={"api": self.api_server_url})
        pass

    @property
    def npm_server_url(self):
        return f"http://{self.npm_server_name}:{self.npm_server_port}"

    @property
    def npm_network_url(self):
        return f"{self.npm_server_url}/network"

    @property
    def npm_network(self):
        from IPython.display import IFrame

        return IFrame(self.npm_network_url, width=800, height=100)

    @property
    def api_server_url(self):
        return f"http://{self.api_server_name}:{self.api_server_port}"

    @property
    def api_docs_url(self):
        return f"{self.api_server_url}/docs"

    @property
    def api_docs(self):
        from IPython.display import IFrame

        return IFrame(self.api_docs_url, width=800, height=600)


@dataclass
class GlobalState:

    network_info: NetworkInfo = None

    def update(self, key: str, value: Any):
        self.__dict__[key] = value

    def weak_update(self, key: str, value: Any):
        self.__dict__[key] = weakref.ref(value)

    def get(self, key: str):
        if isinstance(self.__dict__[key], weakref.ReferenceType):
            return self.__dict__[key]()
        return self.__dict__[key]

    def __getattribute__(self, __name: str) -> Any:
        attr = object.__getattribute__(self, __name)
        if isinstance(attr, weakref.ReferenceType):
            return attr()
        return attr


global state
state = GlobalState()
