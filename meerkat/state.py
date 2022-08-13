import subprocess
import weakref
from ast import Global
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Union

from fastapi import FastAPI, HTTPException

from meerkat.columns.abstract import AbstractColumn
from meerkat.datapanel import DataPanel
from meerkat.interactive.server import Server
from meerkat.tools.utils import WeakMapping

if TYPE_CHECKING:
    from meerkat.mixins.identifiable import IdentifiableMixin


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
class Identifiables:
    """
    We maintain a separate group for each type of identifiable object.
    Objects in the group are identified by a unique id.
    """

    columns: WeakMapping = field(default_factory=WeakMapping)
    datapanels: WeakMapping = field(default_factory=WeakMapping)
    interfaces: Mapping = field(default_factory=dict)
    slicebys: WeakMapping = field(default_factory=WeakMapping)
    aggregations: WeakMapping = field(default_factory=WeakMapping)

    def add(self, obj: "IdentifiableMixin"):
        group = getattr(self, obj.identifiable_group)
        group[obj.id] = obj

    def get(self, id: str, group: str):
        group, group_name = getattr(self, group), group
        try:
            value = group[id]
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"No object in group '{group_name}' with id '{id}'",
            )
        return value


@dataclass
class GlobalState:

    network_info: NetworkInfo = None
    identifiables: Identifiables = field(default_factory=Identifiables)


global state
state = GlobalState()
