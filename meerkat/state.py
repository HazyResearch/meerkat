import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping

from fastapi import FastAPI, HTTPException

from meerkat.tools.utils import WeakMapping

if TYPE_CHECKING:
    from meerkat.interactive.graph import Modification
    from meerkat.interactive.server import Server
    from meerkat.mixins.identifiable import IdentifiableMixin


@dataclass
class Secrets:

    api_keys: Dict[str, str] = field(default_factory=dict)

    def add(self, api: str, api_key: str):
        self.api_keys[api] = api_key

    def get(self, api: str):
        try:
            return self.api_keys[api]
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"No API key found for {api}.\
                 Add one with `secrets.add(api, api_key)`.",
            )


@dataclass
class LanguageModel:

    manifest: Any = None

    def set(self, client: str = "ai21", engine: str = "j1-jumbo"):
        from manifest import Manifest

        self.manifest = Manifest(
            client_name=client,
            client_connection=state.secrets.get(client),
            engine=engine,
            cache_name="sqlite",
            cache_connection="./logs",
        )

    def get(self):
        return self.manifest


@dataclass
class NetworkInfo:

    api: FastAPI
    api_server_port: int
    api_server: "Server"
    npm_server_port: int
    npm_process: subprocess.Popen = None
    api_server_name: str = "localhost"
    npm_server_name: str = "localhost"
    shareable_npm_server_name: str = None
    shareable_api_server_name: str = None
    npm_out_path: str = None
    npm_err_path: str = None

    def __post_init__(self):
        # Hit the npm server _network endpoint with the api url
        # requests.get(url=f"{self.npm_server_url}/network/register",
        # params={"api": self.api_server_url})
        pass

    @property
    def shareable_npm_server_url(self):
        if self.shareable_npm_server_name is None:
            return None
        return f"http://{self.shareable_npm_server_name}"

    @property
    def shareable_api_server_url(self):
        if self.shareable_api_server_name is None:
            return None
        return f"http://{self.shareable_api_server_name}"

    @property
    def npm_server_url(self):
        return f"http://{self.npm_server_name}:{self.npm_server_port}"

    @property
    def npm_network_url(self):
        return f"{self.npm_server_url}/network"

    @property
    def npm_server_out(self) -> str:
        if self.npm_out_path is None:
            return ""
        return open(self.npm_out_path, "r").read()

    @property
    def npm_server_err(self) -> str:
        if self.npm_err_path is None:
            return ""
        return open(self.npm_err_path, "r").read()

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
    """We maintain a separate group for each type of identifiable object.

    Objects in the group are identified by a unique id.
    """

    columns: WeakMapping = field(default_factory=WeakMapping)
    dataframes: WeakMapping = field(default_factory=WeakMapping)
    interfaces: Mapping = field(default_factory=dict)
    slicebys: WeakMapping = field(default_factory=WeakMapping)
    aggregations: WeakMapping = field(default_factory=WeakMapping)
    refs: WeakMapping = field(default_factory=WeakMapping)
    box_operations: WeakMapping = field(default_factory=WeakMapping)
    components: WeakMapping = field(default_factory=WeakMapping)
    refs: WeakMapping = field(default_factory=WeakMapping)
    stores: WeakMapping = field(default_factory=WeakMapping)
    endpoints: WeakMapping = field(default_factory=WeakMapping)

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
class ModificationQueue:
    """A queue of modifications to be applied to a dataframe."""

    queue: List["Modification"] = field(default_factory=list)

    def add(self, modification: "Modification"):
        self.queue.append(modification)


@dataclass
class GlobalState:

    network_info: NetworkInfo = None
    identifiables: Identifiables = field(default_factory=Identifiables)
    secrets: Secrets = field(default_factory=Secrets)
    llm: LanguageModel = field(default_factory=LanguageModel)
    modification_queue: ModificationQueue = field(default_factory=ModificationQueue)


global state
state = GlobalState()


def add_secret(api: str, api_key: str):
    """Add an API key to the global state."""
    state.secrets.add(api, api_key)
