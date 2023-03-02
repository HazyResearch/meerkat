import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from fastapi import FastAPI, HTTPException

from meerkat.interactive.server import Server
from meerkat.tools.utils import WeakMapping

if TYPE_CHECKING:
    from meerkat.interactive.modification import Modification
    from meerkat.mixins.identifiable import IdentifiableMixin

logger = logging.getLogger(__name__)


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
class APIInfo:
    api: Optional[FastAPI]
    port: Optional[int]
    server: Optional[Server] = None
    name: str = "localhost"
    shared: bool = False
    process: Optional[subprocess.Popen] = None
    _url: Optional[str] = None

    @property
    def url(self):
        if self._url:
            return self._url
        if self.shared:
            return f"http://{self.name}"
        return f"http://{self.name}:{self.port}"

    @property
    def docs_url(self):
        return f"{self.url}/docs"

    @property
    def docs(self):
        from IPython.display import IFrame

        return IFrame(self.docs_url, width=800, height=600)


@dataclass
class FrontendInfo:
    package_manager: Optional[str]
    port: Optional[int]
    name: str = "localhost"
    shared: bool = False
    process: Optional[subprocess.Popen] = None
    _url: Optional[str] = None

    @property
    def url(self):
        if self._url:
            return self._url
        if self.shared:
            return f"http://{self.name}"
        return f"http://{self.name}:{self.port}"


@dataclass
class Identifiables:
    """We maintain a separate group for each type of identifiable object.

    Objects in the group are identified by a unique id.
    """

    columns: WeakMapping = field(default_factory=WeakMapping)
    dataframes: WeakMapping = field(default_factory=WeakMapping)
    pages: Mapping = field(default_factory=dict)
    slicebys: WeakMapping = field(default_factory=WeakMapping)
    aggregations: WeakMapping = field(default_factory=WeakMapping)
    box_operations: WeakMapping = field(default_factory=WeakMapping)
    components: WeakMapping = field(default_factory=WeakMapping)
    refs: WeakMapping = field(default_factory=WeakMapping)
    stores: WeakMapping = field(default_factory=WeakMapping)
    endpoints: WeakMapping = field(default_factory=WeakMapping)
    routers: WeakMapping = field(default_factory=WeakMapping)
    nodes: WeakMapping = field(default_factory=WeakMapping)
    states: WeakMapping = field(default_factory=WeakMapping)

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

    # Boolean attribute that controls whether the queue is accepting new
    # modifications
    # When _ready is False, `add` will no-op
    _ready: bool = False

    def add(self, modification: "Modification"):
        if self._ready:
            logger.debug(f"Adding modification {modification} to queue.")
            self.queue.append(modification)
            return
        # Do nothing if not ready
        logger.debug(f"Modification queue not ready. Ignoring {modification}.")

    def clear(self) -> List["Modification"]:
        """Clear the modification queue, and return the old queue."""
        logger.debug("Clearing modification queue.")
        current_queue = self.queue
        self.queue = []
        return current_queue

    def ready(self):
        """Ready the queue for accepting new modifications."""
        count = 0
        while self._ready:
            # Modification queue is already in use
            # Wait for it to be unready
            logger.debug("Modification queue is already in use. Waiting...")
            time.sleep(0.1)
            count += 1
            if count == 1e-3:
                logger.warn(
                    "Modification queue is taking a long time to unready."
                    "Check for deadlocks."
                )

        self._ready = True
        logger.debug("Modification queue is now ready.")

    def unready(self):
        """Unready the queue for accepting new modifications."""
        self._ready = False
        logger.debug("Modification queue is now unready.")


@dataclass
class ProgressQueue:
    """A queue of progress messages to be displayed to the user."""

    queue: list = field(default_factory=list)

    def add(self, message: str):
        self.queue.append(message)

    def clear(self) -> list:
        """Clear the progress queue, and return the old queue."""
        current_queue = self.queue
        self.queue = []
        return current_queue


@dataclass
class GlobalState:
    api_info: Optional[APIInfo] = None
    frontend_info: Optional[FrontendInfo] = None
    identifiables: Identifiables = field(default_factory=Identifiables)
    secrets: Secrets = field(default_factory=Secrets)
    llm: LanguageModel = field(default_factory=LanguageModel)
    modification_queue: ModificationQueue = field(default_factory=ModificationQueue)
    progress_queue: ProgressQueue = field(default_factory=ProgressQueue)


global state
state = GlobalState()


def add_secret(api: str, api_key: str):
    """Add an API key to the global state."""
    state.secrets.add(api, api_key)


def run_on_startup():
    """Run on startup."""
    frontend_url = os.environ.get("MEERKAT_FRONTEND_URL", None)
    if frontend_url:
        state.frontend_info = FrontendInfo(None, None, _url=frontend_url)

    api_url = os.environ.get("MEERKAT_API_URL", None)
    if api_url:
        state.api_info = APIInfo(None, None, _url=api_url)


run_on_startup()
