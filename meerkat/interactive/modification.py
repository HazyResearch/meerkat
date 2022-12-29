from abc import ABC
from typing import TYPE_CHECKING, Any, List

from pydantic import BaseModel

if TYPE_CHECKING:
    from meerkat.interactive.node import Node


class Modification(BaseModel, ABC):
    """Base class for modifications.

    Modifications are used to track changes to Reference and Store nodes
    in the graph.

    Attributes:
        id (str): The id of the Reference or Store.
    """

    id: str

    @property
    def node(self):
        """The Reference or Store node that this modification is for."""
        raise NotImplementedError()

    def add_to_queue(self):
        """Add this modification to the queue."""
        # Get the queue
        from meerkat.state import state

        state.modification_queue.add(self)


# TODO: need to consolidate Modification
# associate them with NodeMixin (Nodeable objects)
class DataFrameModification(Modification):
    scope: List[str]
    type: str = "ref"

    @property
    def node(self) -> "Node":
        from meerkat.state import state

        try:
            df = state.identifiables.get(group="dataframes", id=self.id)
            return df.inode
        except Exception:
            return state.identifiables.get(group="nodes", id=self.id)


class StoreModification(Modification):
    value: Any  # : Storeable # TODO(karan): Storeable prevents
    # us from storing objects in the store
    type: str = "store"

    @property
    def backend_only(self) -> bool:
        """Whether this modification should not be sent to frontend."""
        from meerkat.state import state

        store = state.identifiables.get(group="stores", id=self.id)
        return store._self_backend_only

    @property
    def node(self) -> "Node":
        from meerkat.state import state

        # FIXME: what's going on with this try-except here?
        try:
            store = state.identifiables.get(group="stores", id=self.id)
            return store.inode
        except Exception:
            return state.identifiables.get(group="nodes", id=self.id)
