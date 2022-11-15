from abc import ABC
from typing import TYPE_CHECKING, Any, List

from pydantic import BaseModel

if TYPE_CHECKING:
    from meerkat.interactive.graph import Reference, Store
    from meerkat.interactive.node import Node


class Modification(BaseModel, ABC):
    """
    Base class for modifications.

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


class ReferenceModification(Modification):
    scope: List[str]
    type: str = "ref"

    @property
    def node(self) -> "Reference":
        from meerkat.state import state

        return state.identifiables.get(group="refs", id=self.id)


class StoreModification(Modification):
    value: Any  # : Storeable # TODO(karan): Storeable prevents
    # us from storing objects in the store
    type: str = "store"

    @property
    def node(self) -> "Node":
        from meerkat.state import state

        store: Store = state.identifiables.get(group="stores", id=self.id)
        return store.inode
