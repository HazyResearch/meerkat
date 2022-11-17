from typing import List

from meerkat.state import state 
from meerkat.interactive.endpoint import Endpoint, endpoint
from meerkat.interactive.graph import Modification, Store, StoreModification, trigger


@endpoint(prefix="/store", route="/{store}/trigger/")
def store_trigger(store: Store, value=Endpoint.EmbeddedBody()) -> List[Modification]:
    """
    Triggers the computational graph when a store on the frontend
    changes.
    """
    # TODO: the interface sends store_triggers for all stores when it starts
    # up -- these requests should not be being sent.
    # These requests are indirectly ignored here because we check if the
    # value of the store actually changed (and these initial post requests
    # do not change the value of the store).


    # Check if this request would actually change the value of the store
    # current_store_value = store_modification.node
    if store == value:
        return []

    # Set the new value of the store
    # TODO (Sabri): Need to figure out how to get this to preserve the Pydantic type
    # of the store.
    # store_modification.node.set(value)
    print(value)

    store.set(value)

    state.modification_queue.ready()

    # Trigger on the store modification: leads to modifications on the graph
    modifications = trigger()

    state.modification_queue.unready()


    # Return the modifications
    return modifications
