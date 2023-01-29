import logging
from typing import List

from meerkat.interactive.endpoint import Endpoint, endpoint
from meerkat.interactive.graph import Store, trigger
from meerkat.interactive.modification import Modification, StoreModification
from meerkat.state import state

logger = logging.getLogger(__name__)

@endpoint(prefix="/store", route="/{store}/trigger/")
def store_trigger(store: Store, value=Endpoint.EmbeddedBody()) -> List[Modification]:
    """Triggers the computational graph when a store on the frontend changes."""

    logger.debug(f"Updating store {store} with value {value}.")

    # TODO: the interface sends store_triggers for all stores when it starts
    # up -- these requests should not be being sent.
    # These requests are indirectly ignored here because we check if the
    # value of the store actually changed (and these initial post requests
    # do not change the value of the store).

    # Check if this request would actually change the value of the store
    # current_store_value = store_modification.node
    if store == value:
        logger.debug("Store value did not change. Skipping trigger.")
        return []

    # Ready the modification queue 
    state.modification_queue.ready()

    # Set the new value of the store
    store.set(value)

    # Trigger on the store modification: leads to modifications on the graph
    modifications = trigger()

    # only return modifications that are not backend_only
    modifications = [
        m
        for m in modifications
        if not (isinstance(m, StoreModification) and m.backend_only)
    ]

    # Return the modifications
    return modifications
