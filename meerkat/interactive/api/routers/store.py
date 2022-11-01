import functools
from typing import List

from fastapi import APIRouter, Body

from meerkat.interactive.graph import Modification, StoreModification, trigger

router = APIRouter(
    prefix="/store",
    tags=["store"],
    responses={404: {"description": "Not found"}},
)

EmbeddedBody = functools.partial(Body, embed=True)


@router.post("/{store_id}/trigger/")
def store_trigger(store_id: str, value=EmbeddedBody()) -> List[Modification]:
    """Triggers the computational graph when a store on the frontend
    changes."""
    # Create a store modification
    store_modification = StoreModification(id=store_id, value=value)

    # Check if this request would actually change the value of the store
    current_store_value = store_modification.node.value
    if current_store_value == value:
        return []

    # Set the new value of the store
    # TODO (Sabri): Need to figure out how to get this to preserve the Pydantic type
    # of the store.
    store_modification.node.value = value

    # Trigger on the store modification: leads to modifications on the graph
    modifications = trigger([store_modification])

    # Return the modifications
    return modifications
