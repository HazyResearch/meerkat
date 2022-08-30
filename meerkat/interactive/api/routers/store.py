import functools
from typing import Any, Dict, List, Union

from fastapi import APIRouter, Body

from meerkat.interactive.graph import StoreModification, trigger

from ....tools.utils import convert_to_python

router = APIRouter(
    prefix="/store",
    tags=["store"],
    responses={404: {"description": "Not found"}},
)

EmbeddedBody = functools.partial(Body, embed=True)


@router.post("/{store_id}/trigger")
def store_trigger(store_id: str, value: any = EmbeddedBody()):
    """
    Triggers the computational graph when a store on the frontend changes.
    """
    # Create a store modification
    store_modification = StoreModification(store_id, value)

    # Check if this request would actually change the value of the store
    current_store_value = store_modification.node.value
    if current_store_value == value: return []

    # Trigger on the store modification: leads to modifications on the graph
    modifications = trigger([store_modification])

    # Return the modifications
    return modifications
