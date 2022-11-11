import functools
from typing import Any, List, Tuple

from fastapi import APIRouter, Body

from meerkat.interactive.graph import Endpoint, Modification

router = APIRouter(
    prefix="/endpoint",
    tags=["endpoint"],
    responses={404: {"description": "Not found"}},
)

EmbeddedBody = functools.partial(Body, embed=True)


@router.post("/{endpoint_id}/dispatch/")
def dispatch(
    endpoint_id: str,
    kwargs=EmbeddedBody(),
    payload=EmbeddedBody(default=None),
) -> Tuple[Any, List[Modification]]:
    """Call an endpoint."""
    # Look up the endpoint
    endpoint: Endpoint = Endpoint.get(endpoint_id)

    # Call the endpoint
    result = endpoint(**kwargs)()

    # Get the modifications
    from meerkat.state import state

    modifications = state.modification_queue.queue

    # Reset the modification queue
    state.modification_queue.queue = []

    # Return the modifications and the result to the frontend
    return result, modifications
