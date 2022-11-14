from typing import TYPE_CHECKING, Any, List, Tuple

from meerkat.interactive.endpoint import Endpoint, endpoint
from meerkat.state import state

if TYPE_CHECKING:
    from meerkat.interactive.modification import Modification


@endpoint(prefix="/endpoint", route="/{endpoint}/dispatch/")
def dispatch(
    endpoint: Endpoint,
    kwargs: dict,
    payload: dict = None,
) -> Tuple[Any, List["Modification"]]:
    # TODO: figure out how to use the payload
    """Call an endpoint."""
    # Call the endpoint
    result = endpoint(**kwargs).run()

    # Get the modifications
    # from meerkat.state import state
    modifications = state.modification_queue.queue

    # Reset the modification queue
    state.modification_queue.queue = []

    # Return the modifications and the result to the frontend
    return result, modifications
