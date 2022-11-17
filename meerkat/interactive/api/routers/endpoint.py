from typing import TYPE_CHECKING, Any, List, Tuple

from meerkat.interactive.endpoint import Endpoint, endpoint
from meerkat.state import state
from meerkat.interactive.graph import Modification, trigger


if TYPE_CHECKING:
    from meerkat.interactive.modification import Modification


@endpoint(prefix="/endpoint", route="/{endpoint}/dispatch/")
def dispatch(
    endpoint: Endpoint,
    fn_kwargs: dict,
    payload: dict = None,
) -> Tuple[Any, List["Modification"]]:
    # TODO: figure out how to use the payload
    """Call an endpoint."""
    # Call the endpoint
    result, modifications = endpoint.partial(**fn_kwargs).run()

    # Return the modifications and the result to the frontend
    return result, modifications
