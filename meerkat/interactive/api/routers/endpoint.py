from typing import TYPE_CHECKING, Any, List, Tuple

from meerkat.interactive.endpoint import Endpoint, endpoint
from meerkat.interactive.graph import Modification

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
    from meerkat.interactive.modification import StoreModification

    result, modifications = endpoint.partial(**fn_kwargs).run()

    # only return store modifications that are not backend_only
    modifications = [
        m
        for m in modifications
        if not (isinstance(m, StoreModification) and m.backend_only)
    ]

    # Return the modifications and the result to the frontend
    return result, modifications
