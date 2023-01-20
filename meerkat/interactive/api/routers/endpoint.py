from typing import TYPE_CHECKING, Any, List, Tuple

from meerkat.errors import TriggerError
from meerkat.interactive.endpoint import Endpoint, endpoint

if TYPE_CHECKING:
    from meerkat.interactive.modification import Modification


@endpoint(prefix="/endpoint", route="/{endpoint}/dispatch/")
def dispatch(
    endpoint: Endpoint,
    payload: dict,
) -> Tuple[Any, List["Modification"]]:
    # TODO: figure out how to use the payload
    """Call an endpoint."""
    from meerkat.interactive.modification import StoreModification

    # `payload` is a dict with {detail: {key: value} | primitive}
    # Unpack the payload to build the fn_kwargs
    fn_kwargs = {}
    kwargs = payload["detail"]
    if isinstance(kwargs, dict):
        fn_kwargs = kwargs

    try:
        # Run the endpoint
        result, modifications = endpoint.partial(**fn_kwargs).run()
    except TriggerError as e:
        # TODO: handle case where result is not none
        return {"result": None, "modifications": [], "error": str(e)}

    # Only return store modifications that are not backend_only
    modifications = [
        m
        for m in modifications
        if not (isinstance(m, StoreModification) and m.backend_only)
    ]

    # Return the modifications and the result to the frontend
    return {"result": result, "modifications": modifications, "error": None}
