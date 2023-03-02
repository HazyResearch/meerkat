import logging
import traceback

import numpy as np
import pandas as pd
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from meerkat.columns.abstract import Column
from meerkat.interactive.endpoint import Endpoint, endpoint
from meerkat.interactive.graph import Store, trigger
from meerkat.interactive.modification import StoreModification
from meerkat.state import state
from meerkat.tools.lazy_loader import LazyLoader

torch = LazyLoader("torch")


logger = logging.getLogger(__name__)


# KG: do not use -> List[Modification] as the return type for the `update`
# fn. This causes Pydantic to drop several fields from the
# StoreModification object (it only ends up sending the ids).
@endpoint(prefix="/store", route="/{store}/update/")
def update(store: Store, value=Endpoint.EmbeddedBody()):
    """Triggers the computational graph when a store on the frontend
    changes."""

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
    try:
        modifications = trigger()
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Only return modifications that are not backend_only
    modifications = [
        m
        for m in modifications
        if not (isinstance(m, StoreModification) and m.backend_only)
    ]

    logger.debug(f"Returning modifications: {modifications}.")

    # Return the modifications
    return jsonable_encoder(
        modifications,
        custom_encoder={
            np.ndarray: lambda v: v.tolist(),
            torch.Tensor: lambda v: v.tolist(),
            pd.Series: lambda v: v.tolist(),
            Column: lambda v: v.to_json(),
            np.int64: lambda v: int(v),
            np.float64: lambda v: float(v),
            np.int32: lambda v: int(v),
            np.bool_: lambda v: bool(v),
            np.bool8: lambda v: bool(v),
        },
    )
