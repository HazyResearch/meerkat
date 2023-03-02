import logging
import traceback

import numpy as np
import pandas as pd
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from meerkat.columns.abstract import Column
from meerkat.interactive.endpoint import Endpoint, endpoint
from meerkat.tools.lazy_loader import LazyLoader

torch = LazyLoader("torch")

logger = logging.getLogger(__name__)


@endpoint(prefix="/endpoint", route="/{endpoint}/dispatch/")
def dispatch(
    endpoint: Endpoint,
    payload: dict,
) -> dict:
    """Call an endpoint."""
    logger.debug(f"Dispatching endpoint {endpoint} with payload {payload}.")
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
    except Exception as e:
        # General exception should be converted to a HTTPException
        # that fastapi can handle.
        from meerkat.state import state

        logger.debug("Exception in dispatch", exc_info=True)
        state.progress_queue.add(None)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Only return store modifications that are not backend_only
    modifications = [
        m
        for m in modifications
        if not (isinstance(m, StoreModification) and m.backend_only)
    ]

    # Return the modifications and the result to the frontend
    # Need to support sending back numpy arrays, torch tensors, and pandas series
    return jsonable_encoder(
        {"result": result, "modifications": modifications, "error": None},
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
