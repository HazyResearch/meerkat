import numpy as np
import pandas as pd
from fastapi.encoders import jsonable_encoder

from meerkat.columns.abstract import Column
from meerkat.interactive import Page
from meerkat.interactive.endpoint import endpoint
from meerkat.tools.lazy_loader import LazyLoader

torch = LazyLoader("torch")


@endpoint(prefix="/page", route="/{page}/config/", method="GET")
def config(page: Page):
    return jsonable_encoder(
        # TODO: we should not be doing anything except page.frontend
        # here. This is a temp workaround to avoid getting an
        # exception in the notebook.
        page.frontend if isinstance(page, Page) else page,
        custom_encoder={
            np.ndarray: lambda v: v.tolist(),
            torch.Tensor: lambda v: v.tolist(),
            pd.Series: lambda v: v.tolist(),
            Column: lambda v: v.to_json(),
            np.int64: lambda v: int(v),
            np.float64: lambda v: float(v),
        },
    )
