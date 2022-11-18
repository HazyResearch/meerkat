from dataclasses import dataclass
from ..abstract import Component
import re
from typing import Callable, List, Optional, Tuple
import ast

from fastapi import HTTPException
import numpy as np

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import Endpoint, endpoint

# from meerkat.interactive.modification import Modification


@endpoint
def get_discover_schema(df: DataFrame):
    import meerkat as mk
    from meerkat.interactive.api.routers.dataframe import (
        _get_column_infos,
        SchemaResponse,
    )

    columns = [
        k
        for k, v in df.items()
        if isinstance(v, mk.NumpyArrayColumn) and len(v.shape) == 2
    ]
    return SchemaResponse(
        id=df.id,
        columns=_get_column_infos(df, columns),
        nrows=len(df),
    )


@endpoint
def discover(df: DataFrame, against: str):
    print("discovering")


@dataclass
class Discover(Component):

    df: "DataFrame"
    get_discover_schema: str

    def __post_init__(self):
        super().__post_init__()

        self.get_discover_schema = get_discover_schema.partial(
            df=self.df,
        )

        self.on_match = discover.partial(
            df=self.df,
        )
        # if self.on_match is not None:
        #     on_match = on_match.compose(self.on_match)
