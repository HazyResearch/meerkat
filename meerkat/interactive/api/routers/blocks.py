import functools
from typing import Any, Dict, List, Union

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

import meerkat as mk
from meerkat.datapanel import DataPanel
from meerkat.state import state

from ....tools.utils import convert_to_python

router = APIRouter(
    prefix="/dp",
    tags=["dp"],
    responses={404: {"description": "Not found"}},
)

EmbeddedBody = functools.partial(Body, embed=True)

