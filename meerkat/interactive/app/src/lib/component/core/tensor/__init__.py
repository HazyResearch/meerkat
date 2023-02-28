from typing import Dict, List

from pydantic import BaseModel

from meerkat.interactive.app.src.lib.component.abstract import Component


class TensorInfo(BaseModel):
    data: List
    shape: Dict[str, int]
    dtype: str


class Tensor(Component):
    data: TensorInfo
    dtype: str
