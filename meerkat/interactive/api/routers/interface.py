import base64
from io import BytesIO

import PIL
from fastapi import APIRouter, HTTPException

from meerkat.columns.image_column import ImageColumn
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.datapanel import DataPanel
from meerkat.datasets.imagenette import build_imagenette_dp

from meerkat.interactive.state import interfaces


def get_interface(interface_id: int):
    print(type(interface_id))
    if interface_id not in interfaces:
        raise HTTPException(
            status_code=404, detail="No interface with id {}".format(interface_id)
        )
    return interfaces[interface_id]


router = APIRouter(
    prefix="/interface",
    tags=["interface"],
    responses={404: {"description": "Not found"}},
)


@router.get("/config/")
def get_config(id: int):
    interface = get_interface(id)
    return interface.config
