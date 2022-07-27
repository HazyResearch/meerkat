import base64
from io import BytesIO

import PIL
from fastapi import APIRouter

from meerkat.columns.image_column import ImageColumn
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.datapanel import DataPanel
from meerkat.datasets.imagenette import build_imagenette_dp

router = APIRouter(
    prefix="/dp",
    tags=["dp"],
    responses={404: {"description": "Not found"}},
)


def image_to_base64(image: PIL.Image) -> str:
    with BytesIO() as buffer:
        image.save(buffer, "jpeg")
        return "data:image/jpeg;base64,{im_base_64}".format(
            im_base_64=base64.b64encode(buffer.getvalue()).decode()
        )


# TODO: create Pydantic model to validate the returned data
def _get_rows(dp: DataPanel, limit: int = None, offset: int = 0):
    """
    Get rows from a DataPanel as a JSON object.
    """
    if limit is not None:
        dp = dp.lz[offset : offset + limit]

    # TODO(karan): efficiency, coverage, type conversion
    data = {}
    for col in dp.columns:
        if isinstance(dp.lz[col], PandasSeriesColumn):
            data[col] = dp[col].to_list()
        elif isinstance(dp.lz[col], ImageColumn):
            data[col] = dp.lz[col].map(lambda x: image_to_base64(x))
        else:
            raise NotImplementedError("Column type not implemented.")

    return data


@router.get("/rows/")
def get_rows(limit: int = 32, offset: int = 0):
    """
    Get rows from a DataPanel as a JSON object.
    """

    # TODO(karan): Robust lookup for the DataPanel instance
    # that is pertinent to the request
    dp = build_imagenette_dp(
        dataset_dir="/Users/krandiash/Desktop/"
        "workspace/projects/datasci/data/imagenette/",
        version="320px",
        download=True,
    )

    return _get_rows(dp, limit, offset)
