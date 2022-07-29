import base64
from io import BytesIO

import PIL
from fastapi import APIRouter

from meerkat.columns.image_column import ImageColumn
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.columns.list_column import ListColumn
from meerkat.datapanel import DataPanel

from .interface import get_interface

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


# # TODO: create Pydantic model to validate the returned data
# def _get_rows(dp: DataPanel, limit: int = None, offset: int = 0):
#     """
#     Get rows from a DataPanel as a JSON object.
#     """
#     if limit is not None:
#         dp = dp.lz[offset : offset + limit]

#     # TODO(karan): efficiency, coverage, type conversion
#     data = {}
#     for col in dp.columns:
#         if isinstance(dp.lz[col], PandasSeriesColumn):
#             data[col] = dp[col].to_list()
#         elif isinstance(dp.lz[col], ImageColumn):
#             data[col] = dp.lz[col].map(lambda x: image_to_base64(x))
#         else:
#             raise NotImplementedError("Column type not implemented.")

#     return data

def _get_rows(dp: DataPanel, limit: int = None, offset: int = 0):
    if limit is not None:
        dp = dp.lz[offset : offset + limit]

    # TODO(karan): efficiency, coverage, type conversion
    columns = dp.columns
    types = []
    for col in columns:
        if isinstance(dp[col], ImageColumn):
            dp[col] = dp[col].to_lambda(image_to_base64)
            types.append("image")
        elif isinstance(dp[col], PandasSeriesColumn):
            dp[col] = ListColumn(dp[col].to_list())
            types.append("string")
        else: 
            raise NotImplementedError("Column type not implemented.")

    data = {
        "columns": columns, 
        "types": types, 
        "rows": [], 
    }
    for row in dp:
        data["rows"].append([row[col] for col in columns])
    return data 



@router.get("/rows/")
def get_rows(interface_id: int, limit: int = 32, offset: int = 0):
    """
    Get rows from a DataPanel as a JSON object.
    """
    dp = get_interface(interface_id).data 
    return _get_rows(dp, limit, offset)
