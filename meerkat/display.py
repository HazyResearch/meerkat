from __future__ import annotations

import base64
from io import BytesIO
from typing import TYPE_CHECKING, Any

from IPython.display import Audio
from PIL import Image

import meerkat as mk

if TYPE_CHECKING:
    from meerkat.columns.file_column import FileCell
    from meerkat.columns.lambda_column import LambdaCell


def auto_formatter(cell: Any):
    if isinstance(cell, Image.Image):
        return image_formatter(cell)
    else:
        return repr(cell)


def lambda_cell_formatter(cell: LambdaCell):
    return auto_formatter(cell.get())


def image_formatter(cell: Image.Image):

    if not mk.config.display.show_images:
        return repr(cell)

    max_image_width = mk.config.display.max_image_width
    max_image_height = mk.config.display.max_image_height

    cell.thumbnail((max_image_width, max_image_height))

    with BytesIO() as buffer:
        cell.save(buffer, "jpeg")
        im_base_64 = base64.b64encode(buffer.getvalue()).decode()
        return f'<img src="data:image/jpeg;base64,{im_base_64}">'


def image_file_formatter(cell: FileCell):
    if not mk.config.display.show_images:
        return repr(cell)

    return lambda_cell_formatter(cell)


def audio_file_formatter(cell: FileCell) -> str:
    if not mk.config.display.show_audio:
        return repr(cell)

    # TODO (Sabri): Implement based on audio_formatter so we can include transform
    #
    return Audio(filename=cell.absolute_path)._repr_html_()
