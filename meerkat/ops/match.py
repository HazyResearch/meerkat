from typing import Optional, Union

import numpy as np

from meerkat import AbstractColumn, DataPanel

from .embed import embed


def match(
    data: Union[DataPanel, AbstractColumn],
    to: Union[str, AbstractColumn],
    input: Optional[str] = None,
):
    """Match data to another column.

    Args:
        data: A datapanel or column containing the data to embed.
        to: An
    """
    # Embed the data.
    embeddings = embed(data=data, input=input, encoder="clip")
