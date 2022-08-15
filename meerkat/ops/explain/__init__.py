from typing import Optional, Tuple, Union

import domino
import numpy as np
from domino import Slicer

from meerkat import AbstractColumn, DataPanel, NumpyArrayColumn, embed


def explain(
    data: Union[AbstractColumn, DataPanel],
    input: Optional[str] = None,
    target: Optional[str] = None,
    method: Union[str, Slicer] = "MixtureSlicer",
    encoder: str = "clip",  # add support for auto selection of encoder
    modality: str = None,
    **kwargs,
) -> Tuple[NumpyArrayColumn, Slicer]:
    """Cluster the data in a column. If the column is an unstructured type, (e.g.
    image), the column is first embedded then clustered.

    Args:
        data (Union[DataPanel, AbstractColumn]): The column to cluster or a datapanel
            containing the column to cluster.
        input (Union[str, Sequence[str]]): The column(s) to cluster by. These columns will
            be embedded using the ``encoder`` and the resulting embedding will be used.
            Ignored if ``data`` is a Column.
        method (Union[str, Slicer]): The clustering method to use.
        encoder (str): The encoder to use for the embedding. Defaults to ``clip``.
        modality (Union[str, Sequence[str])): The modality to of the
        **kwargs: Additional keyword arguments to pass to the clustering method.

    Returns:
        (Union[NumpyArrayColumn, DataPanel], Slicer): A tuple containing the
            clustered column and the fit clusterer. If ``data`` is a DataPanel, the
            clustered column is added to the DataPanel and it is returned.
    """
    if isinstance(data, DataPanel):
        # TODO (sabri): Give the user the option to specify the output column.
        output_column = f"{method}({input},{target})"
        embed_col = f"{encoder}({input})"

        # TODO (sabri): Remove this guard once caching is supported.
        if embed_col not in data:
            data = embed(
                data=data,
                input=input,
                encoder=encoder,
                out_col=embed_col,
                modality=modality,
            )
        data_embedding = data[embed_col]
    else:
        raise NotImplementedError

    data_embedding = data[embed_col]
    if isinstance(method, str):
        method = getattr(domino, method)(**kwargs)

    target = data[target]
    method.fit(
        embeddings=data_embedding.data, targets=target, pred_probs=np.zeros_like(target)
    )
    slices = method.predict(
        embeddings=data_embedding.data,
        pred_probs=np.zeros_like(target),
        targets=np.zeros_like(target),
    )

    if isinstance(data, DataPanel):
        data[output_column] = slices
        return data, method
    return slices, method
