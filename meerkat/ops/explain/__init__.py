from typing import TYPE_CHECKING, Mapping, Tuple, Union

from meerkat import AbstractColumn, DataFrame, NumpyArrayColumn, embed

if TYPE_CHECKING:
    from domino import Slicer


def explain(
    data: Union[AbstractColumn, DataFrame],
    input: str,
    target: Union[str, Mapping[str, str]],
    method: Union[str, "Slicer"] = "MixtureSlicer",
    encoder: str = "clip",  # add support for auto selection of encoder
    modality: str = None,
    **kwargs,
) -> Tuple[NumpyArrayColumn, "Slicer"]:
    """Cluster the data in a column. If the column is an unstructured type,
    (e.g. image), the column is first embedded then clustered.

    Args:
        data (Union[DataFrame, AbstractColumn]): The column to cluster or a dataframe
            containing the column to cluster.
        input (Union[str, Sequence[str]]): The column(s) to cluster by. These
            columns will be embedded using the ``encoder`` and the resulting
            embedding will be used. Ignored if ``data`` is a Column.
        method (Union[str, Slicer]): The clustering method to use.
        encoder (str): The encoder to use for the embedding. Defaults to ``clip``.
        modality (Union[str, Sequence[str])): The modality to of the
        **kwargs: Additional keyword arguments to pass to the clustering method.

    Returns:
        (Union[NumpyArrayColumn, DataFrame], Slicer): A tuple containing the
            clustered column and the fit clusterer. If ``data`` is a DataFrame, the
            clustered column is added to the DataFrame and it is returned.
    """
    if isinstance(data, DataFrame):
        # TODO (sabri): Give the user the option to specify the output column.
        output_column = f"{method}({input},{target})"
        #embed_col = f"{encoder}({input})"
        embed_col = input

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
        import domino

        method = getattr(domino, method)(**kwargs)

    if isinstance(target, str):
        # TODO: make this generalizable – this is a hack to make it work for RFW
        target = {"targets": data[target], "pred_probs": None}

    elif isinstance(target, Mapping):
        target = {k: data[v] for k, v in target.items()}

    method.fit(embeddings=data_embedding.data, **target)
    slices = method.predict(
        embeddings=data_embedding.data,
        targets=None,
        pred_probs=None,
    )

    if isinstance(data, DataFrame):
        data[output_column] = slices
        return data, method
    return slices, method
