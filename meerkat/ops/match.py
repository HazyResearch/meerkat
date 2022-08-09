from typing import List, Optional, Tuple, Union

from meerkat import AbstractColumn, DataPanel, PandasSeriesColumn

from .embed import embed


def match(
    data: Union[DataPanel, AbstractColumn],
    query: Union[str, List[str], Tuple[str], PandasSeriesColumn],
    input: Optional[str] = None,
    input_modality: Optional[str] = None,
    query_modality: Optional[str] = None,
):
    """Match data to another column.

    Args:
        data: A datapanel or column containing the data to embed.
        query: A single or multiple query strings to match against.
        input: If ``data`` is a datapanel, the name of the column
            to embed. If ``data`` is a column, then the parameter is ignored.
            Defaults to None.
        input_modality: The input modality. If None, infer from the input column.
        query_modality: The query modality. If None, infer from the query column.

    Returns:
        mk.DataPanel: A view of ``data`` with a new column containing the embeddings.
        This column will be named according to the ``out_col`` parameter.
    """
    encoder = "clip"
    # TODO (arjundd): Give the user the option to specify the output column.
    out_col = f"{encoder}({input})"
    # TODO (arjundd): Remove this guard once caching is supported.
    if out_col not in data.columns:
        embed(data=data, input=input, encoder=encoder, out_col=out_col, modality=input_modality)

    data_embedding = data[out_col]

    if not isinstance(query, AbstractColumn):
        if isinstance(query, str):
            query = [query]
        query = PandasSeriesColumn(query)
    # Text cannot be embedded with num_workers > 0 because the clip text encoder
    # is not pickleable.
    to_embedding = embed(data=query, encoder=encoder, num_workers=0, modality=query_modality)

    scores = data_embedding @ to_embedding.T
    for i, query_item in enumerate(query):
        data[f"_match_{input}_{query_item}"] = scores[:, i]
    return data
