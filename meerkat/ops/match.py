from typing import List, Optional, Tuple, Union

from meerkat import AbstractColumn, DataFrame, PandasSeriesColumn

from .embed import embed


def match(
    data: Union[DataFrame, AbstractColumn],
    query: Union[str, List[str], Tuple[str], PandasSeriesColumn],
    input: Optional[str] = None,
    input_modality: Optional[str] = None,
    query_modality: Optional[str] = None,
    return_column_names: bool = False,
):
    """Match data to another column.

    This operation adds q columns to the dataframe where q is the number of queries.
    Note, if data is a dataframe, this operation is performed in-place.

    Args:
        data: A dataframe or column containing the data to embed.
        query: A single or multiple query strings to match against.
        input: If ``data`` is a dataframe, the name of the column
            to embed. If ``data`` is a column, then the parameter is ignored.
            Defaults to None.
        input_modality: The input modality. If None, infer from the input column.
        query_modality: The query modality. If None, infer from the query column.
        return_column_names: Whether to return the names of columns added based
            on match.

    Returns:
        mk.DataFrame: A view of ``data`` with a new column containing the embeddings.
        This column will be named according to the ``out_col`` parameter.
    """
    encoder = "clip"
    # TODO (arjundd): Give the user the option to specify the output column.
    out_col = f"{encoder}({input})"
    # TODO (arjundd): Remove this guard once caching is supported.
    if out_col not in data.columns:
        embed(
            data=data,
            input=input,
            encoder=encoder,
            out_col=out_col,
            modality=input_modality,
        )

    data_embedding = data[out_col]

    if not isinstance(query, AbstractColumn):
        if isinstance(query, str):
            query = [query]
        query = PandasSeriesColumn(query)
    # Text cannot be embedded with num_workers > 0 because the clip text encoder
    # is not pickleable.
    to_embedding = embed(
        data=query, encoder=encoder, num_workers=0, modality=query_modality, pbar=False
    )

    scores = data_embedding @ to_embedding.T
    column_names = []
    for i, query_item in enumerate(query):
        col_name = f"_match_{input}_{query_item}"
        data[col_name] = scores[:, i]
        column_names.append(col_name)
    return (data, column_names) if return_column_names else data
