from typing import List, Optional, Tuple, Union

from meerkat import Column, DataFrame, ScalarColumn

from .embed import embed


def match(
    data: Union[DataFrame, Column],
    query: Union[str, List[str], Tuple[str], ScalarColumn, DataFrame],
    against: Optional[str] = None,
    against_modality: Optional[str] = None,
    query_modality: Optional[str] = None,
    encoder: str = "clip",
    return_column_names: bool = False,
):
    """Match data to another column.

    This operation adds q columns to the dataframe where q is the number of queries.
    Note, if data is a dataframe, this operation is performed in-place.

    Args:
        data: A dataframe or column containing the data to embed.
        query: A single or multiple query strings to match against.
        against: If ``data`` is a dataframe, the name of the column
            to embed. If ``data`` is a column, then the parameter is ignored.
            Defaults to None.
        against_modality: The modality of the data in the against column. If None,
            infer from the against column.
        query_modality: The query modality. If None, infer from the query column.
        return_column_names: Whether to return the names of columns added based
            on match.

    Returns:
        mk.DataFrame: A view of ``data`` with a new column containing the embeddings.
        This column will be named according to the ``out_col`` parameter.
    """
    if against not in data:
        raise ValueError(f"Column {against} not found in data.")

    encoder = "clip"
    data_embedding = data[against]

    if not isinstance(query, Column):
        if isinstance(query, str):
            query = [query]
        query = ScalarColumn(query)
    # Text cannot be embedded with num_workers > 0 because the clip text encoder
    # is not pickleable.
    to_embedding = embed(
        data=query, encoder=encoder, num_workers=0, modality=query_modality, pbar=False
    )
    scores = data_embedding @ to_embedding.T
    column_names = []
    for i, query_item in enumerate(query):
        col_name = f"match({against}, {query_item})"
        data[col_name] = scores[:, i]
        column_names.append(col_name)
    return (data, column_names) if return_column_names else data
