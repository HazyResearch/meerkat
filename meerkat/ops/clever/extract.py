from meerkat.columns.scalar import ScalarColumn
from meerkat.dataframe import DataFrame
from meerkat.ops.clever.chat import chat

from .impute import _prepare_row

_TRAIN_MASK_COLUMN = "__train_mask__"


def extract(
    df: DataFrame,
    column: str = None,
    missing: str = "nan",
    train_mask: ScalarColumn = None,
):
    """Extract information from the dataframe.

    Given some data (i.e. ``df``), extract out information about the column.

    Args:
        df: The dataframe.
        missing: The value to use for finding missing entries in ``df[column]``.
            This will be ignored if ``train_mask`` is not None.
        train_mask: The mask to use for in-context examples in the model.
            If None, then all rows in ``df`` that are not missing will be used.
    """
    if not isinstance(df[column], ScalarColumn):
        raise TypeError("Extraction is only supported for scalar columns.")

    column_name = column
    column: ScalarColumn = df[column]

    if train_mask is None:
        if missing == "nan":
            missing_mask = column.isna()
        else:
            missing_mask = column == missing
        train_df = df[~missing_mask]
    else:
        train_df = df[train_mask]

    if len(train_df) == 0:
        raise ValueError(
            "Could not find any training examples. "
            "Please specify ``train_mask`` with at least one non-zero entry."
        )

    # Prepare the in-context examples.
    in_ctx_examples = "\n".join(
        train_df.head().map(
            lambda row: _prepare_row(row, column=column_name, missing=False),
            inputs="row",
        )
    )

    # Make a new dataframe where we append the train mask.
    df = df.copy()
    df[_TRAIN_MASK_COLUMN] = train_mask

    return df.map(
        _impute.partial(column_name=column_name, in_ctx_examples=in_ctx_examples),
        inputs="row",
    )


def _impute(row, column_name: str, in_ctx_examples: str):
    """Impute a row."""
    if row[_TRAIN_MASK_COLUMN]:
        return row[column_name]

    return chat(
        in_ctx_examples + "\n" + _prepare_row(row, column=column_name, missing=True)
    )[0]
