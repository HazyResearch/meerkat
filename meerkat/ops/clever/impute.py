from typing import Union

from meerkat.columns.scalar import ScalarColumn
from meerkat.dataframe import DataFrame
from meerkat.ops.clever.chat import chat
from meerkat.row import Row


def impute(df: DataFrame, column: str, missing: str = "nan"):
    """
    Impute missing values in a DataFrame.

    Use a large language model to guess what the missing value is for each
    row of a DataFrame.
    """

    # Strategies
    # Strategy 1: collect all the necessary information up front e.g.
    # names of possible categories.
    # Then you give the LLM a few example rows in-context to show the
    # mapping from rows to values in the column.
    # Then you ask the LLM to fill in the missing values for one or more rows.

    # What data types are we imputing with this? scalar column stuff
    # Strings
    # Numbers
    # Categoricals
    # Booleans
    # Floats
    # Dates
    if not isinstance(df[column], ScalarColumn):
        raise TypeError("Imputation is only supported for scalar columns.")
    column_name = column
    column: ScalarColumn = df[column]

    if missing == "nan":
        missing_mask = column.isna()
    else:
        missing_mask = column == missing

    train_df = df[~missing_mask]

    in_ctx_examples = "\n".join(
        train_df.head().map(
            lambda row: _prepare_row(row, column=column_name, missing=False),
            inputs="row",
        )
    )

    return df.map(
        (
            lambda row: (
                chat(
                    in_ctx_examples
                    + "\n"
                    + _prepare_row(row, column=column_name, missing=True)
                )[0]
            )
            if row[column_name] == missing
            else row[column_name]
        ),
        inputs="row",
    )


def _prepare_row(row: Row, column: str, missing: bool = True):
    CELL_FORMAT = "{name}: {value}\n"

    row_str = "---begin row---\n"

    for name, value in row.items():
        if name == column:
            continue
        row_str += CELL_FORMAT.format(name=name, value=value)

    if not missing:
        row_str += CELL_FORMAT.format(name=column, value=row[column])
    else:
        row_str += CELL_FORMAT.format(name=column, value="")

    return row_str
