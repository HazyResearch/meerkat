from meerkat.columns.scalar import ScalarColumn
from meerkat.dataframe import DataFrame
from meerkat.engines.abstract import TextCompletion
from meerkat.row import Row


def _prepare_row(row: Row, output_column: str, is_test: bool):
    CELL_FORMAT = "{name}: {value}\n"

    row_str = "---begin row---\n"

    for name, value in row.items():
        if name == output_column:
            continue
        row_str += CELL_FORMAT.format(name=name, value=value)

    if not is_test:
        row_str += CELL_FORMAT.format(name=output_column, value=row[output_column])
    else:
        row_str += CELL_FORMAT.format(name=output_column, value="").rstrip()

    return row_str


def extract(
    df: DataFrame,
    *,
    annotated_df: DataFrame,
    column: str,
    to: str,
    engine: TextCompletion = None,
    schema: str = None
):
    """Extract information given some annotated examples.

    Args:
        df: The dataframe with values to extract.
        annotated_df: The dataframe with few-shot examples.
            This dataframe should contain both columns specified in
            ``from`` and ``to``.
        column: The column to extract information from.
        to: The column to extract information to.
        engine: The engine to use for extracting information.
        schema: The schema to validate the extraction against.
            This schema will not be used for prompting the language model.

    Return:

    """
    column_name = column
    if not isinstance(df[to], ScalarColumn):
        raise TypeError("Extraction is only supported for scalar columns.")

    # Annotated examples.
    annot_input = annotated_df[[column_name, to]]

    # Prepare the in-context training examples.
    in_ctx_examples = "\n".join(
        annot_input.map(
            lambda row: _prepare_row(row, output_column=to, is_test=False),
            inputs="row",
        )
    )

    # Run the LLM
    test_df = df[[column]]
    return test_df.map(
        lambda row: engine.run(
            prompt=in_ctx_examples + "\n" + _prepare_row(row, column=to, is_test=True)
        ),
        inputs="row",
    )


def extract_to_schema(
    column: ScalarColumn,
    schema: str,
    engine: TextCompletion = None,
) -> ScalarColumn:
    """Extract information from a column to a specified schema.

    Args:
        column: The column to extract information from.
        schema: The schema to tell the LLM to format the extraction into.
        engine: The engine to use for extracting information.

    Return:
        A column with the extracted information.
    """
    prompt = """
    Extract information from the TEXT to a SCHEMA.

    TEXT: {text}
    SCHEMA: {schema}
    """

    return column.map(
        lambda text: engine.run(prompt=prompt.format(text=text, schema=schema))
    )


def _run_llm(row, to: str, engine: TextCompletion, in_ctx_examples: str):
    """Run extraction for a single test row."""
    return engine.run(
        prompt=in_ctx_examples + "\n" + _prepare_row(row, column=to, is_test=True)
    )
