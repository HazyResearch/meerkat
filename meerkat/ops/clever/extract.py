from functools import partial
from typing import Union
from meerkat.columns.scalar import ScalarColumn
from meerkat.dataframe import DataFrame
from meerkat.engines.abstract import TextCompletion
from meerkat.row import Row
from meerkat import env

if env.package_available("guardrails"):
    import guardrails as gd
    Guard = gd.Guard
else:
    Guard = None


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
            prompt=in_ctx_examples + "\n" + _prepare_row(row, output_column=to, is_test=True)
        ),
        inputs="row",
    )


def extract_to_schema(
    column: ScalarColumn,
    schema: Union[str, Guard] = None,
    engine: TextCompletion = None,
    prompt: str = None,
) -> ScalarColumn:
    """Extract information from a column to a specified schema.

    Args:
        column: The column to extract information from.
        schema: The schema to tell the LLM to format the extraction into.
        engine: The engine to use for extracting information.

    Return:
        A column with the extracted information.
    """
    if isinstance(schema, Guard):
        return column.map(
            lambda text: schema(engine.run, prompt_params=dict(text=text))[0]
        )

    if prompt is None:
        prompt = "Extract information from the TEXT to a SCHEMA.\n\nTEXT: {text}\nSCHEMA: {schema}\nAnswer:"
    
    formatter = partial(prompt.format, schema=schema)
    return column.map(
        lambda text: engine.run(prompt=formatter(text=text))
    )
