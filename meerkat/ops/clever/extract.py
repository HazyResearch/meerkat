import asyncio
from functools import partial
import json
from typing import Union

from meerkat import env
from meerkat.columns.scalar import ScalarColumn
from meerkat.dataframe import DataFrame
from meerkat.engines import TextCompletion
from meerkat.ops.amap import amap
from meerkat.row import Row

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
            prompt=in_ctx_examples
            + "\n"
            + _prepare_row(row, output_column=to, is_test=True)
        ),
        inputs="row",
    )


def extract_to_schema(
    column: ScalarColumn,
    schema: Union[str, Guard] = None,
    engine: TextCompletion = None,
    prompt: str = None,
    async_: bool = False,
    max_concurrent: int = 10,
) -> ScalarColumn:
    """Extract information from a column to a specified schema.

    Args:
        column: The column to extract information from.
        schema: The schema to tell the LLM to format the extraction into.
        engine: The engine to use for extracting information.
        prompt: The prompt to use for the LLM. If not specified, a default
            prompt will be used.

    Return:
        A column with the extracted information.
    """

    def _run(text):
        # clone = engine.clone() # Shouldn't need to do this. Keeping it here for now,
        # in case async map has issues.
        return schema(engine.run, prompt_params=dict(text=text))[0]

    if isinstance(schema, Guard):
        if not async_:
            return column.map(_run)
        return asyncio.run(amap(column, _run, max_concurrent=max_concurrent))

    if prompt is None:
        prompt = "Extract information from the TEXT to a SCHEMA.\n\nTEXT: {text}\nSCHEMA: {schema}\nAnswer:"

    formatter = partial(prompt.format, schema=schema)
    return column.map(lambda text: engine.run(prompt=formatter(text=text)))


def extraction_guard(schema: str) -> Guard:
    output = """
<output>
{schema}
</output>
""".format(
        schema=schema
    )

    prompt = """
<prompt>
Extract data from the given text into the schema below.
Text: {{text}}
@complete_json_suffix_v2\
</prompt>
"""

    rail = """\
<rail version="0.1">
{output}
{prompt}
</rail>
""".format(
        output=output, prompt=prompt
    )

    return gd.Guard.from_rail_string(rail)


def span_citation_guard() -> gd.Guard:
    schema = """
<list name="spans" description="The most relevant spans that contain the extracted information." format="atmost-top-3-spans">
    <string description="A very short span of text, containing atmost 5 words. The span must be verbatim, so that `assert (span in doc)`. If unsure, return an empty string."/>
</list>
"""

    output = """
<output>
{schema}
</output>
""".format(
        schema=schema
    )

    #     prompt = """
    # <prompt>
    # Output the most relevant spans in the text that are relevant to the information extracted.
    # Abstract: {{abstract}}

    # Extracted Information: {{description}} {{extracted}}
    # @complete_json_suffix_v2\
    # </prompt>
    # """
    prompt = """\
<prompt>
Abstract: {{abstract}}

Description of Information: {{description}}
Extracted Information: {{extracted}}

Output the 3 most relevant substrings VERBATIM from the abstract with atmost 5 words that contains the extracted info. ONLY OUTPUT THE VERBATIM SUBSTRINGS SUCH THAT python
assert extraction in substring WILL WORK. IF YOU ARE NOT CONFIDENT, RETURN AN EMPTY STRING.

List of Substrings (as list of substrings):\
</prompt>
"""

    rail = """\
<rail version="0.1">
{output}
{prompt}
</rail>
""".format(
        output=output, prompt=prompt
    )

    return gd.Guard.from_rail_string(rail)
