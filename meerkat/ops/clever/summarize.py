from meerkat.dataframe import DataFrame
from meerkat.engines import TextCompletion
from meerkat.row import Row


def summary_prompt_one_sentence():
    """A prompt for summarizing text."""
    return """\
Summarize the following text in one sentence.
Text: {text}
Summary:\
"""


def summary_prompt_one_paragraph():
    """A prompt for summarizing text."""
    return """\
Summarize the following text in one paragraph.
Text: {text}
Summary:\
"""


def summarize(
    df: DataFrame,
    *,
    column: str,
    engine: TextCompletion,
):
    """Summarize every row in a column."""

    def _summarize(row: Row):
        return engine.run(prompt=summary_prompt_one_sentence().format(text=row[column]))

    return df.map(_summarize, inputs="row")
