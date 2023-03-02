"""Flash-fill."""
import os
from functools import partial

from manifest import Manifest

import meerkat as mk
from meerkat.dataframe import Batch

manifest = Manifest(
    client_name="openai",
    client_connection=os.getenv("OPENAI_API_KEY"),
)


def complete_prompt(row, example_template: mk.Store[str]):
    assert isinstance(row, dict)
    output = example_template.format(**row)
    return output


df = mk.DataFrame(
    {
        "guest": [
            "Col. Mustard",
            "Mrs. White",
            "Mr. Green",
            "Mrs. Peacock",
            "Prof. Plum",
        ],
        "hometown": ["London", "Stockholm", "Bath", "New York", "Paris"],
        "relationship": ["father", "aunt", "cousin", "teacher", "brother"],
        "_status": ["train", "train", "train", "fill", "fill"],
    }
)


@mk.endpoint()
def run_manifest(instruct_cmd: str, df: mk.DataFrame, output_col: str):
    def _run_manifest(example: Batch):
        # Format instruct-example-instruct prompt.
        return manifest.run(
            [f"{instruct_cmd} {in_context_examples} {x}" for x in example]
        )

    # Concat all of the in-context examples.
    train_df = df[df["_status"] == "train"]
    in_context_examples = "\n".join(train_df["example"]())

    fill_df_index = df["_status"] == "fill"
    fill_df = df[fill_df_index]

    # Filter based on train/abstain/fill
    # Only fill in the blanks.
    col = output_col
    flash_fill = mk.map(
        fill_df, function=_run_manifest, is_batched_fn=True, batch_size=4
    )

    # If the dataframe does not have the column, add it.
    if col not in df.columns:
        df[col] = ""

    df[col][fill_df_index] = flash_fill
    df.set(df)


@mk.reactive()
def update_df_with_example_template(df: mk.DataFrame, template: mk.Store[str]):
    """Update the df with the new prompt template.

    This is as simple as returning a view.
    """
    df = df.view()

    # Extract out the example template from the prompt template.
    df["example"] = mk.defer(
        df, function=partial(complete_prompt, example_template=example_template)
    )
    return df


output_col_area = mk.gui.Textbox("", placeholder="Column name here...")
output_col = output_col_area.text


@mk.endpoint()
def set_code(code: mk.Store[str], new_code: str):
    code.set(new_code)


instruction_editor = mk.gui.Editor(code="give me the name of my guest")
example_template_editor = mk.gui.Editor(
    code="Guest name: {guest}, hometown: {hometown}; Result: {guest}"
)

instruction_cmd = instruction_editor.code
mk.gui.print("Instruction commmand:", instruction_cmd)
mk.gui.print("Example template:", example_template_editor.code)

example_template = example_template_editor.code

df_view = update_df_with_example_template(df, example_template)
table = mk.gui.Gallery(df_view, main_column="guest")

run_manifest_button = mk.gui.Button(
    title="Run Manifest",
    on_click=run_manifest.partial(
        instruct_cmd=instruction_cmd, df=df_view, output_col=output_col
    ),
)


page = mk.gui.Page(
    mk.gui.html.flexcol(
        [
            mk.gui.html.flex(
                [
                    mk.gui.Caption("Output Column"),
                    output_col_area,
                ],
                classes="items-center gap-4 mx-4",
            ),
            mk.gui.html.flex(
                [
                    # Make the items fill out the space.
                    mk.gui.html.flexcol(
                        [
                            mk.gui.Caption("Instruction Template"),
                            instruction_editor,
                        ],
                        classes="flex-1 gap-1",
                    ),
                    mk.gui.html.flexcol(
                        [
                            mk.gui.Caption("Example Template"),
                            example_template_editor,
                        ],
                        classes="flex-1 gap-1",
                    ),
                ],
                classes="w-full gap-4 mx-4",
            ),
            run_manifest_button,
            table,
        ],
        classes="gap-4",
    ),
    id="flash-fill",
)
page.launch()
