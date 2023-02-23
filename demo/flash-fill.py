"""Flash-fill"""
from functools import partial

from manifest import Manifest

import meerkat as mk
from meerkat.dataframe import Batch

# manifest = Manifest(
#     client_name="huggingface",
#     client_connection="http://127.0.0.1:8010",
# )


def complete_prompt(row, example_template: mk.Store[str]):
    assert isinstance(row, dict)
    print("prompt template", example_template)
    output = example_template.format(**row)
    print(output)
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
        "note": ["", "", "", "", ""],
        # "_status": ["train", "train", "train", "fill", "fill"],
    }
)
df["note"] = ""

df = df.mark()


@mk.reactive()
def check_example_template(example_template: str, output_col: str):
    if not output_col:
        return
    if not example_template.endswith("{" + output_col + "}"):
        raise ValueError("The example template must end with '{" + output_col + "}'")


@mk.endpoint()
def run_manifest(instruct_cmd: str, df: mk.DataFrame, output_col: str):
    def _run_manifest(example: Batch):
        # Format instruct-example-instruct prompt.
        return ["Response test"] * len(example)
        return manifest.run(
            [f"{instruct_cmd} {in_context_examples} {x}" for x in example]
        )

    if output_col == "":
        raise ValueError("Please enter an output column")

    # Verify the example template ends with {output_col}.
    check_example_template(example_template, output_col)

    # Concat all of the in-context examples.
    train_df = df[df["_status"] == "train"]
    in_context_examples = "\n".join(train_df["example"]())

    print("in_context_examples", in_context_examples)

    fill_df_index = df["_status"] == "fill"
    fill_df = df[fill_df_index]

    # Filter based on train/abstain/fill
    # Only fill in the blanks.
    col = output_col
    flash_fill = mk.map(
        fill_df, function=_run_manifest, is_batched_fn=True, batch_size=4
    )

    print("flash fill", flash_fill)

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


output_col_area = mk.gui.Select(values=df.columns)
output_col = output_col_area.value

# show_prompts = mk.gui.Markdown("")


@mk.endpoint()
def set_code(code: mk.Store[str], new_code: str):
    code.set(new_code)


instruction_editor = mk.gui.Editor(code="give me the name of my guest")
example_template_editor = mk.gui.Editor(
    code="Guest name: {guest}, hometown: {hometown}; Result: {guest}"
)
# prompt_editor = mk.gui.Editor(code="Write the prompt here.")
instruction_cmd = instruction_editor.code
# mk.gui.print("Instruction commmand:", instruction_cmd)
# mk.gui.print("Example template:", example_template_editor.code)

example_template = example_template_editor.code

# Reactively check the value of the example template.
check_example_template(example_template, output_col)

df_view = update_df_with_example_template(df, example_template)


@mk.endpoint
def on_edit(df: mk.DataFrame, column: str, keyidx: any, posidx: int, value: any):
    df.loc[keyidx, column] = value
    print("updating")


# mk.gui.Gallery(df_view, main_column="guest")
table = mk.gui.Table(df_view, on_edit=on_edit.partial(df=df))

run_manifest_button = mk.gui.Button(
    title="Run Manifest",
    on_click=run_manifest.partial(
        instruct_cmd=instruction_cmd, df=df_view, output_col=output_col
    ),
)

# mk.gui.print("Prompt template here:", prompt_template)

page = mk.gui.Page(
    component=mk.gui.html.flexcol(
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
            mk.gui.html.div([table], classes="h-full w-screen"),
        ],
        classes="gap-4 h-screen",
    ),
    id="flash-fill",
)

page.launch()
